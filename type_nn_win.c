#include "type_nn_win.h"
#include "type_nn_layerkit.h"

/*
 * type-nn-win — And/Or net whose shape is moved by back-prop, not by
 * a dataset table.
 *
 * Start: one product layer (k = 2). That is the Type Mechanics prior
 * (rank-2 is the first non-affine), not a UCI special case.
 *
 * After the EMA of |dY| and per-layer error energy ge[i] / activity
 * ae[i] have had a short warmup, at most one move per SETTLE steps:
 *
 *   drop  an idle hidden map (k=1 and W≈I, tiny ge and ae)
 *   drop  a dead extra Or factor (k>2, ||W||² tiny)
 *   grow  width of the k=1 layer with the largest ge, if that ge is
 *         still high *and* the last grow actually lowered it
 *   grow  k of a product that is still carrying error
 *   add   a layer at the interface with the largest site score
 *
 * Site i = "before layer i"; site `depth` = after the tail.
 *   front of a product  → k=1 basis, width = current tail out (then grow)
 *   after a product     → k=1 readout
 *   anywhere else       → identity, so the mapping does not jump
 *
 * No `if (in < 4)`, no `tick == 16`, no "commit k=1/k=2/k=1 and lock".
 * XOR stays depth 1 because its residual falls through the floor
 * before a stall is declared. A wide file keeps growing H or depth
 * while ge stays high.
 */

#define AD_B1   0.9
#define AD_B2   0.999
#define AD_EPS  1e-8
#define SETTLE  48
#define WARMUP  32
#define FREEZE  24
#define EMA_FLOOR 0.04
#define GE_FLOOR  0.03
#define GE_GROW   0.08
#define ID_TAU    0.08
#define W_PRUNE   0.03
#define MAX_H     32
#define MAX_K     4

typedef struct {
    LKLayer L;
    double *mW, *vW, *mb, *vb, *aW, *ab;
    int freeze_col, pending;
    double ge, ae;
} PLayer;

typedef struct {
    PLayer  layer[LK_MAX_DEPTH];
    double *act[LK_MAX_DEPTH + 1];
    size_t  act_w[LK_MAX_DEPTH + 1];
    size_t  depth;
    int     dynamic;
    unsigned n_add, n_drop;
    unsigned tick, last_change;
    double  ema, ema_ref, ge_ref;
    int     refuse_insert;
    int     refuse_grow;
    unsigned long tstep;
    double  b1p, b2p;
} PNet;

static void psync(PLayer *P)
{
    size_t n = P->L.n_or, in = P->L.in;
    P->mW = (double *)realloc(P->mW, n * in * sizeof(double));
    P->vW = (double *)realloc(P->vW, n * in * sizeof(double));
    P->mb = (double *)realloc(P->mb, n * sizeof(double));
    P->vb = (double *)realloc(P->vb, n * sizeof(double));
    P->aW = (double *)realloc(P->aW, n * in * sizeof(double));
    P->ab = (double *)realloc(P->ab, n * sizeof(double));
    if (n && in) {
        memset(P->mW, 0, n * in * sizeof(double));
        memset(P->vW, 0, n * in * sizeof(double));
        memset(P->aW, 0, n * in * sizeof(double));
    }
    if (n) {
        memset(P->mb, 0, n * sizeof(double));
        memset(P->vb, 0, n * sizeof(double));
        memset(P->ab, 0, n * sizeof(double));
    }
    P->pending = 0;
}

static void palloc(PLayer *P, size_t in, size_t out, size_t k)
{
    lk_alloc(&P->L, in, out, k);
    psync(P);
    P->freeze_col = 0;
    P->ge = P->ae = 0.0;
}

static void pfree(PLayer *P)
{
    lk_free(&P->L);
    free(P->mW); free(P->vW); free(P->mb); free(P->vb); free(P->aW); free(P->ab);
    memset(P, 0, sizeof(*P));
}

static void fit(PNet *N, size_t s, size_t w)
{
    if (N->act_w[s] >= w && N->act[s]) return;
    free(N->act[s]);
    N->act[s] = (double *)calloc(w ? w : 1, sizeof(double));
    N->act_w[s] = w ? w : 1;
}

static double ad(double *m, double *v, double g, double lr, double b1p, double b2p)
{
    *m = AD_B1 * *m + (1.0 - AD_B1) * g;
    *v = AD_B2 * *v + (1.0 - AD_B2) * g * g;
    return lr * (*m / (1.0 - b1p)) / (sqrt(*v / (1.0 - b2p)) + AD_EPS);
}

static void flush_layer(PNet *N, PLayer *P, double lr)
{
    if (P->pending <= 0) return;
    LKLayer *L = &P->L;
    const size_t in = L->in;
    double inv = 1.0 / (double)P->pending;
    const double decay = 1e-4;
    N->tstep++;
    N->b1p *= AD_B1;
    N->b2p *= AD_B2;
    for (size_t r = 0; r < L->n_or; r++) {
        double gb = P->ab[r] * inv;
        L->b[r] = tnn_clamp(L->b[r] - ad(&P->mb[r], &P->vb[r], gb, lr, N->b1p, N->b2p),
                            -TNN_WCLIP, TNN_WCLIP);
        P->ab[r] = 0.0;
        for (size_t j = 0; j < in; j++) {
            if (P->freeze_col && j + 1 == in) continue;
            double gw = P->aW[r * in + j] * inv + decay * L->W[r * in + j];
            L->W[r * in + j] = tnn_clamp(
                L->W[r * in + j] - ad(&P->mW[r * in + j], &P->vW[r * in + j],
                                      gw, lr, N->b1p, N->b2p),
                -TNN_WCLIP, TNN_WCLIP);
            P->aW[r * in + j] = 0.0;
        }
    }
    P->pending = 0;
}

static void upd(PNet *N, PLayer *P, const double *x, const double *dy,
                double *dx, double lr)
{
    LKLayer *L = &P->L;
    const size_t in = L->in, k = L->k, out = L->out;
    if (dx) memset(dx, 0, in * sizeof(double));
    double dor[8];
    double e = 0.0;
    for (size_t i = 0; i < out; i++) {
        e += fabs(dy[i]);
        lk_dor(L, i, dy[i], dor);
        for (size_t t = 0; t < k; t++) {
            double g = dor[t];
            size_t r = i * k + t;
            if (dx) {
                const double *w = L->W + r * in;
                for (size_t j = 0; j < in; j++) dx[j] += g * w[j];
            }
            P->ab[r] += g;
            for (size_t j = 0; j < in; j++)
                P->aW[r * in + j] += g * x[j];
        }
    }
    e /= (double)(out ? out : 1);
    P->ge = (P->ge == 0.0) ? e : (0.9 * P->ge + 0.1 * e);
    P->pending++;
    if (P->pending >= 16) flush_layer(N, P, lr);
}

static size_t seed_width(size_t out)
{
    return out < 2 ? 2 : out;
}

static int is_identity_layer(const PLayer *P)
{
    const LKLayer *L = &P->L;
    if (L->k != 1 || L->in != L->out) return 0;
    for (size_t i = 0; i < L->out; i++) {
        for (size_t j = 0; j < L->in; j++) {
            double want = (i == j) ? 1.0 : 0.0;
            if (fabs(L->W[i * L->in + j] - want) > ID_TAU) return 0;
        }
    }
    return 1;
}

/* kind: 1 insert, 2 grow, 0 drop/other */
static void mark_change(PNet *N, int kind)
{
    if (kind == 1 && N->ema_ref > 0.0 && N->ema >= 0.98 * N->ema_ref)
        N->refuse_insert = 1;
    if (kind == 0) {
        N->refuse_insert = 0;
        N->refuse_grow = 0;
    }
    N->last_change = N->tick;
    N->ema_ref = N->ema;
    N->ge_ref = 0.0;
    for (size_t i = 0; i < N->depth; i++)
        if (N->layer[i].ge > N->ge_ref) N->ge_ref = N->layer[i].ge;
}

static void insert_at(PNet *N, size_t idx)
{
    if (N->depth >= LK_MAX_DEPTH) return;
    if (idx > N->depth) idx = N->depth;

    if (idx == 0 && N->layer[0].L.k >= 2) {
        size_t in = N->layer[0].L.in;
        size_t H = seed_width(N->layer[N->depth - 1].L.out);
        memmove(&N->layer[1], &N->layer[0], N->depth * sizeof(PLayer));
        memset(&N->layer[0], 0, sizeof(PLayer));
        palloc(&N->layer[0], in, H, 1);
        lk_resize_in(&N->layer[1].L, H);
        psync(&N->layer[1]);
        N->layer[1].freeze_col = FREEZE;
        N->depth++;
        N->n_add++;
        mark_change(N, 1);
        return;
    }

    if (idx == N->depth && N->layer[N->depth - 1].L.k >= 2) {
        PLayer *tail = &N->layer[N->depth - 1];
        size_t o = tail->L.out;
        size_t feat = seed_width(o);
        lk_resize_out(&tail->L, feat);
        psync(tail);
        palloc(&N->layer[N->depth], feat, o, 1);
        N->layer[N->depth].freeze_col = FREEZE;
        N->depth++;
        N->n_add++;
        mark_change(N, 1);
        return;
    }

    size_t dim = (idx < N->depth) ? N->layer[idx].L.in
                                  : N->layer[N->depth - 1].L.out;
    if (idx < N->depth) {
        memmove(&N->layer[idx + 1], &N->layer[idx],
                (N->depth - idx) * sizeof(PLayer));
    }
    memset(&N->layer[idx], 0, sizeof(PLayer));
    palloc(&N->layer[idx], dim, dim, 1);
    lk_identity(&N->layer[idx].L);
    psync(&N->layer[idx]);
    N->layer[idx].freeze_col = FREEZE;
    N->depth++;
    N->n_add++;
    mark_change(N, 1);
}

static int remove_idx(PNet *N, size_t idx)
{
    if (N->depth < 2 || idx >= N->depth) return -1;
    size_t nin = (idx == 0) ? N->layer[0].L.in : N->layer[idx].L.in;
    pfree(&N->layer[idx]);
    if (idx + 1 < N->depth)
        memmove(&N->layer[idx], &N->layer[idx + 1],
                (N->depth - idx - 1) * sizeof(PLayer));
    memset(&N->layer[N->depth - 1], 0, sizeof(PLayer));
    N->depth--;
    if (idx < N->depth) {
        lk_resize_in(&N->layer[idx].L, nin);
        psync(&N->layer[idx]);
    }
    N->n_drop++;
    mark_change(N, 0);
    return 0;
}

static int drop_idle_hidden(PNet *N)
{
    if (N->depth < 2) return 0;
    for (size_t i = 0; i + 1 < N->depth; i++) {
        PLayer *P = &N->layer[i];
        if (P->freeze_col > 0) continue;
        if (!is_identity_layer(P)) continue;
        if (P->ge > GE_FLOOR || P->ae > 0.15) continue;
        return remove_idx(N, i) == 0;
    }
    return 0;
}

static int drop_dead_or(PNet *N)
{
    for (size_t i = 0; i < N->depth; i++) {
        LKLayer *L = &N->layer[i].L;
        if (L->k <= 2) continue;
        int any = 0;
        for (size_t o = 0; o < L->out; o++) {
            double *w = L->W + ((o * L->k) + (L->k - 1)) * L->in;
            double n2 = 0.0;
            for (size_t j = 0; j < L->in; j++) n2 += w[j] * w[j];
            if (n2 < 0.01) any = 1;
        }
        if (!any) continue;
        lk_set_k(L, L->k - 1);
        psync(&N->layer[i]);
        N->n_drop++;
        mark_change(N, 0);
        return 1;
    }
    return 0;
}

static int grow_width(PNet *N)
{
    int best = -1;
    double best_ge = GE_GROW;
    /* Never widen the tail: the caller owns AltNet.out. */
    size_t hid_end = N->depth ? N->depth - 1 : 0;
    for (size_t i = 0; i < hid_end; i++) {
        if (N->layer[i].L.k != 1) continue;
        if (N->layer[i].L.out >= MAX_H) continue;
        /* A k=1 map of width > in is redundant (column space). */
        if (N->layer[i].L.out >= N->layer[i].L.in) continue;
        if (N->layer[i].ge > best_ge) {
            best_ge = N->layer[i].ge;
            best = (int)i;
        }
    }
    if (best < 0) return 0;
    PLayer *P = &N->layer[best];
    size_t nout = P->L.out + 1;
    lk_resize_out(&P->L, nout);
    psync(P);
    if ((size_t)best + 1 < N->depth) {
        lk_resize_in(&N->layer[best + 1].L, nout);
        psync(&N->layer[best + 1]);
    }
    mark_change(N, 2);
    return 1;
}

static int grow_k(PNet *N)
{
    int best = -1;
    double best_ge = GE_FLOOR;
    for (size_t i = 0; i < N->depth; i++) {
        if (N->layer[i].L.k < 2 || N->layer[i].L.k >= MAX_K) continue;
        if (N->layer[i].ge > best_ge) {
            best_ge = N->layer[i].ge;
            best = (int)i;
        }
    }
    if (best < 0) return 0;
    lk_set_k(&N->layer[best].L, N->layer[best].L.k + 1);
    psync(&N->layer[best]);
    mark_change(N, 0);
    return 1;
}

static size_t pick_site(const PNet *N, double *score_out)
{
    size_t best = N->depth;
    double best_s = -1.0;
    for (size_t i = 0; i <= N->depth; i++) {
        double s = 0.0;
        if (i == 0) {
            const PLayer *L0 = &N->layer[0];
            /* Product of raw coordinates is the expensive case. */
            if (L0->L.k >= 2)
                s = L0->ge * 2.0 + 0.05 * (double)L0->L.in;
            else
                s = L0->ge * 0.2;
        } else if (i == N->depth) {
            const PLayer *T = &N->layer[N->depth - 1];
            if (T->L.k >= 2)
                s = T->ge * 1.5 + 0.05 * (double)T->L.out;
            else
                s = T->ge * 0.2;
        } else {
            const PLayer *U = &N->layer[i - 1];
            const PLayer *D = &N->layer[i];
            double ratio = U->ge / (D->ae + 0.05);
            s = 0.5 * (U->ge + D->ge) * ratio;
        }
        if (s > best_s) { best_s = s; best = i; }
    }
    if (score_out) *score_out = best_s;
    return best;
}

static void prune_idle_weights(PNet *N)
{
    for (size_t i = 0; i < N->depth; i++) {
        if (N->layer[i].freeze_col > 0) continue;
        LKLayer *L = &N->layer[i].L;
        for (size_t t = 0; t < L->n_or * L->in; t++) {
            if (fabs(L->W[t]) < W_PRUNE && fabs(N->layer[i].mW[t]) < 1e-4)
                L->W[t] = 0.0;
        }
    }
}

static void estimate_and_apply(PNet *N)
{
    if (!N->dynamic) return;
    if (N->tick < WARMUP) return;
    if (N->last_change && N->tick < N->last_change + SETTLE) return;

    /* Prefer shrink. */
    if (drop_idle_hidden(N)) return;
    if (drop_dead_or(N)) return;

    int stalled = (N->ema > 0.95 * (N->ema_ref > 0.0 ? N->ema_ref : N->ema))
                  && (N->ema > EMA_FLOOR);

    if (!N->refuse_grow && grow_width(N)) return;

    if (!stalled) {
        if (N->ema < EMA_FLOOR) prune_idle_weights(N);
        return;
    }

    /* One move: the highest-scoring axis. */
    double site_s = 0.0;
    size_t site = pick_site(N, &site_s);

    double grow_s = 0.0;
    size_t last = N->depth ? N->depth - 1 : 0;
    for (size_t i = 0; i < last; i++)
        if (N->layer[i].L.k == 1 && N->layer[i].L.out < MAX_H
            && N->layer[i].ge > grow_s)
            grow_s = N->layer[i].ge * 1.1;

    double k_s = 0.0;
    for (size_t i = 0; i < N->depth; i++)
        if (N->layer[i].L.k >= 2 && N->layer[i].L.k < MAX_K
            && N->layer[i].ge > k_s)
            k_s = N->layer[i].ge * 0.7;

    if (grow_s >= site_s && grow_s >= k_s && grow_s > GE_FLOOR) {
        if (grow_width(N)) return;
    }
    if (k_s >= site_s && k_s > GE_FLOOR) {
        if (grow_k(N)) return;
    }
    if (site_s > GE_FLOOR && N->depth < LK_MAX_DEPTH && !N->refuse_insert)
        insert_at(N, site);
}

static void net_init(void *c)
{
    PNet *N = c;
    N->tstep = 0; N->b1p = N->b2p = 1.0;
    N->tick = N->last_change = 0;
    N->ema = N->ema_ref = N->ge_ref = 0.0;
    N->n_add = N->n_drop = 0;
    N->refuse_insert = 0;
    for (size_t i = 0; i < N->depth; i++) {
        LKLayer *L = &N->layer[i].L;
        double s = 0.15 / sqrt((double)(L->in > 0 ? L->in : 1));
        for (size_t r = 0; r < L->n_or; r++) {
            L->b[r] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * s;
            for (size_t j = 0; j < L->in; j++)
                L->W[r * L->in + j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * s;
        }
        N->layer[i].ge = N->layer[i].ae = 0.0;
    }
}

static void net_fwd(void *c, const double *x, double *y)
{
    PNet *N = c;
    const double *cur = x;
    for (size_t i = 0; i < N->depth; i++) {
        fit(N, i + 1, N->layer[i].L.out);
        lk_fwd(&N->layer[i].L, cur, N->act[i + 1]);
        double a = 0.0;
        size_t o = N->layer[i].L.out;
        for (size_t j = 0; j < o; j++) a += fabs(N->act[i + 1][j]);
        a /= (double)(o ? o : 1);
        N->layer[i].ae = (N->layer[i].ae == 0.0) ? a : (0.9 * N->layer[i].ae + 0.1 * a);
        cur = N->act[i + 1];
    }
    memcpy(y, cur, N->layer[N->depth - 1].L.out * sizeof(double));
}

static void net_bwd(void *c, const double *x, const double *dy, double lr)
{
    PNet *N = c;
    if (!N->act[1]) net_fwd(c, x, N->act[N->depth]);
    if (lr <= 0.0) lr = 0.01;
    const double *dcur = dy;
    double *hold = NULL;
    for (size_t i = N->depth; i-- > 0; ) {
        const double *xin = i ? N->act[i] : x;
        double *dx = NULL;
        if (i) { fit(N, 0, N->layer[i].L.in); dx = N->act[0]; }
        upd(N, &N->layer[i], xin, dcur, dx, lr);
        if (i) {
            size_t w = N->layer[i].L.in;
            hold = (double *)realloc(hold, w * sizeof(double));
            memcpy(hold, dx, w * sizeof(double));
            dcur = hold;
        }
    }
    free(hold);
    double mag = 0.0;
    size_t o = N->layer[N->depth - 1].L.out;
    for (size_t i = 0; i < o; i++) mag += fabs(dy[i]);
    N->tick++;
    N->ema = (N->tick == 1) ? mag : (0.95 * N->ema + 0.05 * mag);
    if (N->ema_ref == 0.0 && N->tick == WARMUP) N->ema_ref = N->ema;
    for (size_t i = 0; i < N->depth; i++)
        if (N->layer[i].freeze_col > 0) N->layer[i].freeze_col--;
    estimate_and_apply(N);
}

static void net_align(void *c, size_t in)
{
    PNet *N = c;
    lk_resize_in(&N->layer[0].L, in);
    psync(&N->layer[0]);
}
static void net_out(void *c, size_t o)
{
    PNet *N = c;
    lk_resize_out(&N->layer[N->depth - 1].L, o);
    psync(&N->layer[N->depth - 1]);
}
static void net_k(void *c, size_t k)
{
    PNet *N = c;
    if (k < 1) k = 1;
    lk_set_k(&N->layer[N->depth - 1].L, k);
    psync(&N->layer[N->depth - 1]);
}
static void net_ins(void *c)
{
    PNet *N = c;
    if (N->depth >= LK_MAX_DEPTH) return;
    PLayer *tail = &N->layer[N->depth - 1];
    size_t dim = tail->L.in;
    memmove(&N->layer[N->depth], tail, sizeof(PLayer));
    memset(tail, 0, sizeof(PLayer));
    palloc(tail, dim, dim, TNN_K0);
    lk_identity(&tail->L);
    psync(tail);
    N->depth++;
}
static int net_rem(void *c)
{
    PNet *N = c;
    if (N->depth < 2) return -1;
    return remove_idx(N, 0);
}
static void net_dyn(void *c, int on) { ((PNet *)c)->dynamic = on; }
static size_t net_depth(void *c) { return ((PNet *)c)->depth; }
static size_t net_kf(void *c)
{
    PNet *N = c;
    return N->layer[N->depth - 1].L.k;
}
static size_t net_params(void *c)
{
    PNet *N = c;
    size_t n = 0;
    for (size_t i = 0; i < N->depth; i++) {
        LKLayer *L = &N->layer[i].L;
        n += L->n_or * (L->in + 1);
    }
    return n;
}
static size_t net_nbytes(void *c)
{
    return net_params(c) * sizeof(double);
}
static void net_free(void *c)
{
    PNet *N = c;
    for (size_t i = 0; i < N->depth; i++) pfree(&N->layer[i]);
    for (size_t i = 0; i <= LK_MAX_DEPTH; i++) free(N->act[i]);
    free(N);
}
static void net_scale(void *c, size_t idx, size_t in, size_t out)
{
    PNet *N = c;
    if (idx >= N->depth) return;
    lk_resize_in(&N->layer[idx].L, in);
    lk_resize_out(&N->layer[idx].L, out);
    psync(&N->layer[idx]);
    if (idx + 1 < N->depth) {
        lk_resize_in(&N->layer[idx + 1].L, out);
        psync(&N->layer[idx + 1]);
    }
}
static size_t net_lin(void *c, size_t idx)
{
    PNet *N = c;
    return idx < N->depth ? N->layer[idx].L.in : 0;
}
static size_t net_lout(void *c, size_t idx)
{
    PNet *N = c;
    return idx < N->depth ? N->layer[idx].L.out : 0;
}
static size_t net_lk(void *c, size_t idx)
{
    PNet *N = c;
    return idx < N->depth ? N->layer[idx].L.k : 0;
}
static size_t net_nadd(void *c) { return ((PNet *)c)->n_add; }
static size_t net_ndrop(void *c) { return ((PNet *)c)->n_drop; }

AltNet type_nn_win_open(size_t in, size_t out)
{
    PNet *N = (PNet *)calloc(1, sizeof(PNet));
    N->dynamic = 1;
    N->b1p = N->b2p = 1.0;
    palloc(&N->layer[0], in, out, TNN_K0);
    N->depth = 1;
    AltNet h = {
        .impl = "type-nn-win", .ctx = N, .in = in, .out = out,
        .init = net_init, .forward = net_fwd, .backward = net_bwd,
        .align_inputs = net_align, .set_outputs = net_out,
        .set_or_factors = net_k, .insert_identity = net_ins,
        .remove_hidden = net_rem, .set_dynamic = net_dyn,
        .depth = net_depth, .or_factors = net_kf,
        .param_count = net_params, .nbytes = net_nbytes, .free = net_free,
        .scale_layer = net_scale, .layer_in = net_lin, .layer_out = net_lout,
        .layer_k = net_lk, .n_add = net_nadd, .n_drop = net_ndrop
    };
    return h;
}
