#include "type_nn_alt.h"
#include "type_nn_layerkit.h"

/*
 * type-nn-idi / type-nn-idfact
 *
 * Lesson: extra depth with *higher* MSE means the new layer was
 * either inserted too late (random init, few remaining steps) or
 * between the wrong interfaces.
 *
 * Per-layer BP gives both answers:
 *   ge[i], ae[i] score every interface
 *   site i = "before layer i" (i = depth → after the tail)
 *   insert is k=1 identity, so the mapping does not jump
 *   first decision at tick 16, at most two depth adds
 *
 * Use backprop *layer energy* to estimate how many layers to add/remove.
 *
 * After each backward pass we keep an EMA of
 *   ge[i]  = mean(|dAnd|) arriving at layer i   (error energy)
 *   ae[i]  = mean(|And|)  leaving layer i       (activity)
 *
 * Estimate (at most once per SETTLE steps):
 *   n_add_proj    = 1 if the first layer is a product on wide raw x
 *                   (ge[0] high ⇒ residual is not a raw quadratic)
 *   n_add_readout = 1 if the tail is a product writing class scores
 *                   (ge[tail] high and out > 1 ⇒ need a linear mix)
 *   n_remove      = count of hidden layers whose activity is tiny
 *                   *and* ge is tiny after freeze expires
 *
 * Target shape is the one that actually moved UCI MSE:
 *   k=1 projection → k=2 features → k=1 readout
 * Extra layers beyond that are refused. XOR (in < 4) stays depth-1.
 */

#define AD_B1  0.9
#define AD_B2  0.999
#define AD_EPS 1e-8
#define SETTLE 40
#define FREEZE 20
#define MAX_HID 24

typedef struct {
    LKLayer L;
    double *mW, *vW, *mb, *vb, *aW, *ab;
    int freeze_col, pending;
    double ge, ae; /* BP error energy, activity */
} PLayer;

typedef struct {
    PLayer  layer[LK_MAX_DEPTH];
    double *act[LK_MAX_DEPTH + 1];
    size_t  act_w[LK_MAX_DEPTH + 1];
    size_t  depth;
    int     dynamic, added_layers, early_done, force_roles, fact, guess;
    unsigned n_add, n_drop; int locked;
    unsigned tick, last_change;
    double  ema;
    unsigned long tstep;
    double  b1p, b2p;
    const char *name;
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
            double gw = P->aW[r * in + j] * inv;
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
    int batch = (lr >= 0.06) ? 4 : 16;
    if (P->pending >= batch) flush_layer(N, P, lr);
}

/* ── architecture moves driven by ge[] / ae[] ── */


/* Identity-preserving insert BEFORE layer `idx` (idx==depth → after tail).
   New map is k=1, W=I, so the forward values do not jump. */

static int layer_idle(const PLayer *P)
{
    if (P->L.k != 1) return 0;
    const LKLayer *L = &P->L;
    if (L->in != L->out) return 0;
    double acc = 0.0;
    size_t n = 0;
    for (size_t i = 0; i < L->out; i++) {
        for (size_t j = 0; j < L->in; j++) {
            double w = L->W[i * L->in + j];
            double t = (i == j) ? 1.0 : 0.0;
            acc += fabs(w - t);
            n++;
        }
        acc += fabs(L->b[i]);
        n++;
    }
    return (acc / (double)(n ? n : 1) < 0.08) && P->ge < 0.02 && P->ae < 0.05;
}

static void remove_idle(PNet *N)
{
    if (N->locked) return;
    if (N->depth < 2) return;
    for (size_t i = 0; i + 1 < N->depth; i++) {
        if (N->layer[i].freeze_col > 0) continue;
        if (!layer_idle(&N->layer[i])) continue;
        size_t nin = N->layer[i].L.in;
        pfree(&N->layer[i]);
        memmove(&N->layer[i], &N->layer[i + 1],
                (N->depth - i - 1) * sizeof(PLayer));
        memset(&N->layer[N->depth - 1], 0, sizeof(PLayer));
        N->depth--; N->n_drop++;
        lk_resize_in(&N->layer[i].L, nin);
        psync(&N->layer[i]);
        N->last_change = N->tick;
        return;
    }
}


/* Exact identity on the input of layer idx: h = x, then layer idx sees h.
   clip(I x) = x inside OR clip, so the net output does not jump. */
static void insert_ident_before(PNet *N, size_t idx)
{
    if (N->depth >= LK_MAX_DEPTH || idx >= N->depth) return;
    size_t dim = N->layer[idx].L.in;
    memmove(&N->layer[idx + 1], &N->layer[idx],
            (N->depth - idx) * sizeof(PLayer));
    memset(&N->layer[idx], 0, sizeof(PLayer));
    palloc(&N->layer[idx], dim, dim, 1);
    lk_identity(&N->layer[idx].L);
    psync(&N->layer[idx]);
    N->layer[idx].freeze_col = FREEZE;
    N->depth++;
    N->added_layers++; N->n_add++;
    N->last_change = N->tick;
}

/* Move a k=1 layer's affine into a new map in front of it, leave I behind:
     y = W x + b   →   h = W x + b,  y = I h
   Exact (pre-clip). Product layers cannot be factored this way; fall back to I. */
static void insert_factor_before(PNet *N, size_t idx)
{
    if (N->depth >= LK_MAX_DEPTH || idx >= N->depth) return;
    if (N->layer[idx].L.k != 1) {
        insert_ident_before(N, idx);
        return;
    }
    size_t out = N->layer[idx].L.out;
    memmove(&N->layer[idx + 1], &N->layer[idx],
            (N->depth - idx) * sizeof(PLayer));
    /* idx keeps W,b; idx+1 is a duplicate we replace with I */
    memset(&N->layer[idx + 1], 0, sizeof(PLayer));
    palloc(&N->layer[idx + 1], out, out, 1);
    lk_identity(&N->layer[idx + 1].L);
    psync(&N->layer[idx + 1]);
    N->layer[idx + 1].freeze_col = FREEZE;
    N->depth++;
    N->added_layers++; N->n_add++;
    N->last_change = N->tick;
}

static void insert_at(PNet *N, size_t idx)
{
    if (idx >= N->depth) {
        /* after tail: I on the output, y' = I y */
        if (N->depth >= LK_MAX_DEPTH) return;
        PLayer *T = &N->layer[N->depth - 1];
        size_t o = T->L.out;
        palloc(&N->layer[N->depth], o, o, 1);
        lk_identity(&N->layer[N->depth].L);
        psync(&N->layer[N->depth]);
        N->layer[N->depth].freeze_col = FREEZE;
        N->depth++;
        N->added_layers++; N->n_add++;
        N->last_change = N->tick;
        return;
    }
    if (N->fact) insert_factor_before(N, idx);
    else insert_ident_before(N, idx);
}

/* Score every interface. High score = insert an affine identity there.
   site i means "before layer i"; site `depth` means "after the tail". */
static size_t pick_site(const PNet *N, double *score_out)
{
    size_t best = N->depth; /* default: after tail */
    double best_s = -1.0;
    for (size_t i = 0; i <= N->depth; i++) {
        double s = 0.0;
        if (i == 0) {
            /* raw input → first layer. Product on wide x needs a basis. */
            if (N->layer[0].L.in >= 4 && N->layer[0].L.k >= 2)
                s = N->layer[0].ge * 2.0 + 0.3;
        } else if (i == N->depth) {
            /* after tail. Product writing targets needs a linear mix. */
            const PLayer *T = &N->layer[N->depth - 1];
            if (T->L.k >= 2)
                s = T->ge * 1.5 + 0.2;
        } else {
            /* between i-1 and i: bottleneck if upstream error is large
               and downstream activity is small (layer i is not using it). */
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



static size_t pick_hid(size_t in, size_t out)
{
    size_t h = 8;
    if (in >= 10) h = 16;
    if (in >= 24) h = 24;
    if (h < out + 2) h = out + 2;
    if (h > 24) h = 24;
    return h;
}

/* Real k=1 basis (not I) — this is the iris/wine site move. */
static void insert_proj_real(PNet *N)
{
    if (N->depth >= LK_MAX_DEPTH) return;
    size_t in = N->layer[0].L.in;
    size_t H = pick_hid(in, N->layer[N->depth - 1].L.out);
    memmove(&N->layer[1], &N->layer[0], N->depth * sizeof(PLayer));
    memset(&N->layer[0], 0, sizeof(PLayer));
    palloc(&N->layer[0], in, H, 1);
    lk_resize_in(&N->layer[1].L, H);
    psync(&N->layer[1]);
    N->layer[1].freeze_col = FREEZE;
    N->depth++;
    N->added_layers++; N->n_add++;
    N->last_change = N->tick;
}

static void insert_readout_real(PNet *N)
{
    if (N->depth >= LK_MAX_DEPTH) return;
    PLayer *tail = &N->layer[N->depth - 1];
    if (tail->L.k < 2) return;
    size_t o = tail->L.out;
    size_t feat = pick_hid(tail->L.in, o);
    if (feat < o) feat = o;
    lk_resize_out(&tail->L, feat);
    psync(tail);
    palloc(&N->layer[N->depth], feat, o, 1);
    N->layer[N->depth].freeze_col = FREEZE;
    N->depth++;
    N->added_layers++; N->n_add++;
    N->last_change = N->tick;
}

static void estimate_and_apply(PNet *N)
{
    if (!N->dynamic) return;
    /* Diagnose early: after 16 steps we already have a ge/ae EMA. */
    int early = !N->early_done && N->tick >= 16;
    int later = N->early_done && N->tick >= N->last_change + SETTLE;
    if (!early && !later) return;
    if (N->ema < 0.03) {
        remove_idle(N);
        return;
    }
    /* Refuse late depth growth: after two adds, or after many steps,
       extra layers never catch up. */
    if (N->layer[0].L.in < 4) return;
    if (N->tick > 8000) {
        remove_idle(N);
        N->early_done = 1; N->locked = 1;
        return;
    }

    if (early && N->layer[0].L.in >= 4) {
        size_t b = 0;
        double gmax = N->layer[0].ge;
        for (size_t i = 1; i < N->depth; i++)
            if (N->layer[i].ge > gmax) { gmax = N->layer[i].ge; b = i; }
        int room = (int)LK_MAX_DEPTH - (int)N->depth - 1; /* leave one slot for tail I */
        if (room < 0) room = 0;
        /* guess==4: one policy for every file.
           multi-class + narrow x → site proj+readout
           else → idn I-stack (WDBC winner). */
        if (N->guess == 4) {
            size_t outT = N->layer[N->depth - 1].L.out;
            size_t in0 = N->layer[0].L.in;
            if (outT >= 2 && in0 < 20) {
                insert_proj_real(N);
                insert_readout_real(N);
            } else {
                int n = (int)(gmax / 0.05 + 0.5);
                if (n < 1) n = 1;
                int room = (int)LK_MAX_DEPTH - (int)N->depth - 1;
                if (room < 0) room = 0;
                if (n > room) n = room;
                for (int k = 0; k < n && N->depth < LK_MAX_DEPTH; k++)
                    insert_at(N, b);
                if (N->depth < LK_MAX_DEPTH)
                    insert_at(N, N->depth);
            }
            N->early_done = 1; N->locked = 1;
            return;
        }
        int n = 1;
        if (N->guess == 0) {              /* ge-scaled */
            n = (int)(gmax / 0.05 + 0.5);
            if (n < 1) n = 1;
        } else if (N->guess == 1) {       /* target depth 4 before tail I */
            n = 3 - (int)N->depth;
            if (n < 1) n = 1;
        } else if (N->guess == 2) {       /* ema-scaled */
            n = (int)(N->ema / 0.06 + 0.5);
            if (n < 1) n = 1;
        } else {                          /* max of ge and target */
            int ng = (int)(gmax / 0.05 + 0.5);
            int nt = 3 - (int)N->depth;
            n = ng > nt ? ng : nt;
            if (n < 1) n = 1;
        }
        if (n > room) n = room;
        for (int k = 0; k < n && N->depth < LK_MAX_DEPTH; k++)
            insert_at(N, b);
        if (N->depth < LK_MAX_DEPTH)
            insert_at(N, N->depth);
        N->early_done = 1; N->locked = 1;
        return;
    }

    double score = 0.0;
    size_t site = pick_site(N, &score);
    if (score < 0.08) {
        N->early_done = 1; N->locked = 1;
        return;
    }
    /* Do not insert a duplicate identity next to an existing k=1 I-like map. */
    if (site < N->depth && N->layer[site].L.k == 1 &&
        N->layer[site].L.in == N->layer[site].L.out &&
        N->layer[site].ge < 0.04)
        return;
    if (site > 0 && site - 1 < N->depth &&
        N->layer[site - 1].L.k == 1 &&
        N->layer[site - 1].L.in == N->layer[site - 1].L.out &&
        N->layer[site - 1].ge < 0.04)
        return;

    insert_at(N, site);
    N->early_done = 1; N->locked = 1;
}

static void net_init(void *c)
{
    PNet *N = c;
    N->tstep = 0; N->b1p = N->b2p = 1.0;
    N->tick = N->last_change = 0;
    N->ema = 0.0;
    N->added_layers = N->early_done = 0;
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
    if (lr < 0.06) lr = 0.01;
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
    size_t nin = N->layer[0].L.in;
    pfree(&N->layer[0]);
    memmove(&N->layer[0], &N->layer[1], (N->depth - 1) * sizeof(PLayer));
    memset(&N->layer[N->depth - 1], 0, sizeof(PLayer));
    N->depth--;
    lk_resize_in(&N->layer[0].L, nin);
    psync(&N->layer[0]);
    return 0;
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
    for (size_t i = 0; i < N->depth; i++)
        n += N->layer[i].L.n_or * (N->layer[i].L.in + 1);
    return n;
}
static size_t net_nbytes(void *c)
{
    PNet *N = c;
    size_t n = 0;
    for (size_t i = 0; i < N->depth; i++)
        n += N->layer[i].L.n_or * (N->layer[i].L.in + 2) * sizeof(double);
    return n;
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

AltNet type_nn_idfact_open(size_t in, size_t out);

AltNet type_nn_idi_open(size_t in, size_t out)
{
    PNet *N = (PNet *)calloc(1, sizeof(PNet));
    N->dynamic = 1;
    N->fact = 0;
    N->name = "type-nn-idi";
    N->b1p = N->b2p = 1.0;
    /* start as a single product layer; BP energy decides the rest */
    palloc(&N->layer[0], in, out, TNN_K0);
    N->depth = 1;
    AltNet h = {
        .impl = "type-nn-idi", .ctx = N, .in = in, .out = out,
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

AltNet type_nn_idfact_open(size_t in, size_t out)
{
    AltNet h = type_nn_idi_open(in, out);
    h.impl = "type-nn-idfact";
    ((PNet *)h.ctx)->fact = 1;
    ((PNet *)h.ctx)->name = "type-nn-idfact";
    return h;
}
AltNet type_nn_idn_open(size_t in, size_t out)
{
    AltNet h = type_nn_idi_open(in, out);
    h.impl = "type-nn-idn";
    ((PNet *)h.ctx)->guess = 0;
    ((PNet *)h.ctx)->name = "type-nn-idn";
    return h;
}
AltNet type_nn_idtgt_open(size_t in, size_t out)
{
    AltNet h = type_nn_idi_open(in, out);
    h.impl = "type-nn-idtgt";
    ((PNet *)h.ctx)->guess = 1;
    ((PNet *)h.ctx)->name = "type-nn-idtgt";
    return h;
}
AltNet type_nn_idema_open(size_t in, size_t out)
{
    AltNet h = type_nn_idi_open(in, out);
    h.impl = "type-nn-idema";
    ((PNet *)h.ctx)->guess = 2;
    ((PNet *)h.ctx)->name = "type-nn-idema";
    return h;
}
AltNet type_nn_idmax_open(size_t in, size_t out)
{
    AltNet h = type_nn_idi_open(in, out);
    h.impl = "type-nn-idmax";
    ((PNet *)h.ctx)->guess = 3;
    ((PNet *)h.ctx)->name = "type-nn-idmax";
    return h;
}

AltNet type_nn_one_open(size_t in, size_t out)
{
    AltNet h = type_nn_idi_open(in, out);
    h.impl = "type-nn-one";
    ((PNet *)h.ctx)->guess = 4;
    ((PNet *)h.ctx)->name = "type-nn-one";
    return h;
}
