#include "type_nn_alt.h"
#include "type_nn_layerkit.h"

/*
 * type-nn-over / type-nn-over-unused
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
    int     dynamic, added_layers, early_done, force_roles, par;
    unsigned n_add, n_drop; int locked;
    unsigned tick, last_change;
    double  ema;
    unsigned long tstep;
    double  b1p, b2p;
    const char *name;
    size_t nnz_cap;
} PNet;

#define PAR_A 1
#define PAR_D 2
#define PAR_E 4
#define PAR_K 8

static size_t pick_hid_par(const PNet *N, size_t in, size_t out)
{
    if (N->par & PAR_K) return (out > 2 ? out : 2);
    if (N->par & PAR_A) {
        size_t h = 4;
        if (in >= 16) h = 8;
        if (h < out) h = out;
        return h;
    }
    size_t h = 8;
    if (in >= 10) h = 16;
    if (in >= 24) h = 24;
    if (h < out + 2) h = out + 2;
    if (h > MAX_HID) h = MAX_HID;
    return h;
}

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

static void insert_at(PNet *N, size_t idx)
{
    if (N->depth >= LK_MAX_DEPTH) return;
    if (idx > N->depth) idx = N->depth;

    if (idx == 0 && N->layer[0].L.in >= 4) {
        /* front: real k=1 basis in → H, stitch the old first layer */
        size_t in = N->layer[0].L.in;
        size_t H = pick_hid_par(N, in, N->layer[N->depth - 1].L.out);
        memmove(&N->layer[1], &N->layer[0], N->depth * sizeof(PLayer));
        memset(&N->layer[0], 0, sizeof(PLayer));
        palloc(&N->layer[0], in, H, 1);
        lk_resize_in(&N->layer[1].L, H);
        psync(&N->layer[1]);
        N->layer[1].freeze_col = FREEZE;
        N->depth++;
        N->added_layers++; N->n_add++;
        N->last_change = N->tick;
        return;
    }

    if (idx == N->depth && N->layer[N->depth - 1].L.k >= 2) {
        /* after product tail: real k=1 readout, mid stays product features */
        PLayer *tail = &N->layer[N->depth - 1];
        size_t o = tail->L.out;
        size_t feat = pick_hid_par(N, tail->L.in, o);
        if (feat < o) feat = o;
        lk_resize_out(&tail->L, feat);
        psync(tail);
        palloc(&N->layer[N->depth], feat, o, 1);
        N->layer[N->depth].freeze_col = FREEZE;
        N->depth++;
        N->added_layers++; N->n_add++;
        N->last_change = N->tick;
        return;
    }

    /* mid interface: identity so the mapping does not jump */
    size_t dim = N->layer[idx].L.in;
    memmove(&N->layer[idx + 1], &N->layer[idx],
            (N->depth - idx) * sizeof(PLayer));
    memset(&N->layer[idx], 0, sizeof(PLayer));
    palloc(&N->layer[idx], dim, dim, 1);
    lk_identity(&N->layer[idx].L);
    psync(&N->layer[idx]);
    N->depth++;
    N->added_layers++; N->n_add++;
    N->last_change = N->tick;
}

/* Score every interface. High score = insert an affine identity there.
   site i means "before layer i"; site `depth` means "after the tail". */


#define ID_W_TAU 0.08

/* Identity-like: square k=1 map whose W is I within tau. Biases ignored. */

static void prune_small_w(PNet *N)
{
    size_t nnz = 0;
    for (size_t i = 0; i < N->depth; i++) {
        LKLayer *L = &N->layer[i].L;
        for (size_t t = 0; t < L->n_or * L->in; t++) {
            if (fabs(L->W[t]) < 0.03) L->W[t] = 0.0;
            else nnz++;
        }
        nnz += L->n_or; /* biases kept */
    }
    N->nnz_cap = nnz;
}

static void drop_dead_or(PNet *N)
{
    for (size_t i = 0; i < N->depth; i++) {
        LKLayer *L = &N->layer[i].L;
        if (L->k <= 2) continue; /* never drop the last product factor */
        /* drop last Or of each And if its W is tiny */
        int any = 0;
        for (size_t o = 0; o < L->out; o++) {
            double *w = L->W + ((o * L->k) + (L->k - 1)) * L->in;
            double n2 = 0.0;
            for (size_t j = 0; j < L->in; j++) n2 += w[j] * w[j];
            if (n2 < 0.01) any = 1;
        }
        if (!any) continue;
        size_t nk = L->k - 1;
        lk_set_k(L, nk);
        psync(&N->layer[i]);
        N->n_drop++;
        N->last_change = N->tick;
        return;
    }
}

static void grow_h_if_needed(PNet *N)
{
    if (N->depth < 1) return;
    PLayer *P = &N->layer[0];
    if (P->L.k != 1) return;
    if (P->ge < 0.08) return;
    {
        size_t cap = (N->par & PAR_A) ? 8 : 16;
        if (P->L.in >= 20 && cap < 12) cap = 12;
        if (P->L.out >= cap) return;
    }
    size_t nout = P->L.out + 2;
    lk_resize_out(&P->L, nout);
    psync(P);
    if (N->depth > 1) {
        lk_resize_in(&N->layer[1].L, nout);
        psync(&N->layer[1]);
    }
    N->last_change = N->tick;
}

static void estimate_and_apply(PNet *N)
{
    if (!N->dynamic) return;
    if (N->layer[0].L.in < 4) return;
    if (!N->early_done && N->tick >= 16) {
        if (N->layer[0].L.k >= 2)
            insert_at(N, 0);
        if (N->layer[N->depth - 1].L.k >= 2)
            insert_at(N, N->depth);
        N->early_done = 1;
        N->locked = 1;
        return;
    }
    if (!N->early_done) return;
    if ((N->par & PAR_D) && N->tick % 64 == 0) prune_small_w(N);
    if ((N->par & PAR_E) && N->tick >= N->last_change + SETTLE) drop_dead_or(N);
    if ((N->par & PAR_K) && N->tick >= N->last_change + SETTLE) grow_h_if_needed(N);
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
    if ((N->par & PAR_D) && N->nnz_cap) return N->nnz_cap;
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

static AltNet par_open(const char *name, int par, size_t in, size_t out)
{
    PNet *N = (PNet *)calloc(1, sizeof(PNet));
    N->dynamic = 1;
    N->par = par;
    N->name = name;
    N->b1p = N->b2p = 1.0;
    palloc(&N->layer[0], in, out, TNN_K0);
    N->depth = 1;
    AltNet h = {
        .impl = name, .ctx = N, .in = in, .out = out,
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
AltNet type_nn_pA_open(size_t in, size_t out) { return par_open("type-nn-pA", PAR_A, in, out); }
AltNet type_nn_pD_open(size_t in, size_t out) { return par_open("type-nn-pD", PAR_D, in, out); }
AltNet type_nn_pE_open(size_t in, size_t out) { return par_open("type-nn-pE", PAR_E, in, out); }
AltNet type_nn_pK_open(size_t in, size_t out) { return par_open("type-nn-pK", PAR_K, in, out); }
AltNet type_nn_pAD_open(size_t in, size_t out) { return par_open("type-nn-pAD", PAR_A|PAR_D, in, out); }
AltNet type_nn_pAE_open(size_t in, size_t out) { return par_open("type-nn-pAE", PAR_A|PAR_E, in, out); }
AltNet type_nn_pAK_open(size_t in, size_t out) { return par_open("type-nn-pAK", PAR_A|PAR_K, in, out); }
AltNet type_nn_pDE_open(size_t in, size_t out) { return par_open("type-nn-pDE", PAR_D|PAR_E, in, out); }
AltNet type_nn_pDK_open(size_t in, size_t out) { return par_open("type-nn-pDK", PAR_D|PAR_K, in, out); }
AltNet type_nn_pEK_open(size_t in, size_t out) { return par_open("type-nn-pEK", PAR_E|PAR_K, in, out); }
AltNet type_nn_pADE_open(size_t in, size_t out) { return par_open("type-nn-pADE", PAR_A|PAR_D|PAR_E, in, out); }
AltNet type_nn_pADK_open(size_t in, size_t out) { return par_open("type-nn-pADK", PAR_A|PAR_D|PAR_K, in, out); }
AltNet type_nn_pAEK_open(size_t in, size_t out) { return par_open("type-nn-pAEK", PAR_A|PAR_E|PAR_K, in, out); }
AltNet type_nn_pDEK_open(size_t in, size_t out) { return par_open("type-nn-pDEK", PAR_D|PAR_E|PAR_K, in, out); }
AltNet type_nn_pADEK_open(size_t in, size_t out) { return par_open("type-nn-pADEK", PAR_A|PAR_D|PAR_E|PAR_K, in, out); }
AltNet type_nn_win_open(size_t in, size_t out) { return par_open("type-nn-win", PAR_A|PAR_D|PAR_E|PAR_K, in, out); }

