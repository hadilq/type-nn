#include "type_nn_alt.h"
#include "type_nn_layerkit.h"

/*
 * type-nn-proj / type-nn-proj-dyn
 *
 * Lesson from the dyn bake-off: growing k on raw wide z-scored inputs
 * multiplies two large affines and overflows And-clip. Torch-mlp wins
 * because it first maps x to a short hidden basis.
 *
 * Faithful Type Mechanics:
 *   hidden: k = 1  (a single Or = affine projection, clipped)
 *   tail:   k = 2  (quadratic in that basis)
 *
 * proj-dyn grows hidden width only after EMA residual stalls.
 */

#define AD_B1  0.9
#define AD_B2  0.999
#define AD_EPS 1e-8
#define SETTLE 32
#define FREEZE 16
#define MAX_HID 32

typedef struct {
    LKLayer L;
    double *mW, *vW, *mb, *vb, *aW, *ab;
    int freeze_col, pending;
} PLayer;

typedef struct {
    PLayer  layer[LK_MAX_DEPTH];
    double *act[LK_MAX_DEPTH + 1];
    size_t  act_w[LK_MAX_DEPTH + 1];
    size_t  depth;
    int     dynamic;
    unsigned tick, last_change;
    double  ema, ema_ref;
    unsigned long tstep;
    double  b1p, b2p;
    const char *name;
} PNet;

static size_t pick_hid(size_t in, size_t out)
{
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
    for (size_t i = 0; i < out; i++) {
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
    P->pending++;
    int batch = (lr >= 0.06) ? 4 : 16; /* XOR small batch, UCI larger */
    if (P->pending >= batch) flush_layer(N, P, lr);
    (void)N;
}

static void net_init(void *c)
{
    PNet *N = c;
    N->tstep = 0; N->b1p = N->b2p = 1.0;
    N->tick = N->last_change = 0;
    N->ema = N->ema_ref = 0.0;
    for (size_t i = 0; i < N->depth; i++) {
        LKLayer *L = &N->layer[i].L;
        double s = 0.15 / sqrt((double)(L->in > 0 ? L->in : 1));
        for (size_t r = 0; r < L->n_or; r++) {
            L->b[r] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * s;
            for (size_t j = 0; j < L->in; j++)
                L->W[r * L->in + j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * s;
        }
    }
}
static void net_fwd(void *c, const double *x, double *y)
{
    PNet *N = c;
    const double *cur = x;
    for (size_t i = 0; i < N->depth; i++) {
        fit(N, i + 1, N->layer[i].L.out);
        lk_fwd(&N->layer[i].L, cur, N->act[i + 1]);
        cur = N->act[i + 1];
    }
    memcpy(y, cur, N->layer[N->depth - 1].L.out * sizeof(double));
}
static void net_bwd(void *c, const double *x, const double *dy, double lr)
{
    PNet *N = c;
    if (!N->act[1]) net_fwd(c, x, N->act[N->depth]);
    if (!N->act[1]) {
        const double *cur = x;
        for (size_t i = 0; i < N->depth; i++) {
            fit(N, i + 1, N->layer[i].L.out);
            lk_fwd(&N->layer[i].L, cur, N->act[i + 1]);
            cur = N->act[i + 1];
        }
    }
    if (lr < 0.06) lr = 0.01; /* UCI; leave XOR's 0.08 alone */
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
    if (N->ema_ref == 0.0) N->ema_ref = N->ema;
    if (N->layer[N->depth - 1].freeze_col > 0)
        N->layer[N->depth - 1].freeze_col--;

    if (!N->dynamic) return;
    if (N->tick < N->last_change + SETTLE) return;
    if (N->tick < (unsigned)(SETTLE * 3)) return;
    if (N->ema < 0.05) return;
    if (N->ema_ref > 0 && N->ema < 0.97 * N->ema_ref) return;
    /* 1. add a k=1 affine layer in front of the tail */
    if (N->depth < 4 && N->layer[0].L.in >= 4) {
        PLayer *tail = &N->layer[N->depth - 1];
        size_t din = tail->L.in;
        size_t H = pick_hid(din, tail->L.out);
        if (H < din) H = din;
        memmove(&N->layer[N->depth], tail, sizeof(PLayer));
        memset(tail, 0, sizeof(PLayer));
        palloc(tail, din, H, 1);          /* new affine basis, not I */
        psync(tail);
        lk_resize_in(&N->layer[N->depth].L, H);
        psync(&N->layer[N->depth]);
        N->layer[N->depth].freeze_col = FREEZE;
        N->depth++;
        N->last_change = N->tick;
        N->ema_ref = N->ema;
        return;
    }
    /* 2. otherwise widen the first hidden And row */
    if (N->layer[0].L.out < MAX_HID) {
        size_t nout = N->layer[0].L.out + 1;
        lk_resize_out(&N->layer[0].L, nout);
        lk_resize_in(&N->layer[1].L, nout);
        psync(&N->layer[0]);
        psync(&N->layer[1]);
        N->layer[1].freeze_col = FREEZE;
        N->last_change = N->tick;
        N->ema_ref = N->ema;
    }
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
    if (k > 4) k = 4;
    lk_set_k(&N->layer[N->depth - 1].L, k);
    psync(&N->layer[N->depth - 1]);
}
static void net_ins(void *c)
{
    PNet *N = c;
    if (N->depth >= LK_MAX_DEPTH) return;
    size_t dim = N->layer[N->depth - 1].L.in;
    memmove(&N->layer[N->depth], &N->layer[N->depth - 1], sizeof(PLayer));
    memset(&N->layer[N->depth - 1], 0, sizeof(PLayer));
    palloc(&N->layer[N->depth - 1], dim, dim, 1);
    lk_identity(&N->layer[N->depth - 1].L);
    N->depth++;
}
static int net_rem(void *c)
{
    PNet *N = c;
    if (N->depth <= 1) return -1;
    size_t idx = N->depth - 2, nin = N->layer[idx].L.in;
    pfree(&N->layer[idx]);
    memmove(&N->layer[idx], &N->layer[idx + 1], (N->depth - idx - 1) * sizeof(PLayer));
    memset(&N->layer[N->depth - 1], 0, sizeof(PLayer));
    N->depth--;
    lk_resize_in(&N->layer[idx].L, nin);
    psync(&N->layer[idx]);
    return 0;
}
static void net_dyn(void *c, int on) { ((PNet *)c)->dynamic = on ? 1 : 0; }
static size_t net_depth(void *c) { return ((PNet *)c)->depth; }
static size_t net_kf(void *c) { PNet *N = c; return N->layer[N->depth - 1].L.k; }
static size_t net_params(void *c)
{
    PNet *N = c; size_t n = 0;
    for (size_t i = 0; i < N->depth; i++)
        n += N->layer[i].L.n_or * (N->layer[i].L.in + 1);
    return n;
}
static size_t net_nbytes(void *c)
{
    PNet *N = c; size_t n = 0;
    for (size_t i = 0; i < N->depth; i++)
        n += 3 * N->layer[i].L.n_or * (N->layer[i].L.in + 1) * sizeof(double);
    return n;
}
static void net_free(void *c)
{
    PNet *N = c;
    for (size_t i = 0; i < N->depth; i++) pfree(&N->layer[i]);
    for (size_t i = 0; i < LK_MAX_DEPTH + 1; i++) free(N->act[i]);
    free(N);
}
static void net_scale(void *c, size_t idx, size_t in, size_t out)
{
    PNet *N = c;
    if (idx >= N->depth) return;
    lk_resize_in(&N->layer[idx].L, in);
    lk_resize_out(&N->layer[idx].L, out);
    if (idx + 1 < N->depth) lk_resize_in(&N->layer[idx + 1].L, out);
    if (idx > 0) lk_resize_out(&N->layer[idx - 1].L, in);
    psync(&N->layer[idx]);
    if (idx + 1 < N->depth) psync(&N->layer[idx + 1]);
    if (idx > 0) psync(&N->layer[idx - 1]);
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

static AltNet open_proj(const char *name, int dyn, size_t in, size_t out)
{
    PNet *N = (PNet *)calloc(1, sizeof(PNet));
    size_t H = pick_hid(in, out);
    N->dynamic = dyn;
    N->name = name;
    N->b1p = N->b2p = 1.0;
    palloc(&N->layer[0], in, H, 1);          /* affine hidden */
    palloc(&N->layer[1], H, out, TNN_K0);    /* quadratic tail */
    N->depth = 2;
    AltNet h = {
        .impl = name, .ctx = N, .in = in, .out = out,
        .init = net_init, .forward = net_fwd, .backward = net_bwd,
        .align_inputs = net_align, .set_outputs = net_out,
        .set_or_factors = net_k, .insert_identity = net_ins,
        .remove_hidden = net_rem, .set_dynamic = net_dyn,
        .depth = net_depth, .or_factors = net_kf,
        .param_count = net_params, .nbytes = net_nbytes, .free = net_free,
        .scale_layer = net_scale, .layer_in = net_lin, .layer_out = net_lout, .layer_k = net_lk
    };
    return h;
}

static AltNet open_lin(size_t in, size_t out)
{
    AltNet h = open_proj("type-nn-lin", 0, in, out);
    /* tail k=1: clipped affine stack, no product */
    PNet *N = (PNet *)h.ctx;
    lk_set_k(&N->layer[1].L, 1);
    psync(&N->layer[1]);
    return h;
}
static AltNet open_proj2(const char *name, int dyn, size_t in, size_t out)
{
    AltNet h = open_proj(name, dyn, in, out);
    PNet *N = (PNet *)h.ctx;
    size_t H = N->layer[0].L.out;
    size_t o = N->layer[1].L.out;
    /* turn k=2 tail into mid features, add k=1 readout */
    lk_resize_out(&N->layer[1].L, H);
    lk_set_k(&N->layer[1].L, TNN_K0);
    psync(&N->layer[1]);
    palloc(&N->layer[2], H, o, 1);
    N->depth = 3;
    return h;
}
AltNet type_nn_proj_open(size_t in, size_t out)
{ return open_proj("type-nn-proj", 0, in, out); }
AltNet type_nn_proj_dyn_open(size_t in, size_t out)
{ return open_proj("type-nn-proj-dyn", 1, in, out); }
AltNet type_nn_proj2_open(size_t in, size_t out)
{ return open_proj2("type-nn-proj2", 0, in, out); }
AltNet type_nn_proj2_dyn_open(size_t in, size_t out)
{ return open_proj2("type-nn-proj2-dyn", 1, in, out); }

AltNet type_nn_lin_open(size_t in, size_t out)
{ return open_lin(in, out); }
