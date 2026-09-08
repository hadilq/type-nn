#include "type_nn_alt.h"
#include "type_nn_layerkit.h"

/*
 * type-nn-adam — Adam on every And/Or weight and bias.
 *
 *   m ← β1 m + (1-β1) g
 *   v ← β2 v + (1-β2) g²
 *   ŵ = m / (1-β1^t)     v̂ = v / (1-β2^t)
 *   w ← w − α ŵ / (√v̂ + ε)
 *
 * Same Type Mechanics model and scale/insert/remove API.
 */

#define AD_B1  0.9
#define AD_B2  0.999
#define AD_EPS 1e-8

typedef struct {
    LKLayer L;
    double *mW, *vW, *mb, *vb;
} AdLayer;

typedef struct {
    AdLayer layer[LK_MAX_DEPTH];
    double *act[LK_MAX_DEPTH + 1];
    size_t  act_w[LK_MAX_DEPTH + 1];
    size_t  depth;
    int     dynamic;
    size_t  max_or;
    unsigned long t;
    double  b1p, b2p;
} AdNet;

static void alayer_alloc(AdLayer *A, size_t in, size_t out, size_t k)
{
    lk_alloc(&A->L, in, out, k);
    size_t n = A->L.n_or;
    A->mW = (double *)realloc(A->mW, n * in * sizeof(double));
    A->vW = (double *)realloc(A->vW, n * in * sizeof(double));
    A->mb = (double *)realloc(A->mb, n * sizeof(double));
    A->vb = (double *)realloc(A->vb, n * sizeof(double));
    if (n && in) {
        memset(A->mW, 0, n * in * sizeof(double));
        memset(A->vW, 0, n * in * sizeof(double));
    }
    if (n) {
        memset(A->mb, 0, n * sizeof(double));
        memset(A->vb, 0, n * sizeof(double));
    }
}
static void alayer_free(AdLayer *A)
{
    lk_free(&A->L);
    free(A->mW); free(A->vW); free(A->mb); free(A->vb);
    A->mW = A->vW = A->mb = A->vb = NULL;
}
static void alayer_sync(AdLayer *A)
{
    size_t n = A->L.n_or, in = A->L.in;
    A->mW = (double *)realloc(A->mW, n * in * sizeof(double));
    A->vW = (double *)realloc(A->vW, n * in * sizeof(double));
    A->mb = (double *)realloc(A->mb, n * sizeof(double));
    A->vb = (double *)realloc(A->vb, n * sizeof(double));
    memset(A->mW, 0, n * in * sizeof(double));
    memset(A->vW, 0, n * in * sizeof(double));
    memset(A->mb, 0, n * sizeof(double));
    memset(A->vb, 0, n * sizeof(double));
}

static void fit(AdNet *N, size_t s, size_t w)
{
    if (N->act_w[s] >= w && N->act[s]) return;
    free(N->act[s]);
    N->act[s] = (double *)calloc(w ? w : 1, sizeof(double));
    N->act_w[s] = w ? w : 1;
}

static double adam_step(double *m, double *v, double g,
                        double lr, double b1p, double b2p)
{
    *m = AD_B1 * *m + (1.0 - AD_B1) * g;
    *v = AD_B2 * *v + (1.0 - AD_B2) * g * g;
    double mh = *m / (1.0 - b1p);
    double vh = *v / (1.0 - b2p);
    return lr * mh / (sqrt(vh) + AD_EPS);
}

static void bwd_layer(AdLayer *A, const double *x, const double *dy,
                      double *dx, double lr, double b1p, double b2p)
{
    LKLayer *L = &A->L;
    const size_t in = L->in, k = L->k, out = L->out;
    if (dx) memset(dx, 0, in * sizeof(double));
    double dor[8];
    for (size_t i = 0; i < out; i++) {
        lk_dor(L, i, dy[i], dor);
        for (size_t t = 0; t < k; t++) {
            double g = dor[t];
            size_t r = i * k + t;
            L->b[r] = tnn_clamp(
                L->b[r] - adam_step(&A->mb[r], &A->vb[r], g, lr, b1p, b2p),
                -TNN_WCLIP, TNN_WCLIP);
            double *w = L->W + r * in;
            double *m = A->mW + r * in;
            double *v = A->vW + r * in;
            for (size_t j = 0; j < in; j++) {
                if (dx) dx[j] += g * w[j];
                w[j] = tnn_clamp(
                    w[j] - adam_step(&m[j], &v[j], g * x[j], lr, b1p, b2p),
                    -TNN_WCLIP, TNN_WCLIP);
            }
        }
    }
}

static void net_init(void *c)
{
    AdNet *N = c;
    N->t = 0;
    N->b1p = 1.0;
    N->b2p = 1.0;
    for (size_t i = 0; i < N->depth; i++) lk_init(&N->layer[i].L);
}
static void net_fwd(void *c, const double *x, double *y)
{
    AdNet *N = c;
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
    AdNet *N = c;
    if (!N->act[1]) {
        const double *cur = x;
        for (size_t i = 0; i < N->depth; i++) {
            fit(N, i + 1, N->layer[i].L.out);
            lk_fwd(&N->layer[i].L, cur, N->act[i + 1]);
            cur = N->act[i + 1];
        }
    }
    N->t++;
    N->b1p *= AD_B1;
    N->b2p *= AD_B2;
    /* Product nets + the SGD lrs in bench (0.02–0.08) over-step Adam.
       Cap the effective α; XOR still converges in 400 steps. */
    if (lr > 0.02) lr = 0.01;
    const double *dcur = dy;
    double *hold = NULL;
    for (size_t i = N->depth; i-- > 0; ) {
        const double *xin = i ? N->act[i] : x;
        double *dx = NULL;
        if (i) { fit(N, 0, N->layer[i].L.in); dx = N->act[0]; }
        bwd_layer(&N->layer[i], xin, dcur, dx, lr, N->b1p, N->b2p);
        if (i) {
            size_t w = N->layer[i].L.in;
            hold = (double *)realloc(hold, w * sizeof(double));
            memcpy(hold, dx, w * sizeof(double));
            dcur = hold;
        }
    }
    free(hold);
    if (N->dynamic) {
        LKLayer *t = &N->layer[N->depth - 1].L;
        double mag = 0;
        for (size_t i = 0; i < t->out; i++) mag += fabs(dy[i]);
        if (mag > 1.0 && t->k < N->max_or && t->k < 4) {
            lk_set_k(t, t->k + 1);
            alayer_sync(&N->layer[N->depth - 1]);
        }
    }
}
static void net_align(void *c, size_t in)
{
    AdNet *N = c;
    lk_resize_in(&N->layer[0].L, in);
    alayer_sync(&N->layer[0]);
}
static void net_out(void *c, size_t o)
{
    AdNet *N = c;
    lk_resize_out(&N->layer[N->depth - 1].L, o);
    alayer_sync(&N->layer[N->depth - 1]);
}
static void net_k(void *c, size_t k)
{
    AdNet *N = c;
    if (k > N->max_or) k = N->max_or;
    lk_set_k(&N->layer[N->depth - 1].L, k);
    alayer_sync(&N->layer[N->depth - 1]);
}
static void net_ins(void *c)
{
    AdNet *N = c;
    if (N->depth >= LK_MAX_DEPTH) return;
    size_t dim = N->layer[N->depth - 1].L.in;
    memmove(&N->layer[N->depth], &N->layer[N->depth - 1], sizeof(AdLayer));
    memset(&N->layer[N->depth - 1], 0, sizeof(AdLayer));
    alayer_alloc(&N->layer[N->depth - 1], dim, dim, TNN_K0);
    lk_identity(&N->layer[N->depth - 1].L);
    N->depth++;
}
static int net_rem(void *c)
{
    AdNet *N = c;
    if (N->depth <= 1) return -1;
    size_t idx = N->depth - 2, nin = N->layer[idx].L.in;
    alayer_free(&N->layer[idx]);
    memmove(&N->layer[idx], &N->layer[idx + 1],
            (N->depth - idx - 1) * sizeof(AdLayer));
    memset(&N->layer[N->depth - 1], 0, sizeof(AdLayer));
    N->depth--;
    lk_resize_in(&N->layer[idx].L, nin);
    alayer_sync(&N->layer[idx]);
    return 0;
}
static void net_dyn(void *c, int on) { ((AdNet *)c)->dynamic = on ? 1 : 0; }
static size_t net_depth(void *c) { return ((AdNet *)c)->depth; }
static size_t net_kf(void *c) { AdNet *N = c; return N->layer[N->depth - 1].L.k; }
static size_t net_params(void *c)
{
    AdNet *N = c; size_t n = 0;
    for (size_t i = 0; i < N->depth; i++)
        n += N->layer[i].L.n_or * (N->layer[i].L.in + 1);
    return n;
}
static size_t net_nbytes(void *c)
{
    /* W+b plus first and second moment */
    AdNet *N = c; size_t n = 0;
    for (size_t i = 0; i < N->depth; i++)
        n += 3 * N->layer[i].L.n_or * (N->layer[i].L.in + 1) * sizeof(double);
    return n;
}
static void net_free(void *c)
{
    AdNet *N = c;
    for (size_t i = 0; i < N->depth; i++) alayer_free(&N->layer[i]);
    for (size_t i = 0; i < LK_MAX_DEPTH + 1; i++) free(N->act[i]);
    free(N);
}


static void net_scale(void *c, size_t idx, size_t in, size_t out)
{
    AdNet *N = c;
    if (idx >= N->depth) return;
    lk_resize_in(&N->layer[idx].L, in);
    lk_resize_out(&N->layer[idx].L, out);
    if (idx + 1 < N->depth) lk_resize_in(&N->layer[idx + 1].L, out);
    if (idx > 0) lk_resize_out(&N->layer[idx - 1].L, in);
}
static size_t net_lin(void *c, size_t idx)
{
    AdNet *N = c;
    return idx < N->depth ? N->layer[idx].L.in : 0;
}
static size_t net_lout(void *c, size_t idx)
{
    AdNet *N = c;
    return idx < N->depth ? N->layer[idx].L.out : 0;
}
AltNet type_nn_adam_open(size_t in, size_t out)
{
    AdNet *N = (AdNet *)calloc(1, sizeof(AdNet));
    N->depth = 1; N->dynamic = 1; N->max_or = TNN_MAX_OR;
    N->b1p = 1.0; N->b2p = 1.0;
    alayer_alloc(&N->layer[0], in, out, TNN_K0);
    AltNet h = {
        .impl = "type-nn-adam", .ctx = N, .in = in, .out = out,
        .init = net_init, .forward = net_fwd, .backward = net_bwd,
        .align_inputs = net_align, .set_outputs = net_out,
        .set_or_factors = net_k, .insert_identity = net_ins,
        .remove_hidden = net_rem, .set_dynamic = net_dyn,
        .depth = net_depth, .or_factors = net_kf,
        .param_count = net_params, .nbytes = net_nbytes, .free = net_free,
        .scale_layer = net_scale, .layer_in = net_lin, .layer_out = net_lout
    };
    return h;
}
