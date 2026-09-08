#include "type_nn_alt.h"
#include "type_nn_layerkit.h"

/*
 * type-nn-bpgemm — second backward iteration.
 *
 * Same And/Or model. Instead of walking each Or row for dx:
 *
 *   dOr  = prefix/suffix product rule          (O(k) per And)
 *   dx   = Wᵀ dOr                              blocked GEMV
 *   W    ← W − lr dOr xᵀ                       blocked rank-1
 *   b    ← b − lr dOr
 *
 * Scratch for dOr / dx is reused so the hot path does not malloc.
 */

typedef struct {
    LKLayer layer[LK_MAX_DEPTH];
    double *act[LK_MAX_DEPTH + 1];
    size_t  act_w[LK_MAX_DEPTH + 1];
    double *dor, *dxa, *dxb;
    size_t  dor_w, dxa_w, dxb_w;
    size_t  depth;
    int     dynamic;
    size_t  max_or;
} GNet;

static void fit_act(GNet *N, size_t s, size_t w)
{
    if (N->act_w[s] >= w && N->act[s]) return;
    free(N->act[s]);
    N->act[s] = (double *)calloc(w ? w : 1, sizeof(double));
    N->act_w[s] = w ? w : 1;
}
static void fit_vec(double **p, size_t *cap, size_t w)
{
    if (*cap >= w && *p) return;
    free(*p);
    *p = (double *)calloc(w ? w : 1, sizeof(double));
    *cap = w ? w : 1;
}

static void gemv_t(const double *W, const double *dor,
                   size_t n_or, size_t in, double *dx)
{
    memset(dx, 0, in * sizeof(double));
    const size_t CB = 16;
    for (size_t r = 0; r < n_or; r++) {
        double g = dor[r];
        if (g == 0.0) continue;
        const double *w = W + r * in;
        size_t j = 0;
        for (; j + 3 < in; j += 4) {
            dx[j  ] += g * w[j];
            dx[j+1] += g * w[j+1];
            dx[j+2] += g * w[j+2];
            dx[j+3] += g * w[j+3];
        }
        for (; j < in; j++) dx[j] += g * w[j];
        (void)CB;
    }
}

static void ger(double *W, const double *dor, const double *x,
                size_t n_or, size_t in, double lr)
{
    for (size_t r = 0; r < n_or; r++) {
        double g = lr * dor[r];
        if (g == 0.0) continue;
        double *w = W + r * in;
        size_t j = 0;
        for (; j + 3 < in; j += 4) {
            w[j  ] = tnn_clamp(w[j  ] - g * x[j  ], -TNN_WCLIP, TNN_WCLIP);
            w[j+1] = tnn_clamp(w[j+1] - g * x[j+1], -TNN_WCLIP, TNN_WCLIP);
            w[j+2] = tnn_clamp(w[j+2] - g * x[j+2], -TNN_WCLIP, TNN_WCLIP);
            w[j+3] = tnn_clamp(w[j+3] - g * x[j+3], -TNN_WCLIP, TNN_WCLIP);
        }
        for (; j < in; j++)
            w[j] = tnn_clamp(w[j] - g * x[j], -TNN_WCLIP, TNN_WCLIP);
    }
}

static void bwd_layer(GNet *N, LKLayer *L, const double *x,
                      const double *dy, double *dx, double lr)
{
    const size_t in = L->in, k = L->k, out = L->out;
    fit_vec(&N->dor, &N->dor_w, L->n_or);
    double *dor = N->dor;
    for (size_t i = 0; i < out; i++)
        lk_dor(L, i, dy[i], dor + i * k);
    if (dx) gemv_t(L->W, dor, L->n_or, in, dx);
    ger(L->W, dor, x, L->n_or, in, lr);
    for (size_t r = 0; r < L->n_or; r++)
        L->b[r] = tnn_clamp(L->b[r] - lr * dor[r], -TNN_WCLIP, TNN_WCLIP);
}

static void net_init(void *c)
{
    GNet *N = c;
    for (size_t i = 0; i < N->depth; i++) lk_init(&N->layer[i]);
}
static void net_fwd(void *c, const double *x, double *y)
{
    GNet *N = c;
    const double *cur = x;
    for (size_t i = 0; i < N->depth; i++) {
        fit_act(N, i + 1, N->layer[i].out);
        lk_fwd(&N->layer[i], cur, N->act[i + 1]);
        cur = N->act[i + 1];
    }
    memcpy(y, cur, N->layer[N->depth - 1].out * sizeof(double));
}
static void net_bwd(void *c, const double *x, const double *dy, double lr)
{
    GNet *N = c;
    if (!N->act[1]) {
        const double *cur = x;
        for (size_t i = 0; i < N->depth; i++) {
            fit_act(N, i + 1, N->layer[i].out);
            lk_fwd(&N->layer[i], cur, N->act[i + 1]);
            cur = N->act[i + 1];
        }
    }
    const double *dcur = dy;
    int ping = 0;
    for (size_t i = N->depth; i-- > 0; ) {
        const double *xin = i ? N->act[i] : x;
        double *dx = NULL;
        if (i) {
            if (ping) fit_vec(&N->dxb, &N->dxb_w, N->layer[i].in);
            else      fit_vec(&N->dxa, &N->dxa_w, N->layer[i].in);
            dx = ping ? N->dxb : N->dxa;
            ping ^= 1;
        }
        bwd_layer(N, &N->layer[i], xin, dcur, dx, lr);
        if (i) dcur = dx;
    }
    if (N->dynamic) {
        LKLayer *t = &N->layer[N->depth - 1];
        double mag = 0;
        for (size_t i = 0; i < t->out; i++) mag += fabs(dy[i]);
        if (mag > 1.0 && t->k < N->max_or && t->k < 4) lk_set_k(t, t->k + 1);
    }
}
static void net_align(void *c, size_t in) { lk_resize_in(&((GNet *)c)->layer[0], in); }
static void net_out(void *c, size_t o)
{
    GNet *N = c;
    lk_resize_out(&N->layer[N->depth - 1], o);
}
static void net_k(void *c, size_t k)
{
    GNet *N = c;
    if (k > N->max_or) k = N->max_or;
    lk_set_k(&N->layer[N->depth - 1], k);
}
static void net_ins(void *c)
{
    GNet *N = c;
    if (N->depth >= LK_MAX_DEPTH) return;
    size_t dim = N->layer[N->depth - 1].in;
    memmove(&N->layer[N->depth], &N->layer[N->depth - 1], sizeof(LKLayer));
    memset(&N->layer[N->depth - 1], 0, sizeof(LKLayer));
    lk_alloc(&N->layer[N->depth - 1], dim, dim, TNN_K0);
    lk_identity(&N->layer[N->depth - 1]);
    N->depth++;
}
static int net_rem(void *c)
{
    GNet *N = c;
    if (N->depth <= 1) return -1;
    size_t idx = N->depth - 2, nin = N->layer[idx].in;
    lk_free(&N->layer[idx]);
    memmove(&N->layer[idx], &N->layer[idx + 1],
            (N->depth - idx - 1) * sizeof(LKLayer));
    memset(&N->layer[N->depth - 1], 0, sizeof(LKLayer));
    N->depth--;
    lk_resize_in(&N->layer[idx], nin);
    return 0;
}
static void net_dyn(void *c, int on) { ((GNet *)c)->dynamic = on ? 1 : 0; }
static size_t net_depth(void *c) { return ((GNet *)c)->depth; }
static size_t net_kf(void *c) { GNet *N = c; return N->layer[N->depth - 1].k; }
static size_t net_params(void *c)
{
    GNet *N = c; size_t n = 0;
    for (size_t i = 0; i < N->depth; i++)
        n += N->layer[i].n_or * (N->layer[i].in + 1);
    return n;
}
static size_t net_nbytes(void *c)
{
    GNet *N = c; size_t n = 0;
    for (size_t i = 0; i < N->depth; i++)
        n += N->layer[i].n_or * (N->layer[i].in + 1) * sizeof(double);
    return n;
}
static void net_free(void *c)
{
    GNet *N = c;
    for (size_t i = 0; i < N->depth; i++) lk_free(&N->layer[i]);
    for (size_t i = 0; i < LK_MAX_DEPTH + 1; i++) free(N->act[i]);
    free(N->dor); free(N->dxa); free(N->dxb);
    free(N);
}


static void net_scale(void *c, size_t idx, size_t in, size_t out)
{
    GNet *N = c;
    lk_scale_arr(N->layer, N->depth, idx, in, out);
}
static size_t net_lin(void *c, size_t idx)
{
    GNet *N = c;
    return idx < N->depth ? N->layer[idx].in : 0;
}
static size_t net_lout(void *c, size_t idx)
{
    GNet *N = c;
    return idx < N->depth ? N->layer[idx].out : 0;
}


static size_t net_lk(void *c, size_t idx)
{
    GNet *N = c;
    return idx < N->depth ? N->layer[idx].k : 0;
}
AltNet type_nn_bpgemm_open(size_t in, size_t out)
{
    GNet *N = (GNet *)calloc(1, sizeof(GNet));
    N->depth = 1; N->dynamic = 1; N->max_or = TNN_MAX_OR;
    lk_alloc(&N->layer[0], in, out, TNN_K0);
    AltNet h = {
        .impl = "type-nn-bpgemm", .ctx = N, .in = in, .out = out,
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
