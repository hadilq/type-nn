#include "type_nn_alt.h"
#include "type_nn_stack.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * type-nn-opt — third pass.
 *
 * The int8 packed net (now type-nn-opt-q8) cannot hit XOR MSE 0 because
 * every update is snapped to 1/127. This one keeps full double weights
 * so the And/Or arithmetic matches type-nn-soa / type-nn-gemm exactly,
 * and takes the speed lessons:
 *
 *   • SoA 4-wide loop when in < 16
 *   • blocked GEMV when in >= 16
 *   • k==2 And is one multiply
 *   • activations kept; backward does not re-forward
 *   • nbytes is the payload only: W + b
 */

#define OPT_MAX_DEPTH 6

typedef struct {
    size_t in, out, k, n_or;
    double *W, *b, *or_val;
} OptLayer;

typedef struct {
    OptLayer layer[OPT_MAX_DEPTH];
    double  *act[OPT_MAX_DEPTH + 1];
    size_t   act_w[OPT_MAX_DEPTH + 1];
    size_t   depth;
    int      dynamic;
    size_t   max_or;
} OptNet;

static void layer_alloc(OptLayer *L, size_t in, size_t out, size_t k)
{
    L->in = in; L->out = out; L->k = k;
    L->n_or = out * k;
    L->W = (double *)realloc(L->W, L->n_or * in * sizeof(double));
    L->b = (double *)realloc(L->b, L->n_or * sizeof(double));
    L->or_val = (double *)realloc(L->or_val, L->n_or * sizeof(double));
    if (L->n_or && in) memset(L->W, 0, L->n_or * in * sizeof(double));
    if (L->n_or) memset(L->b, 0, L->n_or * sizeof(double));
}

static void layer_free(OptLayer *L)
{
    free(L->W); free(L->b); free(L->or_val);
    memset(L, 0, sizeof(*L));
}

static void fit_act(OptNet *N, size_t slot, size_t w)
{
    if (N->act_w[slot] >= w && N->act[slot]) return;
    free(N->act[slot]);
    N->act[slot] = (double *)calloc(w ? w : 1, sizeof(double));
    N->act_w[slot] = w ? w : 1;
}

static void layer_init(OptLayer *L)
{
    for (size_t r = 0; r < L->n_or; r++) {
        L->b[r] = tnn_rand();
        for (size_t j = 0; j < L->in; j++)
            L->W[r * L->in + j] = tnn_rand();
    }
}

static void layer_identity(OptLayer *L)
{
    memset(L->W, 0, L->n_or * L->in * sizeof(double));
    memset(L->b, 0, L->n_or * sizeof(double));
    size_t n = L->in < L->out ? L->in : L->out;
    for (size_t i = 0; i < n; i++) {
        L->W[(i * L->k + 0) * L->in + i] = 1.0;
        if (L->k > 1) L->b[i * L->k + 1] = 1.0;
        for (size_t t = 2; t < L->k; t++) L->b[i * L->k + t] = 1.0;
    }
}

static void layer_fwd(OptLayer *L, const double *x, double *y)
{
    const size_t in = L->in, k = L->k, out = L->out;
    if (in >= 16 && L->n_or >= 2) {
        const size_t RB = 8, CB = 16;
        for (size_t r = 0; r < L->n_or; r++) L->or_val[r] = L->b[r];
        for (size_t r0 = 0; r0 < L->n_or; r0 += RB) {
            size_t r1 = r0 + RB; if (r1 > L->n_or) r1 = L->n_or;
            for (size_t c0 = 0; c0 < in; c0 += CB) {
                size_t c1 = c0 + CB; if (c1 > in) c1 = in;
                for (size_t r = r0; r < r1; r++) {
                    const double *w = L->W + r * in + c0;
                    double acc = 0.0;
                    size_t c = c0;
                    for (; c + 3 < c1; c += 4)
                        acc += w[c-c0]*x[c] + w[c-c0+1]*x[c+1]
                             + w[c-c0+2]*x[c+2] + w[c-c0+3]*x[c+3];
                    for (; c < c1; c++) acc += w[c-c0]*x[c];
                    L->or_val[r] += acc;
                }
            }
        }
        for (size_t r = 0; r < L->n_or; r++)
            L->or_val[r] = tnn_clamp(L->or_val[r], -TNN_ORCLIP, TNN_ORCLIP);
    } else {
        for (size_t r = 0; r < L->n_or; r++) {
            const double *w = L->W + r * in;
            double acc = L->b[r];
            size_t j = 0;
            for (; j + 3 < in; j += 4)
                acc += w[j]*x[j] + w[j+1]*x[j+1] + w[j+2]*x[j+2] + w[j+3]*x[j+3];
            for (; j < in; j++) acc += w[j]*x[j];
            L->or_val[r] = tnn_clamp(acc, -TNN_ORCLIP, TNN_ORCLIP);
        }
    }
    if (k == 2) {
        for (size_t i = 0; i < out; i++)
            y[i] = tnn_clamp(L->or_val[i*2] * L->or_val[i*2+1],
                             -TNN_ANDCLIP, TNN_ANDCLIP);
    } else {
        for (size_t i = 0; i < out; i++) {
            double p = 1.0;
            for (size_t t = 0; t < k; t++) p *= L->or_val[i * k + t];
            y[i] = tnn_clamp(p, -TNN_ANDCLIP, TNN_ANDCLIP);
        }
    }
}

static void layer_bwd(OptLayer *L, const double *x, const double *dy,
                      double *dx, double lr)
{
    const size_t in = L->in, k = L->k, out = L->out;
    if (dx) memset(dx, 0, in * sizeof(double));
    for (size_t i = 0; i < out; i++) {
        for (size_t t = 0; t < k; t++) {
            double accum = (k == 2) ? L->or_val[i * 2 + (t ^ 1)] : 1.0;
            if (k != 2) {
                for (size_t u = 0; u < k; u++)
                    if (u != t) accum *= L->or_val[i * k + u];
            }
            double d_or = dy[i] * accum;
            size_t r = i * k + t;
            L->b[r] = tnn_clamp(L->b[r] - lr * d_or, -TNN_WCLIP, TNN_WCLIP);
            double *w = L->W + r * in;
            for (size_t j = 0; j < in; j++) {
                if (dx) dx[j] += d_or * w[j];
                w[j] = tnn_clamp(w[j] - lr * d_or * x[j], -TNN_WCLIP, TNN_WCLIP);
            }
        }
    }
}

static void layer_resize_in(OptLayer *L, size_t nin)
{
    if (nin == L->in) return;
    double *nW = (double *)calloc(L->n_or * nin, sizeof(double));
    size_t copy = L->in < nin ? L->in : nin;
    for (size_t r = 0; r < L->n_or; r++)
        memcpy(nW + r * nin, L->W + r * L->in, copy * sizeof(double));
    free(L->W);
    L->W = nW;
    L->in = nin;
}

static void layer_resize_out(OptLayer *L, size_t nout)
{
    if (nout == L->out) return;
    size_t k = L->k, in = L->in;
    size_t n1 = nout * k, keep = (L->out < nout ? L->out : nout) * k;
    double *nW = (double *)calloc(n1 * in, sizeof(double));
    double *nb = (double *)calloc(n1, sizeof(double));
    double *no = (double *)calloc(n1, sizeof(double));
    memcpy(nW, L->W, keep * in * sizeof(double));
    memcpy(nb, L->b, keep * sizeof(double));
    for (size_t r = keep; r < n1; r++) {
        nb[r] = tnn_rand();
        for (size_t j = 0; j < in; j++) nW[r * in + j] = tnn_rand();
    }
    free(L->W); free(L->b); free(L->or_val);
    L->W = nW; L->b = nb; L->or_val = no;
    L->out = nout;
    L->n_or = n1;
}

static void layer_set_k(OptLayer *L, size_t nk)
{
    if (nk < 1) nk = 1;
    if (nk == L->k) return;
    size_t in = L->in, out = L->out, k0 = L->k, n1 = out * nk;
    double *nW = (double *)calloc(n1 * in, sizeof(double));
    double *nb = (double *)calloc(n1, sizeof(double));
    double *no = (double *)calloc(n1, sizeof(double));
    size_t kc = k0 < nk ? k0 : nk;
    for (size_t i = 0; i < out; i++) {
        for (size_t t = 0; t < kc; t++) {
            memcpy(nW + (i * nk + t) * in, L->W + (i * k0 + t) * in,
                   in * sizeof(double));
            nb[i * nk + t] = L->b[i * k0 + t];
        }
        for (size_t t = kc; t < nk; t++) nb[i * nk + t] = 1.0;
    }
    free(L->W); free(L->b); free(L->or_val);
    L->W = nW; L->b = nb; L->or_val = no;
    L->k = nk;
    L->n_or = n1;
}

static void optn_init(void *ctx)
{
    OptNet *N = (OptNet *)ctx;
    for (size_t i = 0; i < N->depth; i++) layer_init(&N->layer[i]);
}

static void optn_forward(void *ctx, const double *x, double *y)
{
    OptNet *N = (OptNet *)ctx;
    const double *cur = x;
    for (size_t i = 0; i < N->depth; i++) {
        fit_act(N, i + 1, N->layer[i].out);
        layer_fwd(&N->layer[i], cur, N->act[i + 1]);
        cur = N->act[i + 1];
    }
    memcpy(y, cur, N->layer[N->depth - 1].out * sizeof(double));
}

static void optn_backward(void *ctx, const double *x, const double *dy, double lr)
{
    OptNet *N = (OptNet *)ctx;
    if (!N->act[1]) {
        const double *cur = x;
        for (size_t i = 0; i < N->depth; i++) {
            fit_act(N, i + 1, N->layer[i].out);
            layer_fwd(&N->layer[i], cur, N->act[i + 1]);
            cur = N->act[i + 1];
        }
    }
    const double *dcur = dy;
    double *hold = NULL;
    for (size_t i = N->depth; i-- > 0; ) {
        const double *xin = (i == 0) ? x : N->act[i];
        double *dx = NULL;
        if (i > 0) {
            fit_act(N, 0, N->layer[i].in);
            dx = N->act[0];
        }
        layer_bwd(&N->layer[i], xin, dcur, dx, lr);
        if (i > 0) {
            size_t w = N->layer[i].in;
            hold = (double *)realloc(hold, w * sizeof(double));
            memcpy(hold, dx, w * sizeof(double));
            dcur = hold;
        }
    }
    free(hold);
    if (!N->dynamic) return;
    OptLayer *tail = &N->layer[N->depth - 1];
    double mag = 0.0;
    for (size_t i = 0; i < tail->out; i++) mag += fabs(dy[i]);
    if (mag > 1.0 && tail->k < N->max_or && tail->k < 4)
        layer_set_k(tail, tail->k + 1);
}

static void optn_align(void *ctx, size_t in)
{
    layer_resize_in(&((OptNet *)ctx)->layer[0], in);
}
static void optn_set_out(void *ctx, size_t out)
{
    OptNet *N = (OptNet *)ctx;
    layer_resize_out(&N->layer[N->depth - 1], out);
}
static void optn_set_k(void *ctx, size_t k)
{
    OptNet *N = (OptNet *)ctx;
    if (k > N->max_or) k = N->max_or;
    layer_set_k(&N->layer[N->depth - 1], k);
}
static void optn_insert(void *ctx)
{
    OptNet *N = (OptNet *)ctx;
    if (N->depth >= OPT_MAX_DEPTH) return;
    OptLayer *tail = &N->layer[N->depth - 1];
    size_t dim = tail->in;
    memmove(&N->layer[N->depth], tail, sizeof(OptLayer));
    memset(&N->layer[N->depth - 1], 0, sizeof(OptLayer));
    layer_alloc(&N->layer[N->depth - 1], dim, dim, TNN_K0);
    layer_identity(&N->layer[N->depth - 1]);
    N->depth++;
}
static int optn_remove(void *ctx)
{
    OptNet *N = (OptNet *)ctx;
    if (N->depth <= 1) return -1;
    size_t idx = N->depth - 2;
    size_t new_in = N->layer[idx].in;
    layer_free(&N->layer[idx]);
    memmove(&N->layer[idx], &N->layer[idx + 1],
            (N->depth - idx - 1) * sizeof(OptLayer));
    memset(&N->layer[N->depth - 1], 0, sizeof(OptLayer));
    N->depth--;
    layer_resize_in(&N->layer[idx], new_in);
    return 0;
}
static void optn_dyn(void *ctx, int on) { ((OptNet *)ctx)->dynamic = on ? 1 : 0; }
static size_t optn_depth(void *ctx) { return ((OptNet *)ctx)->depth; }
static size_t optn_k(void *ctx)
{
    OptNet *N = (OptNet *)ctx;
    return N->layer[N->depth - 1].k;
}
static size_t optn_params(void *ctx)
{
    OptNet *N = (OptNet *)ctx;
    size_t n = 0;
    for (size_t i = 0; i < N->depth; i++)
        n += N->layer[i].n_or * (N->layer[i].in + 1);
    return n;
}
static size_t optn_nbytes(void *ctx)
{
    OptNet *N = (OptNet *)ctx;
    size_t n = 0;
    for (size_t i = 0; i < N->depth; i++)
        n += N->layer[i].n_or * (N->layer[i].in + 1) * sizeof(double);
    return n;
}
static void optn_free(void *ctx)
{
    OptNet *N = (OptNet *)ctx;
    for (size_t i = 0; i < N->depth; i++) layer_free(&N->layer[i]);
    for (size_t i = 0; i < OPT_MAX_DEPTH + 1; i++) free(N->act[i]);
    free(N);
}


static void optn_scale(void *c, size_t idx, size_t in, size_t out)
{
    OptNet *N = c;
    if (idx >= N->depth) return;
    layer_resize_in(&N->layer[idx], in);
    layer_resize_out(&N->layer[idx], out);
    if (idx + 1 < N->depth) layer_resize_in(&N->layer[idx + 1], out);
    if (idx > 0) layer_resize_out(&N->layer[idx - 1], in);
}
static size_t optn_lin(void *c, size_t idx)
{
    OptNet *N = c;
    return idx < N->depth ? N->layer[idx].in : 0;
}
static size_t optn_lout(void *c, size_t idx)
{
    OptNet *N = c;
    return idx < N->depth ? N->layer[idx].out : 0;
}

AltNet type_nn_opt_open(size_t in, size_t out)
{
    OptNet *N = (OptNet *)calloc(1, sizeof(OptNet));
    N->depth = 1;
    N->dynamic = 1;
    N->max_or = TNN_MAX_OR;
    layer_alloc(&N->layer[0], in, out, TNN_K0);
    AltNet h = {
        .impl = "type-nn-opt",
        .ctx = N, .in = in, .out = out,
        .init = optn_init, .forward = optn_forward, .backward = optn_backward,
        .align_inputs = optn_align, .set_outputs = optn_set_out,
        .set_or_factors = optn_set_k, .insert_identity = optn_insert,
        .remove_hidden = optn_remove, .set_dynamic = optn_dyn,
        .depth = optn_depth, .or_factors = optn_k,
        .param_count = optn_params, .nbytes = optn_nbytes, .free = optn_free,
        .scale_layer = optn_scale, .layer_in = optn_lin, .layer_out = optn_lout,
    };
    return h;
}
