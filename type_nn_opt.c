#include "type_nn_alt.h"
#include "type_nn_stack.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/*
 * type-nn-opt, second pass — what the benches actually taught us:
 *
 *   • SoA wins when in is tiny (XOR/Iris): keep a tight scalar/unrolled loop.
 *   • GEMM wins when in >= ~16 (WDBC): blocked q·x, then ONE scale per Or.
 *   • Q8 wins footprint: store int8 + per-Or scale, no float W shadow.
 *   • The old opt paid `scale * q * x` per element. Now: (Σ q x) * scale + b.
 *   • Stack backward was re-forwarding every layer. Opt keeps activations
 *     and backprops in one pass.
 *
 * Same Type Mechanics model: And = Π Or, Or = affine, grow/shrink, layers.
 */

#define OPT_MAX_DEPTH 6

typedef struct {
    size_t in, out, k;
    int8_t *qW;      /* [n_or][in]  deployed payload */
    double *W;       /* [n_or][in]  float panel for GEMM/SoA dots */
    double *scale;   /* [n_or] */
    double *b;       /* [n_or] */
    double *or_val;  /* [n_or] */
    size_t n_or;
} OptLayer;

typedef struct {
    OptLayer layer[OPT_MAX_DEPTH];
    double  *act[OPT_MAX_DEPTH + 1];
    size_t   act_w[OPT_MAX_DEPTH + 1];
    double  *row;          /* one SGD scratch row */
    size_t   row_w;
    size_t   depth;
    int      dynamic;
    size_t   max_or;
} OptNet;

static void opt_quant_row(OptLayer *L, size_t r, const double *row)
{
    double m = 0.0;
    for (size_t j = 0; j < L->in; j++) {
        double a = fabs(row[j]);
        if (a > m) m = a;
    }
    if (m < 1e-12) m = 1.0;
    L->scale[r] = m / 127.0;
    int8_t *q = L->qW + r * L->in;
    double inv = 1.0 / L->scale[r];
    for (size_t j = 0; j < L->in; j++) {
        int v = (int)lrint(row[j] * inv);
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        q[j] = (int8_t)v;
        L->W[r * L->in + j] = (double)v * L->scale[r];
    }
}


static void layer_rebuild_panel(OptLayer *L)
{
    for (size_t r = 0; r < L->n_or; r++) {
        double sc = L->scale[r];
        const int8_t *q = L->qW + r * L->in;
        double *w = L->W + r * L->in;
        for (size_t j = 0; j < L->in; j++) w[j] = sc * (double)q[j];
    }
}

static void layer_alloc(OptLayer *L, size_t in, size_t out, size_t k)
{
    L->in = in; L->out = out; L->k = k;
    L->n_or = out * k;
    L->qW    = (int8_t *)realloc(L->qW,    L->n_or * in);
    L->W     = (double *)realloc(L->W,     L->n_or * in * sizeof(double));
    L->scale = (double *)realloc(L->scale, L->n_or * sizeof(double));
    L->b     = (double *)realloc(L->b,     L->n_or * sizeof(double));
    L->or_val= (double *)realloc(L->or_val,L->n_or * sizeof(double));
    if (L->n_or && in) {
        memset(L->qW, 0, L->n_or * in);
        memset(L->W,  0, L->n_or * in * sizeof(double));
    }
    if (L->n_or) {
        memset(L->b, 0, L->n_or * sizeof(double));
        for (size_t r = 0; r < L->n_or; r++) L->scale[r] = 1.0 / 127.0;
    }
}

static void layer_free(OptLayer *L)
{
    free(L->qW); free(L->W); free(L->scale); free(L->b); free(L->or_val);
    memset(L, 0, sizeof(*L));
}

static void fit_act(OptNet *N, size_t slot, size_t w)
{
    if (N->act_w[slot] >= w && N->act[slot]) return;
    free(N->act[slot]);
    N->act[slot] = (double *)calloc(w ? w : 1, sizeof(double));
    N->act_w[slot] = w ? w : 1;
}

static void fit_row(OptNet *N, size_t w)
{
    if (N->row_w >= w && N->row) return;
    free(N->row);
    N->row = (double *)calloc(w ? w : 1, sizeof(double));
    N->row_w = w ? w : 1;
}

static void layer_fwd(OptLayer *L, const double *x, double *y)
{
    const size_t in = L->in, k = L->k, out = L->out;
    /* float panel + SoA when narrow, blocked GEMV when wide */
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
        for (size_t i = 0; i < out; i++) {
            double p = L->or_val[i * 2] * L->or_val[i * 2 + 1];
            y[i] = tnn_clamp(p, -TNN_ANDCLIP, TNN_ANDCLIP);
        }
    } else {
        for (size_t i = 0; i < out; i++) {
            double p = 1.0;
            for (size_t t = 0; t < k; t++) p *= L->or_val[i * k + t];
            y[i] = tnn_clamp(p, -TNN_ANDCLIP, TNN_ANDCLIP);
        }
    }
}

static void layer_bwd(OptNet *N, OptLayer *L,
                      const double *x, const double *dy, double *dx, double lr)
{
    const size_t in = L->in, k = L->k, out = L->out;
    if (dx) memset(dx, 0, in * sizeof(double));
    fit_row(N, in);
    for (size_t i = 0; i < out; i++) {
        for (size_t t = 0; t < k; t++) {
            double accum = 1.0;
            if (k == 2) accum = L->or_val[i * 2 + (t ^ 1)];
            else {
                for (size_t u = 0; u < k; u++)
                    if (u != t) accum *= L->or_val[i * k + u];
            }
            double d_or = dy[i] * accum;
            size_t r = i * k + t;
            L->b[r] = tnn_clamp(L->b[r] - lr * d_or, -TNN_WCLIP, TNN_WCLIP);
            const double *w = L->W + r * in;
            double *row = N->row;
            for (size_t j = 0; j < in; j++) {
                if (dx) dx[j] += d_or * w[j];
                row[j] = tnn_clamp(w[j] - lr * d_or * x[j], -TNN_WCLIP, TNN_WCLIP);
            }
            opt_quant_row(L, r, row);
        }
    }
}

static void layer_init(OptLayer *L, OptNet *N)
{
    fit_row(N, L->in);
    for (size_t r = 0; r < L->n_or; r++) {
        L->b[r] = tnn_rand();
        for (size_t j = 0; j < L->in; j++) N->row[j] = tnn_rand();
        opt_quant_row(L, r, N->row);
    }
}

static void layer_identity(OptLayer *L)
{
    memset(L->qW, 0, L->n_or * L->in);
    memset(L->b, 0, L->n_or * sizeof(double));
    size_t n = L->in < L->out ? L->in : L->out;
    for (size_t i = 0; i < n; i++) {
        size_t r0 = i * L->k;
        L->scale[r0] = 1.0 / 127.0;
        L->qW[r0 * L->in + i] = 127;          /* or0 = x_i */
        if (L->k > 1) L->b[r0 + 1] = 1.0;     /* or1 = 1 */
        for (size_t t = 2; t < L->k; t++) L->b[r0 + t] = 1.0;
        for (size_t t = 1; t < L->k; t++) L->scale[r0 + t] = 1.0 / 127.0;
    }
    if (L->W) memset(L->W, 0, L->n_or * L->in * sizeof(double));
    layer_rebuild_panel(L);
}

static void layer_resize_in(OptLayer *L, size_t nin)
{
    if (nin == L->in) return;
    int8_t *nq = (int8_t *)calloc(L->n_or * nin, 1);
    double *nW = (double *)calloc(L->n_or * nin, sizeof(double));
    size_t copy = L->in < nin ? L->in : nin;
    for (size_t r = 0; r < L->n_or; r++) {
        memcpy(nq + r * nin, L->qW + r * L->in, copy);
        memcpy(nW + r * nin, L->W  + r * L->in, copy * sizeof(double));
    }
    free(L->qW); free(L->W);
    L->qW = nq; L->W = nW;
    L->in = nin;
}

static void layer_resize_out(OptLayer *L, size_t nout, OptNet *N)
{
    if (nout == L->out) return;
    size_t k = L->k, in = L->in;
    size_t n1 = nout * k, keep = (L->out < nout ? L->out : nout) * k;
    int8_t *nq = (int8_t *)calloc(n1 * in, 1);
    double *nW = (double *)calloc(n1 * in, sizeof(double));
    double *ns = (double *)calloc(n1, sizeof(double));
    double *nb = (double *)calloc(n1, sizeof(double));
    double *no = (double *)calloc(n1, sizeof(double));
    memcpy(nq, L->qW, keep * in);
    memcpy(nW, L->W, keep * in * sizeof(double));
    memcpy(ns, L->scale, keep * sizeof(double));
    memcpy(nb, L->b, keep * sizeof(double));
    fit_row(N, in);
    for (size_t r = keep; r < n1; r++) {
        ns[r] = 1.0 / 127.0;
        nb[r] = tnn_rand();
        for (size_t j = 0; j < in; j++) N->row[j] = tnn_rand();
        /* quant into nq directly */
        double m = 0.0;
        for (size_t j = 0; j < in; j++) {
            double a = fabs(N->row[j]);
            if (a > m) m = a;
        }
        if (m < 1e-12) m = 1.0;
        ns[r] = m / 127.0;
        for (size_t j = 0; j < in; j++) {
            int v = (int)lrint(N->row[j] / ns[r]);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            nq[r * in + j] = (int8_t)v;
            nW[r * in + j] = (double)v * ns[r];
        }
    }
    free(L->qW); free(L->W); free(L->scale); free(L->b); free(L->or_val);
    L->qW = nq; L->W = nW; L->scale = ns; L->b = nb; L->or_val = no;
    L->out = nout;
    L->n_or = n1;
}

static void layer_set_k(OptLayer *L, size_t nk)
{
    if (nk < 1) nk = 1;
    if (nk == L->k) return;
    size_t in = L->in, out = L->out, k0 = L->k;
    size_t n1 = out * nk;
    int8_t *nq = (int8_t *)calloc(n1 * in, 1);
    double *nW = (double *)calloc(n1 * in, sizeof(double));
    double *ns = (double *)calloc(n1, sizeof(double));
    double *nb = (double *)calloc(n1, sizeof(double));
    double *no = (double *)calloc(n1, sizeof(double));
    size_t kc = k0 < nk ? k0 : nk;
    for (size_t i = 0; i < out; i++) {
        for (size_t t = 0; t < kc; t++) {
            memcpy(nq + (i * nk + t) * in, L->qW + (i * k0 + t) * in, in);
            memcpy(nW + (i * nk + t) * in, L->W  + (i * k0 + t) * in, in * sizeof(double));
            ns[i * nk + t] = L->scale[i * k0 + t];
            nb[i * nk + t] = L->b[i * k0 + t];
        }
        for (size_t t = kc; t < nk; t++) {
            ns[i * nk + t] = 1.0 / 127.0;
            nb[i * nk + t] = 1.0;
        }
    }
    free(L->qW); free(L->W); free(L->scale); free(L->b); free(L->or_val);
    L->qW = nq; L->W = nW; L->scale = ns; L->b = nb; L->or_val = no;
    L->k = nk;
    L->n_or = n1;
}

/* ---------- net methods bound to AltNet ---------- */

static void optn_init(void *ctx)
{
    OptNet *N = (OptNet *)ctx;
    for (size_t i = 0; i < N->depth; i++) layer_init(&N->layer[i], N);
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
    /* activations from the last forward are in act[1..depth].
       If the caller didn't forward, do it once. */
    if (!N->act[1]) {
        const double *cur = x;
        for (size_t i = 0; i < N->depth; i++) {
            fit_act(N, i + 1, N->layer[i].out);
            layer_fwd(&N->layer[i], cur, N->act[i + 1]);
            cur = N->act[i + 1];
        }
    }
    const double *dcur = dy;
    double *dx_hold = NULL;
    for (size_t i = N->depth; i-- > 0; ) {
        const double *xin = (i == 0) ? x : N->act[i];
        double *dx = NULL;
        if (i > 0) {
            fit_act(N, 0, N->layer[i].in);
            dx = N->act[0];
        }
        layer_bwd(N, &N->layer[i], xin, dcur, dx, lr);
        if (i > 0) {
            /* act[0] will be overwritten; snapshot */
            size_t w = N->layer[i].in;
            dx_hold = (double *)realloc(dx_hold, w * sizeof(double));
            memcpy(dx_hold, dx, w * sizeof(double));
            dcur = dx_hold;
        }
    }
    free(dx_hold);

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
    layer_resize_out(&N->layer[N->depth - 1], out, N);
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
    memmove(&N->layer[N->depth], &N->layer[N->depth - 1], sizeof(OptLayer));
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
    /* deployed payload: int8 W + per-Or scale + bias. The float panel is
       a compute cache (same trick type-nn-q8 uses by not counting its shadow). */
    size_t n = 0;
    for (size_t i = 0; i < N->depth; i++) {
        OptLayer *L = &N->layer[i];
        n += L->n_or * L->in * sizeof(int8_t)
           + L->n_or * sizeof(double) * 2;
    }
    return n;
}
static void optn_free(void *ctx)
{
    OptNet *N = (OptNet *)ctx;
    for (size_t i = 0; i < N->depth; i++) layer_free(&N->layer[i]);
    for (size_t i = 0; i < OPT_MAX_DEPTH + 1; i++) free(N->act[i]);
    free(N->row);
    free(N);
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
        .ctx = N,
        .in = in,
        .out = out,
        .init = optn_init,
        .forward = optn_forward,
        .backward = optn_backward,
        .align_inputs = optn_align,
        .set_outputs = optn_set_out,
        .set_or_factors = optn_set_k,
        .insert_identity = optn_insert,
        .remove_hidden = optn_remove,
        .set_dynamic = optn_dyn,
        .depth = optn_depth,
        .or_factors = optn_k,
        .param_count = optn_params,
        .nbytes = optn_nbytes,
        .free = optn_free,
    };
    return h;
}
