#ifndef TYPE_NN_DENSE_H
#define TYPE_NN_DENSE_H

#include "type_nn_stack.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    size_t in, out, k;
    double *W, *b, *or_val;
    int kind; /* 0 soa, 1 gemm, 2 q8-shadow, 3 tape */
} DenseLayer;

static inline DenseLayer *dense_new(size_t in, size_t out, size_t k, int kind)
{
    DenseLayer *L = (DenseLayer *)calloc(1, sizeof(DenseLayer));
    L->in = in; L->out = out; L->k = k; L->kind = kind;
    L->W = (double *)calloc(out * k * in, sizeof(double));
    L->b = (double *)calloc(out * k, sizeof(double));
    L->or_val = (double *)calloc(out * k, sizeof(double));
    return L;
}

static inline void dense_init(void *p)
{
    DenseLayer *L = (DenseLayer *)p;
    size_t n_or = L->out * L->k;
    for (size_t r = 0; r < n_or; r++) {
        L->b[r] = tnn_rand();
        for (size_t j = 0; j < L->in; j++)
            L->W[r * L->in + j] = tnn_rand();
    }
}

static inline void dense_fwd_soa(void *p, const double *x, double *y)
{
    DenseLayer *L = (DenseLayer *)p;
    tnn_dense_fwd(L->in, L->out, L->k, L->W, L->b, x, y, L->or_val);
}

static inline void dense_fwd_gemm(void *p, const double *x, double *y)
{
    DenseLayer *L = (DenseLayer *)p;
    size_t rows = L->out * L->k, cols = L->in;
    const size_t RB = 8, CB = 16;
    for (size_t r = 0; r < rows; r++) L->or_val[r] = L->b[r];
    for (size_t r0 = 0; r0 < rows; r0 += RB) {
        size_t r1 = r0 + RB; if (r1 > rows) r1 = rows;
        for (size_t c0 = 0; c0 < cols; c0 += CB) {
            size_t c1 = c0 + CB; if (c1 > cols) c1 = cols;
            for (size_t r = r0; r < r1; r++) {
                const double *w = L->W + r * cols + c0;
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
    for (size_t r = 0; r < rows; r++)
        L->or_val[r] = tnn_clamp(L->or_val[r], -TNN_ORCLIP, TNN_ORCLIP);
    for (size_t i = 0; i < L->out; i++) {
        double prod = 1.0;
        for (size_t t = 0; t < L->k; t++) prod *= L->or_val[i * L->k + t];
        y[i] = tnn_clamp(prod, -TNN_ANDCLIP, TNN_ANDCLIP);
    }
}

static inline void dense_bwd(void *p, const double *x, const double *dy,
                             double *dx, double lr)
{
    DenseLayer *L = (DenseLayer *)p;
    if (dx) memset(dx, 0, L->in * sizeof(double));
    for (size_t i = 0; i < L->out; i++) {
        for (size_t t = 0; t < L->k; t++) {
            double accum = 1.0;
            for (size_t u = 0; u < L->k; u++)
                if (u != t) accum *= L->or_val[i * L->k + u];
            double d_or = dy[i] * accum;
            size_t r = i * L->k + t;
            L->b[r] = tnn_clamp(L->b[r] - lr * d_or, -TNN_WCLIP, TNN_WCLIP);
            double *w = L->W + r * L->in;
            for (size_t j = 0; j < L->in; j++) {
                if (dx) dx[j] += d_or * w[j];
                w[j] = tnn_clamp(w[j] - lr * d_or * x[j], -TNN_WCLIP, TNN_WCLIP);
            }
        }
    }
}

static inline void dense_resize_in(void *p, size_t in)
{
    DenseLayer *L = (DenseLayer *)p;
    tnn_dense_resize_in(&L->W, &L->in, L->out, L->k, in);
}
static inline void dense_resize_out(void *p, size_t out)
{
    DenseLayer *L = (DenseLayer *)p;
    tnn_dense_resize_out(&L->W, &L->b, &L->or_val, L->in, &L->out, L->k, out);
}
static inline void dense_set_k(void *p, size_t k)
{
    DenseLayer *L = (DenseLayer *)p;
    tnn_dense_set_k(&L->W, &L->b, &L->or_val, L->in, L->out, &L->k, k);
}
static inline void dense_identity(void *p)
{
    DenseLayer *L = (DenseLayer *)p;
    tnn_dense_identity(L->W, L->b, L->in, L->out, L->k);
}
static inline size_t dense_in(void *p)  { return ((DenseLayer *)p)->in; }
static inline size_t dense_out(void *p) { return ((DenseLayer *)p)->out; }
static inline size_t dense_k(void *p)   { return ((DenseLayer *)p)->k; }
static inline size_t dense_params(void *p)
{
    DenseLayer *L = (DenseLayer *)p;
    return L->out * L->k * (L->in + 1);
}
static inline size_t dense_nbytes(void *p)
{
    DenseLayer *L = (DenseLayer *)p;
    return sizeof(DenseLayer)
        + sizeof(double) * (L->out * L->k * L->in + 2 * L->out * L->k);
}
static inline void dense_free(void *p)
{
    DenseLayer *L = (DenseLayer *)p;
    free(L->W); free(L->b); free(L->or_val); free(L);
}

#endif
