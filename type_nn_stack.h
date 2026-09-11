#ifndef TYPE_NN_STACK_H
#define TYPE_NN_STACK_H

#include "type_nn_alt.h"
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

#define TNN_K0          2
#define TNN_MAX_OR      8
#define TNN_MAX_DEPTH   6
#define TNN_WCLIP       4.0
#define TNN_ORCLIP      4.0
#define TNN_ANDCLIP     32.0

static inline double tnn_clip_gate(double clipped, double C)
{
    return (fabs(clipped) >= C - 1e-12) ? 0.05 : 1.0;
}

static inline double tnn_huber(double e)
{
    if (e > 1.0) return 1.0;
    if (e < -1.0) return -1.0;
    return e;
}

typedef struct TLayerOps {
    void  *(*new)(size_t in, size_t out, size_t k);
    void   (*init)(void *L);
    void   (*fwd)(void *L, const double *x, double *y);
    /* dx may be NULL (input layer). */
    void   (*bwd)(void *L, const double *x, const double *dy, double *dx, double lr);
    void   (*resize_in)(void *L, size_t in);
    void   (*resize_out)(void *L, size_t out);
    void   (*set_k)(void *L, size_t k);
    void   (*identity)(void *L);
    size_t (*in)(void *L);
    size_t (*out)(void *L);
    size_t (*k)(void *L);
    size_t (*params)(void *L);
    size_t (*nbytes)(void *L);
    void   (*free)(void *L);
} TLayerOps;

typedef struct TStack {
    const char *impl;
    const TLayerOps *ops;
    void **layer;
    size_t depth, cap;
    int dynamic;
    size_t max_depth, max_or;
    double *scratch[TNN_MAX_DEPTH + 2];
    size_t scratch_w[TNN_MAX_DEPTH + 2];
} TStack;

TStack *tstack_open(const char *impl, const TLayerOps *ops, size_t in, size_t out);
void    tstack_bind(AltNet *dst, TStack *s);

static inline double tnn_clamp(double x, double lo, double hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
static inline double tnn_rand(void)
{
    return ((double)rand() / (double)RAND_MAX * 2.0 - 1.0) * 0.15;
}
void   tnn_dense_fwd(size_t in, size_t out, size_t k,
                     const double *W, const double *b,
                     const double *x, double *y, double *or_val);
void   tnn_dense_bwd(size_t in, size_t out, size_t k,
                     double *W, double *b,
                     const double *x, const double *dy, double *dx,
                     const double *or_val, double lr);
void   tnn_dense_resize_in(double **W, size_t *in, size_t out, size_t k, size_t nin);
void   tnn_dense_resize_out(double **W, double **b, double **or_val,
                            size_t in, size_t *out, size_t k, size_t nout);
void   tnn_dense_set_k(double **W, double **b, double **or_val,
                       size_t in, size_t out, size_t *k, size_t nk);
void   tnn_dense_identity(double *W, double *b, size_t in, size_t out, size_t k);

#endif
