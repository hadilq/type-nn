#include "type_nn_dense.h"
#include <stdint.h>
#include <math.h>

typedef struct {
    DenseLayer *d;
    int8_t *qW;
    double *scale;
} Q8Layer;

static void q8_sync(Q8Layer *Q)
{
    DenseLayer *L = Q->d;
    size_t n_or = L->out * L->k;
    Q->qW = (int8_t *)realloc(Q->qW, n_or * L->in);
    Q->scale = (double *)realloc(Q->scale, n_or * sizeof(double));
    for (size_t r = 0; r < n_or; r++) {
        double m = 0.0;
        double *w = L->W + r * L->in;
        for (size_t j = 0; j < L->in; j++) {
            double a = fabs(w[j]);
            if (a > m) m = a;
        }
        if (m < 1e-12) m = 1.0;
        Q->scale[r] = m / 127.0;
        for (size_t j = 0; j < L->in; j++) {
            int v = (int)lrint(w[j] / Q->scale[r]);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            Q->qW[r * L->in + j] = (int8_t)v;
        }
    }
}

static void *q8_new(size_t in, size_t out, size_t k)
{
    Q8Layer *Q = (Q8Layer *)calloc(1, sizeof(Q8Layer));
    Q->d = dense_new(in, out, k, 2);
    return Q;
}
static void q8_init(void *p)
{
    Q8Layer *Q = (Q8Layer *)p;
    dense_init(Q->d);
    q8_sync(Q);
}
static void q8_fwd(void *p, const double *x, double *y)
{
    Q8Layer *Q = (Q8Layer *)p;
    DenseLayer *L = Q->d;
    size_t n_or = L->out * L->k;
    for (size_t r = 0; r < n_or; r++) {
        double acc = L->b[r], sc = Q->scale[r];
        const int8_t *q = Q->qW + r * L->in;
        for (size_t j = 0; j < L->in; j++) acc += sc * (double)q[j] * x[j];
        L->or_val[r] = tnn_clamp(acc, -TNN_ORCLIP, TNN_ORCLIP);
    }
    for (size_t i = 0; i < L->out; i++) {
        double prod = 1.0;
        for (size_t t = 0; t < L->k; t++) prod *= L->or_val[i * L->k + t];
        y[i] = tnn_clamp(prod, -TNN_ANDCLIP, TNN_ANDCLIP);
    }
}
static void q8_bwd(void *p, const double *x, const double *dy, double *dx, double lr)
{
    Q8Layer *Q = (Q8Layer *)p;
    dense_bwd(Q->d, x, dy, dx, lr);
    q8_sync(Q);
}
static void q8_rin(void *p, size_t in)  { Q8Layer *Q=p; dense_resize_in(Q->d,in); q8_sync(Q); }
static void q8_rout(void *p, size_t o)  { Q8Layer *Q=p; dense_resize_out(Q->d,o); q8_sync(Q); }
static void q8_sk(void *p, size_t k)    { Q8Layer *Q=p; dense_set_k(Q->d,k); q8_sync(Q); }
static void q8_id(void *p)              { Q8Layer *Q=p; dense_identity(Q->d); q8_sync(Q); }
static size_t q8_in(void *p)  { return dense_in(((Q8Layer *)p)->d); }
static size_t q8_out(void *p) { return dense_out(((Q8Layer *)p)->d); }
static size_t q8_k(void *p)   { return dense_k(((Q8Layer *)p)->d); }
static size_t q8_params(void *p) { return dense_params(((Q8Layer *)p)->d); }
static size_t q8_nbytes(void *p)
{
    Q8Layer *Q = (Q8Layer *)p;
    DenseLayer *L = Q->d;
    size_t n_or = L->out * L->k;
    return sizeof(Q8Layer) + sizeof(DenseLayer)
        + n_or * L->in * sizeof(int8_t)
        + n_or * sizeof(double)  /* scale */
        + n_or * sizeof(double)  /* bias  */
        + n_or * sizeof(double); /* or_val */
}
static void q8_free(void *p)
{
    Q8Layer *Q = (Q8Layer *)p;
    dense_free(Q->d); free(Q->qW); free(Q->scale); free(Q);
}

static const TLayerOps OPS = {
    q8_new, q8_init, q8_fwd, q8_bwd,
    q8_rin, q8_rout, q8_sk, q8_id,
    q8_in, q8_out, q8_k, q8_params, q8_nbytes, q8_free
};

AltNet type_nn_q8_open(size_t in, size_t out)
{
    AltNet h;
    tstack_bind(&h, tstack_open("type-nn-q8", &OPS, in, out));
    return h;
}
