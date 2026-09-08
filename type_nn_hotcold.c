#include "type_nn_dense.h"

#define HOT_MAX 16

typedef struct {
    DenseLayer *d;
    size_t hot, cold;
} HotLayer;

static void hot_split(HotLayer *H)
{
    H->hot = H->d->in < HOT_MAX ? H->d->in : HOT_MAX;
    H->cold = H->d->in - H->hot;
}

static void *hot_new(size_t in, size_t out, size_t k)
{
    HotLayer *H = (HotLayer *)calloc(1, sizeof(HotLayer));
    H->d = dense_new(in, out, k, 0);
    hot_split(H);
    return H;
}
static void hot_init(void *p) { dense_init(((HotLayer *)p)->d); }
static void hot_fwd(void *p, const double *x, double *y)
{
    HotLayer *H = (HotLayer *)p;
    DenseLayer *L = H->d;
    size_t n_or = L->out * L->k;
    for (size_t r = 0; r < n_or; r++) {
        double acc = L->b[r];
        const double *w = L->W + r * L->in;
        for (size_t j = 0; j < H->hot; j++) acc += w[j] * x[j];
        for (size_t j = H->hot; j < L->in; j++) acc += w[j] * x[j];
        L->or_val[r] = tnn_clamp(acc, -TNN_ORCLIP, TNN_ORCLIP);
    }
    for (size_t i = 0; i < L->out; i++) {
        double prod = 1.0;
        for (size_t t = 0; t < L->k; t++) prod *= L->or_val[i * L->k + t];
        y[i] = tnn_clamp(prod, -TNN_ANDCLIP, TNN_ANDCLIP);
    }
}
static void hot_bwd(void *p, const double *x, const double *dy, double *dx, double lr)
{
    dense_bwd(((HotLayer *)p)->d, x, dy, dx, lr);
}
static void hot_rin(void *p, size_t in)
{
    HotLayer *H = (HotLayer *)p;
    dense_resize_in(H->d, in);
    hot_split(H);
}
static void hot_rout(void *p, size_t o) { dense_resize_out(((HotLayer *)p)->d, o); }
static void hot_sk(void *p, size_t k)   { dense_set_k(((HotLayer *)p)->d, k); }
static void hot_id(void *p)             { dense_identity(((HotLayer *)p)->d); }
static size_t hot_in(void *p)  { return dense_in(((HotLayer *)p)->d); }
static size_t hot_out(void *p) { return dense_out(((HotLayer *)p)->d); }
static size_t hot_k(void *p)   { return dense_k(((HotLayer *)p)->d); }
static size_t hot_params(void *p) { return dense_params(((HotLayer *)p)->d); }
static size_t hot_nbytes(void *p)
{
    HotLayer *H = (HotLayer *)p;
    return sizeof(HotLayer) + dense_nbytes(H->d);
}
static void hot_free(void *p)
{
    HotLayer *H = (HotLayer *)p;
    dense_free(H->d); free(H);
}

static const TLayerOps OPS = {
    hot_new, hot_init, hot_fwd, hot_bwd,
    hot_rin, hot_rout, hot_sk, hot_id,
    hot_in, hot_out, hot_k, hot_params, hot_nbytes, hot_free
};

AltNet type_nn_hotcold_open(size_t in, size_t out)
{
    AltNet h;
    tstack_bind(&h, tstack_open("type-nn-hotcold", &OPS, in, out));
    return h;
}
