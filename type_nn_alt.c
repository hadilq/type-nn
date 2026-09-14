#include "type_nn_alt.h"
#include "type_nn_cmlp.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void type_nn_alt_train(AltNet *a, double **X, double **Y,
                       size_t n, size_t epochs, double lr)
{
    double *pred = (double *)calloc(a->out, sizeof(double));
    double *dy = (double *)calloc(a->out, sizeof(double));
    size_t *ord = (size_t *)malloc(n * sizeof(size_t));
    if (a->set_corpus) a->set_corpus(a->ctx, n);
    for (size_t ep = 0; ep < epochs; ep++) {
        for (size_t i = 0; i < n; i++) ord[i] = i;
        unsigned st = 34972u ^ (unsigned)((ep + 1u) * 0x9E3779B9u);
        if (st == 0) st = 1;
        for (size_t i = n; i > 1; i--) {
            unsigned x = st;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            st = x;
            size_t j = (size_t)(x % (unsigned)i);
            size_t tmp = ord[i - 1];
            ord[i - 1] = ord[j];
            ord[j] = tmp;
        }
        for (size_t s = 0; s < n; s++) {
            size_t si = ord[s];
            a->forward(a->ctx, X[si], pred);
            double inv = 1.0 / (double)(a->out ? a->out : 1);
            for (size_t k = 0; k < a->out; k++) {
                double e = pred[k] - Y[si][k];
                if (e > 1.0) e = 1.0;
                if (e < -1.0) e = -1.0;
                dy[k] = e * inv;
            }
            a->backward(a->ctx, X[si], dy, lr);
        }
    }
    free(ord);
    free(pred);
    free(dy);
}

void type_nn_alt_snap(const AltNet *a, TnnSnap *s)
{
    size_t i;
    memset(s, 0, sizeof(*s));
    if (!a || !a->depth) return;
    s->depth = a->depth(a->ctx);
    if (s->depth > TNN_SNAP_MAX) s->depth = TNN_SNAP_MAX;
    for (i = 0; i < s->depth; i++) {
        size_t out = 0, k = 2;
        if (a->layer_out) out = a->layer_out(a->ctx, i);
        else if (i + 1 == s->depth) out = a->out;
        if (a->layer_k) k = a->layer_k(a->ctx, i);
        else if (a->or_factors && i + 1 == s->depth) k = a->or_factors(a->ctx);
        if (k < 1) k = 1;
        s->n_and[i] = out;
        s->n_or[i] = out * k;
    }
}

void type_nn_alt_dyn_score(const TnnSnap *b, const TnnSnap *a,
                           double *dyn_scale, int *dyn_depth)
{
    size_t n = b->depth > a->depth ? b->depth : a->depth;
    double m = 0.0;
    size_t i;
    if (n > TNN_SNAP_MAX) n = TNN_SNAP_MAX;
    for (i = 0; i < n; i++) {
        long d_and = (long)(i < a->depth ? a->n_and[i] : 0) -
                     (long)(i < b->depth ? b->n_and[i] : 0);
        long d_or  = (long)(i < a->depth ? a->n_or[i]  : 0) -
                     (long)(i < b->depth ? b->n_or[i]  : 0);
        if (d_and < 0) d_and = -d_and;
        if (d_or < 0) d_or = -d_or;
        m += (double)d_or + (double)d_and + (double)d_or * (double)d_and;
    }
    if (dyn_scale) *dyn_scale = m;
    if (dyn_depth) *dyn_depth = (int)a->depth - (int)b->depth;
}

size_t type_nn_alt_count(void) { return 1; }

const char *type_nn_alt_name(size_t i)
{
    return i == 0 ? "c-mlp" : NULL;
}

AltNet (*type_nn_alt_opener(size_t i))(size_t, size_t)
{
    return i == 0 ? type_nn_cmlp_open : NULL;
}
