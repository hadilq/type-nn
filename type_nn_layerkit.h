#ifndef TYPE_NN_LAYERKIT_H
#define TYPE_NN_LAYERKIT_H

#include "type_nn_stack.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Shared And/Or layer used by the optimizer experiments.
   Or = affine sum, And = product of Ors. Grow/shrink + identity live here. */

#define LK_MAX_DEPTH 6

typedef struct {
    size_t in, out, k, n_or;
    double *W, *b, *or_val;
} LKLayer;

static inline void lk_alloc(LKLayer *L, size_t in, size_t out, size_t k)
{
    L->in = in; L->out = out; L->k = k;
    L->n_or = out * k;
    L->W = (double *)realloc(L->W, L->n_or * in * sizeof(double));
    L->b = (double *)realloc(L->b, L->n_or * sizeof(double));
    L->or_val = (double *)realloc(L->or_val, L->n_or * sizeof(double));
    if (L->n_or && in) memset(L->W, 0, L->n_or * in * sizeof(double));
    if (L->n_or) memset(L->b, 0, L->n_or * sizeof(double));
}

static inline void lk_free(LKLayer *L)
{
    free(L->W); free(L->b); free(L->or_val);
    memset(L, 0, sizeof(*L));
}

static inline void lk_init(LKLayer *L)
{
    for (size_t r = 0; r < L->n_or; r++) {
        L->b[r] = tnn_rand();
        for (size_t j = 0; j < L->in; j++)
            L->W[r * L->in + j] = tnn_rand();
    }
}

static inline void lk_identity(LKLayer *L)
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

static inline void lk_fwd(LKLayer *L, const double *x, double *y)
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
                acc += w[j]*x[j]+w[j+1]*x[j+1]+w[j+2]*x[j+2]+w[j+3]*x[j+3];
            for (; j < in; j++) acc += w[j]*x[j];
            L->or_val[r] = tnn_clamp(acc, -TNN_ORCLIP, TNN_ORCLIP);
        }
    }
    if (k == 2) {
        for (size_t i = 0; i < out; i++)
            y[i] = tnn_clamp(L->or_val[i*2]*L->or_val[i*2+1],
                             -TNN_ANDCLIP, TNN_ANDCLIP);
    } else {
        for (size_t i = 0; i < out; i++) {
            double p = 1.0;
            for (size_t t = 0; t < k; t++) p *= L->or_val[i*k+t];
            y[i] = tnn_clamp(p, -TNN_ANDCLIP, TNN_ANDCLIP);
        }
    }
}

/* d_or[t] = dy * Π_{u≠t} Or_u  computed with prefix/suffix, O(k) not O(k²). */
static inline void lk_dor(const LKLayer *L, size_t i, double dy, double *dor)
{
    const size_t k = L->k;
    const double *ov = L->or_val + i * k;
    if (k == 2) {
        dor[0] = dy * ov[1];
        dor[1] = dy * ov[0];
        return;
    }
    double pre[8], suf[8];
    if (k > 8) { /* fall back */
        for (size_t t = 0; t < k && t < 8; t++) {
            double a = 1.0;
            for (size_t u = 0; u < k; u++) if (u != t) a *= ov[u];
            dor[t] = dy * a;
        }
        return;
    }
    pre[0] = 1.0;
    for (size_t t = 1; t < k; t++) pre[t] = pre[t-1] * ov[t-1];
    suf[k-1] = 1.0;
    for (size_t t = k-1; t-- > 0; ) suf[t] = suf[t+1] * ov[t+1];
    for (size_t t = 0; t < k; t++) dor[t] = dy * pre[t] * suf[t];
}

static inline void lk_resize_in(LKLayer *L, size_t nin)
{
    if (nin == L->in) return;
    double *nW = (double *)calloc(L->n_or * nin, sizeof(double));
    size_t copy = L->in < nin ? L->in : nin;
    for (size_t r = 0; r < L->n_or; r++)
        memcpy(nW + r * nin, L->W + r * L->in, copy * sizeof(double));
    free(L->W); L->W = nW; L->in = nin;
}

static inline void lk_resize_out(LKLayer *L, size_t nout)
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
        for (size_t j = 0; j < in; j++) nW[r*in+j] = tnn_rand();
    }
    free(L->W); free(L->b); free(L->or_val);
    L->W = nW; L->b = nb; L->or_val = no;
    L->out = nout; L->n_or = n1;
}

static inline void lk_set_k(LKLayer *L, size_t nk)
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
            memcpy(nW+(i*nk+t)*in, L->W+(i*k0+t)*in, in*sizeof(double));
            nb[i*nk+t] = L->b[i*k0+t];
        }
        for (size_t t = kc; t < nk; t++) nb[i*nk+t] = 1.0;
    }
    free(L->W); free(L->b); free(L->or_val);
    L->W = nW; L->b = nb; L->or_val = no;
    L->k = nk; L->n_or = n1;
}


static inline void lk_scale_arr(LKLayer *L, size_t depth, size_t idx, size_t in, size_t out)
{
    if (idx >= depth) return;
    lk_resize_in(&L[idx], in);
    lk_resize_out(&L[idx], out);
    if (idx + 1 < depth) lk_resize_in(&L[idx + 1], out);
    if (idx > 0) lk_resize_out(&L[idx - 1], in);
}

#endif

