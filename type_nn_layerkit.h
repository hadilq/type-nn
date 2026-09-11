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
    size_t in, out, k, R, n_or;
    double *W, *b, *or_val;
} LKLayer;

static inline void lk_alloc(LKLayer *L, size_t in, size_t out, size_t k)
{
    L->in = in; L->out = out; L->k = k;
    if (!L->R) L->R = 1;
    L->n_or = out * L->R * k;
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
    size_t R = L->R ? L->R : 1, k = L->k;
    for (size_t i = 0; i < n; i++) {
        L->W[((i * R + 0) * k + 0) * L->in + i] = 1.0;
        if (k > 1) L->b[(i * R + 0) * k + 1] = 1.0;
        for (size_t t = 2; t < k; t++) L->b[(i * R + 0) * k + t] = 1.0;
        for (size_t p = 1; p < R; p++)
            for (size_t t = 0; t < k; t++)
                L->b[(i * R + p) * k + t] = 1.0;
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
    size_t R = L->R ? L->R : 1;
    for (size_t i = 0; i < out; i++) {
        double acc = 1.0;
        for (size_t p = 0; p < R; p++) {
            double pr = 1.0;
            size_t base = (i * R + p) * k;
            for (size_t t = 0; t < k; t++) pr *= L->or_val[base + t];
            acc *= pr;
        }
        y[i] = tnn_clamp(acc, -TNN_ANDCLIP, TNN_ANDCLIP);
    }
}

/* d_or[t] = dy * Π_{u≠t} Or_u  computed with prefix/suffix, O(k) not O(k²). */
static inline void lk_dor_term(const LKLayer *L, size_t i, size_t p,
                               double dy, double *dor)
{
    const size_t k = L->k;
    size_t R = L->R ? L->R : 1;
    if (p >= R) p = 0;
    const double *ov = L->or_val + (i * R + p) * k;
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

static inline void lk_dor(const LKLayer *L, size_t i, double dy, double *dor)
{
    lk_dor_term(L, i, 0, dy, dor);
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
    size_t R = L->R ? L->R : 1;
    size_t n1 = nout * R * k, keep = (L->out < nout ? L->out : nout) * R * k;
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
    size_t in = L->in, out = L->out, k0 = L->k, R = L->R ? L->R : 1;
    size_t n1 = out * R * nk;
    double *nW = (double *)calloc(n1 * in, sizeof(double));
    double *nb = (double *)calloc(n1, sizeof(double));
    double *no = (double *)calloc(n1, sizeof(double));
    size_t kc = k0 < nk ? k0 : nk;
    for (size_t i = 0; i < out; i++) {
        for (size_t p = 0; p < R; p++) {
            for (size_t t = 0; t < kc; t++) {
                memcpy(nW+((i*R+p)*nk+t)*in, L->W+((i*R+p)*k0+t)*in, in*sizeof(double));
                nb[(i*R+p)*nk+t] = L->b[(i*R+p)*k0+t];
            }
            for (size_t t = kc; t < nk; t++) nb[(i*R+p)*nk+t] = 1.0;
        }
    }
    free(L->W); free(L->b); free(L->or_val);
    L->W = nW; L->b = nb; L->or_val = no;
    L->k = nk; L->n_or = n1;
}

/* Extra And (product of Ors). New And ≡ 1: every Or is (b=1, W=0). */
static inline void lk_set_R(LKLayer *L, size_t nR)
{
    if (nR < 1) nR = 1;
    if (!L->R) L->R = 1;
    if (nR == L->R) return;
    size_t in = L->in, out = L->out, k = L->k, R0 = L->R;
    size_t n1 = out * nR * k;
    double *nW = (double *)calloc(n1 * in, sizeof(double));
    double *nb = (double *)calloc(n1, sizeof(double));
    double *no = (double *)calloc(n1, sizeof(double));
    size_t Rc = R0 < nR ? R0 : nR;
    for (size_t i = 0; i < out; i++) {
        for (size_t p = 0; p < Rc; p++) {
            for (size_t t = 0; t < k; t++) {
                memcpy(nW+((i*nR+p)*k+t)*in, L->W+((i*R0+p)*k+t)*in, in*sizeof(double));
                nb[(i*nR+p)*k+t] = L->b[(i*R0+p)*k+t];
            }
        }
        for (size_t p = Rc; p < nR; p++)
            for (size_t t = 0; t < k; t++)
                nb[(i*nR+p)*k+t] = 1.0;
    }
    free(L->W); free(L->b); free(L->or_val);
    L->W = nW; L->b = nb; L->or_val = no;
    L->R = nR; L->n_or = n1;
}


static inline int lk_or_is_ones(const LKLayer *L, size_t r)
{
    if (fabs(L->b[r] - 1.0) > 0.2) return 0;
    for (size_t j = 0; j < L->in; j++)
        if (fabs(L->W[r * L->in + j]) > 0.2) return 0;
    return 1;
}

static inline int lk_or_is_dead(const LKLayer *L, size_t r)
{
    if (fabs(L->b[r]) > 0.05) return 0;
    for (size_t j = 0; j < L->in; j++)
        if (fabs(L->W[r * L->in + j]) > 0.05) return 0;
    return 1;
}

static inline int lk_and_is_ones(const LKLayer *L, size_t i, size_t p)
{
    size_t R = L->R ? L->R : 1;
    for (size_t t = 0; t < L->k; t++)
        if (!lk_or_is_ones(L, (i * R + p) * L->k + t)) return 0;
    return 1;
}

static inline void lk_paint_or_ones(LKLayer *L, size_t r)
{
    L->b[r] = 1.0;
    if (L->in) memset(L->W + r * L->in, 0, L->in * sizeof(double));
}

static inline void lk_paint_dummies(LKLayer *L)
{
    size_t R = L->R ? L->R : 1;
    if (!L->k || !L->out) return;
    for (size_t i = 0; i < L->out; i++) {
        for (size_t p = 0; p < R; p++)
            lk_paint_or_ones(L, (i * R + p) * L->k + (L->k - 1));
        if (R >= 2)
            for (size_t t = 0; t < L->k; t++)
                lk_paint_or_ones(L, (i * R + (R - 1)) * L->k + t);
    }
}

static inline void lk_ensure_or_dummy(LKLayer *L, unsigned *or_add)
{
    size_t R = L->R ? L->R : 1;
    if (L->k < 1) return;
    int has = 0;
    for (size_t i = 0; i < L->out && !has; i++)
        for (size_t p = 0; p < R; p++)
            if (lk_or_is_ones(L, (i * R + p) * L->k + (L->k - 1))) has = 1;
    if (!has && L->k < TNN_MAX_OR) {
        lk_set_k(L, L->k + 1);
        R = L->R ? L->R : 1;
        for (size_t i = 0; i < L->out; i++)
            for (size_t p = 0; p < R; p++)
                lk_paint_or_ones(L, (i * R + p) * L->k + (L->k - 1));
        if (or_add) (*or_add)++;
    }
}

static inline void lk_prune_or_dummy(LKLayer *L, unsigned *or_drop)
{
    size_t R = L->R ? L->R : 1;
    if (L->k <= 2) return;
    int last_id = 1, extra = 0, last_dead = 1;
    for (size_t i = 0; i < L->out; i++) {
        for (size_t p = 0; p < R; p++) {
            size_t last = (i * R + p) * L->k + (L->k - 1);
            if (!lk_or_is_ones(L, last)) last_id = 0;
            if (!lk_or_is_dead(L, last)) last_dead = 0;
            for (size_t t = 0; t + 1 < L->k; t++)
                if (lk_or_is_ones(L, (i * R + p) * L->k + t)) extra = 1;
        }
    }
    if ((last_id && extra) || (last_dead && !last_id)) {
        lk_set_k(L, L->k - 1);
        if (or_drop) (*or_drop)++;
    }
}

static inline void lk_ensure_and_dummy(LKLayer *L, unsigned *and_add)
{
    size_t R = L->R ? L->R : 1;
    if (!L->out) return;
    int has = 0;
    for (size_t i = 0; i < L->out; i++)
        if (lk_and_is_ones(L, i, R - 1)) has = 1;
    if (!has && R < 4) {
        lk_set_R(L, R + 1);
        for (size_t i = 0; i < L->out; i++)
            for (size_t t = 0; t < L->k; t++)
                lk_paint_or_ones(L, (i * L->R + (L->R - 1)) * L->k + t);
        if (and_add) (*and_add)++;
    }
}

static inline void lk_prune_and_dummy(LKLayer *L, unsigned *and_drop)
{
    size_t R = L->R ? L->R : 1;
    if (R <= 1) return;
    int last_id = 1, extra = 0;
    for (size_t i = 0; i < L->out; i++) {
        if (!lk_and_is_ones(L, i, R - 1)) last_id = 0;
        for (size_t p = 0; p + 1 < R; p++)
            if (lk_and_is_ones(L, i, p)) extra = 1;
    }
    if (last_id && extra) {
        lk_set_R(L, R - 1);
        if (and_drop) (*and_drop)++;
    }
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

