#include "type_nn_stack.h"
#include "type_nn_alt.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void tnn_dense_fwd(size_t in, size_t out, size_t k,
                   const double *W, const double *b,
                   const double *x, double *y, double *or_val)
{
    size_t n_or = out * k;
    for (size_t r = 0; r < n_or; r++) {
        double s = b[r];
        const double *w = W + r * in;
        for (size_t j = 0; j < in; j++) s += w[j] * x[j];
        or_val[r] = tnn_clamp(s, -TNN_ORCLIP, TNN_ORCLIP);
    }
    for (size_t i = 0; i < out; i++) {
        double p = 1.0;
        for (size_t t = 0; t < k; t++) p *= or_val[i * k + t];
        y[i] = tnn_clamp(p, -TNN_ANDCLIP, TNN_ANDCLIP);
    }
}

void tnn_dense_bwd(size_t in, size_t out, size_t k,
                   double *W, double *b,
                   const double *x, const double *dy, double *dx,
                   const double *or_val, double lr)
{
    if (dx) {
        for (size_t j = 0; j < in; j++) dx[j] = 0.0;
    }
    for (size_t i = 0; i < out; i++) {
        for (size_t t = 0; t < k; t++) {
            double accum = 1.0;
            for (size_t u = 0; u < k; u++)
                if (u != t) accum *= or_val[i * k + u];
            double d_or = dy[i] * accum;
            size_t r = i * k + t;
            b[r] = tnn_clamp(b[r] - lr * d_or, -TNN_WCLIP, TNN_WCLIP);
            double *w = W + r * in;
            for (size_t j = 0; j < in; j++) {
                w[j] = tnn_clamp(w[j] - lr * d_or * x[j], -TNN_WCLIP, TNN_WCLIP);
                if (dx) dx[j] += d_or * w[j];
            }
        }
    }
}

void tnn_dense_resize_in(double **W, size_t *in, size_t out, size_t k, size_t nin)
{
    if (nin == *in) return;
    size_t n_or = out * k;
    double *nw = (double *)calloc(n_or * nin, sizeof(double));
    size_t copy = *in < nin ? *in : nin;
    for (size_t r = 0; r < n_or; r++)
        for (size_t j = 0; j < copy; j++)
            nw[r * nin + j] = (*W)[r * (*in) + j];
    free(*W);
    *W = nw;
    *in = nin;
}

void tnn_dense_resize_out(double **W, double **b, double **or_val,
                          size_t in, size_t *out, size_t k, size_t nout)
{
    if (nout == *out) return;
    size_t o0 = *out, n_or1 = nout * k;
    double *nw = (double *)calloc(n_or1 * in, sizeof(double));
    double *nb = (double *)calloc(n_or1, sizeof(double));
    double *no = (double *)calloc(n_or1, sizeof(double));
    size_t keep = o0 < nout ? o0 : nout;
    for (size_t r = 0; r < keep * k; r++) {
        nb[r] = (*b)[r];
        for (size_t j = 0; j < in; j++) nw[r * in + j] = (*W)[r * in + j];
    }
    for (size_t r = keep * k; r < n_or1; r++) {
        nb[r] = tnn_rand();
        for (size_t j = 0; j < in; j++) nw[r * in + j] = tnn_rand();
    }
    free(*W); free(*b); free(*or_val);
    *W = nw; *b = nb; *or_val = no;
    *out = nout;
}

void tnn_dense_set_k(double **W, double **b, double **or_val,
                     size_t in, size_t out, size_t *k, size_t nk)
{
    if (nk < 1) nk = 1;
    if (nk == *k) return;
    size_t k0 = *k;
    double *nw = (double *)calloc(out * nk * in, sizeof(double));
    double *nb = (double *)calloc(out * nk, sizeof(double));
    double *no = (double *)calloc(out * nk, sizeof(double));
    size_t kc = k0 < nk ? k0 : nk;
    for (size_t i = 0; i < out; i++) {
        for (size_t t = 0; t < kc; t++) {
            nb[i * nk + t] = (*b)[i * k0 + t];
            for (size_t j = 0; j < in; j++)
                nw[(i * nk + t) * in + j] = (*W)[(i * k0 + t) * in + j];
        }
        for (size_t t = kc; t < nk; t++) {
            /* new factor ≈ 1 so the product is unchanged */
            nb[i * nk + t] = 1.0;
        }
    }
    free(*W); free(*b); free(*or_val);
    *W = nw; *b = nb; *or_val = no;
    *k = nk;
}

void tnn_dense_identity(double *W, double *b, size_t in, size_t out, size_t k)
{
    size_t n_or = out * k;
    memset(W, 0, n_or * in * sizeof(double));
    memset(b, 0, n_or * sizeof(double));
    size_t n = in < out ? in : out;
    for (size_t i = 0; i < n; i++) {
        /* or0 = x_i, or1 = 1, rest = 1 */
        W[(i * k + 0) * in + i] = 1.0;
        if (k > 1) b[i * k + 1] = 1.0;
        for (size_t t = 2; t < k; t++) b[i * k + t] = 1.0;
    }
}

static void scratch_fit(TStack *s, size_t slot, size_t w)
{
    if (s->scratch_w[slot] >= w && s->scratch[slot]) return;
    free(s->scratch[slot]);
    s->scratch[slot] = (double *)calloc(w, sizeof(double));
    s->scratch_w[slot] = w;
}

static void stack_init(void *ctx)
{
    TStack *s = (TStack *)ctx;
    for (size_t i = 0; i < s->depth; i++) s->ops->init(s->layer[i]);
}

static void stack_forward(void *ctx, const double *x, double *y)
{
    TStack *s = (TStack *)ctx;
    const double *cur = x;
    for (size_t i = 0; i < s->depth; i++) {
        size_t out = s->ops->out(s->layer[i]);
        scratch_fit(s, i + 1, out);
        s->ops->fwd(s->layer[i], cur, s->scratch[i + 1]);
        cur = s->scratch[i + 1];
    }
    memcpy(y, cur, s->ops->out(s->layer[s->depth - 1]) * sizeof(double));
}

static void stack_backward(void *ctx, const double *x, const double *dy, double lr)
{
    TStack *s = (TStack *)ctx;
    /* activations: scratch[0] unused; we re-forward to fill scratch[1..] */
    const double *cur = x;
    for (size_t i = 0; i < s->depth; i++) {
        size_t out = s->ops->out(s->layer[i]);
        scratch_fit(s, i + 1, out);
        s->ops->fwd(s->layer[i], cur, s->scratch[i + 1]);
        cur = s->scratch[i + 1];
    }
    const double *dcur = dy;
    double *dx_own = NULL;
    for (size_t i = s->depth; i-- > 0; ) {
        const double *xin = (i == 0) ? x : s->scratch[i];
        size_t in = s->ops->in(s->layer[i]);
        double *dx = NULL;
        if (i > 0) {
            scratch_fit(s, 0, in);
            dx = s->scratch[0];
        }
        s->ops->bwd(s->layer[i], xin, dcur, dx, lr);
        if (i > 0) {
            /* stash dx into a private buffer because scratch[0] is reused */
            free(dx_own);
            dx_own = (double *)malloc(in * sizeof(double));
            memcpy(dx_own, dx, in * sizeof(double));
            dcur = dx_own;
        }
    }
    free(dx_own);

    if (!s->dynamic) return;
    void *tail = s->layer[s->depth - 1];
    /* grow an Or factor when the residual is large */
    double mag = 0.0;
    for (size_t i = 0; i < s->ops->out(tail); i++) mag += fabs(dy[i]);
    size_t k = s->ops->k(tail);
    if (mag > 1.0 && k < s->max_or && k < 4)
        s->ops->set_k(tail, k + 1);
}


static void stack_align(void *ctx, size_t in)
{
    TStack *s = (TStack *)ctx;
    s->ops->resize_in(s->layer[0], in);
}

static void stack_set_out(void *ctx, size_t out)
{
    TStack *s = (TStack *)ctx;
    s->ops->resize_out(s->layer[s->depth - 1], out);
}

static void stack_set_k(void *ctx, size_t k)
{
    TStack *s = (TStack *)ctx;
    if (k > s->max_or) k = s->max_or;
    s->ops->set_k(s->layer[s->depth - 1], k);
}

static void stack_insert(void *ctx)
{
    TStack *s = (TStack *)ctx;
    if (s->depth >= s->max_depth) return;
    void *tail = s->layer[s->depth - 1];
    size_t dim = s->ops->in(tail);
    void *hid = s->ops->new(dim, dim, TNN_K0);
    s->ops->identity(hid);
    if (s->depth + 1 > s->cap) {
        s->cap *= 2;
        s->layer = (void **)realloc(s->layer, s->cap * sizeof(void *));
    }
    size_t idx = s->depth - 1;
    memmove(&s->layer[idx + 1], &s->layer[idx],
            (s->depth - idx) * sizeof(void *));
    s->layer[idx] = hid;
    s->depth++;
}

static int stack_remove(void *ctx)
{
    TStack *s = (TStack *)ctx;
    if (s->depth <= 1) return -1;
    /* drop the layer before tail (hidden) */
    size_t idx = s->depth - 2;
    void *dead = s->layer[idx];
    size_t new_in = s->ops->in(dead);
    s->ops->free(dead);
    memmove(&s->layer[idx], &s->layer[idx + 1],
            (s->depth - idx - 1) * sizeof(void *));
    s->depth--;
    s->ops->resize_in(s->layer[idx], new_in);
    return 0;
}

static void stack_set_dyn(void *ctx, int on)
{
    ((TStack *)ctx)->dynamic = on ? 1 : 0;
}

static void stack_scale(void *ctx, size_t idx, size_t in, size_t out)
{
    TStack *s = (TStack *)ctx;
    if (idx >= s->depth) return;
    s->ops->resize_in(s->layer[idx], in);
    s->ops->resize_out(s->layer[idx], out);
    if (idx + 1 < s->depth)
        s->ops->resize_in(s->layer[idx + 1], out);
    if (idx > 0)
        s->ops->resize_out(s->layer[idx - 1], in);
}
static size_t stack_lin(void *ctx, size_t idx)
{
    TStack *s = (TStack *)ctx;
    if (idx >= s->depth) return 0;
    return s->ops->in(s->layer[idx]);
}
static size_t stack_lout(void *ctx, size_t idx)
{
    TStack *s = (TStack *)ctx;
    if (idx >= s->depth) return 0;
    return s->ops->out(s->layer[idx]);
}
static size_t stack_lk(void *ctx, size_t idx)
{
    TStack *s = (TStack *)ctx;
    if (idx >= s->depth) return 0;
    return s->ops->k(s->layer[idx]);
}

static size_t stack_depth(void *ctx) { return ((TStack *)ctx)->depth; }

static size_t stack_k(void *ctx)
{
    TStack *s = (TStack *)ctx;
    return s->ops->k(s->layer[s->depth - 1]);
}

static size_t stack_params(void *ctx)
{
    TStack *s = (TStack *)ctx;
    size_t n = 0;
    for (size_t i = 0; i < s->depth; i++) n += s->ops->params(s->layer[i]);
    return n;
}

static size_t stack_nbytes(void *ctx)
{
    TStack *s = (TStack *)ctx;
    size_t n = sizeof(TStack) + s->cap * sizeof(void *);
    for (size_t i = 0; i < s->depth; i++) n += s->ops->nbytes(s->layer[i]);
    return n;
}

static void stack_free(void *ctx)
{
    TStack *s = (TStack *)ctx;
    for (size_t i = 0; i < s->depth; i++) s->ops->free(s->layer[i]);
    for (size_t i = 0; i < TNN_MAX_DEPTH + 2; i++) free(s->scratch[i]);
    free(s->layer);
    free(s);
}

TStack *tstack_open(const char *impl, const TLayerOps *ops, size_t in, size_t out)
{
    TStack *s = (TStack *)calloc(1, sizeof(TStack));
    s->impl = impl;
    s->ops = ops;
    s->cap = 4;
    s->layer = (void **)calloc(s->cap, sizeof(void *));
    s->layer[0] = ops->new(in, out, TNN_K0);
    s->depth = 1;
    s->dynamic = 1;
    s->max_depth = TNN_MAX_DEPTH;
    s->max_or = TNN_MAX_OR;
    return s;
}

void tstack_bind(AltNet *dst, TStack *s)
{
    dst->impl = s->impl;
    dst->ctx = s;
    dst->in = s->ops->in(s->layer[0]);
    dst->out = s->ops->out(s->layer[s->depth - 1]);
    dst->init = stack_init;
    dst->forward = stack_forward;
    dst->backward = stack_backward;
    dst->align_inputs = stack_align;
    dst->set_outputs = stack_set_out;
    dst->set_or_factors = stack_set_k;
    dst->insert_identity = stack_insert;
    dst->remove_hidden = stack_remove;
    dst->set_dynamic = stack_set_dyn;
    dst->depth = stack_depth;
    dst->or_factors = stack_k;
    dst->param_count = stack_params;
    dst->nbytes = stack_nbytes;
    dst->free = stack_free;
    dst->scale_layer = stack_scale;
    dst->layer_in = stack_lin;
    dst->layer_out = stack_lout;
    dst->layer_k = stack_lk;
}

void type_nn_alt_train(AltNet *a, double **X, double **Y,
                       size_t n, size_t epochs, double lr)
{
    double *pred = (double *)calloc(a->out, sizeof(double));
    double *dy = (double *)calloc(a->out, sizeof(double));
    size_t *ord = (size_t *)malloc(n * sizeof(size_t));
    for (size_t ep = 0; ep < epochs; ep++) {
        for (size_t i = 0; i < n; i++) ord[i] = i;
        /* Epoch order is a dedicated xorshift, not libc rand(), so every
           impl sees the same permutation regardless of how many rand()
           calls its init consumed. */
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
            /* torch MSE grad is (pred-y)/out per channel (mean over elements).
               Without this, a 3-class head gets 3× the pull of WDBC's 1-class
               head and cannot drive the unused classes to 0 independently. */
            double inv = 1.0 / (double)(a->out ? a->out : 1);
            for (size_t k = 0; k < a->out; k++)
                dy[k] = (pred[k] - Y[si][k]) * inv;
            a->backward(a->ctx, X[si], dy, lr);
        }
    }
    free(ord);
    free(pred);
    free(dy);
}

extern AltNet type_nn_win_open(size_t, size_t);

static AltNet (*const OPENERS[])(size_t, size_t) = {
    type_nn_win_open,
};
static const char *const NAMES[] = {
    "type-nn-win",
};

size_t type_nn_alt_count(void)
{
    return sizeof(OPENERS) / sizeof(OPENERS[0]);
}

const char *type_nn_alt_name(size_t i)
{
    return i < type_nn_alt_count() ? NAMES[i] : NULL;
}

AltNet (*type_nn_alt_opener(size_t i))(size_t, size_t)
{
    return i < type_nn_alt_count() ? OPENERS[i] : NULL;
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
        /* add and multiply sum-type and product-type size changes */
        m += (double)d_or + (double)d_and + (double)d_or * (double)d_and;
    }
    if (dyn_scale) *dyn_scale = m;
    if (dyn_depth) *dyn_depth = (int)a->depth - (int)b->depth;
}
