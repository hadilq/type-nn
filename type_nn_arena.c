#include "type_nn_dense.h"

/* Arena layout: And/Or/Weight nodes in slabs, `right` is an index.
   Resize rebuilds the chains so grow/shrink stays faithful. */

#define NIL (-1)

typedef struct { double value; size_t idx; int right; } AW;
typedef struct { double value, accum, bias; int w_head, right; } AO;
typedef struct { double value; int o_head, right; } AA;

typedef struct {
    size_t in, out, k;
    AW *W; int w_n, w_cap;
    AO *O; int o_n, o_cap;
    AA *A; int a_n, a_cap;
    double *or_val;
} ArenaLayer;

static int aw(ArenaLayer *s)
{
    if (s->w_n >= s->w_cap) {
        s->w_cap = s->w_cap ? s->w_cap * 2 : 8;
        s->W = (AW *)realloc(s->W, (size_t)s->w_cap * sizeof(AW));
    }
    int id = s->w_n++;
    memset(&s->W[id], 0, sizeof(AW));
    s->W[id].right = NIL;
    return id;
}
static int ao(ArenaLayer *s)
{
    if (s->o_n >= s->o_cap) {
        s->o_cap = s->o_cap ? s->o_cap * 2 : 4;
        s->O = (AO *)realloc(s->O, (size_t)s->o_cap * sizeof(AO));
    }
    int id = s->o_n++;
    memset(&s->O[id], 0, sizeof(AO));
    s->O[id].w_head = s->O[id].right = NIL;
    return id;
}
static int aa(ArenaLayer *s)
{
    if (s->a_n >= s->a_cap) {
        s->a_cap = s->a_cap ? s->a_cap * 2 : 2;
        s->A = (AA *)realloc(s->A, (size_t)s->a_cap * sizeof(AA));
    }
    int id = s->a_n++;
    memset(&s->A[id], 0, sizeof(AA));
    s->A[id].o_head = s->A[id].right = NIL;
    return id;
}

static void arena_build(ArenaLayer *s)
{
    s->w_n = s->o_n = s->a_n = 0;
    int prev_a = NIL;
    for (size_t i = 0; i < s->out; i++) {
        int a = aa(s);
        if (prev_a != NIL) s->A[prev_a].right = a;
        prev_a = a;
        int prev_o = NIL;
        for (size_t t = 0; t < s->k; t++) {
            int o = ao(s);
            if (prev_o == NIL) s->A[a].o_head = o;
            else s->O[prev_o].right = o;
            prev_o = o;
            int prev_w = NIL;
            for (size_t j = 0; j < s->in; j++) {
                int w = aw(s);
                s->W[w].idx = j;
                if (prev_w == NIL) s->O[o].w_head = w;
                else s->W[prev_w].right = w;
                prev_w = w;
            }
        }
    }
    s->or_val = (double *)realloc(s->or_val, s->out * s->k * sizeof(double));
}

static void *arena_new(size_t in, size_t out, size_t k)
{
    ArenaLayer *s = (ArenaLayer *)calloc(1, sizeof(ArenaLayer));
    s->in = in; s->out = out; s->k = k;
    arena_build(s);
    return s;
}
static void arena_init(void *p)
{
    ArenaLayer *s = (ArenaLayer *)p;
    for (int o = 0; o < s->o_n; o++) {
        s->O[o].bias = tnn_rand();
        for (int w = s->O[o].w_head; w != NIL; w = s->W[w].right)
            s->W[w].value = tnn_rand();
    }
}
static void arena_fwd(void *p, const double *x, double *y)
{
    ArenaLayer *s = (ArenaLayer *)p;
    int ai = 0;
    for (int a = 0; a != NIL; a = s->A[a].right, ai++) {
        double prod = 1.0;
        int oi = 0;
        for (int o = s->A[a].o_head; o != NIL; o = s->O[o].right, oi++) {
            double sum = s->O[o].bias;
            for (int w = s->O[o].w_head; w != NIL; w = s->W[w].right)
                sum += s->W[w].value * x[s->W[w].idx];
            sum = tnn_clamp(sum, -TNN_ORCLIP, TNN_ORCLIP);
            s->O[o].value = sum;
            s->or_val[ai * (int)s->k + oi] = sum;
            prod *= sum;
        }
        y[ai] = tnn_clamp(prod, -TNN_ANDCLIP, TNN_ANDCLIP);
    }
}
static void arena_bwd(void *p, const double *x, const double *dy, double *dx, double lr)
{
    ArenaLayer *s = (ArenaLayer *)p;
    if (dx) memset(dx, 0, s->in * sizeof(double));
    int ai = 0;
    for (int a = 0; a != NIL; a = s->A[a].right, ai++) {
        int ors[8], n = 0;
        for (int o = s->A[a].o_head; o != NIL && n < 8; o = s->O[o].right)
            ors[n++] = o;
        for (int t = 0; t < n; t++) {
            double accum = 1.0;
            for (int u = 0; u < n; u++)
                if (u != t) accum *= s->O[ors[u]].value;
            double d_or = dy[ai] * accum;
            s->O[ors[t]].bias = tnn_clamp(s->O[ors[t]].bias - lr * d_or,
                                          -TNN_WCLIP, TNN_WCLIP);
            for (int w = s->O[ors[t]].w_head; w != NIL; w = s->W[w].right) {
                if (dx) dx[s->W[w].idx] += d_or * s->W[w].value;
                s->W[w].value = tnn_clamp(s->W[w].value - lr * d_or * x[s->W[w].idx],
                                          -TNN_WCLIP, TNN_WCLIP);
            }
        }
    }
}
static void arena_rin(void *p, size_t in)
{
    ArenaLayer *s = (ArenaLayer *)p;
    if (in == s->in) return;
    /* keep existing values in a dense snapshot */
    size_t n_or = s->out * s->k;
    double *snap = (double *)calloc(n_or * s->in + n_or, sizeof(double));
    double *bs = snap + n_or * s->in;
    int r = 0;
    for (int a = 0; a != NIL; a = s->A[a].right)
        for (int o = s->A[a].o_head; o != NIL; o = s->O[o].right, r++) {
            bs[r] = s->O[o].bias;
            for (int w = s->O[o].w_head; w != NIL; w = s->W[w].right)
                if (s->W[w].idx < s->in)
                    snap[r * s->in + s->W[w].idx] = s->W[w].value;
        }
    size_t old = s->in;
    s->in = in;
    arena_build(s);
    r = 0;
    for (int a = 0; a != NIL; a = s->A[a].right)
        for (int o = s->A[a].o_head; o != NIL; o = s->O[o].right, r++) {
            s->O[o].bias = bs[r];
            for (int w = s->O[o].w_head; w != NIL; w = s->W[w].right)
                if (s->W[w].idx < old)
                    s->W[w].value = snap[r * old + s->W[w].idx];
        }
    free(snap);
}
static void arena_rout(void *p, size_t out)
{
    ArenaLayer *s = (ArenaLayer *)p;
    s->out = out;
    arena_build(s);
    arena_init(s);
}
static void arena_sk(void *p, size_t k)
{
    ArenaLayer *s = (ArenaLayer *)p;
    if (k < 1) k = 1;
    if (k == s->k) return;
    size_t n_or = s->out * s->k;
    double *snap = (double *)calloc(n_or * s->in + n_or, sizeof(double));
    double *bs = snap + n_or * s->in;
    int r = 0;
    for (int a = 0; a != NIL; a = s->A[a].right)
        for (int o = s->A[a].o_head; o != NIL; o = s->O[o].right, r++) {
            bs[r] = s->O[o].bias;
            for (int w = s->O[o].w_head; w != NIL; w = s->W[w].right)
                if (s->W[w].idx < s->in)
                    snap[r * s->in + s->W[w].idx] = s->W[w].value;
        }
    size_t k0 = s->k, in = s->in, out = s->out;
    s->k = k;
    arena_build(s);
    int ai = 0;
    for (int a = 0; a != NIL; a = s->A[a].right, ai++) {
        int oi = 0;
        for (int o = s->A[a].o_head; o != NIL; o = s->O[o].right, oi++) {
            if ((size_t)oi < k0) {
                size_t rr = (size_t)ai * k0 + (size_t)oi;
                s->O[o].bias = bs[rr];
                for (int w = s->O[o].w_head; w != NIL; w = s->W[w].right)
                    s->W[w].value = snap[rr * in + s->W[w].idx];
            } else {
                s->O[o].bias = 1.0; /* product-preserving new Or */
                for (int w = s->O[o].w_head; w != NIL; w = s->W[w].right)
                    s->W[w].value = 0.0;
            }
        }
    }
    (void)out;
    free(snap);
}
static void arena_id(void *p)
{
    ArenaLayer *s = (ArenaLayer *)p;
    int ai = 0;
    for (int a = 0; a != NIL; a = s->A[a].right, ai++) {
        int oi = 0;
        for (int o = s->A[a].o_head; o != NIL; o = s->O[o].right, oi++) {
            s->O[o].bias = (oi == 1) ? 1.0 : 0.0;
            for (int w = s->O[o].w_head; w != NIL; w = s->W[w].right)
                s->W[w].value = (oi == 0 && s->W[w].idx == (size_t)ai) ? 1.0 : 0.0;
        }
    }
}
static size_t arena_in(void *p)  { return ((ArenaLayer *)p)->in; }
static size_t arena_out(void *p) { return ((ArenaLayer *)p)->out; }
static size_t arena_k(void *p)   { return ((ArenaLayer *)p)->k; }
static size_t arena_params(void *p)
{
    ArenaLayer *s = (ArenaLayer *)p;
    return (size_t)s->w_n + (size_t)s->o_n;
}
static size_t arena_nbytes(void *p)
{
    ArenaLayer *s = (ArenaLayer *)p;
    return sizeof(ArenaLayer)
        + (size_t)s->w_cap * sizeof(AW)
        + (size_t)s->o_cap * sizeof(AO)
        + (size_t)s->a_cap * sizeof(AA);
}
static void arena_free(void *p)
{
    ArenaLayer *s = (ArenaLayer *)p;
    free(s->W); free(s->O); free(s->A); free(s->or_val); free(s);
}

static const TLayerOps OPS = {
    arena_new, arena_init, arena_fwd, arena_bwd,
    arena_rin, arena_rout, arena_sk, arena_id,
    arena_in, arena_out, arena_k, arena_params, arena_nbytes, arena_free
};

AltNet type_nn_arena_open(size_t in, size_t out)
{
    AltNet h;
    tstack_bind(&h, tstack_open("type-nn-arena", &OPS, in, out));
    return h;
}
