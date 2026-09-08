#include "type_nn_alt.h"
#include "type_nn_layerkit.h"

/*
 * type-nn-bp — backprop experiment, still vanilla SGD.
 * Faithful And/Or net. Backward changes:
 *   • prefix/suffix products so dOr is O(k) not O(k²)
 *   • dx and weight update in one pass over W
 *   • kept activations, no re-forward
 */

typedef struct {
    LKLayer layer[LK_MAX_DEPTH];
    double *act[LK_MAX_DEPTH + 1];
    size_t  act_w[LK_MAX_DEPTH + 1];
    size_t  depth;
    int     dynamic;
    size_t  max_or;
} BPNet;

static void fit(BPNet *N, size_t s, size_t w)
{
    if (N->act_w[s] >= w && N->act[s]) return;
    free(N->act[s]);
    N->act[s] = (double *)calloc(w ? w : 1, sizeof(double));
    N->act_w[s] = w ? w : 1;
}

static void bwd_layer(LKLayer *L, const double *x, const double *dy,
                      double *dx, double lr)
{
    const size_t in = L->in, k = L->k, out = L->out;
    if (dx) memset(dx, 0, in * sizeof(double));
    double dor[8];
    for (size_t i = 0; i < out; i++) {
        lk_dor(L, i, dy[i], dor);
        for (size_t t = 0; t < k; t++) {
            double g = dor[t];
            size_t r = i * k + t;
            L->b[r] = tnn_clamp(L->b[r] - lr * g, -TNN_WCLIP, TNN_WCLIP);
            double *w = L->W + r * in;
            for (size_t j = 0; j < in; j++) {
                if (dx) dx[j] += g * w[j];
                w[j] = tnn_clamp(w[j] - lr * g * x[j], -TNN_WCLIP, TNN_WCLIP);
            }
        }
    }
}

static void net_init(void *c)
{
    BPNet *N = (BPNet *)c;
    for (size_t i = 0; i < N->depth; i++) lk_init(&N->layer[i]);
}
static void net_fwd(void *c, const double *x, double *y)
{
    BPNet *N = (BPNet *)c;
    const double *cur = x;
    for (size_t i = 0; i < N->depth; i++) {
        fit(N, i + 1, N->layer[i].out);
        lk_fwd(&N->layer[i], cur, N->act[i + 1]);
        cur = N->act[i + 1];
    }
    memcpy(y, cur, N->layer[N->depth-1].out * sizeof(double));
}
static void net_bwd(void *c, const double *x, const double *dy, double lr)
{
    BPNet *N = (BPNet *)c;
    if (!N->act[1]) {
        const double *cur = x;
        for (size_t i = 0; i < N->depth; i++) {
            fit(N, i+1, N->layer[i].out);
            lk_fwd(&N->layer[i], cur, N->act[i+1]);
            cur = N->act[i+1];
        }
    }
    const double *dcur = dy;
    double *hold = NULL;
    for (size_t i = N->depth; i-- > 0; ) {
        const double *xin = i ? N->act[i] : x;
        double *dx = NULL;
        if (i) { fit(N, 0, N->layer[i].in); dx = N->act[0]; }
        bwd_layer(&N->layer[i], xin, dcur, dx, lr);
        if (i) {
            size_t w = N->layer[i].in;
            hold = (double *)realloc(hold, w * sizeof(double));
            memcpy(hold, dx, w * sizeof(double));
            dcur = hold;
        }
    }
    free(hold);
    if (N->dynamic) {
        LKLayer *t = &N->layer[N->depth-1];
        double mag = 0;
        for (size_t i = 0; i < t->out; i++) mag += fabs(dy[i]);
        if (mag > 1.0 && t->k < N->max_or && t->k < 4) lk_set_k(t, t->k+1);
    }
}
static void net_align(void *c, size_t in) { lk_resize_in(&((BPNet*)c)->layer[0], in); }
static void net_out(void *c, size_t o)
{
    BPNet *N = (BPNet *)c;
    lk_resize_out(&N->layer[N->depth-1], o);
}
static void net_k(void *c, size_t k)
{
    BPNet *N = (BPNet *)c;
    if (k > N->max_or) k = N->max_or;
    lk_set_k(&N->layer[N->depth-1], k);
}
static void net_ins(void *c)
{
    BPNet *N = (BPNet *)c;
    if (N->depth >= LK_MAX_DEPTH) return;
    size_t dim = N->layer[N->depth-1].in;
    memmove(&N->layer[N->depth], &N->layer[N->depth-1], sizeof(LKLayer));
    memset(&N->layer[N->depth-1], 0, sizeof(LKLayer));
    lk_alloc(&N->layer[N->depth-1], dim, dim, TNN_K0);
    lk_identity(&N->layer[N->depth-1]);
    N->depth++;
}
static int net_rem(void *c)
{
    BPNet *N = (BPNet *)c;
    if (N->depth <= 1) return -1;
    size_t idx = N->depth-2, nin = N->layer[idx].in;
    lk_free(&N->layer[idx]);
    memmove(&N->layer[idx], &N->layer[idx+1], (N->depth-idx-1)*sizeof(LKLayer));
    memset(&N->layer[N->depth-1], 0, sizeof(LKLayer));
    N->depth--;
    lk_resize_in(&N->layer[idx], nin);
    return 0;
}
static void net_dyn(void *c, int on) { ((BPNet*)c)->dynamic = on?1:0; }
static size_t net_depth(void *c) { return ((BPNet*)c)->depth; }
static size_t net_kf(void *c) { BPNet *N=c; return N->layer[N->depth-1].k; }
static size_t net_params(void *c)
{
    BPNet *N = c; size_t n=0;
    for (size_t i=0;i<N->depth;i++) n += N->layer[i].n_or*(N->layer[i].in+1);
    return n;
}
static size_t net_nbytes(void *c)
{
    BPNet *N = c; size_t n=0;
    for (size_t i=0;i<N->depth;i++)
        n += N->layer[i].n_or*(N->layer[i].in+1)*sizeof(double);
    return n;
}
static void net_free(void *c)
{
    BPNet *N=c;
    for (size_t i=0;i<N->depth;i++) lk_free(&N->layer[i]);
    for (size_t i=0;i<LK_MAX_DEPTH+1;i++) free(N->act[i]);
    free(N);
}

AltNet type_nn_bp_open(size_t in, size_t out)
{
    BPNet *N = (BPNet *)calloc(1, sizeof(BPNet));
    N->depth = 1; N->dynamic = 1; N->max_or = TNN_MAX_OR;
    lk_alloc(&N->layer[0], in, out, TNN_K0);
    AltNet h = {
        .impl="type-nn-bp", .ctx=N, .in=in, .out=out,
        .init=net_init, .forward=net_fwd, .backward=net_bwd,
        .align_inputs=net_align, .set_outputs=net_out,
        .set_or_factors=net_k, .insert_identity=net_ins,
        .remove_hidden=net_rem, .set_dynamic=net_dyn,
        .depth=net_depth, .or_factors=net_kf,
        .param_count=net_params, .nbytes=net_nbytes, .free=net_free
    };
    return h;
}
