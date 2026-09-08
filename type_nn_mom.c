#include "type_nn_alt.h"
#include "type_nn_layerkit.h"

/*
 * type-nn-mom — SGD with momentum on every And/Or weight and bias.
 *   v ← μ v + g
 *   w ← w − lr v
 * Faithful layer API. Extra state is one velocity per parameter.
 */

#define MOM_MU 0.5

typedef struct {
    LKLayer L;
    double *vW, *vb;
} MomLayer;

typedef struct {
    MomLayer layer[LK_MAX_DEPTH];
    double *act[LK_MAX_DEPTH + 1];
    size_t  act_w[LK_MAX_DEPTH + 1];
    size_t  depth;
    int     dynamic;
    size_t  max_or;
} MomNet;

static void mlayer_alloc(MomLayer *M, size_t in, size_t out, size_t k)
{
    lk_alloc(&M->L, in, out, k);
    size_t n = M->L.n_or;
    M->vW = (double *)realloc(M->vW, n * in * sizeof(double));
    M->vb = (double *)realloc(M->vb, n * sizeof(double));
    if (n && in) memset(M->vW, 0, n * in * sizeof(double));
    if (n) memset(M->vb, 0, n * sizeof(double));
}
static void mlayer_free(MomLayer *M)
{
    lk_free(&M->L);
    free(M->vW); free(M->vb);
    M->vW = M->vb = NULL;
}
static void mlayer_sync(MomLayer *M)
{
    size_t n = M->L.n_or, in = M->L.in;
    M->vW = (double *)realloc(M->vW, n * in * sizeof(double));
    M->vb = (double *)realloc(M->vb, n * sizeof(double));
    memset(M->vW, 0, n * in * sizeof(double));
    memset(M->vb, 0, n * sizeof(double));
}

static void fit(MomNet *N, size_t s, size_t w)
{
    if (N->act_w[s] >= w && N->act[s]) return;
    free(N->act[s]);
    N->act[s] = (double *)calloc(w ? w : 1, sizeof(double));
    N->act_w[s] = w ? w : 1;
}

static void bwd_layer(MomLayer *M, const double *x, const double *dy,
                      double *dx, double lr)
{
    LKLayer *L = &M->L;
    const size_t in = L->in, k = L->k, out = L->out;
    if (dx) memset(dx, 0, in * sizeof(double));
    double dor[8];
    for (size_t i = 0; i < out; i++) {
        lk_dor(L, i, dy[i], dor);
        for (size_t t = 0; t < k; t++) {
            double g = dor[t];
            size_t r = i * k + t;
            /* Polyak: v ← μ v − lr g ; w ← w + v */
            M->vb[r] = MOM_MU * M->vb[r] - lr * g;
            L->b[r] = tnn_clamp(L->b[r] + M->vb[r], -TNN_WCLIP, TNN_WCLIP);
            double *w = L->W + r * in;
            double *v = M->vW + r * in;
            for (size_t j = 0; j < in; j++) {
                if (dx) dx[j] += g * w[j];
                v[j] = MOM_MU * v[j] - lr * g * x[j];
                w[j] = tnn_clamp(w[j] + v[j], -TNN_WCLIP, TNN_WCLIP);
            }
        }
    }
}

static void net_init(void *c)
{
    MomNet *N = c;
    for (size_t i = 0; i < N->depth; i++) lk_init(&N->layer[i].L);
}
static void net_fwd(void *c, const double *x, double *y)
{
    MomNet *N = c;
    const double *cur = x;
    for (size_t i = 0; i < N->depth; i++) {
        fit(N, i+1, N->layer[i].L.out);
        lk_fwd(&N->layer[i].L, cur, N->act[i+1]);
        cur = N->act[i+1];
    }
    memcpy(y, cur, N->layer[N->depth-1].L.out * sizeof(double));
}
static void net_bwd(void *c, const double *x, const double *dy, double lr)
{
    MomNet *N = c;
    if (!N->act[1]) {
        const double *cur = x;
        for (size_t i = 0; i < N->depth; i++) {
            fit(N, i+1, N->layer[i].L.out);
            lk_fwd(&N->layer[i].L, cur, N->act[i+1]);
            cur = N->act[i+1];
        }
    }
    const double *dcur = dy;
    double *hold = NULL;
    for (size_t i = N->depth; i-- > 0; ) {
        const double *xin = i ? N->act[i] : x;
        double *dx = NULL;
        if (i) { fit(N, 0, N->layer[i].L.in); dx = N->act[0]; }
        bwd_layer(&N->layer[i], xin, dcur, dx, lr);
        if (i) {
            size_t w = N->layer[i].L.in;
            hold = (double *)realloc(hold, w*sizeof(double));
            memcpy(hold, dx, w*sizeof(double));
            dcur = hold;
        }
    }
    free(hold);
    if (N->dynamic) {
        LKLayer *t = &N->layer[N->depth-1].L;
        double mag = 0;
        for (size_t i = 0; i < t->out; i++) mag += fabs(dy[i]);
        if (mag > 1.0 && t->k < N->max_or && t->k < 4) {
            lk_set_k(t, t->k+1);
            mlayer_sync(&N->layer[N->depth-1]);
        }
    }
}
static void net_align(void *c, size_t in)
{
    MomNet *N = c;
    lk_resize_in(&N->layer[0].L, in);
    mlayer_sync(&N->layer[0]);
}
static void net_out(void *c, size_t o)
{
    MomNet *N = c;
    lk_resize_out(&N->layer[N->depth-1].L, o);
    mlayer_sync(&N->layer[N->depth-1]);
}
static void net_k(void *c, size_t k)
{
    MomNet *N = c;
    if (k > N->max_or) k = N->max_or;
    lk_set_k(&N->layer[N->depth-1].L, k);
    mlayer_sync(&N->layer[N->depth-1]);
}
static void net_ins(void *c)
{
    MomNet *N = c;
    if (N->depth >= LK_MAX_DEPTH) return;
    size_t dim = N->layer[N->depth-1].L.in;
    memmove(&N->layer[N->depth], &N->layer[N->depth-1], sizeof(MomLayer));
    memset(&N->layer[N->depth-1], 0, sizeof(MomLayer));
    mlayer_alloc(&N->layer[N->depth-1], dim, dim, TNN_K0);
    lk_identity(&N->layer[N->depth-1].L);
    N->depth++;
}
static int net_rem(void *c)
{
    MomNet *N = c;
    if (N->depth <= 1) return -1;
    size_t idx = N->depth-2, nin = N->layer[idx].L.in;
    mlayer_free(&N->layer[idx]);
    memmove(&N->layer[idx], &N->layer[idx+1], (N->depth-idx-1)*sizeof(MomLayer));
    memset(&N->layer[N->depth-1], 0, sizeof(MomLayer));
    N->depth--;
    lk_resize_in(&N->layer[idx].L, nin);
    mlayer_sync(&N->layer[idx]);
    return 0;
}
static void net_dyn(void *c, int on) { ((MomNet*)c)->dynamic = on?1:0; }
static size_t net_depth(void *c) { return ((MomNet*)c)->depth; }
static size_t net_kf(void *c) { MomNet *N=c; return N->layer[N->depth-1].L.k; }
static size_t net_params(void *c)
{
    MomNet *N=c; size_t n=0;
    for (size_t i=0;i<N->depth;i++) n += N->layer[i].L.n_or*(N->layer[i].L.in+1);
    return n;
}
static size_t net_nbytes(void *c)
{
    /* W+b plus velocity */
    MomNet *N=c; size_t n=0;
    for (size_t i=0;i<N->depth;i++)
        n += 2 * N->layer[i].L.n_or * (N->layer[i].L.in+1) * sizeof(double);
    return n;
}
static void net_free(void *c)
{
    MomNet *N=c;
    for (size_t i=0;i<N->depth;i++) mlayer_free(&N->layer[i]);
    for (size_t i=0;i<LK_MAX_DEPTH+1;i++) free(N->act[i]);
    free(N);
}

AltNet type_nn_mom_open(size_t in, size_t out)
{
    MomNet *N = (MomNet *)calloc(1, sizeof(MomNet));
    N->depth = 1; N->dynamic = 1; N->max_or = TNN_MAX_OR;
    mlayer_alloc(&N->layer[0], in, out, TNN_K0);
    AltNet h = {
        .impl="type-nn-mom", .ctx=N, .in=in, .out=out,
        .init=net_init, .forward=net_fwd, .backward=net_bwd,
        .align_inputs=net_align, .set_outputs=net_out,
        .set_or_factors=net_k, .insert_identity=net_ins,
        .remove_hidden=net_rem, .set_dynamic=net_dyn,
        .depth=net_depth, .or_factors=net_kf,
        .param_count=net_params, .nbytes=net_nbytes, .free=net_free
    };
    return h;
}
