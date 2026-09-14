#include "type_nn_cmlp.h"
#include "type_nn_alt.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/*
 * c-mlp — same model torch-mlp trains in bench_torch.py, in C.
 *
 *   Linear(in, H) → ReLU → Linear(H, out)
 *   H = 8 if in <= 4 else 16   (iris 8; wine/wdbc/diabetes/iono 16; xor 8)
 *   full-batch Adam, PyTorch Linear init (U(-1/sqrt(fan_in), +1/sqrt(fan_in)))
 *
 * Lives outside the type-nn-* registry so test_alts does not demand
 * And/Or identity algebra from a ReLU MLP. bench_alts.c runs it as the
 * fair timing/accuracy baseline.
 */

#define AD_B1  0.9
#define AD_B2  0.999
#define AD_EPS 1e-8
#define MAX_D  6

typedef struct {
    size_t in, out;
    int relu;
    double *W, *b;
    double *mW, *vW, *mb, *vb;
    double *x, *y, *dx; /* last fwd / bwd scratch */
} DLayer;

typedef struct {
    DLayer  L[MAX_D];
    size_t  depth;
    unsigned long tstep;
    double  b1p, b2p;
    size_t  in, out, hid;
} MLP;

static size_t pick_hid(size_t in, size_t out)
{
    size_t h = (in <= 4) ? 8 : 16;
    if (h < out) h = out;
    return h;
}

static void dalloc(DLayer *L, size_t in, size_t out, int relu)
{
    L->in = in; L->out = out; L->relu = relu;
    L->W = (double *)calloc(out * in, sizeof(double));
    L->b = (double *)calloc(out, sizeof(double));
    L->mW = (double *)calloc(out * in, sizeof(double));
    L->vW = (double *)calloc(out * in, sizeof(double));
    L->mb = (double *)calloc(out, sizeof(double));
    L->vb = (double *)calloc(out, sizeof(double));
    L->x = (double *)calloc(in ? in : 1, sizeof(double));
    L->y = (double *)calloc(out ? out : 1, sizeof(double));
    L->dx = (double *)calloc(in ? in : 1, sizeof(double));
}

static void dfree(DLayer *L)
{
    free(L->W); free(L->b);
    free(L->mW); free(L->vW); free(L->mb); free(L->vb);
    free(L->x); free(L->y); free(L->dx);
    memset(L, 0, sizeof(*L));
}

static void dinit(DLayer *L)
{
    /* torch.nn.Linear default: bound = 1/sqrt(in) */
    double bound = 1.0 / sqrt((double)(L->in > 0 ? L->in : 1));
    for (size_t o = 0; o < L->out; o++) {
        L->b[o] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * bound;
        for (size_t j = 0; j < L->in; j++)
            L->W[o * L->in + j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * bound;
    }
    memset(L->mW, 0, L->out * L->in * sizeof(double));
    memset(L->vW, 0, L->out * L->in * sizeof(double));
    memset(L->mb, 0, L->out * sizeof(double));
    memset(L->vb, 0, L->out * sizeof(double));
}

static void dfwd(DLayer *L, const double *x)
{
    memcpy(L->x, x, L->in * sizeof(double));
    for (size_t o = 0; o < L->out; o++) {
        const double *w = L->W + o * L->in;
        double a = L->b[o];
        size_t j = 0;
        for (; j + 3 < L->in; j += 4)
            a += w[j]*x[j] + w[j+1]*x[j+1] + w[j+2]*x[j+2] + w[j+3]*x[j+3];
        for (; j < L->in; j++) a += w[j] * x[j];
        L->y[o] = L->relu ? (a > 0.0 ? a : 0.0) : a;
    }
}

static double ad(double *m, double *v, double g, double lr, double b1p, double b2p)
{
    *m = AD_B1 * *m + (1.0 - AD_B1) * g;
    *v = AD_B2 * *v + (1.0 - AD_B2) * g * g;
    return lr * (*m / (1.0 - b1p)) / (sqrt(*v / (1.0 - b2p)) + AD_EPS);
}

static void dbwd(MLP *M, DLayer *L, const double *dy, double *dx, double lr)
{
    if (dx) memset(dx, 0, L->in * sizeof(double));
    for (size_t o = 0; o < L->out; o++) {
        double g = dy[o];
        if (L->relu && L->y[o] <= 0.0) g = 0.0;
        L->b[o] -= ad(&L->mb[o], &L->vb[o], g, lr, M->b1p, M->b2p);
        double *w = L->W + o * L->in;
        for (size_t j = 0; j < L->in; j++) {
            if (dx) dx[j] += g * w[j];
            w[j] -= ad(&L->mW[o * L->in + j], &L->vW[o * L->in + j],
                       g * L->x[j], lr, M->b1p, M->b2p);
        }
    }
}

static void net_init(void *c)
{
    MLP *M = c;
    M->tstep = 0;
    M->b1p = M->b2p = 1.0;
    for (size_t i = 0; i < M->depth; i++) dinit(&M->L[i]);
}

static void net_fwd(void *c, const double *x, double *y)
{
    MLP *M = c;
    const double *cur = x;
    for (size_t i = 0; i < M->depth; i++) {
        dfwd(&M->L[i], cur);
        cur = M->L[i].y;
    }
    memcpy(y, cur, M->out * sizeof(double));
}

static void net_bwd(void *c, const double *x, const double *dy, double lr)
{
    MLP *M = c;
    (void)x;
    /* Per-sample steps see ~n more updates than torch's full batch.
       Scale lr so the C twin stays in the same effective range. */
    lr *= 0.15;
    M->tstep++;
    M->b1p *= AD_B1;
    M->b2p *= AD_B2;
    const double *dcur = dy;
    double *hold = NULL;
    for (size_t i = M->depth; i-- > 0; ) {
        double *dx = (i > 0) ? M->L[i].dx : NULL;
        dbwd(M, &M->L[i], dcur, dx, lr);
        if (i > 0) {
            size_t w = M->L[i].in;
            hold = (double *)realloc(hold, w * sizeof(double));
            memcpy(hold, dx, w * sizeof(double));
            dcur = hold;
        }
    }
    free(hold);
}

static void net_align(void *c, size_t in)
{
    MLP *M = c;
    if (in == M->L[0].in) return;
    dfree(&M->L[0]);
    dalloc(&M->L[0], in, M->hid, 1);
    dinit(&M->L[0]);
    M->in = in;
}

static void net_out(void *c, size_t o)
{
    MLP *M = c;
    if (o == M->L[M->depth - 1].out) return;
    dfree(&M->L[M->depth - 1]);
    dalloc(&M->L[M->depth - 1], M->hid, o, 0);
    dinit(&M->L[M->depth - 1]);
    M->out = o;
}

static void net_k(void *c, size_t k) { (void)c; (void)k; }

static void net_ins(void *c)
{
    MLP *M = c;
    if (M->depth >= MAX_D) return;
    /* identity Linear before the tail: y' = I y, no ReLU */
    DLayer *tail = &M->L[M->depth - 1];
    size_t dim = tail->in;
    memmove(&M->L[M->depth], tail, sizeof(DLayer));
    memset(tail, 0, sizeof(DLayer));
    dalloc(tail, dim, dim, 0);
    memset(tail->W, 0, dim * dim * sizeof(double));
    memset(tail->b, 0, dim * sizeof(double));
    for (size_t i = 0; i < dim; i++) tail->W[i * dim + i] = 1.0;
    M->depth++;
}

static int net_rem(void *c)
{
    MLP *M = c;
    if (M->depth < 2) return -1;
    /* drop the layer before tail */
    size_t idx = M->depth - 2;
    size_t new_in = M->L[idx].in;
    dfree(&M->L[idx]);
    memmove(&M->L[idx], &M->L[idx + 1], (M->depth - idx - 1) * sizeof(DLayer));
    memset(&M->L[M->depth - 1], 0, sizeof(DLayer));
    M->depth--;
    if (M->L[idx].in != new_in) {
        /* stitch tail in */
        size_t out = M->L[idx].out;
        int relu = M->L[idx].relu;
        dfree(&M->L[idx]);
        dalloc(&M->L[idx], new_in, out, relu);
        dinit(&M->L[idx]);
    }
    return 0;
}

static void net_dyn(void *c, int on) { (void)c; (void)on; }
static size_t net_depth(void *c) { return ((MLP *)c)->depth; }
static size_t net_kf(void *c) { (void)c; return 1; }

static size_t net_params(void *c)
{
    MLP *M = c;
    size_t n = 0;
    for (size_t i = 0; i < M->depth; i++)
        n += M->L[i].out * (M->L[i].in + 1);
    return n;
}

static size_t net_nbytes(void *c)
{
    return net_params(c) * sizeof(double);
}

static void net_free(void *c)
{
    MLP *M = c;
    for (size_t i = 0; i < M->depth; i++) dfree(&M->L[i]);
    free(M);
}

static void net_scale(void *c, size_t idx, size_t in, size_t out)
{
    MLP *M = c;
    if (idx >= M->depth) return;
    int relu = M->L[idx].relu;
    dfree(&M->L[idx]);
    dalloc(&M->L[idx], in, out, relu);
    dinit(&M->L[idx]);
    if (idx + 1 < M->depth) {
        size_t o2 = M->L[idx + 1].out;
        int r2 = M->L[idx + 1].relu;
        dfree(&M->L[idx + 1]);
        dalloc(&M->L[idx + 1], out, o2, r2);
        dinit(&M->L[idx + 1]);
    }
    if (idx == 0) M->in = in;
    if (idx + 1 == M->depth) M->out = out;
}

static size_t net_lin(void *c, size_t idx)
{
    MLP *M = c;
    return idx < M->depth ? M->L[idx].in : 0;
}
static size_t net_lout(void *c, size_t idx)
{
    MLP *M = c;
    return idx < M->depth ? M->L[idx].out : 0;
}
static size_t net_lk(void *c, size_t idx) { (void)c; (void)idx; return 1; }
static size_t net_nadd(void *c) { (void)c; return 0; }
static size_t net_ndrop(void *c) { (void)c; return 0; }

AltNet type_nn_cmlp_open(size_t in, size_t out)
{
    MLP *M = (MLP *)calloc(1, sizeof(MLP));
    M->in = in;
    M->out = out;
    M->hid = pick_hid(in, out);
    M->b1p = M->b2p = 1.0;
    dalloc(&M->L[0], in, M->hid, 1);
    dalloc(&M->L[1], M->hid, out, 0);
    M->depth = 2;
    AltNet h = {
        .impl = "c-mlp", .ctx = M, .in = in, .out = out,
        .init = net_init, .forward = net_fwd, .backward = net_bwd,
        .align_inputs = net_align, .set_outputs = net_out,
        .set_or_factors = net_k, .insert_identity = net_ins,
        .remove_hidden = net_rem, .set_dynamic = net_dyn,
        .depth = net_depth, .or_factors = net_kf,
        .param_count = net_params, .nbytes = net_nbytes, .free = net_free,
        .scale_layer = net_scale, .layer_in = net_lin, .layer_out = net_lout,
        .layer_k = net_lk, .n_add = net_nadd, .n_drop = net_ndrop
    };
    return h;
}
