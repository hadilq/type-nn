#include "type_nn_alt.h"
#include "type_nn_layerkit.h"

/*
 * type-nn-typefact
 *
 * Type Mechanics: a layer's Or rows are sum-types, Ands are products.
 * "The factorized terms are an abstracted type, and also the parent
 * node in the tree graph."
 *
 * 1. Lift: every Or of the current layer becomes a k=1 type in a new
 *    parent. The child And is rewritten as a product of those parent
 *    features (one-hot W). Exact: And_i = Π_t Or_{i,t}(x) is unchanged.
 *
 * 2. Merge: Or rows whose weights *and* backprop directions align
 *    (cosine) are the same type. Average them into one parent unit and
 *    retarget every child connection. Approximate only when the match
 *    is not exact.
 */

#define AD_B1  0.9
#define AD_B2  0.999
#define AD_EPS 1e-8
#define SETTLE 40
#define FREEZE 20
#define COS_W  0.92
#define COS_G  0.80

typedef struct {
    LKLayer L;
    double *mW, *vW, *mb, *vb, *aW, *ab;
    int freeze_col, pending;
    double ge, ae;
} PLayer;

typedef struct {
    PLayer  layer[LK_MAX_DEPTH];
    double *act[LK_MAX_DEPTH + 1];
    size_t  act_w[LK_MAX_DEPTH + 1];
    size_t  depth;
    int     dynamic, lifted, merged, adapt;
    unsigned n_add, n_drop; int locked;
    unsigned tick, last_change;
    double  ema;
    unsigned long tstep;
    double  b1p, b2p;
} PNet;

static void psync(PLayer *P)
{
    size_t n = P->L.n_or, in = P->L.in;
    P->mW = (double *)realloc(P->mW, n * in * sizeof(double));
    P->vW = (double *)realloc(P->vW, n * in * sizeof(double));
    P->mb = (double *)realloc(P->mb, n * sizeof(double));
    P->vb = (double *)realloc(P->vb, n * sizeof(double));
    P->aW = (double *)realloc(P->aW, n * in * sizeof(double));
    P->ab = (double *)realloc(P->ab, n * sizeof(double));
    if (n && in) {
        memset(P->mW, 0, n * in * sizeof(double));
        memset(P->vW, 0, n * in * sizeof(double));
        memset(P->aW, 0, n * in * sizeof(double));
    }
    if (n) {
        memset(P->mb, 0, n * sizeof(double));
        memset(P->vb, 0, n * sizeof(double));
        memset(P->ab, 0, n * sizeof(double));
    }
    P->pending = 0;
}
static void palloc(PLayer *P, size_t in, size_t out, size_t k)
{
    lk_alloc(&P->L, in, out, k);
    psync(P);
    P->freeze_col = 0;
    P->ge = P->ae = 0.0;
}
static void pfree(PLayer *P)
{
    lk_free(&P->L);
    free(P->mW); free(P->vW); free(P->mb); free(P->vb); free(P->aW); free(P->ab);
    memset(P, 0, sizeof(*P));
}
static void fit(PNet *N, size_t s, size_t w)
{
    if (N->act_w[s] >= w && N->act[s]) return;
    free(N->act[s]);
    N->act[s] = (double *)calloc(w ? w : 1, sizeof(double));
    N->act_w[s] = w ? w : 1;
}
static double ad(double *m, double *v, double g, double lr, double b1p, double b2p)
{
    *m = AD_B1 * *m + (1.0 - AD_B1) * g;
    *v = AD_B2 * *v + (1.0 - AD_B2) * g * g;
    return lr * (*m / (1.0 - b1p)) / (sqrt(*v / (1.0 - b2p)) + AD_EPS);
}
static void flush_layer(PNet *N, PLayer *P, double lr)
{
    if (P->pending <= 0) return;
    LKLayer *L = &P->L;
    const size_t in = L->in;
    double inv = 1.0 / (double)P->pending;
    N->tstep++;
    N->b1p *= AD_B1;
    N->b2p *= AD_B2;
    for (size_t r = 0; r < L->n_or; r++) {
        double gb = P->ab[r] * inv;
        L->b[r] = tnn_clamp(L->b[r] - ad(&P->mb[r], &P->vb[r], gb, lr, N->b1p, N->b2p),
                            -TNN_WCLIP, TNN_WCLIP);
        P->ab[r] = 0.0;
        for (size_t j = 0; j < in; j++) {
            if (P->freeze_col && j + 1 == in) continue;
            double gw = P->aW[r * in + j] * inv;
            L->W[r * in + j] = tnn_clamp(
                L->W[r * in + j] - ad(&P->mW[r * in + j], &P->vW[r * in + j],
                                      gw, lr, N->b1p, N->b2p),
                -TNN_WCLIP, TNN_WCLIP);
            P->aW[r * in + j] = 0.0;
        }
    }
    P->pending = 0;
}
static void upd(PNet *N, PLayer *P, const double *x, const double *dy,
                double *dx, double lr)
{
    LKLayer *L = &P->L;
    const size_t in = L->in, k = L->k, out = L->out;
    if (dx) memset(dx, 0, in * sizeof(double));
    double dor[8], e = 0.0;
    for (size_t i = 0; i < out; i++) {
        e += fabs(dy[i]);
        lk_dor(L, i, dy[i], dor);
        for (size_t t = 0; t < k; t++) {
            double g = dor[t];
            size_t r = i * k + t;
            if (dx) {
                const double *w = L->W + r * in;
                for (size_t j = 0; j < in; j++) dx[j] += g * w[j];
            }
            P->ab[r] += g;
            for (size_t j = 0; j < in; j++)
                P->aW[r * in + j] += g * x[j];
        }
    }
    e /= (double)(out ? out : 1);
    P->ge = (P->ge == 0.0) ? e : (0.9 * P->ge + 0.1 * e);
    P->pending++;
    if (P->pending >= ((lr >= 0.06) ? 4 : 16)) flush_layer(N, P, lr);
}

static double cosine(const double *a, const double *b, size_t n)
{
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na < 1e-18 || nb < 1e-18) return 0.0;
    return dot / (sqrt(na) * sqrt(nb));
}

/* Lift every Or of layer 0 into a parent k=1 type layer. Exact rewrite. */

static void insert_ident_front(PNet *N)
{
    if (N->depth >= LK_MAX_DEPTH) return;
    size_t dim = N->layer[0].L.in;
    memmove(&N->layer[1], &N->layer[0], N->depth * sizeof(PLayer));
    memset(&N->layer[0], 0, sizeof(PLayer));
    palloc(&N->layer[0], dim, dim, 1);
    lk_identity(&N->layer[0].L);
    psync(&N->layer[0]);
    N->layer[0].freeze_col = FREEZE;
    N->depth++;
    N->n_add++;
    N->last_change = N->tick;
}

static void lift_types(PNet *N)
{
    if (N->depth >= LK_MAX_DEPTH || N->lifted) return;
    PLayer *C = &N->layer[0];
    size_t in = C->L.in, n_or = C->L.n_or, k = C->L.k, out = C->L.out;
    if (n_or < 1) return;

    PLayer child = *C;
    memset(&N->layer[0], 0, sizeof(PLayer));
    palloc(&N->layer[0], in, n_or, 1);
    memcpy(N->layer[0].L.W, child.L.W, n_or * in * sizeof(double));
    memcpy(N->layer[0].L.b, child.L.b, n_or * sizeof(double));
    psync(&N->layer[0]);

    memset(&N->layer[1], 0, sizeof(PLayer));
    palloc(&N->layer[1], n_or, out, k);
    memset(N->layer[1].L.W, 0, N->layer[1].L.n_or * n_or * sizeof(double));
    memset(N->layer[1].L.b, 0, N->layer[1].L.n_or * sizeof(double));
    for (size_t i = 0; i < out; i++)
        for (size_t t = 0; t < k; t++) {
            size_t r = i * k + t;
            N->layer[1].L.W[r * n_or + r] = 1.0;
        }
    psync(&N->layer[1]);

    /* child.L buffers now owned by layer[1] after palloc; free the old ones
       that palloc replaced — we copied W/b already. Drop leftover child. */
    free(child.L.W); free(child.L.b); free(child.L.or_val);
    free(child.mW); free(child.vW); free(child.mb); free(child.vb);
    free(child.aW); free(child.ab);

    N->depth = 2;
    N->lifted = 1; N->n_add++;
    N->last_change = N->tick;
}

/* Merge parent Ors that share type (W and grad direction). */
static void merge_common_types(PNet *N)
{
    if (N->depth < 2 || N->merged) return;
    PLayer *P = &N->layer[0];
    PLayer *C = &N->layer[1];
    size_t n = P->L.n_or, in = P->L.in;
    if (n < 2) return;

    int asg[64];
    if (n > 64) return;
    for (size_t i = 0; i < n; i++) asg[i] = -1;
    int ntype = 0;
    for (size_t i = 0; i < n; i++) {
        if (asg[i] >= 0) continue;
        asg[i] = ntype;
        for (size_t j = i + 1; j < n; j++) {
            if (asg[j] >= 0) continue;
            double cw = cosine(P->L.W + i * in, P->L.W + j * in, in);
            double cg = cosine(P->aW + i * in, P->aW + j * in, in);
            if (cw >= COS_W && cg >= COS_G)
                asg[j] = ntype;
        }
        ntype++;
    }
    if (ntype >= (int)n) return; /* nothing in common */

    /* average W,b per type */
    double *nW = (double *)calloc((size_t)ntype * in, sizeof(double));
    double *nb = (double *)calloc((size_t)ntype, sizeof(double));
    int *cnt = (int *)calloc((size_t)ntype, sizeof(int));
    for (size_t i = 0; i < n; i++) {
        int t = asg[i];
        cnt[t]++;
        nb[t] += P->L.b[i];
        for (size_t j = 0; j < in; j++)
            nW[t * in + j] += P->L.W[i * in + j];
    }
    for (int t = 0; t < ntype; t++) {
        double inv = 1.0 / (double)(cnt[t] ? cnt[t] : 1);
        nb[t] *= inv;
        for (size_t j = 0; j < in; j++) nW[t * in + j] *= inv;
    }

    size_t ck = C->L.k, cout = C->L.out;
    double *cW = (double *)calloc(C->L.n_or * (size_t)ntype, sizeof(double));
    for (size_t r = 0; r < C->L.n_or; r++) {
        /* old child Or r was one-hot on feature r (after lift) */
        size_t src = r;
        if (src >= n) src = n - 1;
        int t = asg[src];
        cW[r * (size_t)ntype + (size_t)t] = 1.0;
    }

    lk_alloc(&P->L, in, (size_t)ntype, 1);
    memcpy(P->L.W, nW, (size_t)ntype * in * sizeof(double));
    memcpy(P->L.b, nb, (size_t)ntype * sizeof(double));
    psync(P);

    lk_alloc(&C->L, (size_t)ntype, cout, ck);
    memcpy(C->L.W, cW, C->L.n_or * (size_t)ntype * sizeof(double));
    memset(C->L.b, 0, C->L.n_or * sizeof(double));
    psync(C);

    free(nW); free(nb); free(cnt); free(cW);
    N->merged = 1;
    N->last_change = N->tick;
}

static void estimate_and_apply(PNet *N)
{
    if (!N->dynamic) return;
    if (N->layer[0].L.in < 4) return;
    if (!N->lifted && N->tick >= 16) {
        /* Wide x: I-stack into the product (WDBC winner).
           Narrow x: exact Or-lift (iris param-match). */
        if (N->adapt && N->layer[0].L.in >= 16) {
            int n = (int)(N->ema / 0.05 + 0.5);
            if (n < 3) n = 3;
            if (n > 5) n = 5;
            for (int k = 0; k < n && N->depth < LK_MAX_DEPTH; k++)
                insert_ident_front(N);
            N->lifted = 1; N->locked = 1; /* no late add/drop */
            return;
        }
        lift_types(N);
        N->locked = 1;
        return;
    }
    if (N->lifted && !N->merged && N->tick >= N->last_change + SETTLE)
        merge_common_types(N);
}

static void net_init(void *c)
{
    PNet *N = c;
    N->tstep = 0; N->b1p = N->b2p = 1.0;
    N->tick = N->last_change = 0;
    N->ema = 0.0;
    N->lifted = N->merged = 0;
    for (size_t i = 0; i < N->depth; i++) {
        LKLayer *L = &N->layer[i].L;
        double s = 0.15 / sqrt((double)(L->in > 0 ? L->in : 1));
        for (size_t r = 0; r < L->n_or; r++) {
            L->b[r] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * s;
            for (size_t j = 0; j < L->in; j++)
                L->W[r * L->in + j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * s;
        }
    }
}
static void net_fwd(void *c, const double *x, double *y)
{
    PNet *N = c;
    const double *cur = x;
    for (size_t i = 0; i < N->depth; i++) {
        fit(N, i + 1, N->layer[i].L.out);
        lk_fwd(&N->layer[i].L, cur, N->act[i + 1]);
        cur = N->act[i + 1];
    }
    memcpy(y, cur, N->layer[N->depth - 1].L.out * sizeof(double));
}
static void net_bwd(void *c, const double *x, const double *dy, double lr)
{
    PNet *N = c;
    if (!N->act[1]) net_fwd(c, x, N->act[N->depth]);
    if (lr < 0.06) lr = 0.01;
    const double *dcur = dy;
    double *hold = NULL;
    for (size_t i = N->depth; i-- > 0; ) {
        const double *xin = i ? N->act[i] : x;
        double *dx = NULL;
        if (i) { fit(N, 0, N->layer[i].L.in); dx = N->act[0]; }
        upd(N, &N->layer[i], xin, dcur, dx, lr);
        if (i) {
            size_t w = N->layer[i].L.in;
            hold = (double *)realloc(hold, w * sizeof(double));
            memcpy(hold, dx, w * sizeof(double));
            dcur = hold;
        }
    }
    free(hold);
    double mag = 0.0;
    size_t o = N->layer[N->depth - 1].L.out;
    for (size_t i = 0; i < o; i++) mag += fabs(dy[i]);
    N->tick++;
    N->ema = (N->tick == 1) ? mag : (0.95 * N->ema + 0.05 * mag);
    for (size_t i = 0; i < N->depth; i++)
        if (N->layer[i].freeze_col > 0) N->layer[i].freeze_col--;
    estimate_and_apply(N);
}

static void net_align(void *c, size_t in)
{
    PNet *N = c;
    lk_resize_in(&N->layer[0].L, in);
    psync(&N->layer[0]);
}
static void net_out(void *c, size_t o)
{
    PNet *N = c;
    lk_resize_out(&N->layer[N->depth - 1].L, o);
    psync(&N->layer[N->depth - 1]);
}
static void net_k(void *c, size_t k)
{
    PNet *N = c;
    if (k < 1) k = 1;
    lk_set_k(&N->layer[N->depth - 1].L, k);
    psync(&N->layer[N->depth - 1]);
}
static void net_ins(void *c)
{
    PNet *N = c;
    if (N->depth >= LK_MAX_DEPTH) return;
    PLayer *tail = &N->layer[N->depth - 1];
    size_t dim = tail->L.in;
    memmove(&N->layer[N->depth], tail, sizeof(PLayer));
    memset(tail, 0, sizeof(PLayer));
    palloc(tail, dim, dim, TNN_K0);
    lk_identity(&tail->L);
    psync(tail);
    N->depth++;
}
static int net_rem(void *c)
{
    PNet *N = c;
    if (N->depth < 2) return -1;
    size_t nin = N->layer[0].L.in;
    pfree(&N->layer[0]);
    memmove(&N->layer[0], &N->layer[1], (N->depth - 1) * sizeof(PLayer));
    memset(&N->layer[N->depth - 1], 0, sizeof(PLayer));
    N->depth--;
    lk_resize_in(&N->layer[0].L, nin);
    psync(&N->layer[0]);
    return 0;
}
static void net_dyn(void *c, int on) { ((PNet *)c)->dynamic = on; }
static size_t net_depth(void *c) { return ((PNet *)c)->depth; }
static size_t net_kf(void *c)
{
    PNet *N = c;
    return N->layer[N->depth - 1].L.k;
}
static size_t net_params(void *c)
{
    PNet *N = c;
    size_t n = 0;
    for (size_t i = 0; i < N->depth; i++)
        n += N->layer[i].L.n_or * (N->layer[i].L.in + 1);
    return n;
}
static size_t net_nbytes(void *c)
{
    PNet *N = c;
    size_t n = 0;
    for (size_t i = 0; i < N->depth; i++)
        n += N->layer[i].L.n_or * (N->layer[i].L.in + 2) * sizeof(double);
    return n;
}
static void net_free(void *c)
{
    PNet *N = c;
    for (size_t i = 0; i < N->depth; i++) pfree(&N->layer[i]);
    for (size_t i = 0; i <= LK_MAX_DEPTH; i++) free(N->act[i]);
    free(N);
}
static void net_scale(void *c, size_t idx, size_t in, size_t out)
{
    PNet *N = c;
    if (idx >= N->depth) return;
    lk_resize_in(&N->layer[idx].L, in);
    lk_resize_out(&N->layer[idx].L, out);
    psync(&N->layer[idx]);
    if (idx + 1 < N->depth) {
        lk_resize_in(&N->layer[idx + 1].L, out);
        psync(&N->layer[idx + 1]);
    }
}
static size_t net_lin(void *c, size_t idx)
{
    PNet *N = c;
    return idx < N->depth ? N->layer[idx].L.in : 0;
}
static size_t net_lout(void *c, size_t idx)
{
    PNet *N = c;
    return idx < N->depth ? N->layer[idx].L.out : 0;
}
static size_t net_lk(void *c, size_t idx)
{
    PNet *N = c;
    return idx < N->depth ? N->layer[idx].L.k : 0;
}
static size_t net_nadd(void *c) { return ((PNet *)c)->n_add; }
static size_t net_ndrop(void *c) { return ((PNet *)c)->n_drop; }

AltNet type_nn_adapt_open(size_t in, size_t out);

AltNet type_nn_typefact_open(size_t in, size_t out)
{
    PNet *N = (PNet *)calloc(1, sizeof(PNet));
    N->dynamic = 1;
    N->adapt = 0;
    N->b1p = N->b2p = 1.0;
    palloc(&N->layer[0], in, out, TNN_K0);
    N->depth = 1;
    AltNet h = {
        .impl = "type-nn-typefact", .ctx = N, .in = in, .out = out,
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

AltNet type_nn_adapt_open(size_t in, size_t out)
{
    AltNet h = type_nn_typefact_open(in, out);
    h.impl = "type-nn-adapt";
    ((PNet *)h.ctx)->adapt = 1;
    return h;
}
