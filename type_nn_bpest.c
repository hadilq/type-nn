#include "type_nn_alt.h"
#include "type_nn_layerkit.h"

/*
 * type-nn-bpest
 *
 * Use backprop *layer energy* to estimate how many layers to add/remove.
 *
 * After each backward pass we keep an EMA of
 *   ge[i]  = mean(|dAnd|) arriving at layer i   (error energy)
 *   ae[i]  = mean(|And|)  leaving layer i       (activity)
 *
 * Estimate (at most once per SETTLE steps):
 *   n_add_proj    = 1 if the first layer is a product on wide raw x
 *                   (ge[0] high ⇒ residual is not a raw quadratic)
 *   n_add_readout = 1 if the tail is a product writing class scores
 *                   (ge[tail] high and out > 1 ⇒ need a linear mix)
 *   n_remove      = count of hidden layers whose activity is tiny
 *                   *and* ge is tiny after freeze expires
 *
 * Target shape is the one that actually moved UCI MSE:
 *   k=1 projection → k=2 features → k=1 readout
 * Extra layers beyond that are refused. XOR (in < 4) stays depth-1.
 */

#define AD_B1  0.9
#define AD_B2  0.999
#define AD_EPS 1e-8
#define SETTLE 40
#define FREEZE 20
#define MAX_HID 32
#define MODE_GAP   1
#define MODE_CURV  2
#define MODE_READ1 4  /* readout even when out==1 */
#define MODE_K     8  /* grow mid/tail k from gap */
#define MODE_W     16 /* grow hidden width from ge */
#define MODE_COMBO (MODE_GAP|MODE_CURV|MODE_READ1|MODE_K|MODE_W)
#define MODE_CUBE  (MODE_COMBO)
#define MODE_WIDE  (MODE_COMBO)


typedef struct {
    LKLayer L;
    double *mW, *vW, *mb, *vb, *aW, *ab;
    int freeze_col, pending;
    double ge, ae; /* BP error energy, activity */
} PLayer;

typedef struct {
    PLayer  layer[LK_MAX_DEPTH];
    double *act[LK_MAX_DEPTH + 1];
    size_t  act_w[LK_MAX_DEPTH + 1];
    size_t  depth;
    int     dynamic, mode;
    unsigned tick, last_change;
    double  ema;
    unsigned long tstep;
    double  b1p, b2p;
    double  gap;      /* product gap of current tail */
    double  flat;     /* mean Adam-v (curvature / noise) */
    const char *name;
} PNet;

static size_t pick_hid(size_t in, size_t out)
{
    size_t h = 8;
    if (in >= 10) h = 16;
    if (in >= 24) h = 24;
    if (h < out + 2) h = out + 2;
    if (h > MAX_HID) h = MAX_HID;
    return h;
}

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
    double dor[8];
    double e = 0.0;
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
    int batch = (lr >= 0.06) ? 4 : 16;
    if (P->pending >= batch) flush_layer(N, P, lr);
}

/* ── architecture moves driven by ge[] / ae[] ── */

static int has_k1_front(const PNet *N)
{
    return N->depth >= 2 && N->layer[0].L.k == 1;
}
static int has_k1_readout(const PNet *N)
{
    return N->depth >= 2 && N->layer[N->depth - 1].L.k == 1;
}

static void insert_proj(PNet *N)
{
    if (N->depth >= LK_MAX_DEPTH) return;
    PLayer *tail = &N->layer[0];
    size_t in = tail->L.in, out = tail->L.out;
    size_t H = pick_hid(in, out);
    memmove(&N->layer[1], tail, N->depth * sizeof(PLayer));
    memset(&N->layer[0], 0, sizeof(PLayer));
    palloc(&N->layer[0], in, H, 1);
    lk_resize_in(&N->layer[1].L, H);
    psync(&N->layer[1]);
    N->layer[1].freeze_col = FREEZE;
    N->depth++;
    N->last_change = N->tick;
}

static void insert_readout(PNet *N)
{
    if (N->depth >= LK_MAX_DEPTH) return;
    PLayer *tail = &N->layer[N->depth - 1];
    size_t o = tail->L.out;
    /* mid keeps current out as feature width; readout maps H → o with k=1.
       If tail already writes class dim, first widen it to a feature block. */
    size_t feat = pick_hid(tail->L.in, o);
    if (feat < o) feat = o;
    lk_resize_out(&tail->L, feat);
    psync(tail);
    palloc(&N->layer[N->depth], feat, o, 1);
    N->layer[N->depth].freeze_col = FREEZE;
    N->depth++;
    N->last_change = N->tick;
}

static int layer_idle(const PLayer *P)
{
    if (P->L.k != 1) return 0;
    const LKLayer *L = &P->L;
    if (L->in != L->out) return 0;
    double acc = 0.0;
    size_t n = 0;
    for (size_t i = 0; i < L->out; i++) {
        for (size_t j = 0; j < L->in; j++) {
            double w = L->W[i * L->in + j];
            double t = (i == j) ? 1.0 : 0.0;
            acc += fabs(w - t);
            n++;
        }
        acc += fabs(L->b[i]);
        n++;
    }
    return (acc / (double)(n ? n : 1) < 0.08) && P->ge < 0.02 && P->ae < 0.05;
}

static void remove_idle(PNet *N)
{
    if (N->depth < 2) return;
    for (size_t i = 0; i + 1 < N->depth; i++) {
        if (N->layer[i].freeze_col > 0) continue;
        if (!layer_idle(&N->layer[i])) continue;
        size_t nin = N->layer[i].L.in;
        pfree(&N->layer[i]);
        memmove(&N->layer[i], &N->layer[i + 1],
                (N->depth - i - 1) * sizeof(PLayer));
        memset(&N->layer[N->depth - 1], 0, sizeof(PLayer));
        N->depth--;
        lk_resize_in(&N->layer[i].L, nin);
        psync(&N->layer[i]);
        N->last_change = N->tick;
        return;
    }
}

static void grow_width(PNet *N)
{
    if (N->depth < 2) return;
    PLayer *H = &N->layer[0];
    if (H->L.out >= MAX_HID) return;
    size_t nout = H->L.out + 2;
    if (nout > MAX_HID) nout = MAX_HID;
    lk_resize_out(&H->L, nout);
    psync(H);
    lk_resize_in(&N->layer[1].L, nout);
    psync(&N->layer[1]);
    N->layer[1].freeze_col = FREEZE;
    N->last_change = N->tick;
}

static void grow_mid_k(PNet *N)
{
    /* prefer the product layer */
    size_t idx = N->depth >= 2 ? N->depth - 2 : N->depth - 1;
    if (N->layer[N->depth - 1].L.k >= 2) idx = N->depth - 1;
    if (N->depth >= 3) idx = 1;
    LKLayer *L = &N->layer[idx].L;
    if (L->k >= 3) return;
    lk_set_k(L, L->k + 1);
    psync(&N->layer[idx]);
    N->last_change = N->tick;
}

static void estimate_and_apply(PNet *N)
{
    if (!N->dynamic) return;
    if (N->tick < N->last_change + SETTLE) return;
    if (N->ema < 0.03) {
        remove_idle(N);
        return;
    }

    int n_add_proj = 0, n_add_readout = 0, n_remove = 0;
    size_t in0 = N->layer[0].L.in;
    size_t outT = N->layer[N->depth - 1].L.out;
    int gap_low = (N->mode & MODE_GAP) ? (N->gap < 0.15) : 0;
    int curv_flat = (N->mode & MODE_CURV) ? (N->flat < 1e-4 && N->ema > 0.08) : 0;

    if (in0 >= 4 && !has_k1_front(N) && N->layer[0].ge > 0.05)
        n_add_proj = 1;

    /* readout: multi-class OR 1-d target sitting on a k=2 tail after a proj */
    if (!has_k1_readout(N) && N->layer[N->depth - 1].L.k >= 2) {
        int multi = outT >= 2 && N->layer[N->depth - 1].ge > 0.04;
        int unit  = (N->mode & MODE_READ1) && has_k1_front(N) && outT == 1
                    && N->layer[N->depth - 1].ge > 0.03;
        if (multi || unit) n_add_readout = 1;
    }

    for (size_t i = 0; i + 1 < N->depth; i++)
        if (layer_idle(&N->layer[i])) n_remove++;

    if (n_add_proj && N->depth < 3) { insert_proj(N); return; }
    if (n_add_readout && N->depth < 4) { insert_readout(N); return; }

    /* degree too low: product barely differs from first Or, error still high */
    if ((N->mode & MODE_K) && gap_low && N->ema > 0.06 && N->depth >= 2) {
        grow_mid_k(N);
        return;
    }
    /* features insufficient: error energy + flat Adam-v → widen basis */
    if ((N->mode & MODE_W) && N->depth >= 2 &&
        (N->layer[0].ge > 0.06 || curv_flat) && N->ema > 0.05) {
        grow_width(N);
        return;
    }
    if (n_remove) remove_idle(N);
}

static void net_init(void *c)
{
    PNet *N = c;
    N->tstep = 0; N->b1p = N->b2p = 1.0;
    N->tick = N->last_change = 0;
    N->ema = 0.0;
    for (size_t i = 0; i < N->depth; i++) {
        LKLayer *L = &N->layer[i].L;
        double s = 0.15 / sqrt((double)(L->in > 0 ? L->in : 1));
        for (size_t r = 0; r < L->n_or; r++) {
            L->b[r] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * s;
            for (size_t j = 0; j < L->in; j++)
                L->W[r * L->in + j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * s;
        }
        N->layer[i].ge = N->layer[i].ae = 0.0;
    }
}

static void net_fwd(void *c, const double *x, double *y)
{
    PNet *N = c;
    const double *cur = x;
    for (size_t i = 0; i < N->depth; i++) {
        fit(N, i + 1, N->layer[i].L.out);
        lk_fwd(&N->layer[i].L, cur, N->act[i + 1]);
        double a = 0.0;
        size_t o = N->layer[i].L.out;
        for (size_t j = 0; j < o; j++) a += fabs(N->act[i + 1][j]);
        a /= (double)(o ? o : 1);
        N->layer[i].ae = (N->layer[i].ae == 0.0) ? a : (0.9 * N->layer[i].ae + 0.1 * a);
        if (N->layer[i].L.k >= 2) {
            LKLayer *L = &N->layer[i].L;
            double g = 0.0;
            for (size_t u = 0; u < L->out; u++) {
                double o0 = L->or_val[u * L->k + 0];
                g += fabs(N->act[i + 1][u] - o0);
            }
            g /= (double)(L->out ? L->out : 1);
            N->gap = (N->gap == 0.0) ? g : (0.9 * N->gap + 0.1 * g);
        }
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


static AltNet open_est(const char *name, int mode, size_t in, size_t out)
{
    PNet *N = (PNet *)calloc(1, sizeof(PNet));
    N->dynamic = 1;
    N->mode = mode;
    N->name = name;
    N->b1p = N->b2p = 1.0;
    palloc(&N->layer[0], in, out, TNN_K0);
    N->depth = 1;
    AltNet h = {
        .impl = name, .ctx = N, .in = in, .out = out,
        .init = net_init, .forward = net_fwd, .backward = net_bwd,
        .align_inputs = net_align, .set_outputs = net_out,
        .set_or_factors = net_k, .insert_identity = net_ins,
        .remove_hidden = net_rem, .set_dynamic = net_dyn,
        .depth = net_depth, .or_factors = net_kf,
        .param_count = net_params, .nbytes = net_nbytes, .free = net_free,
        .scale_layer = net_scale, .layer_in = net_lin, .layer_out = net_lout,
        .layer_k = net_lk
    };
    return h;
}

static AltNet open_est_wide(const char *name, int mode, size_t in, size_t out)
{
    AltNet h = open_est(name, mode, in, out);
    /* start already in proj2 shape with a wider basis */
    PNet *N = (PNet *)h.ctx;
    size_t H = pick_hid(in, out);
    if (H < 16) H = 16;
    pfree(&N->layer[0]);
    palloc(&N->layer[0], in, H, 1);
    palloc(&N->layer[1], H, H, TNN_K0);
    palloc(&N->layer[2], H, out, 1);
    N->depth = 3;
    return h;
}

AltNet type_nn_bpgap_open(size_t in, size_t out)
{ return open_est("type-nn-bpgap", MODE_GAP | MODE_READ1, in, out); }
AltNet type_nn_bpcurv_open(size_t in, size_t out)
{ return open_est("type-nn-bpcurv", MODE_CURV | MODE_READ1 | MODE_W, in, out); }
AltNet type_nn_bpcombo_open(size_t in, size_t out)
{ return open_est("type-nn-bpcombo", MODE_COMBO, in, out); }
AltNet type_nn_bpcube_open(size_t in, size_t out)
{ return open_est("type-nn-bpcube", MODE_CUBE, in, out); }
AltNet type_nn_bpwide_open(size_t in, size_t out)
{ return open_est_wide("type-nn-bpwide", MODE_WIDE, in, out); }
