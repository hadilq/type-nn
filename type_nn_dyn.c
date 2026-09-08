#include "type_nn_alt.h"
#include "type_nn_layerkit.h"

/*
 * Dynamic Type-NN: grow only when EMA residual has stalled, shrink when idle.
 * New factors start as identity and stay frozen for FREEZE steps.
 *
 *   type-nn-dyn-k     Or-count only
 *   type-nn-dyn-w     hidden width only (inserts a frozen identity first)
 *   type-nn-dyn-l     depth only
 *   type-nn-dyn       all three, SGD
 *   type-nn-dyn-adam  all three, Adam
 */

#define SETTLE      48
#define FREEZE      24
#define EMA_A       0.95
#define STALL       0.97
#define FLOOR_K     0.12
#define FLOOR_W     0.20
#define FLOOR_L     0.25
#define IDLE_B      0.12
#define IDLE_W      0.04
#define MAX_K       4
#define MAX_HID     8
#define AD_B1       0.9
#define AD_B2       0.999
#define AD_EPS      1e-8

enum { POL_K = 1, POL_W = 2, POL_L = 4, POL_ADAM = 8 };

typedef struct {
    LKLayer L;
    double *mW, *vW, *mb, *vb;
    int freeze_or, freeze_col, freeze_all;
} PLayer;

typedef struct {
    PLayer  layer[LK_MAX_DEPTH];
    double *act[LK_MAX_DEPTH + 1];
    size_t  act_w[LK_MAX_DEPTH + 1];
    size_t  depth;
    int     dynamic, flags;
    unsigned tick, last_change;
    double  ema, ema_ref;
    unsigned long tstep;
    double  b1p, b2p;
    const char *name;
} PNet;

static void player_sync(PLayer *P)
{
    size_t n = P->L.n_or, in = P->L.in;
    P->mW = (double *)realloc(P->mW, n * in * sizeof(double));
    P->vW = (double *)realloc(P->vW, n * in * sizeof(double));
    P->mb = (double *)realloc(P->mb, n * sizeof(double));
    P->vb = (double *)realloc(P->vb, n * sizeof(double));
    if (n && in) {
        memset(P->mW, 0, n * in * sizeof(double));
        memset(P->vW, 0, n * in * sizeof(double));
    }
    if (n) { memset(P->mb, 0, n * sizeof(double)); memset(P->vb, 0, n * sizeof(double)); }
}

static void player_alloc(PLayer *P, size_t in, size_t out, size_t k)
{
    lk_alloc(&P->L, in, out, k);
    player_sync(P);
    P->freeze_or = P->freeze_col = P->freeze_all = 0;
}

static void player_free(PLayer *P)
{
    lk_free(&P->L);
    free(P->mW); free(P->vW); free(P->mb); free(P->vb);
    memset(P, 0, sizeof(*P));
}

static void fit(PNet *N, size_t s, size_t w)
{
    if (N->act_w[s] >= w && N->act[s]) return;
    free(N->act[s]);
    N->act[s] = (double *)calloc(w ? w : 1, sizeof(double));
    N->act_w[s] = w ? w : 1;
}

static double adam_delta(double *m, double *v, double g, double lr, double b1p, double b2p)
{
    *m = AD_B1 * *m + (1.0 - AD_B1) * g;
    *v = AD_B2 * *v + (1.0 - AD_B2) * g * g;
    return lr * (*m / (1.0 - b1p)) / (sqrt(*v / (1.0 - b2p)) + AD_EPS);
}

static void layer_update(PNet *N, PLayer *P, const double *x,
                         const double *dy, double *dx, double lr)
{
    LKLayer *L = &P->L;
    const size_t in = L->in, k = L->k, out = L->out;
    if (dx) memset(dx, 0, in * sizeof(double));
    double dor[8];
    int skip_all = P->freeze_all > 0;
    int skip_last_or = P->freeze_or > 0;
    int skip_last_col = P->freeze_col > 0;
    int adam = N->flags & POL_ADAM;
    for (size_t i = 0; i < out; i++) {
        lk_dor(L, i, dy[i], dor);
        for (size_t t = 0; t < k; t++) {
            double g = dor[t];
            size_t r = i * k + t;
            int freeze = skip_all || (skip_last_or && t == k - 1);
            if (dx) {
                const double *w = L->W + r * in;
                for (size_t j = 0; j < in; j++) dx[j] += g * w[j];
            }
            if (freeze) continue;
            if (adam)
                L->b[r] = tnn_clamp(L->b[r] - adam_delta(&P->mb[r], &P->vb[r], g, lr, N->b1p, N->b2p),
                                    -TNN_WCLIP, TNN_WCLIP);
            else
                L->b[r] = tnn_clamp(L->b[r] - lr * g, -TNN_WCLIP, TNN_WCLIP);
            double *w = L->W + r * in;
            double *m = P->mW + r * in;
            double *v = P->vW + r * in;
            for (size_t j = 0; j < in; j++) {
                if (skip_last_col && j + 1 == in) continue;
                double gw = g * x[j];
                if (adam)
                    w[j] = tnn_clamp(w[j] - adam_delta(&m[j], &v[j], gw, lr, N->b1p, N->b2p),
                                     -TNN_WCLIP, TNN_WCLIP);
                else
                    w[j] = tnn_clamp(w[j] - lr * gw, -TNN_WCLIP, TNN_WCLIP);
            }
        }
    }
}

static int or_idle(const LKLayer *L, size_t i, size_t t)
{
    size_t r = i * L->k + t;
    if (fabs(L->b[r] - 1.0) > IDLE_B) return 0;
    for (size_t j = 0; j < L->in; j++)
        if (fabs(L->W[r * L->in + j]) > IDLE_W) return 0;
    return 1;
}

static int and_idle(const PNet *N, size_t li, size_t a)
{
    const LKLayer *L = &N->layer[li].L;
    for (size_t t = 0; t < L->k; t++) {
        size_t r = a * L->k + t;
        if (fabs(L->b[r]) > IDLE_W) return 0;
        for (size_t j = 0; j < L->in; j++)
            if (fabs(L->W[r * L->in + j]) > IDLE_W) return 0;
    }
    if (li + 1 < N->depth) {
        const LKLayer *T = &N->layer[li + 1].L;
        for (size_t r = 0; r < T->n_or; r++)
            if (a < T->in && fabs(T->W[r * T->in + a]) > IDLE_W) return 0;
    }
    return 1;
}

static int layer_is_id(const LKLayer *L)
{
    if (L->k < 2 || L->in != L->out) return 0;
    for (size_t i = 0; i < L->out; i++) {
        size_t r0 = i * L->k, r1 = r0 + 1;
        if (fabs(L->b[r0]) > IDLE_B) return 0;
        if (fabs(L->b[r1] - 1.0) > IDLE_B) return 0;
        for (size_t j = 0; j < L->in; j++) {
            double want = (j == i) ? 1.0 : 0.0;
            if (fabs(L->W[r0 * L->in + j] - want) > 0.15) return 0;
            if (fabs(L->W[r1 * L->in + j]) > IDLE_W) return 0;
        }
    }
    return 1;
}

static int stalled(PNet *N, double floor)
{
    if (N->tick < N->last_change + SETTLE) return 0;
    if (N->ema < floor) return 0;
    if (N->ema_ref <= 0.0) return 1;
    return N->ema > STALL * N->ema_ref;
}

static void note_change(PNet *N)
{
    N->last_change = N->tick;
    N->ema_ref = N->ema;
}

static void insert_frozen_id(PNet *N)
{
    if (N->depth >= LK_MAX_DEPTH) return;
    PLayer tail = N->layer[0];
    memset(&N->layer[0], 0, sizeof(PLayer));
    N->layer[1] = tail;
    player_alloc(&N->layer[0], tail.L.in, tail.L.in, TNN_K0);
    lk_identity(&N->layer[0].L);
    player_sync(&N->layer[0]);
    N->layer[0].freeze_all = FREEZE;
    N->depth = 2;
    note_change(N);
}

static void try_shrink(PNet *N)
{
    if ((N->flags & POL_K) && (N->tick % 16 == 0)) {
        LKLayer *T = &N->layer[N->depth - 1].L;
        if (T->k > 2) {
            int idle = 1;
            for (size_t i = 0; i < T->out; i++)
                if (!or_idle(T, i, T->k - 1)) { idle = 0; break; }
            if (idle) {
                lk_set_k(T, T->k - 1);
                player_sync(&N->layer[N->depth - 1]);
                note_change(N);
            }
        }
    }
    if ((N->flags & POL_W) && N->depth >= 2 && (N->tick % 16 == 0)) {
        PLayer *H = &N->layer[0];
        if (H->L.out > H->L.in && and_idle(N, 0, H->L.out - 1)) {
            size_t nout = H->L.out - 1;
            lk_resize_out(&H->L, nout);
            lk_resize_in(&N->layer[1].L, nout);
            player_sync(H);
            player_sync(&N->layer[1]);
            note_change(N);
        }
    }
    if ((N->flags & POL_L) && N->depth >= 2 && (N->tick % 32 == 0)) {
        if (N->layer[0].freeze_all == 0 && layer_is_id(&N->layer[0].L)) {
            size_t nin = N->layer[0].L.in;
            player_free(&N->layer[0]);
            memmove(&N->layer[0], &N->layer[1], (N->depth - 1) * sizeof(PLayer));
            memset(&N->layer[N->depth - 1], 0, sizeof(PLayer));
            N->depth--;
            lk_resize_in(&N->layer[0].L, nin);
            player_sync(&N->layer[0]);
            note_change(N);
        }
    }
}

static void try_grow(PNet *N)
{
    if ((N->flags & POL_K) && stalled(N, FLOOR_K)) {
        LKLayer *T = &N->layer[N->depth - 1].L;
        if (T->k < MAX_K) {
            lk_set_k(T, T->k + 1);
            player_sync(&N->layer[N->depth - 1]);
            N->layer[N->depth - 1].freeze_or = FREEZE;
            note_change(N);
            return;
        }
    }
    if (N->flags & POL_W) {
        if (N->depth == 1 && stalled(N, FLOOR_W)) {
            insert_frozen_id(N);
            return;
        }
        if (N->depth >= 2 && stalled(N, FLOOR_W)) {
            PLayer *H = &N->layer[0];
            if (H->L.out < MAX_HID) {
                size_t nout = H->L.out + 1;
                lk_resize_out(&H->L, nout);
                lk_resize_in(&N->layer[1].L, nout);
                player_sync(H);
                player_sync(&N->layer[1]);
                N->layer[1].freeze_col = FREEZE;
                note_change(N);
                return;
            }
        }
    }
    if ((N->flags & POL_L) && N->depth == 1 && stalled(N, FLOOR_L))
        insert_frozen_id(N);
}

static void net_init(void *c)
{
    PNet *N = c;
    N->tick = N->last_change = 0;
    N->ema = N->ema_ref = 0.0;
    N->tstep = 0;
    N->b1p = N->b2p = 1.0;
    for (size_t i = 0; i < N->depth; i++) lk_init(&N->layer[i].L);
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
    if (!N->act[1]) {
        const double *cur = x;
        for (size_t i = 0; i < N->depth; i++) {
            fit(N, i + 1, N->layer[i].L.out);
            lk_fwd(&N->layer[i].L, cur, N->act[i + 1]);
            cur = N->act[i + 1];
        }
    }
    if (N->flags & POL_ADAM) {
        N->tstep++;
        N->b1p *= AD_B1;
        N->b2p *= AD_B2;
        if (lr > 0.02) lr = 0.01;
    }
    const double *dcur = dy;
    double *hold = NULL;
    for (size_t i = N->depth; i-- > 0; ) {
        const double *xin = i ? N->act[i] : x;
        double *dx = NULL;
        if (i) { fit(N, 0, N->layer[i].L.in); dx = N->act[0]; }
        layer_update(N, &N->layer[i], xin, dcur, dx, lr);
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
    N->ema = (N->tick == 1) ? mag : (EMA_A * N->ema + (1.0 - EMA_A) * mag);
    if (N->ema_ref == 0.0) N->ema_ref = N->ema;
    for (size_t i = 0; i < N->depth; i++) {
        if (N->layer[i].freeze_or > 0) N->layer[i].freeze_or--;
        if (N->layer[i].freeze_col > 0) N->layer[i].freeze_col--;
        if (N->layer[i].freeze_all > 0) N->layer[i].freeze_all--;
    }
    if (!N->dynamic) return;
    try_shrink(N);
    try_grow(N);
}

static void net_align(void *c, size_t in)
{
    PNet *N = c;
    lk_resize_in(&N->layer[0].L, in);
    player_sync(&N->layer[0]);
}
static void net_out(void *c, size_t o)
{
    PNet *N = c;
    lk_resize_out(&N->layer[N->depth - 1].L, o);
    player_sync(&N->layer[N->depth - 1]);
}
static void net_k(void *c, size_t k)
{
    PNet *N = c;
    if (k < 1) k = 1;
    if (k > MAX_K) k = MAX_K;
    lk_set_k(&N->layer[N->depth - 1].L, k);
    player_sync(&N->layer[N->depth - 1]);
}
static void net_ins(void *c)
{
    PNet *N = c;
    if (N->depth >= LK_MAX_DEPTH) return;
    size_t dim = N->layer[N->depth - 1].L.in;
    memmove(&N->layer[N->depth], &N->layer[N->depth - 1], sizeof(PLayer));
    memset(&N->layer[N->depth - 1], 0, sizeof(PLayer));
    player_alloc(&N->layer[N->depth - 1], dim, dim, TNN_K0);
    lk_identity(&N->layer[N->depth - 1].L);
    N->depth++;
}
static int net_rem(void *c)
{
    PNet *N = c;
    if (N->depth <= 1) return -1;
    size_t idx = N->depth - 2, nin = N->layer[idx].L.in;
    player_free(&N->layer[idx]);
    memmove(&N->layer[idx], &N->layer[idx + 1], (N->depth - idx - 1) * sizeof(PLayer));
    memset(&N->layer[N->depth - 1], 0, sizeof(PLayer));
    N->depth--;
    lk_resize_in(&N->layer[idx].L, nin);
    player_sync(&N->layer[idx]);
    return 0;
}
static void net_dyn(void *c, int on) { ((PNet *)c)->dynamic = on ? 1 : 0; }
static size_t net_depth(void *c) { return ((PNet *)c)->depth; }
static size_t net_kf(void *c) { PNet *N = c; return N->layer[N->depth - 1].L.k; }
static size_t net_params(void *c)
{
    PNet *N = c; size_t n = 0;
    for (size_t i = 0; i < N->depth; i++)
        n += N->layer[i].L.n_or * (N->layer[i].L.in + 1);
    return n;
}
static size_t net_nbytes(void *c)
{
    PNet *N = c; size_t n = 0;
    int mul = (N->flags & POL_ADAM) ? 3 : 1;
    for (size_t i = 0; i < N->depth; i++)
        n += (size_t)mul * N->layer[i].L.n_or * (N->layer[i].L.in + 1) * sizeof(double);
    return n;
}
static void net_free(void *c)
{
    PNet *N = c;
    for (size_t i = 0; i < N->depth; i++) player_free(&N->layer[i]);
    for (size_t i = 0; i < LK_MAX_DEPTH + 1; i++) free(N->act[i]);
    free(N);
}
static void net_scale(void *c, size_t idx, size_t in, size_t out)
{
    PNet *N = c;
    if (idx >= N->depth) return;
    lk_resize_in(&N->layer[idx].L, in);
    lk_resize_out(&N->layer[idx].L, out);
    if (idx + 1 < N->depth) lk_resize_in(&N->layer[idx + 1].L, out);
    if (idx > 0) lk_resize_out(&N->layer[idx - 1].L, in);
    player_sync(&N->layer[idx]);
    if (idx + 1 < N->depth) player_sync(&N->layer[idx + 1]);
    if (idx > 0) player_sync(&N->layer[idx - 1]);
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

static AltNet open_pol(const char *name, int flags, size_t in, size_t out)
{
    PNet *N = (PNet *)calloc(1, sizeof(PNet));
    N->depth = 1; N->dynamic = 1; N->flags = flags; N->name = name;
    N->b1p = N->b2p = 1.0;
    player_alloc(&N->layer[0], in, out, TNN_K0);
    AltNet h = {
        .impl = name, .ctx = N, .in = in, .out = out,
        .init = net_init, .forward = net_fwd, .backward = net_bwd,
        .align_inputs = net_align, .set_outputs = net_out,
        .set_or_factors = net_k, .insert_identity = net_ins,
        .remove_hidden = net_rem, .set_dynamic = net_dyn,
        .depth = net_depth, .or_factors = net_kf,
        .param_count = net_params, .nbytes = net_nbytes, .free = net_free,
        .scale_layer = net_scale, .layer_in = net_lin, .layer_out = net_lout
    };
    return h;
}

AltNet type_nn_dyn_open(size_t in, size_t out)
{ return open_pol("type-nn-dyn", POL_K | POL_W | POL_L | POL_ADAM, in, out); }
AltNet type_nn_dyn_sgd_open(size_t in, size_t out)
{ return open_pol("type-nn-dyn-sgd", POL_K | POL_W | POL_L, in, out); }
AltNet type_nn_dyn_k_open(size_t in, size_t out)
{ return open_pol("type-nn-dyn-k", POL_K, in, out); }
AltNet type_nn_dyn_w_open(size_t in, size_t out)
{ return open_pol("type-nn-dyn-w", POL_W, in, out); }
AltNet type_nn_dyn_l_open(size_t in, size_t out)
{ return open_pol("type-nn-dyn-l", POL_L, in, out); }
AltNet type_nn_dyn_adam_open(size_t in, size_t out)
{ return open_pol("type-nn-dyn-adam", POL_K | POL_W | POL_L | POL_ADAM, in, out); }
