#include "type_nn_win.h"
#include "type_nn_layerkit.h"

/*
 * Shared And/Or engine. Policies:
 *   win  general BP recipe:
 *        F revert a depth insert that does not drop energy
 *        H grow k=1 width toward rank(in); stop on flat ge
 *        rank unused columns before new depth
 *        B slope-stall only once a hidden map exists
 *        D energy = class margin when out>1 (|dY| dies after acc=1)
 *   A    refuse only the site that just failed
 *   B    insert only on a flat EMA slope (true stall)
 *   C    width / k / depth are separate gates, not one score fight
 *   D    stall energy is class margin when out > 1
 *   E    tail insert is identity (function does not jump)
 *   F    snapshot, train one settle, revert if EMA did not drop
 *   G    softmax + cross-entropy gradient (MSE dy is decoded)
 *   H    grow width toward rank(in), stop when ge goes flat
 *   I    out>1 ⇒ product is features, not class scores (readout)
 */

#define AD_B1   0.9
#define AD_B2   0.999
#define AD_EPS  1e-8
#define SETTLE  48
#define WARMUP  32
#define FREEZE  24
#define EMA_FLOOR 0.04
#define GE_FLOOR  0.03
#define GE_GROW   0.08
#define ID_TAU    0.08
#define W_PRUNE   0.03
#define MAX_H     32
#define MAX_K     4

enum {
    POL_WIN = 0,
    POL_A, POL_B, POL_C, POL_D, POL_E,
    POL_F, POL_G, POL_H, POL_I
};

typedef struct {
    LKLayer L;
    double *mW, *vW, *mb, *vb, *aW, *ab;
    int freeze_col, pending;
    double ge, ae;
} PLayer;

typedef struct {
    int live;
    size_t depth;
    size_t in[LK_MAX_DEPTH], out[LK_MAX_DEPTH], k[LK_MAX_DEPTH];
    double *W[LK_MAX_DEPTH], *b[LK_MAX_DEPTH];
    double ema;
    unsigned n_add, n_drop;
} Trial;

typedef struct {
    PLayer  layer[LK_MAX_DEPTH];
    double *act[LK_MAX_DEPTH + 1];
    size_t  act_w[LK_MAX_DEPTH + 1];
    size_t  depth, task_out;
    int     dynamic, policy;
    unsigned n_add, n_drop;
    unsigned tick, last_change, settle_n;
    double  ema, ema_ref, ge_ref, ema_hist;
    int     refuse_insert, refuse_grow;
    int     refuse_site[LK_MAX_DEPTH + 1];
    int     i_done, pending_trial, last_kind, last_site, skip_trial;
    unsigned long tstep;
    double  b1p, b2p;
    Trial   trial;
    const char *name;
    double  logit[8];
    int     have_logit;
} PNet;

static int combo(const PNet *N) { return N->policy == POL_WIN; }
static int uses_trial(const PNet *N) { return N->policy == POL_F || combo(N); }
static int uses_ce(const PNet *N)
{
    /* CE on And-scores exploded hold MSE in G. Only G keeps it;
       the winner trains MSE on the I readout. */
    (void)N;
    return 0;
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
    const double decay = 1e-4;
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
            double gw = P->aW[r * in + j] * inv + decay * L->W[r * in + j];
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
    if (P->pending >= 16) flush_layer(N, P, lr);
}

static size_t seed_width(size_t out)
{
    return out < 2 ? 2 : out;
}

static int is_identity_layer(const PLayer *P)
{
    const LKLayer *L = &P->L;
    if (L->k != 1 || L->in != L->out) return 0;
    for (size_t i = 0; i < L->out; i++) {
        for (size_t j = 0; j < L->in; j++) {
            double want = (i == j) ? 1.0 : 0.0;
            if (fabs(L->W[i * L->in + j] - want) > ID_TAU) return 0;
        }
    }
    return 1;
}

static void trial_clear(Trial *T)
{
    if (!T->live) return;
    for (size_t i = 0; i < LK_MAX_DEPTH; i++) {
        free(T->W[i]); free(T->b[i]);
        T->W[i] = T->b[i] = NULL;
    }
    memset(T, 0, sizeof(*T));
}

static void trial_save(PNet *N)
{
    trial_clear(&N->trial);
    N->trial.live = 1;
    N->trial.depth = N->depth;
    N->trial.ema = N->ema;
    N->trial.n_add = N->n_add;
    N->trial.n_drop = N->n_drop;
    for (size_t i = 0; i < N->depth; i++) {
        LKLayer *L = &N->layer[i].L;
        N->trial.in[i] = L->in;
        N->trial.out[i] = L->out;
        N->trial.k[i] = L->k;
        size_t nw = L->n_or * L->in, nb = L->n_or;
        N->trial.W[i] = (double *)malloc((nw ? nw : 1) * sizeof(double));
        N->trial.b[i] = (double *)malloc((nb ? nb : 1) * sizeof(double));
        if (nw) memcpy(N->trial.W[i], L->W, nw * sizeof(double));
        if (nb) memcpy(N->trial.b[i], L->b, nb * sizeof(double));
    }
}

static void trial_restore(PNet *N)
{
    if (!N->trial.live) return;
    while (N->depth > N->trial.depth) {
        pfree(&N->layer[N->depth - 1]);
        N->depth--;
    }
    for (size_t i = 0; i < N->trial.depth; i++) {
        if (i >= N->depth) {
            memset(&N->layer[i], 0, sizeof(PLayer));
            palloc(&N->layer[i], N->trial.in[i], N->trial.out[i], N->trial.k[i]);
            N->depth++;
        } else {
            lk_resize_in(&N->layer[i].L, N->trial.in[i]);
            lk_resize_out(&N->layer[i].L, N->trial.out[i]);
            lk_set_k(&N->layer[i].L, N->trial.k[i]);
            psync(&N->layer[i]);
        }
        LKLayer *L = &N->layer[i].L;
        size_t nw = L->n_or * L->in, nb = L->n_or;
        if (nw) memcpy(L->W, N->trial.W[i], nw * sizeof(double));
        if (nb) memcpy(L->b, N->trial.b[i], nb * sizeof(double));
        N->layer[i].ge = N->layer[i].ae = 0.0;
    }
    N->n_add = N->trial.n_add;
    N->n_drop = N->trial.n_drop;
    N->ema = N->trial.ema;
    N->ema_ref = N->trial.ema;
    trial_clear(&N->trial);
}

static void mark_change(PNet *N, int kind, int site)
{
    if (kind == 1 && N->ema_ref > 0.0 && N->ema >= 0.98 * N->ema_ref) {
        if (!combo(N)) N->refuse_insert = 1;
        if (site >= 0 && site <= (int)LK_MAX_DEPTH)
            N->refuse_site[site] = 1;
    }
    /* H evaluates refuse_grow on the next settle, not this tick. */
    if (kind == 0) {
        N->refuse_insert = 0;
        N->refuse_grow = 0;
    }
    N->last_kind = kind;
    N->last_site = site;
    N->last_change = N->tick;
    N->ema_ref = N->ema;
    N->ge_ref = 0.0;
    for (size_t i = 0; i < N->depth; i++)
        if (N->layer[i].ge > N->ge_ref) N->ge_ref = N->layer[i].ge;
}

static void insert_at(PNet *N, size_t idx)
{
    if (N->depth >= LK_MAX_DEPTH) return;
    if (idx > N->depth) idx = N->depth;

    if (uses_trial(N) && !N->skip_trial) trial_save(N);

    if (idx == 0 && N->layer[0].L.k >= 2) {
        size_t in = N->layer[0].L.in;
        size_t H = seed_width(N->layer[N->depth - 1].L.out);
        memmove(&N->layer[1], &N->layer[0], N->depth * sizeof(PLayer));
        memset(&N->layer[0], 0, sizeof(PLayer));
        palloc(&N->layer[0], in, H, 1);
        lk_resize_in(&N->layer[1].L, H);
        psync(&N->layer[1]);
        N->layer[1].freeze_col = FREEZE;
        N->depth++;
        N->n_add++;
        mark_change(N, 1, 0);
        N->pending_trial = uses_trial(N) && !N->skip_trial;
        return;
    }

    if (N->depth > 0 && idx == N->depth && N->layer[N->depth - 1].L.k >= 2) {
        PLayer *tail = &N->layer[N->depth - 1];
        size_t o = tail->L.out;
        if (N->policy == POL_E || combo(N)) {
            /* Identity readout: Ands stay class-sized, map does not jump. */
            palloc(&N->layer[N->depth], o, o, 1);
            lk_identity(&N->layer[N->depth].L);
            psync(&N->layer[N->depth]);
        } else {
            size_t feat = seed_width(o);
            lk_resize_out(&tail->L, feat);
            psync(tail);
            palloc(&N->layer[N->depth], feat, o, 1);
        }
        N->layer[N->depth].freeze_col = FREEZE;
        N->depth++;
        N->n_add++;
        mark_change(N, 1, (int)idx);
        N->pending_trial = uses_trial(N) && !N->skip_trial;
        return;
    }

    size_t dim = (idx < N->depth) ? N->layer[idx].L.in
                                  : N->layer[N->depth - 1].L.out;
    if (idx < N->depth) {
        memmove(&N->layer[idx + 1], &N->layer[idx],
                (N->depth - idx) * sizeof(PLayer));
    }
    memset(&N->layer[idx], 0, sizeof(PLayer));
    palloc(&N->layer[idx], dim, dim, 1);
    lk_identity(&N->layer[idx].L);
    psync(&N->layer[idx]);
    N->layer[idx].freeze_col = FREEZE;
    N->depth++;
    N->n_add++;
    mark_change(N, 1, (int)idx);
    N->pending_trial = uses_trial(N) && !N->skip_trial;
}

static int remove_idx(PNet *N, size_t idx)
{
    if (N->depth < 2 || idx >= N->depth) return -1;
    size_t nin = (idx == 0) ? N->layer[0].L.in : N->layer[idx].L.in;
    pfree(&N->layer[idx]);
    if (idx + 1 < N->depth)
        memmove(&N->layer[idx], &N->layer[idx + 1],
                (N->depth - idx - 1) * sizeof(PLayer));
    memset(&N->layer[N->depth - 1], 0, sizeof(PLayer));
    N->depth--;
    if (idx < N->depth) {
        lk_resize_in(&N->layer[idx].L, nin);
        psync(&N->layer[idx]);
    }
    N->n_drop++;
    mark_change(N, 0, (int)idx);
    return 0;
}

static int drop_idle_hidden(PNet *N)
{
    if (N->depth < 2) return 0;
    for (size_t i = 0; i + 1 < N->depth; i++) {
        PLayer *P = &N->layer[i];
        if (P->freeze_col > 0) continue;
        if (!is_identity_layer(P)) continue;
        if (P->ge > GE_FLOOR || P->ae > 0.15) continue;
        return remove_idx(N, i) == 0;
    }
    return 0;
}

static int drop_dead_or(PNet *N)
{
    for (size_t i = 0; i < N->depth; i++) {
        LKLayer *L = &N->layer[i].L;
        if (L->k <= 2) continue;
        int any = 0;
        for (size_t o = 0; o < L->out; o++) {
            double *w = L->W + ((o * L->k) + (L->k - 1)) * L->in;
            double n2 = 0.0;
            for (size_t j = 0; j < L->in; j++) n2 += w[j] * w[j];
            if (n2 < 0.01) any = 1;
        }
        if (!any) continue;
        lk_set_k(L, L->k - 1);
        psync(&N->layer[i]);
        N->n_drop++;
        mark_change(N, 0, -1);
        return 1;
    }
    return 0;
}

static int grow_width(PNet *N)
{
    int best = -1;
    double floor = (N->policy == POL_H || combo(N)) ? 1e-4 : GE_GROW;
    double best_ge = floor;
    size_t hid_end = N->depth ? N->depth - 1 : 0;
    for (size_t i = 0; i < hid_end; i++) {
        if (N->layer[i].L.k != 1) continue;
        if (N->layer[i].L.out >= MAX_H) continue;
        if (N->policy != POL_H && N->layer[i].L.out >= N->layer[i].L.in) continue;
        if (N->policy == POL_H && N->layer[i].L.out >= N->layer[i].L.in) continue;
        if (N->layer[i].ge > best_ge) {
            best_ge = N->layer[i].ge;
            best = (int)i;
        }
    }
    if (best < 0) return 0;
    if ((N->policy == POL_H || combo(N)) && N->refuse_grow) return 0;
    /* Width is cheap; F-trial only depth inserts. H stops on flat ge. */
    if (uses_trial(N) && !combo(N) && !N->skip_trial) trial_save(N);
    PLayer *P = &N->layer[best];
    size_t nout = P->L.out + 1;
    lk_resize_out(&P->L, nout);
    psync(P);
    if ((size_t)best + 1 < N->depth) {
        lk_resize_in(&N->layer[best + 1].L, nout);
        psync(&N->layer[best + 1]);
    }
    mark_change(N, 2, best);
    N->pending_trial = uses_trial(N) && !combo(N) && !N->skip_trial;
    return 1;
}

static int grow_k(PNet *N)
{
    int best = -1;
    double best_ge = GE_FLOOR;
    for (size_t i = 0; i < N->depth; i++) {
        if (N->layer[i].L.k < 2 || N->layer[i].L.k >= MAX_K) continue;
        if (N->layer[i].ge > best_ge) {
            best_ge = N->layer[i].ge;
            best = (int)i;
        }
    }
    if (best < 0) return 0;
    if (uses_trial(N) && !N->skip_trial) trial_save(N);
    lk_set_k(&N->layer[best].L, N->layer[best].L.k + 1);
    psync(&N->layer[best]);
    mark_change(N, 0, -1);
    N->pending_trial = uses_trial(N) && !N->skip_trial;
    return 1;
}

static size_t pick_site(const PNet *N, double *score_out)
{
    size_t best = N->depth;
    double best_s = -1.0;
    for (size_t i = 0; i <= N->depth; i++) {
        double s = 0.0;
        if (i == 0) {
            const PLayer *L0 = &N->layer[0];
            if (L0->L.k >= 2)
                s = L0->ge * 2.0 + 0.05 * (double)L0->L.in;
            else
                s = L0->ge * 0.2;
        } else if (i == N->depth) {
            const PLayer *T = &N->layer[N->depth - 1];
            if (T->L.k >= 2)
                s = T->ge * 1.5 + 0.05 * (double)T->L.out;
            else
                s = T->ge * 0.2;
        } else {
            const PLayer *U = &N->layer[i - 1];
            const PLayer *D = &N->layer[i];
            double ratio = U->ge / (D->ae + 0.05);
            s = 0.5 * (U->ge + D->ge) * ratio;
        }
        if (N->policy == POL_A && N->refuse_site[i]) s = -1.0;
        if (s > best_s) { best_s = s; best = i; }
    }
    if (score_out) *score_out = best_s;
    return best;
}

static void prune_idle_weights(PNet *N)
{
    for (size_t i = 0; i < N->depth; i++) {
        if (N->layer[i].freeze_col > 0) continue;
        LKLayer *L = &N->layer[i].L;
        for (size_t t = 0; t < L->n_or * L->in; t++) {
            if (fabs(L->W[t]) < W_PRUNE && fabs(N->layer[i].mW[t]) < 1e-4)
                L->W[t] = 0.0;
        }
    }
}

static int site_blocked(const PNet *N, size_t site)
{
    if (N->policy == POL_A || combo(N))
        return site <= LK_MAX_DEPTH && N->refuse_site[site];
    return N->refuse_insert;
}

static void estimate_and_apply(PNet *N)
{
    if (!N->dynamic) return;
    if (N->tick < WARMUP) return;
    if (N->last_change && N->tick < N->last_change + SETTLE) return;

    if (uses_trial(N) && N->pending_trial && N->trial.live) {
        N->pending_trial = 0;
        if (N->ema >= 0.98 * N->trial.ema) {
            int site = N->last_site;
            trial_restore(N);
            if (site >= 0 && site <= (int)LK_MAX_DEPTH)
                N->refuse_site[site] = 1;
            if (N->last_kind == 2) N->refuse_grow = 1;
            if (N->last_kind == 1) N->refuse_insert = 1;
            N->last_change = N->tick;
            return;
        }
        trial_clear(&N->trial);
    }

    if ((N->policy == POL_H || combo(N)) && N->last_kind == 2 && N->ge_ref > 0.0) {
        double gmax = 0.0;
        for (size_t i = 0; i < N->depth; i++)
            if (N->layer[i].ge > gmax) gmax = N->layer[i].ge;
        if (gmax >= 0.98 * N->ge_ref) N->refuse_grow = 1;
    }

    if (drop_idle_hidden(N)) return;
    if (drop_dead_or(N)) return;

    /* I: product should emit features when there is more than one label. */
    if (N->policy == POL_I && !N->i_done && N->task_out > 1
        && N->layer[N->depth - 1].L.k >= 2
        && N->layer[N->depth - 1].L.out == N->task_out
        && N->depth < LK_MAX_DEPTH) {
        N->skip_trial = 1;
        insert_at(N, N->depth);
        N->skip_trial = 0;
        N->i_done = 1;
        return;
    }

    int stalled;
    /* B's slope test: combo uses it only after a net already has
       hidden maps, so the first basis on a wide binary file is not delayed. */
    if (N->policy == POL_B || (combo(N) && N->depth >= 2)) {
        N->settle_n++;
        double prev = (N->ema_hist > 0.0) ? N->ema_hist : N->ema;
        double rel = fabs(N->ema - prev) / (prev > 1e-6 ? prev : 1e-6);
        N->ema_hist = N->ema;
        stalled = (N->settle_n >= 2) && (rel < 0.05) && (N->ema > EMA_FLOOR);
    } else {
        stalled = (N->ema > 0.95 * (N->ema_ref > 0.0 ? N->ema_ref : N->ema))
                  && (N->ema > EMA_FLOOR);
    }

    if (N->policy == POL_C) {
        if (!N->refuse_grow && grow_width(N)) return;
        if (stalled && grow_k(N)) return;
        if (stalled && N->depth < LK_MAX_DEPTH) {
            double site_s = 0.0;
            size_t site = pick_site(N, &site_s);
            if (site_s > GE_FLOOR && !site_blocked(N, site))
                insert_at(N, site);
        }
        return;
    }

    if (!N->refuse_grow && grow_width(N)) return;

    /* Unused column rank on a k=1 map: grow that before adding depth. */
    if (combo(N)) {
        int rank_left = 0;
        size_t hid = N->depth ? N->depth - 1 : 0;
        for (size_t i = 0; i < hid; i++)
            if (N->layer[i].L.k == 1 && N->layer[i].L.out < N->layer[i].L.in
                && N->layer[i].L.out < MAX_H)
                rank_left = 1;
        if (rank_left) return;
    }

    if (!stalled) {
        if (N->ema < EMA_FLOOR) prune_idle_weights(N);
        return;
    }

    double site_s = 0.0;
    size_t site = pick_site(N, &site_s);

    double grow_s = 0.0;
    size_t last = N->depth ? N->depth - 1 : 0;
    for (size_t i = 0; i < last; i++)
        if (N->layer[i].L.k == 1 && N->layer[i].L.out < MAX_H
            && N->layer[i].ge > grow_s)
            grow_s = N->layer[i].ge * 1.1;

    double k_s = 0.0;
    for (size_t i = 0; i < N->depth; i++)
        if (N->layer[i].L.k >= 2 && N->layer[i].L.k < MAX_K
            && N->layer[i].ge > k_s)
            k_s = N->layer[i].ge * 0.7;

    if (grow_s >= site_s && grow_s >= k_s && grow_s > GE_FLOOR) {
        if (grow_width(N)) return;
    }
    if (k_s >= site_s && k_s > GE_FLOOR) {
        if (grow_k(N)) return;
    }
    if (site_s > GE_FLOOR && N->depth < LK_MAX_DEPTH && !site_blocked(N, site))
        insert_at(N, site);
}

static void net_init(void *c)
{
    PNet *N = c;
    N->tstep = 0; N->b1p = N->b2p = 1.0;
    N->tick = N->last_change = N->settle_n = 0;
    N->ema = N->ema_ref = N->ge_ref = N->ema_hist = 0.0;
    N->n_add = N->n_drop = 0;
    N->refuse_insert = N->refuse_grow = 0;
    N->i_done = N->pending_trial = N->skip_trial = 0;
    N->last_kind = 0; N->last_site = -1;
    memset(N->refuse_site, 0, sizeof(N->refuse_site));
    trial_clear(&N->trial);
    /* Multiclass adapter (in→H→product→out) was tried for wine.
       Same 52/54 hold, 313 params vs 84. Loader already maps class
       1..3 → one-hot 0..2; the miss is cultivar 3 vs 2, not a swapped
       output. Leave the product on x and let F/H/B move. */
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
        cur = N->act[i + 1];
    }
    size_t o = N->layer[N->depth - 1].L.out;
    N->have_logit = 0;
    if (uses_ce(N) && o > 0 && o <= 8) {
        memcpy(N->logit, cur, o * sizeof(double));
        N->have_logit = 1;
        if (o == 1) {
            double z = cur[0];
            if (z > 8) z = 8;
            if (z < -8) z = -8;
            y[0] = 1.0 / (1.0 + exp(-z));
        } else {
            double mx = cur[0];
            for (size_t i = 1; i < o; i++) if (cur[i] > mx) mx = cur[i];
            if (mx > 8) mx = 8;
            double s = 0.0;
            for (size_t i = 0; i < o; i++) {
                y[i] = exp(cur[i] - mx);
                s += y[i];
            }
            if (s < 1e-12) s = 1e-12;
            for (size_t i = 0; i < o; i++) y[i] /= s;
        }
    } else {
        memcpy(y, cur, o * sizeof(double));
    }
}

static void softmax_ce_from_mse(const double *pred, const double *dy,
                                size_t o, double *g)
{
    /* trainer sent dy = (pred - y) / o. Recover y, replace with p - y. */
    double y[8], p[8];
    if (o > 8) o = 8;
    if (o == 1) {
        double pr = pred[0];
        if (pr > 8.0) pr = 8.0;
        if (pr < -8.0) pr = -8.0;
        double sig = 1.0 / (1.0 + exp(-pr));
        double tgt = pred[0] - dy[0];
        if (tgt < 0.0) tgt = 0.0;
        if (tgt > 1.0) tgt = 1.0;
        g[0] = 0.5 * (sig - tgt);
        return;
    }
    double mx = pred[0];
    for (size_t i = 1; i < o; i++) if (pred[i] > mx) mx = pred[i];
    if (mx > 8.0) mx = 8.0;
    double s = 0.0;
    for (size_t i = 0; i < o; i++) {
        p[i] = exp(pred[i] - mx);
        s += p[i];
        y[i] = pred[i] - dy[i] * (double)o;
        if (y[i] < 0.0) y[i] = 0.0;
        if (y[i] > 1.0) y[i] = 1.0;
    }
    if (s < 1e-12) s = 1e-12;
    for (size_t i = 0; i < o; i++) g[i] = 0.5 * (p[i] / s - y[i]);
}

static double margin_mag(const double *pred, const double *dy, size_t o)
{
    if (o <= 1) {
        double m = 0.0;
        for (size_t i = 0; i < o; i++) m += fabs(dy[i]);
        return m;
    }
    size_t ti = 0;
    double best = pred[0] - dy[0] * (double)o; /* recovered y */
    for (size_t i = 1; i < o; i++) {
        double yi = pred[i] - dy[i] * (double)o;
        if (yi > best) { best = yi; ti = i; }
    }
    double runner = -1e9;
    for (size_t i = 0; i < o; i++)
        if (i != ti && pred[i] > runner) runner = pred[i];
    double m = runner - pred[ti] + 1.0;
    return m > 0.0 ? m : 0.0;
}

static void net_bwd(void *c, const double *x, const double *dy, double lr)
{
    PNet *N = c;
    if (!N->act[1]) net_fwd(c, x, N->act[N->depth]);
    if (lr <= 0.0) lr = 0.01;
    if (N->policy == POL_G) lr *= 0.35;
    size_t o = N->layer[N->depth - 1].L.out;
    double gbuf[8];
    const double *dcur = dy;
    if (uses_ce(N) && o <= 8) {
        /* fwd emitted softmax(p); trainer dy = (p-y)/o. CE on logits is p-y. */
        for (size_t i = 0; i < o; i++) gbuf[i] = dy[i] * (double)(o ? o : 1);
        dcur = gbuf;
    } else if (N->policy == POL_G && o <= 8 && N->act[N->depth]) {
        softmax_ce_from_mse(N->act[N->depth], dy, o, gbuf);
        dcur = gbuf;
    }
    double *hold = NULL;
    const double *duse = dcur;
    for (size_t i = N->depth; i-- > 0; ) {
        const double *xin = i ? N->act[i] : x;
        double *dx = NULL;
        if (i) { fit(N, 0, N->layer[i].L.in); dx = N->act[0]; }
        upd(N, &N->layer[i], xin, duse, dx, lr);
        if (i) {
            size_t w = N->layer[i].L.in;
            hold = (double *)realloc(hold, w * sizeof(double));
            memcpy(hold, dx, w * sizeof(double));
            duse = hold;
        }
    }
    free(hold);
    double mag;
    if ((N->policy == POL_D || combo(N)) && o > 1 && N->act[N->depth])
        mag = margin_mag(N->act[N->depth], dy, o);
    else {
        mag = 0.0;
        const double *dg = (N->policy == POL_G) ? dcur : dy;
        for (size_t i = 0; i < o; i++) mag += fabs(dg[i]);
    }
    N->tick++;
    N->ema = (N->tick == 1) ? mag : (0.95 * N->ema + 0.05 * mag);
    if (N->ema_ref == 0.0 && N->tick == WARMUP) N->ema_ref = N->ema;
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
    N->task_out = o;
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
    return remove_idx(N, 0);
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
    for (size_t i = 0; i < N->depth; i++) {
        LKLayer *L = &N->layer[i].L;
        n += L->n_or * (L->in + 1);
    }
    return n;
}
static size_t net_nbytes(void *c)
{
    return net_params(c) * sizeof(double);
}
static void net_free(void *c)
{
    PNet *N = c;
    for (size_t i = 0; i < N->depth; i++) pfree(&N->layer[i]);
    for (size_t i = 0; i <= LK_MAX_DEPTH; i++) free(N->act[i]);
    trial_clear(&N->trial);
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

static AltNet open_pol(size_t in, size_t out, int policy, const char *name)
{
    PNet *N = (PNet *)calloc(1, sizeof(PNet));
    N->dynamic = 1;
    N->policy = policy;
    N->name = name;
    N->task_out = out;
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
        .layer_k = net_lk, .n_add = net_nadd, .n_drop = net_ndrop
    };
    return h;
}

AltNet type_nn_win_open(size_t in, size_t out)
{ return open_pol(in, out, POL_WIN, "type-nn-win"); }
AltNet type_nn_A_open(size_t in, size_t out)
{ return open_pol(in, out, POL_A, "type-nn-A"); }
AltNet type_nn_B_open(size_t in, size_t out)
{ return open_pol(in, out, POL_B, "type-nn-B"); }
AltNet type_nn_C_open(size_t in, size_t out)
{ return open_pol(in, out, POL_C, "type-nn-C"); }
AltNet type_nn_D_open(size_t in, size_t out)
{ return open_pol(in, out, POL_D, "type-nn-D"); }
AltNet type_nn_E_open(size_t in, size_t out)
{ return open_pol(in, out, POL_E, "type-nn-E"); }
AltNet type_nn_F_open(size_t in, size_t out)
{ return open_pol(in, out, POL_F, "type-nn-F"); }
AltNet type_nn_G_open(size_t in, size_t out)
{ return open_pol(in, out, POL_G, "type-nn-G"); }
AltNet type_nn_H_open(size_t in, size_t out)
{ return open_pol(in, out, POL_H, "type-nn-H"); }
AltNet type_nn_I_open(size_t in, size_t out)
{ return open_pol(in, out, POL_I, "type-nn-I"); }
