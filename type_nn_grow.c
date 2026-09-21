#include "type_nn_grow.h"
#include "type_nn_layer.h"
#include "type_nn_ln.h"

#include <math.h>
#include <string.h>

static Network *g_grow = NULL;

void tnn_grow_bind(Network *net)
{
    g_grow = net;
}

int tnn_grow_on(const Network *net)
{
    return net && net->growpol != 0;
}

int tnn_grow_apply(Network *net, const char *name)
{
    if (!net || !name || !name[0]) return 0;
    unsigned p = 0;
    if (!strcmp(name, "scale-off") || !strcmp(name, "Goff"))
        p = 0;
    else if (!strcmp(name, "scale-keep") || !strcmp(name, "Gkeep"))
        p = TNN_G_OR_KEEP | TNN_G_AND_KEEP;
    else if (!strcmp(name, "scale-resid") || !strcmp(name, "Gresid"))
        p = TNN_G_OR_RESID | TNN_G_AND_RESID;
    else if (!strcmp(name, "scale-ratio") || !strcmp(name, "Gratio"))
        p = TNN_G_OR_RATIO | TNN_G_AND_RATIO;
    else if (!strcmp(name, "scale-dead") || !strcmp(name, "Gdead"))
        p = TNN_G_OR_DEAD | TNN_G_AND_DEAD;
    else if (!strcmp(name, "scale-or") || !strcmp(name, "Gor"))
        p = TNN_G_OR_RATIO;
    else if (!strcmp(name, "scale-and") || !strcmp(name, "Gand"))
        p = TNN_G_AND_RATIO;
    else if (!strcmp(name, "scale-refuse") || !strcmp(name, "Grefuse"))
        p = TNN_G_OR_RATIO | TNN_G_AND_RATIO | TNN_G_REFUSE;
    else if (!strcmp(name, "scale-energy") || !strcmp(name, "Genergy"))
        p = TNN_G_OR_ENERGY | TNN_G_AND_ENERGY;
    else if (!strcmp(name, "scale-jac") || !strcmp(name, "Gjac"))
        p = TNN_G_OR_JAC | TNN_G_AND_JAC;
    else if (!strcmp(name, "scale-slack") || !strcmp(name, "Gslack"))
        p = TNN_G_OR_SLACK | TNN_G_AND_SLACK;
    else if (!strcmp(name, "scale-sign") || !strcmp(name, "Gsign"))
        p = TNN_G_OR_SIGN | TNN_G_AND_SIGN;
    else if (!strcmp(name, "scale-mix") || !strcmp(name, "Gmix"))
        p = TNN_G_OR_ENERGY | TNN_G_AND_JAC;
    else if (!strcmp(name, "scale-ej") || !strcmp(name, "Gej"))
        p = TNN_G_OR_ENERGY | TNN_G_AND_ENERGY | TNN_G_OR_JAC | TNN_G_AND_JAC;
    else if (!strcmp(name, "scale-layer") || !strcmp(name, "Glayer"))
        p = TNN_G_OR_ENERGY | TNN_G_AND_ENERGY | TNN_G_OR_JAC | TNN_G_AND_JAC
          | TNN_G_REFUSE;
    else if (!strcmp(name, "slim-cap")) {
        net->growpol |= TNN_G_CAP;
        net->max_or = TNN_G_CAP_OR;
        g_grow = net;
        return 1;
    } else if (!strcmp(name, "slim-prune")) {
        net->growpol |= TNN_G_PRUNE;
        g_grow = net;
        return 1;
    } else if (!strcmp(name, "slim-k") || !strcmp(name, "slim-topk")) {
        net->growpol |= TNN_G_TOPK;
        g_grow = net;
        return 1;
    } else if (!strcmp(name, "scale-sched") || !strcmp(name, "Gsched")) {
        net->growpol |= TNN_G_SCHED | TNN_G_OR_ENERGY | TNN_G_AND_ENERGY
                      | TNN_G_OR_JAC | TNN_G_AND_JAC;
        net->sched_grow = 0.40;
        net->sched_cut  = 0.60;
        g_grow = net;
        return 1;
    } else if (!strcmp(name, "scale-sched-tight") || !strcmp(name, "Gsched-t")) {
        net->growpol |= TNN_G_SCHED | TNN_G_OR_ENERGY | TNN_G_AND_ENERGY
                      | TNN_G_OR_JAC | TNN_G_AND_JAC;
        net->sched_grow = 0.25;
        net->sched_cut  = 0.50;
        g_grow = net;
        return 1;
    } else if (!strcmp(name, "scale-sched-wide") || !strcmp(name, "Gsched-w")) {
        net->growpol |= TNN_G_SCHED | TNN_G_OR_ENERGY | TNN_G_AND_ENERGY
                      | TNN_G_OR_JAC | TNN_G_AND_JAC;
        net->sched_grow = 0.55;
        net->sched_cut  = 0.75;
        g_grow = net;
        return 1;
    } else if (!strcmp(name, "scale-sched-bal") || !strcmp(name, "Gsched-b")) {
        /* Cap width at 2. Early: energy AND jac. Late cut starts late
           so mid jac can finish the fit before prune. */
        net->growpol |= TNN_G_SCHED | TNN_G_CAP
                      | TNN_G_OR_ENERGY | TNN_G_AND_ENERGY
                      | TNN_G_OR_JAC | TNN_G_AND_JAC;
        net->max_or = TNN_G_CAP_OR;
        net->sched_grow = 0.30;
        net->sched_cut  = 0.80;
        g_grow = net;
        return 1;
    } else if (!strcmp(name, "scale-phase") || !strcmp(name, "Gphase")) {
        /* No count cap. Grow / cut / shrink from u. */
        net->growpol |= TNN_G_PHASE | TNN_G_SIGNAL | TNN_G_FREE
                      | TNN_G_OR_ENERGY | TNN_G_AND_ENERGY
                      | TNN_G_OR_JAC | TNN_G_AND_JAC;
        net->max_or = (size_t)-1 / 4;
        net->sched_grow = 0.40;
        net->sched_cut  = 0.70;
        g_grow = net;
        return 1;
    } else if (!strcmp(name, "scale-signal") || !strcmp(name, "Gsignal")) {
        net->growpol |= TNN_G_PHASE | TNN_G_SIGNAL | TNN_G_FREE
                      | TNN_G_OR_ENERGY | TNN_G_AND_ENERGY
                      | TNN_G_OR_JAC | TNN_G_AND_JAC;
        net->max_or = (size_t)-1 / 4;
        net->sched_grow = 0.40;
        net->sched_cut  = 0.70;
        g_grow = net;
        return 1;
    } else if (!strcmp(name, "scale-norm") || !strcmp(name, "Gnorm")) {
        net->growpol |= TNN_G_PHASE | TNN_G_SIGNAL | TNN_G_NORM | TNN_G_FREE
                      | TNN_G_OR_ENERGY | TNN_G_AND_ENERGY
                      | TNN_G_OR_JAC | TNN_G_AND_JAC;
        net->max_or = (size_t)-1 / 4;
        net->sched_grow = 0.40;
        net->sched_cut  = 0.70;
        g_grow = net;
        return 1;
    } else if (!strcmp(name, "scale-assemble") || !strcmp(name, "Gassemble")) {
        net->growpol |= TNN_G_PHASE | TNN_G_SIGNAL | TNN_G_ASSEMBLE | TNN_G_FREE
                      | TNN_G_OR_ENERGY | TNN_G_AND_ENERGY
                      | TNN_G_OR_JAC | TNN_G_AND_JAC;
        net->max_or = (size_t)-1 / 4;
        net->sched_grow = 0.40;
        net->sched_cut  = 0.70;
        g_grow = net;
        return 1;
    } else if (!strcmp(name, "scale-compose") || !strcmp(name, "Gcompose")) {
        net->growpol |= TNN_G_PHASE | TNN_G_SIGNAL | TNN_G_COMPOSE | TNN_G_NORM | TNN_G_FREE
                      | TNN_G_OR_ENERGY | TNN_G_AND_ENERGY
                      | TNN_G_OR_JAC | TNN_G_AND_JAC;
        net->max_or = (size_t)-1 / 4;
        net->sched_grow = 0.40;
        net->sched_cut  = 0.70;
        g_grow = net;
        return 1;
    } else if (!strcmp(name, "scale") || !strcmp(name, "Gscale")
            || !strcmp(name, "scale-pulse") || !strcmp(name, "Gpulse")) {
        /* Canonical type-nn: dummy-Or And scale (degree), dummy-width
           Or scale (previous-layer coordinate), signal cut. No
           dummy-And clause spawn — A_k is one product of Ors. No
           count cap. No top-k: live typed factors keep BP weights. */
        net->growpol |= TNN_G_PHASE | TNN_G_SIGNAL | TNN_G_NORM
                      | TNN_G_ASSEMBLE | TNN_G_FREE | TNN_G_WIDTH
                      | TNN_G_TOPK
                      | TNN_G_OR_ENERGY | TNN_G_OR_JAC;
        net->max_or = (size_t)-1 / 4;
        net->sched_grow = 0.40;
        net->sched_cut  = 0.75;
        g_grow = net;
        return 1;
    } else if (!strcmp(name, "scale-forge") || !strcmp(name, "Gforge")) {
        /* Pulse + compose (degree → identity hidden) + residual clock. */
        net->growpol |= TNN_G_PHASE | TNN_G_SIGNAL | TNN_G_NORM
                      | TNN_G_ASSEMBLE | TNN_G_COMPOSE | TNN_G_FREE
                      | TNN_G_CLOCK
                      | TNN_G_OR_ENERGY | TNN_G_AND_ENERGY
                      | TNN_G_OR_JAC | TNN_G_AND_JAC;
        net->max_or = (size_t)-1 / 4;
        net->sched_grow = 0.40;
        net->sched_cut  = 0.75;
        g_grow = net;
        return 1;
    } else
        return 0;

    {
        unsigned slim = net->growpol & (TNN_G_CAP | TNN_G_PRUNE | TNN_G_TOPK);
        net->growpol = p | slim;
    }
    g_grow = net;
    if (net->growpol & TNN_G_CAP)
        net->max_or = TNN_G_CAP_OR;
    else if (p)
        net->max_or = (size_t)-1 / 4;
    return 1;
}

int tnn_grow_phase(const Network *net)
{
    if (!net) return 2;
    double u = network_progress(net);
    double grow = net->sched_grow > 0 ? net->sched_grow : 0.35;
    double cut  = net->sched_cut  > 0 ? net->sched_cut  : 0.75;
    /* Residual EMA stretches grow on loud files (wine 3-way, wdbc)
       and shortens it when the type has already collapsed. */
    if (net->growpol & TNN_G_CLOCK) {
        double e = net->ema_rms;
        if (e < 0.0) e = 0.0;
        if (e > 1.0) e = 1.0;
        grow = 0.32 + 0.30 * e;
        cut  = grow + 0.20 + 0.08 * e;
        if (grow > 0.62) grow = 0.62;
        if (cut > 0.88) cut = 0.88;
    }
    if (u < grow) return 0;
    if (u < cut)  return 1;
    return 2;
}

double tnn_grow_signal_or(const OrNode *o)
{
    if (!o) return 0.0;
    /* Forward contribution of this factor to its And: |Or| · |cofactor|. */
    return fabs(o->value) * (fabs(o->accum) + 1e-12);
}

double tnn_grow_signal_and(const AndNode *a)
{
    if (!a) return 0.0;
    double aa = (a->expn > 0.0) ? a->expn : 1.0;
    /* |A|^a times the incoming gradient magnitude (how loud the head is). */
    return pow(fabs(a->value) + 1e-12, aa) * (fabs(a->grad) + 1e-12);
}

double tnn_grow_thresh(const Network *net, double signal)
{
    int ph = tnn_grow_phase(net);
    double lo, hi;
    if (ph <= 0) {
        /* Grow: dummy must leave ×1 by a real margin. */
        lo = 0.06;
        hi = 0.12;
    } else if (ph == 1) {
        lo = 0.08;
        hi = 0.16;
    } else {
        lo = 0.10;
        hi = 0.20;
    }
    double s = signal / (signal + 1.0);
    double t = hi + (lo - hi) * s;
    /* Cut/shrink only: a saturated factor is not precious precision. */
    if (ph >= 1 && signal > 20.0)
        t = hi;
    return t;
}

static int or_is_dummy(const OrNode *o)
{
    if (!o) return 0;
    if (fabs(o->bias.value - 1.0) > 0.2) return 0;
    const WeightNode *w = o->weight;
    while (w) {
        if (fabs(w->value) > 0.2) return 0;
        w = w->right;
    }
    return 1;
}

static int has_dummy_or(const AndNode *a)
{
    const OrNode *o = a ? a->or_row : NULL;
    while (o) {
        if (or_is_dummy(o)) return 1;
        o = o->right;
    }
    return 0;
}

static int and_is_dummy_g(const AndNode *a)
{
    if (!a || !a->or_row) return 0;
    const OrNode *o = a->or_row;
    while (o) {
        if (!or_is_dummy(o)) return 0;
        o = o->right;
    }
    return 1;
}

static double max_live_or_grad(const AndNode *a)
{
    double m = 0.0;
    const OrNode *o = a ? a->or_row : NULL;
    while (o) {
        if (!or_is_dummy(o) && fabs(o->grad) > m) m = fabs(o->grad);
        o = o->right;
    }
    return m;
}

static double sum_live_or_grad(const AndNode *a)
{
    double s = 0.0;
    const OrNode *o = a ? a->or_row : NULL;
    while (o) {
        if (!or_is_dummy(o)) s += fabs(o->grad);
        o = o->right;
    }
    return s;
}

static void live_or_wgrad(const AndNode *a, double *sum, size_t *n)
{
    double s = 0.0;
    size_t k = 0;
    const OrNode *o = a ? a->or_row : NULL;
    while (o) {
        if (!or_is_dummy(o)) {
            s += fabs(o->bias.grad);
            k++;
            const WeightNode *w = o->weight;
            while (w) {
                s += fabs(w->grad);
                k++;
                w = w->right;
            }
        }
        o = o->right;
    }
    if (sum) *sum = s;
    if (n) *n = k;
}

static double sum_live_or_wgrad(const AndNode *a)
{
    double s = 0.0;
    live_or_wgrad(a, &s, NULL);
    return s;
}

static double mean_live_or_wgrad(const AndNode *a)
{
    double s = 0.0;
    size_t k = 0;
    live_or_wgrad(a, &s, &k);
    return k ? s / (double)k : 0.0;
}

static int live_ors_specialized(const AndNode *a)
{
    int n_live = 0;
    const OrNode *o = a ? a->or_row : NULL;
    while (o) {
        if (!or_is_dummy(o)) {
            n_live++;
            double wl1 = fabs(o->bias.value - 1.0);
            const WeightNode *w = o->weight;
            while (w) {
                wl1 += fabs(w->value);
                w = w->right;
            }
            if (wl1 < TNN_G_SPEC) return 0;
        }
        o = o->right;
    }
    return n_live > 0;
}

static int live_or_sign_conflict(const AndNode *a, double d_and)
{
    int pos = 0, neg = 0;
    const OrNode *o = a ? a->or_row : NULL;
    while (o) {
        if (!or_is_dummy(o)) {
            if (o->grad > TNN_G_DEAD) pos = 1;
            if (o->grad < -TNN_G_DEAD) neg = 1;
        }
        o = o->right;
    }
    if (pos && neg) return 1;
    if (!pos && !neg) return 1;
    if (d_and > TNN_G_DEAD && !pos) return 1;
    if (d_and < -TNN_G_DEAD && !neg) return 1;
    return 0;
}

static double max_live_and_grad(const Layer *l, size_t index)
{
    double m = 0.0;
    const AndNode *a = l ? l->and_row : NULL;
    while (a) {
        if (a->right_index == index && !and_is_dummy_g(a)
            && fabs(a->grad) > m)
            m = fabs(a->grad);
        a = a->right;
    }
    return m;
}

static double sum_live_and_grad(const Layer *l, size_t index)
{
    double s = 0.0;
    const AndNode *a = l ? l->and_row : NULL;
    while (a) {
        if (a->right_index == index && !and_is_dummy_g(a))
            s += fabs(a->grad);
        a = a->right;
    }
    return s;
}

static void live_and_wgrad(const Layer *l, size_t index, double *sum, size_t *n)
{
    double s = 0.0;
    size_t k = 0;
    const AndNode *a = l ? l->and_row : NULL;
    while (a) {
        if (a->right_index == index && !and_is_dummy_g(a)) {
            double os = 0.0;
            size_t on = 0;
            live_or_wgrad(a, &os, &on);
            s += os + fabs(a->expn_grad);
            k += on + 1;
        }
        a = a->right;
    }
    if (sum) *sum = s;
    if (n) *n = k;
}

static double sum_live_and_wgrad(const Layer *l, size_t index)
{
    double s = 0.0;
    live_and_wgrad(l, index, &s, NULL);
    return s;
}

static double mean_live_and_wgrad(const Layer *l, size_t index)
{
    double s = 0.0;
    size_t k = 0;
    live_and_wgrad(l, index, &s, &k);
    return k ? s / (double)k : 0.0;
}

static int live_ands_specialized(const Layer *l, size_t index)
{
    int n_live = 0;
    const AndNode *a = l ? l->and_row : NULL;
    while (a) {
        if (a->right_index == index && !and_is_dummy_g(a)) {
            n_live++;
            if (fabs(a->value - 1.0) < TNN_G_SPEC) return 0;
        }
        a = a->right;
    }
    return n_live > 0;
}

static int live_and_sign_conflict(const Layer *l, size_t index, double d_z)
{
    int pos = 0, neg = 0;
    const AndNode *a = l ? l->and_row : NULL;
    while (a) {
        if (a->right_index == index && !and_is_dummy_g(a)) {
            if (a->grad > TNN_G_DEAD) pos = 1;
            if (a->grad < -TNN_G_DEAD) neg = 1;
        }
        a = a->right;
    }
    if (pos && neg) return 1;
    if (!pos && !neg) return 1;
    if (d_z > TNN_G_DEAD && !pos) return 1;
    if (d_z < -TNN_G_DEAD && !neg) return 1;
    return 0;
}

static int gate_or(unsigned p, const AndNode *a, double ad, double d_and)
{
    int energy = ad > TNN_G_K * (sum_live_or_grad(a) + 1e-12);
    int jac;
    if (g_grow && (g_grow->growpol & TNN_G_NORM)) {
        double rms = g_grow->last_dloss_rms;
        if (rms < 1e-6) rms = 1e-6;
        jac = (ad / rms) > TNN_G_T
           && mean_live_or_wgrad(a) < TNN_G_JAC_K * (ad / rms);
    } else {
        jac = ad > TNN_G_T && sum_live_or_wgrad(a) < TNN_G_JAC_K * ad;
    }
    int both_ej = (p & TNN_G_OR_ENERGY) && (p & TNN_G_OR_JAC);

    if (p & TNN_G_OR_KEEP) return 1;
    if (both_ej) return energy && jac;
    if (p & TNN_G_OR_RESID) return ad > TNN_G_T;
    if (p & TNN_G_OR_RATIO) {
        double mo = max_live_or_grad(a);
        if (mo < 1e-12) mo = 1e-12;
        return ad > TNN_G_K * mo;
    }
    if (p & TNN_G_OR_DEAD)
        return ad > TNN_G_T && max_live_or_grad(a) < TNN_G_DEAD;
    if (p & TNN_G_OR_ENERGY) return energy;
    if (p & TNN_G_OR_JAC) return jac;
    if (p & TNN_G_OR_SLACK)
        return ad > TNN_G_T && live_ors_specialized(a);
    if (p & TNN_G_OR_SIGN)
        return ad > TNN_G_T && live_or_sign_conflict(a, d_and);
    return 0;
}

static int gate_and(unsigned p, const Layer *l, size_t index,
                    double ad, double d_z)
{
    int energy = ad > TNN_G_K * (sum_live_and_grad(l, index) + 1e-12);
    int jac;
    if (g_grow && (g_grow->growpol & TNN_G_NORM)) {
        double rms = g_grow->last_dloss_rms;
        if (rms < 1e-6) rms = 1e-6;
        jac = (ad / rms) > TNN_G_T
           && mean_live_and_wgrad(l, index) < TNN_G_JAC_K * (ad / rms);
    } else {
        jac = ad > TNN_G_T
           && sum_live_and_wgrad(l, index) < TNN_G_JAC_K * ad;
    }
    int both_ej = (p & TNN_G_AND_ENERGY) && (p & TNN_G_AND_JAC);

    if (p & TNN_G_AND_KEEP) return 1;
    if (both_ej) return energy && jac;
    if (p & TNN_G_AND_RESID) return ad > TNN_G_T;
    if (p & TNN_G_AND_RATIO) {
        double ma = max_live_and_grad(l, index);
        if (ma < 1e-12) ma = 1e-12;
        return ad > TNN_G_K * ma;
    }
    if (p & TNN_G_AND_DEAD)
        return ad > TNN_G_T && max_live_and_grad(l, index) < TNN_G_DEAD;
    if (p & TNN_G_AND_ENERGY) return energy;
    if (p & TNN_G_AND_JAC) return jac;
    if (p & TNN_G_AND_SLACK)
        return ad > TNN_G_T && live_ands_specialized(l, index);
    if (p & TNN_G_AND_SIGN)
        return ad > TNN_G_T && live_and_sign_conflict(l, index, d_z);
    return 0;
}

static size_t live_or_count(const AndNode *a)
{
    size_t n = 0;
    const OrNode *o = a ? a->or_row : NULL;
    while (o) {
        if (!or_is_dummy(o)) n++;
        o = o->right;
    }
    return n;
}

int tnn_grow_want_or(AndNode *a, double d_and)
{
    if (!g_grow || !a) return 0;
    unsigned p = g_grow->growpol;
    if (!isfinite(d_and)) return 0;
    if (p & TNN_G_PHASE) {
        int ph = tnn_grow_phase(g_grow);
        if (has_dummy_or(a)) return 0;
        if (!isfinite(a->value) || fabs(a->value) > 20.0) return 0;
        if (a->probe_cool > 0) {
            a->probe_cool--;
            return 0;
        }
        /* And scaling = dummy Or rule. n_ones==0 means the dummy
           just got weight. Grow / cut: always append a new ×1 Or
           so the product can still leave 1. Shrink: no new dummy. */
        if (ph >= 2) return 0;
        {
            /* Live-Or fence from the incoming type size, not a
               dataset name. XOR (n=2) stays degree-2. No extra>2
               clamp — that was a special number. */
            size_t d = g_grow->in_size ? g_grow->in_size : 1;
            size_t cap = 1 + (size_t)log(1.0 + (double)d);
            if (cap < 2) cap = 2;
            if (live_or_count(a) >= cap)
                return 0;
        }
        if ((p & TNN_G_COMPOSE) && ph == 0 && live_or_count(a) >= 2)
            g_grow->ask_depth = 1;
        a->probe_cool = ph == 0 ? 8 : 16;
        return 1;
    }
    if ((p & TNN_G_CAP) && (p & TNN_G_SCHED)) {
        double u = network_progress(g_grow);
        double grow = g_grow->sched_grow > 0 ? g_grow->sched_grow : 0.30;
        /* Early: raise the Or cap on the clock (2 → 4). Residual
           can push one more, up to 6, before the cut. */
        size_t floor = 2;
        if (grow > 0.0 && u < grow)
            floor = 2 + (size_t)(u / grow * 3.0);
        if (floor > 4) floor = 4;
        if (g_grow->max_or < floor)
            g_grow->max_or = floor;
        size_t cap = g_grow->max_or ? g_grow->max_or : TNN_G_CAP_OR;
        if (u < grow && g_grow->last_dloss_l1 > 0.15 && cap < 6)
            g_grow->max_or = cap + 1;
    }
    if (has_dummy_or(a)) return 0;
    if (p & TNN_G_CAP) {
        size_t n = 0;
        const OrNode *o = a->or_row;
        while (o) { n++; o = o->right; }
        size_t cap = g_grow->max_or ? g_grow->max_or : TNN_G_CAP_OR;
        if (n >= cap)
            return 0;
    }
    if (p & TNN_G_SCHED) {
        double u = network_progress(g_grow);
        double grow = g_grow->sched_grow > 0 ? g_grow->sched_grow : 0.40;
        double cut  = g_grow->sched_cut  > 0 ? g_grow->sched_cut  : 0.60;
        if (u >= cut) return 0;
        unsigned g;
        if (u < grow)
            g = (p & TNN_G_CAP)
                ? (TNN_G_OR_ENERGY | TNN_G_OR_JAC)   /* both: selective grow */
                : (TNN_G_OR_ENERGY | TNN_G_OR_RESID);
        else
            g = TNN_G_OR_JAC;
        return gate_or(g, a, fabs(d_and), d_and);
    }
    return gate_or(p, a, fabs(d_and), d_and);
}

int tnn_grow_want_and(const Layer *l, size_t index, double d_z)
{
    if (!g_grow || !l) return 0;
    unsigned p = g_grow->growpol;
    /* And-scale on the board is a dummy Or inside the existing
       product, not a new And clause. Skip clause spawn unless a
       recipe actually asked for an And gate. */
    unsigned and_bits = TNN_G_AND_KEEP | TNN_G_AND_RESID | TNN_G_AND_RATIO
                      | TNN_G_AND_DEAD | TNN_G_AND_ENERGY | TNN_G_AND_JAC
                      | TNN_G_AND_SLACK | TNN_G_AND_SIGN;
    if ((p & TNN_G_PHASE) && !(p & and_bits))
        return 0;
    if (p & TNN_G_PHASE) {
        int ph = tnn_grow_phase(g_grow);
        if (ph >= 2) return 0;
        const AndNode *it = l->and_row;
        while (it) {
            if (it->right_index == index && and_is_dummy_g(it))
                return 0;
            it = it->right;
        }
        if (!isfinite(d_z)) return 0;
        {
            AndNode *slot = l->and_row;
            while (slot) {
                if (slot->right_index == index) {
                    if (slot->probe_cool > 0) {
                        slot->probe_cool--;
                        return 0;
                    }
                    break;
                }
                slot = slot->right;
            }
        }
        /* a_{i,r} has no upper cap. Assembly-index descent is free
           to raise a on every live clause; that does not block a
           new generator. A type-size fence (not an a-cap) keeps
           the number of live Ands from running away. */
        {
            size_t nlive = 0;
            const AndNode *c = l->and_row;
            while (c) {
                if (c->right_index == index && !and_is_dummy_g(c))
                    nlive++;
                c = c->right;
            }
            size_t d = g_grow->in_size ? g_grow->in_size : 1;
            size_t cap = 2;
            if (d > 4)
                cap = 3;
            if (nlive >= cap)
                return 0;
        }
        double ad = fabs(d_z);
        double rms = g_grow->last_dloss_rms;
        if (rms < 1e-6) rms = 1e-6;
        int want;
        if (ph == 0) {
            want = gate_and(TNN_G_AND_ENERGY | TNN_G_AND_JAC, l, index, ad, d_z);
            if ((p & TNN_G_CLOCK) && want == 0 && g_grow->out_size > 1
                && l->in_size <= 16) {
                size_t nlive = 0;
                const AndNode *c = l->and_row;
                while (c) {
                    if (c->right_index == index && !and_is_dummy_g(c))
                        nlive++;
                    c = c->right;
                }
                if ((ad / rms) > 0.25 && nlive < 2)
                    want = 1;
            }
        } else
            want = gate_and(TNN_G_AND_JAC, l, index, ad, d_z);
        if (want) {
            AndNode *slot = l->and_row;
            while (slot) {
                if (slot->right_index == index) {
                    slot->probe_cool = 8;
                    break;
                }
                slot = slot->right;
            }
        }
        return want;
    }
    const AndNode *a = l->and_row;
    while (a) {
        if (a->right_index == index && and_is_dummy_g(a))
            return 0;
        a = a->right;
    }
    if (!isfinite(d_z)) return 0;
    if (p & TNN_G_SCHED) {
        double u = network_progress(g_grow);
        double grow = g_grow->sched_grow > 0 ? g_grow->sched_grow : 0.40;
        double cut  = g_grow->sched_cut  > 0 ? g_grow->sched_cut  : 0.60;
        if (u >= cut) return 0;
        unsigned g;
        if (u < grow)
            g = (p & TNN_G_CAP)
                ? (TNN_G_AND_ENERGY | TNN_G_AND_JAC)
                : (TNN_G_AND_ENERGY | TNN_G_AND_RESID);
        else
            g = TNN_G_AND_JAC;
        return gate_and(g, l, index, fabs(d_z), d_z);
    }
    return gate_and(p, l, index, fabs(d_z), d_z);
}

int tnn_grow_probes_refused(const Layer *l, size_t index)
{
    if (!l) return 0;
    int dummy_and = 0, dummy_and_dead = 0;
    int dummy_or = 0, dummy_or_dead = 0;
    const AndNode *a = l->and_row;
    while (a) {
        if (a->right_index == index) {
            if (and_is_dummy_g(a)) {
                dummy_and = 1;
                if (fabs(a->grad) < TNN_G_DEAD) dummy_and_dead = 1;
            }
            const OrNode *o = a->or_row;
            while (o) {
                if (or_is_dummy(o)) {
                    dummy_or = 1;
                    if (fabs(o->grad) < TNN_G_DEAD) dummy_or_dead = 1;
                }
                o = o->right;
            }
        }
        a = a->right;
    }
    return dummy_and && dummy_and_dead && dummy_or && dummy_or_dead;
}
