#include "type_nn_layer.h"
#include "type_nn_grow.h"
#include "type_nn_ln.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int tnn_layer_on(const Network *net)
{
    return net && net->layerpol != 0;
}

int tnn_layer_apply(Network *net, const char *name)
{
    if (!net || !name || !name[0]) return 0;
    unsigned p = 0;
    if (!strcmp(name, "Loff") || !strcmp(name, "layer-off"))
        p = 0;
    else if (!strcmp(name, "Lgrad") || !strcmp(name, "layer-grad"))
        p = TNN_LP_GRAD;
    else if (!strcmp(name, "Lresid") || !strcmp(name, "layer-resid"))
        p = TNN_LP_RESID;
    else if (!strcmp(name, "Lratio") || !strcmp(name, "layer-ratio"))
        p = TNN_LP_RATIO;
    else if (!strcmp(name, "Ldummy") || !strcmp(name, "layer-dummy"))
        p = TNN_LP_DUMMY | TNN_LP_DROP;
    else if (!strcmp(name, "Ldrop") || !strcmp(name, "layer-drop"))
        p = TNN_LP_DROP;
    else if (!strcmp(name, "Lwide") || !strcmp(name, "layer-wide"))
        p = TNN_LP_GRAD | TNN_LP_WIDE;
    else if (!strcmp(name, "Lnarrow") || !strcmp(name, "layer-narrow"))
        p = TNN_LP_GRAD | TNN_LP_NARROW;
    else if (!strcmp(name, "Lboth") || !strcmp(name, "layer-both"))
        p = TNN_LP_GRAD | TNN_LP_DROP;
    else if (!strcmp(name, "Lcap") || !strcmp(name, "layer-cap"))
        p = TNN_LP_CAP | TNN_LP_STUCK | TNN_LP_RESID | TNN_LP_DROP;
    else if (!strcmp(name, "Lfull") || !strcmp(name, "layer-full"))
        p = TNN_LP_CAP | TNN_LP_RESID | TNN_LP_DROP;
    else if (!strcmp(name, "Lstuck") || !strcmp(name, "layer-stuck"))
        p = TNN_LP_STUCK | TNN_LP_RESID | TNN_LP_DROP;
    else if (!strcmp(name, "Lrefuse") || !strcmp(name, "layer-refuse"))
        p = TNN_LP_REFUSE | TNN_LP_RESID | TNN_LP_DROP;
    else if (!strcmp(name, "Ljac") || !strcmp(name, "layer-jac"))
        p = TNN_LP_REFUSE | TNN_LP_STUCK | TNN_LP_RESID | TNN_LP_DROP;
    else if (!strcmp(name, "Learly") || !strcmp(name, "depth-early")
          || !strcmp(name, "layer-early"))
        p = TNN_LP_EARLY | TNN_LP_HOLD;
    else if (!strcmp(name, "Lhold") || !strcmp(name, "depth-hold")
          || !strcmp(name, "layer-hold"))
        p = TNN_LP_HOLD | TNN_LP_RESID | TNN_LP_DROP;
    else if (!strcmp(name, "Lborn") || !strcmp(name, "depth-born")
          || !strcmp(name, "layer-born"))
        p = TNN_LP_BORN | TNN_LP_EARLY | TNN_LP_HOLD | TNN_LP_WIDE;
    else if (!strcmp(name, "slim-narrow") || !strcmp(name, "Snarrow")) {
        net->layerpol |= TNN_LP_NARROW;
        net->layer_probe = 1;
        return 1;
    } else if (!strcmp(name, "Lsched") || !strcmp(name, "depth-sched")
          || !strcmp(name, "layer-sched"))
        p = TNN_LP_SCHED | TNN_LP_RESID | TNN_LP_DROP;
    else if (!strcmp(name, "Lphase") || !strcmp(name, "depth-phase")
          || !strcmp(name, "layer-phase"))
        p = TNN_LP_PHASE | TNN_LP_LINEAR | TNN_LP_DEPTH | TNN_LP_MULTI
          | TNN_LP_RESID | TNN_LP_DROP;
    else if (!strcmp(name, "Lcompose") || !strcmp(name, "depth-compose")
          || !strcmp(name, "layer-compose"))
        p = TNN_LP_PHASE | TNN_LP_COMPOSE | TNN_LP_LINEAR
          | TNN_LP_DEPTH | TNN_LP_MULTI | TNN_LP_RESID | TNN_LP_DROP;
    else if (!strcmp(name, "Ldepth") || !strcmp(name, "depth-scale")
          || !strcmp(name, "layer-depth"))
        p = TNN_LP_PHASE | TNN_LP_LINEAR | TNN_LP_DEPTH | TNN_LP_MULTI
          | TNN_LP_RESID | TNN_LP_DROP;
    else
        return 0;

    net->layerpol = p;
    if (p) {
        net->layer_probe = 1;
        /* Phase models may open a second hidden; others match c-mlp. */
        /* No task cap. 16 is only a runaway fence; insert is gated
           by the residual, not by this number. */
        net->max_depth = (p & TNN_LP_DEPTH) ? 16 : ((p & TNN_LP_PHASE) ? 8 : 2);
        if (p & TNN_LP_BORN)
            tnn_layer_birth(net);
    }
    return 1;
}

double tnn_layer_l1(const InOutNode *n)
{
    double s = 0.0;
    while (n) {
        s += fabs(n->value);
        n = n->right;
    }
    return s;
}

double tnn_layer_dw_l1(const Layer *l)
{
    if (!l) return 0.0;
    double s = 0.0;
    const AndNode *a = l->and_row;
    while (a) {
        s += fabs(a->grad) + fabs(a->expn_grad);
        const OrNode *o = a->or_row;
        while (o) {
            s += fabs(o->grad) + fabs(o->bias.grad);
            const WeightNode *w = o->weight;
            while (w) {
                s += fabs(w->grad);
                w = w->right;
            }
            o = o->right;
        }
        a = a->right;
    }
    return s;
}

/*
 * Tail cannot grow Or/And by appending: every output head holds
 * MAX_AND Ands and every And holds max_or Ors. Dummy factors
 * occupy those slots; leaving 1 is the STUCK test.
 */
int tnn_layer_at_cap(const Layer *l, size_t max_or)
{
    if (!l || !l->and_row) return 0;
    if (max_or < 1) max_or = 1;
    size_t n_out = l->out_size ? l->out_size : 1;
    for (size_t i = 0; i < n_out; i++) {
        size_t n_and = 0, min_or = max_or;
        const AndNode *a = l->and_row;
        while (a) {
            if (a->right_index == i) {
                n_and++;
                size_t n_or = 0;
                const OrNode *o = a->or_row;
                while (o) { n_or++; o = o->right; }
                if (n_or < min_or) min_or = n_or;
            }
            a = a->right;
        }
        if (n_and < TNN_LP_MAX_AND) return 0;
        if (min_or < max_or) return 0;
    }
    return 1;
}

static int layer_is_id_band(const Layer *l, double t)
{
    if (!l || l->in_size != l->out_size || !l->and_row) return 0;
    if (t <= 0.0) t = TNN_LP_ID_BAND;
    size_t seen = 0;
    const AndNode *a = l->and_row;
    while (a) {
        if (a->right_index >= l->out_size) return 0;
        int picks = 0;
        int dummy = 1;
        const OrNode *o = a->or_row;
        while (o) {
            int live = 0;
            if (fabs(o->bias.value) > t && fabs(o->bias.value - 1.0) > t)
                live = 1;
            const WeightNode *w = o->weight;
            while (w) {
                if (fabs(w->value) > t) {
                    live = 1;
                    if (w->right_index != a->right_index ||
                        fabs(w->value - 1.0) > t)
                        dummy = 0;
                    else
                        picks = 1;
                }
                w = w->right;
            }
            if (live && fabs(o->bias.value) > t)
                dummy = 0;
            o = o->right;
        }
        if (!picks && !dummy) return 0;
        if (picks) seen++;
        a = a->right;
    }
    return seen >= l->out_size;
}

int tnn_layer_is_identity(const Layer *l)
{
    return layer_is_id_band(l, TNN_LP_ID_BAND);
}

static size_t hidden_wanted_out(const Network *net, const Layer *at)
{
    size_t in = at && at->in_size ? at->in_size : net->in_size;
    size_t out = net->tail ? net->tail->out_size : net->out_size;
    /* Born (random hidden): match c-mlp width. Early identity insert
       stays square so the map does not jump at step 0. */
    if (net->layerpol & TNN_LP_BORN) {
        if (net->layerpol & TNN_LP_WIN) {
            size_t h = 4;
            if (h < out) h = out;
            return h;
        }
        size_t h = (in <= 4) ? 8 : 16;
        if (h < out) h = out;
        if (net->layerpol & TNN_LP_WIDE) {
            size_t w = in * 2;
            if (w < h) w = h;
            if (w > 32) w = 32;
            return w;
        }
        return h;
    }
    if (net->layerpol & TNN_LP_WIDE) {
        size_t w = in * 2;
        if (w < in + out) w = in + out;
        if (w > 32) w = 32;
        return w;
    }
    if (net->layerpol & TNN_LP_NARROW) {
        size_t w = (in + 1) / 2;
        if (w < out) w = out;
        if (w < 2) w = 2;
        return w;
    }
    return in;
}

static void layer_drop_zero_weights(Layer *l)
{
    if (!l) return;
    AndNode *a = l->and_row;
    while (a) {
        OrNode *o = a->or_row;
        while (o) {
            WeightNode dummy = {0};
            dummy.right = o->weight;
            WeightNode *prev = &dummy;
            while (prev->right) {
                WeightNode *cur = prev->right;
                if (fabs(cur->value) < 1e-12) {
                    prev->right = cur->right;
                    free(cur);
                } else {
                    prev = cur;
                }
            }
            o->weight = dummy.right;
            o = o->right;
        }
        a = a->right;
    }
}

static Layer *insert_hidden(Network *net, Layer *at)
{
    if (!net || !at) return NULL;
    if (net->depth >= net->max_depth) return NULL;
    size_t want = hidden_wanted_out(net, at);
    if (net->layerpol & TNN_LP_DEPTH)
        want = at->in_size ? at->in_size : net->in_size;
    Layer *hid;
    if (net->layerpol & TNN_LP_DEPTH) {
        /* Same typed product as the tail, same random init. Not ×1. */
        hid = network_insert_similar(net, at);
    } else {
        hid = network_insert_identity(net, at);
        if (hid)
            layer_drop_zero_weights(hid);
    }
    if (!hid) return NULL;
    if (want != hid->out_size) {
        layer_set_outputs(hid, want);
        if (hid->next)
            layer_align_inputs(hid->next, want);
    }
    net->layer_add++;
    return hid;
}

static int count_identity_hidden(const Network *net)
{
    int n = 0;
    for (const Layer *l = net->head; l && l != net->tail; l = l->next)
        if (tnn_layer_is_identity(l)) n++;
    return n;
}

static int count_hiddens(const Network *net)
{
    int n = 0;
    for (const Layer *l = net->head; l && l != net->tail; l = l->next)
        n++;
    return n;
}

int tnn_layer_init_depth(size_t in, size_t out)
{
    if (in < 1) in = 1;
    if (out < 1) out = 1;
    double nm = (double)in * (double)out;
    /* Floor: xor 1+ln2 → 1; wine 1+ln39 → 4; wdbc 1+ln30 → 4. */
    int d = (int)(1.0 + log(nm));
    if (d < 1) d = 1;
    return d;
}

/* Extra maps the residual clock may still open on top of 1+ln(nm). */
static int depth_extra(const Network *net)
{
    size_t in = net->in_size ? net->in_size : (net->tail ? net->tail->in_size : 0);
    size_t out = net->out_size ? net->out_size : (net->tail ? net->tail->out_size : 0);
    int init = tnn_layer_init_depth(in, out);
    int extra = init > 1 ? init - 1 : 0;
    return extra;
}

static double schedule_u(const Network *net)
{
    if (!net || net->orcool_span == 0) return 1.0;
    double u = (double)net->orcool_step / (double)net->orcool_span;
    if (u < 0.0) return 0.0;
    if (u > 1.0) return 1.0;
    return u;
}

static int drop_identity_hiddens(Network *net, int keep_one)
{
    Layer *l = net->head;
    int kept = 0;
    int dropped = 0;
    unsigned p = net->layerpol;
    double dw_cut = (p & TNN_LP_HOLD) ? TNN_LP_HOLD_DW : TNN_LP_KEEP_DW;
    int must_keep_depth2 = (p & (TNN_LP_EARLY | TNN_LP_BORN | TNN_LP_HOLD)) ? 1 : 0;

    if ((p & TNN_LP_HOLD) && schedule_u(net) < TNN_LP_HOLD_U)
        return 0;
    if (p & TNN_LP_SCHED) {
        double cut = net->sched_cut > 0 ? net->sched_cut : 0.60;
        if (schedule_u(net) < cut)
            return 0;
        must_keep_depth2 = 0; /* late: identity hidden may go */
    }
    if (p & TNN_LP_PHASE) {
        int ph = tnn_grow_phase(net);
        if (ph < 2)
            return 0;                 /* grow/cut: keep whatever we opened */
        must_keep_depth2 = 0;
        /* Shrink: an identity hidden is a ×1 factor. Drop it even if
           a little gradient still sits on the diagonal. */
        dw_cut = (p & TNN_LP_DEPTH) ? 1.0 : TNN_LP_KEEP_DW;
    }

    while (l && l != net->tail) {
        Layer *next = l->next;
        if ((tnn_layer_is_identity(l) && net->last_dloss_rms < 0.08)
            || ((p & TNN_LP_DEPTH) && net->last_dloss_rms < 0.04
                && layer_is_id_band(l, 0.15))) {
            /* DEPTH shrink: drop only a true ×1 hidden after the
               residual has collapsed. Keep depth while the type
               is still explaining the sample. */
            if (!(p & TNN_LP_DEPTH) && tnn_layer_dw_l1(l) > dw_cut) {
                l = next;
                continue;
            }
            if (must_keep_depth2 && net->depth <= 2) {
                l = next;
                continue;
            }
            if (keep_one && kept == 0) {
                kept = 1;
            } else {
                if (network_remove_layer(net, l) == 0) {
                    net->layer_drop++;
                    dropped = 1;
                }
            }
        }
        l = next;
    }
    return dropped;
}

static int residual_large(const Network *net)
{
    return net->last_dloss_l1 > TNN_LP_RESID_T;
}

static int weights_stuck(const Network *net)
{
    if (!net->tail) return 0;
    double gw = tnn_layer_dw_l1(net->tail);
    double r = net->last_dloss_l1;
    if (r < 1e-12) r = 1e-12;
    return gw < TNN_LP_STUCK_K * r;
}

static int should_insert(const Network *net)
{
    unsigned p = net->layerpol;
    if (net->depth >= net->max_depth) return 0;
    if (!(p & TNN_LP_MULTI) && net->depth != 1 && !(p & TNN_LP_DUMMY))
        return 0;
    if (count_identity_hidden(net) > 0 && !(p & TNN_LP_MULTI))
        return 0;

    if (p & TNN_LP_DUMMY)
        return 1;

    if (p & TNN_LP_SCHED) {
        double grow = net->sched_grow > 0 ? net->sched_grow : 0.40;
        if (schedule_u(net) >= grow) return 0;
        if (net->depth != 1) return 0;
        if (!residual_large(net) || !net->tail) return 0;
        if (!tnn_layer_at_cap(net->tail, net->max_or)) return 0;
        if (!weights_stuck(net)) return 0;
        return 1;
    }
    if (p & TNN_LP_PHASE) {
        if (tnn_grow_phase(net) != 0) return 0;
        if (schedule_u(net) < 0.05) return 0;
        if (net->depth >= net->max_depth) return 0;
        if (!net->tail) return 0;
        if (count_identity_hidden(net) > 0 && !(p & TNN_LP_MULTI))
            return 0;
        if (p & TNN_LP_DEPTH) {
            if (net->last_dloss_rms <= 0.08 && !net->ask_depth)
                return 0;
            int extra = depth_extra(net);
            int have = count_hiddens(net);
            /* Birth already stacked floor(1+ln(nm)). Grow adds only
               when compose asked for a typed linear map. */
            int want = extra;
            if (net->ask_depth && want < have + 1)
                want = have + 1;
            if (want < 1 && net->ask_depth)
                want = 1;
            if (have >= want)
                return 0;
            unsigned extra_u = extra > 0 ? (unsigned)extra : 1u;
            unsigned gap = 24;
            if (net->orcool_span)
                gap = net->orcool_span / (4u * extra_u + 1u);
            if (gap < 24) gap = 24;
            if (net->orcool_step < gap * (unsigned)(have + 1))
                return 0;
            return 1;
        }
        if ((p & TNN_LP_COMPOSE) && net->ask_depth)
            return residual_large(net) && weights_stuck(net);
        if (!residual_large(net)) return 0;
        if (!weights_stuck(net)) return 0;
        return 1;
    }
    /* Early depth: put a hidden in as soon as the first residual is
       non-trivial. Do not wait for Or/And lists to fill. */
    if ((p & TNN_LP_EARLY) && net->depth == 1)
        return 1;
    if ((p & TNN_LP_HOLD) && net->depth == 1 && residual_large(net))
        return 1;

    if (p & TNN_LP_REFUSE) {
        if (!residual_large(net) || !net->tail) return 0;
        const InOutNode *o = net->tail->out;
        int any = 0;
        while (o) {
            if (tnn_grow_probes_refused(net->tail, o->right_index))
                any = 1;
            o = o->right;
        }
        /* No out list yet: try index 0. */
        if (!net->tail->out)
            any = tnn_grow_probes_refused(net->tail, 0);
        return any;
    }

    /* Capacity-exhausted family: every clause is required. */
    if (p & (TNN_LP_CAP | TNN_LP_STUCK)) {
        if ((p & TNN_LP_RESID) && !residual_large(net))
            return 0;
        if ((p & TNN_LP_CAP) &&
            !tnn_layer_at_cap(net->tail, net->max_or))
            return 0;
        if ((p & TNN_LP_STUCK) && !weights_stuck(net))
            return 0;
        return 1;
    }

    if (p & TNN_LP_GRAD) {
        double mag = net->tail ? tnn_layer_l1(net->tail->din) : 0.0;
        if (mag > TNN_LP_GRAD_T) return 1;
    }
    if (p & TNN_LP_RESID) {
        if (residual_large(net)) return 1;
    }
    if (p & TNN_LP_RATIO) {
        double gx = net->tail ? tnn_layer_l1(net->tail->din) : 0.0;
        double gw = net->tail ? tnn_layer_dw_l1(net->tail) : 0.0;
        if (gw < 1e-12) gw = 1e-12;
        if (gx > TNN_LP_RATIO_K * gw) return 1;
    }
    return 0;
}

void tnn_layer_birth(Network *net)
{
    if (!net || !net->tail) return;
    int want = 1;
    if (tnn_ln_on() || (net->layerpol & TNN_LP_DEPTH))
        want = tnn_layer_init_depth(net->in_size, net->out_size);
    else if (net->layerpol & (TNN_LP_EARLY | TNN_LP_BORN))
        want = 2;
    if (want < 1) want = 1;
    if (want > 16) want = 16;
    if ((size_t)want > net->max_depth)
        net->max_depth = (size_t)want;
    net->init_depth = (size_t)want;
    /* Birth is not a train-time L+. Insert similar product layers
       with the tail's constructor; do not count them as or/layer adds. */
    unsigned add0 = net->layer_add;
    while ((int)net->depth < want) {
        if (net->layerpol & TNN_LP_DEPTH)
            network_insert_identity(net, net->tail);
        else if (tnn_ln_on())
            network_insert_similar(net, net->tail);
        else
            insert_hidden(net, net->tail);
    }
    net->layer_add = add0;
}

static int and_head_dummy(const AndNode *a)
{
    if (!a || !a->or_row) return 0;
    const OrNode *o = a->or_row;
    while (o) {
        if (fabs(o->bias.value - 1.0) > 0.2) return 0;
        const WeightNode *w = o->weight;
        while (w) {
            if (fabs(w->value) > 0.2) return 0;
            w = w->right;
        }
        o = o->right;
    }
    return 1;
}

static int index_is_dummy_out(const Layer *l, size_t idx)
{
    const AndNode *a = l ? l->and_row : NULL;
    int any = 0;
    while (a) {
        if (a->right_index == idx) {
            any = 1;
            if (!and_head_dummy(a)) return 0;
        }
        a = a->right;
    }
    return any;
}

static size_t next_out_index(const Layer *l)
{
    size_t m = 0;
    const AndNode *a = l ? l->and_row : NULL;
    while (a) {
        if (a->right_index + 1 > m) m = a->right_index + 1;
        a = a->right;
    }
    return m;
}

static size_t count_dummy_outs(const Layer *l)
{
    size_t n = 0;
    size_t m = next_out_index(l);
    for (size_t i = 0; i < m; i++)
        if (index_is_dummy_out(l, i)) n++;
    return n;
}

static void make_head_dummy(AndNode *a)
{
    if (!a) return;
    OrNode *o = a->or_row;
    while (o) {
        o->bias.value = 1.0;
        WeightNode *w = o->weight;
        while (w) {
            w->value = 0.0;
            w = w->right;
        }
        o = o->right;
    }
}

static AndNode *and_at_index(Layer *l, size_t idx)
{
    AndNode *a = l ? l->and_row : NULL;
    AndNode *last = NULL;
    while (a) {
        if (a->right_index == idx) return a;
        last = a;
        a = a->right;
    }
    return last;
}

void tnn_layer_width_step(Network *net)
{
    if (!net || !(net->growpol & TNN_G_WIDTH)) return;
    int ph = tnn_grow_phase(net);
    /* Never change the tail's task width. Or-scale is the type
       between two layers: prev.out grows, cur gets a dummy weight. */
    for (Layer *prev = net->head; prev && prev->next; prev = prev->next) {
        Layer *cur = prev->next;
        size_t nd = count_dummy_outs(prev);
        if (ph < 2) {
            /* Grow / cut: keep exactly one dummy outgoing coordinate.
               An identity hidden is the depth dummy; do not widen it
               until it has started to move. */
            if (nd == 0 && !tnn_layer_is_identity(prev)) {
                size_t idx = next_out_index(prev);
                size_t cap = prev->in_size + 4;
                if (idx > cap) continue; /* runaway fence, not a task cap */
                layer_set_outputs(prev, idx + 1);
                make_head_dummy(and_at_index(prev, idx));
                layer_align_inputs(cur, idx + 1);
            }
        } else if (nd >= 2) {
            /* Shrink: drop extra dummy coordinates, keep one.
               Only the last index is cheap to trim. */
            size_t m = next_out_index(prev);
            if (m > 1 && index_is_dummy_out(prev, m - 1)) {
                layer_set_outputs(prev, m - 1);
                layer_align_inputs(cur, m - 1);
            }
        }
    }
}

void tnn_layer_step(Network *net)
{
    if (!net || !net->layerpol) return;

    tnn_layer_width_step(net);

    /* One structural edit per backward: drop XOR insert.
       The next sample's Jacobian decides the other half. */
    int dropped = 0;
    if (net->layerpol & TNN_LP_DROP)
        dropped = drop_identity_hiddens(net, (net->layerpol & TNN_LP_DUMMY) ? 1 : 0);
    else if (net->layerpol & TNN_LP_HOLD)
        dropped = drop_identity_hiddens(net, 1);

    if (!dropped && should_insert(net)) {
        /* Insert between any two layers: pick the junction whose
           incoming residual is loudest. NULL-at = before tail. */
        Layer *at = net->tail;
        double best = -1.0;
        for (Layer *l = net->head; l; l = l->next) {
            double g = tnn_layer_l1(l->din);
            if (g > best) {
                best = g;
                at = l;
            }
        }
        insert_hidden(net, at);
    }
    net->ask_depth = 0;
}
