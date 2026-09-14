#include "type_nn_layer.h"
#include "type_nn_grow.h"

#include <math.h>
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
    else
        return 0;

    net->layerpol = p;
    if (p) {
        net->layer_probe = 1;
        /* One hidden, same depth as c-mlp. */
        net->max_depth = 2;
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

int tnn_layer_is_identity(const Layer *l)
{
    if (!l || l->in_size != l->out_size || !l->and_row) return 0;
    const double t = TNN_LP_ID_BAND;
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

static size_t hidden_wanted_out(const Network *net, const Layer *at)
{
    size_t in = at && at->in_size ? at->in_size : net->in_size;
    size_t out = net->tail ? net->tail->out_size : net->out_size;
    /* Born (random hidden): match c-mlp width. Early identity insert
       stays square so the map does not jump at step 0. */
    if (net->layerpol & TNN_LP_BORN) {
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

static Layer *insert_hidden(Network *net, Layer *at)
{
    if (!net || !at) return NULL;
    if (net->depth >= net->max_depth) return NULL;
    size_t want = hidden_wanted_out(net, at);
    Layer *hid = network_insert_identity(net, at);
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

    while (l && l != net->tail) {
        Layer *next = l->next;
        if (tnn_layer_is_identity(l)) {
            if (tnn_layer_dw_l1(l) > dw_cut) {
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
    if (count_identity_hidden(net) > 0) return 0;

    if (p & TNN_LP_DUMMY)
        return 1;

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
    if (!net || !net->layerpol) return;
    if (!(net->layerpol & (TNN_LP_EARLY | TNN_LP_BORN))) return;
    if (net->depth != 1 || !net->tail) return;
    insert_hidden(net, net->tail);
}

void tnn_layer_step(Network *net)
{
    if (!net || !net->layerpol) return;

    /* One structural edit per backward: drop XOR insert.
       The next sample's Jacobian decides the other half. */
    int dropped = 0;
    if (net->layerpol & TNN_LP_DROP)
        dropped = drop_identity_hiddens(net, (net->layerpol & TNN_LP_DUMMY) ? 1 : 0);
    else if (net->layerpol & TNN_LP_HOLD)
        dropped = drop_identity_hiddens(net, 1);

    if (!dropped && should_insert(net))
        insert_hidden(net, net->tail);
}
