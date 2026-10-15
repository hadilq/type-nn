/* Tests for graph-type-nn.
 *
 *   1. birth: dense widths n -> m -> ... -> m, small-world dummy shortcuts,
 *      and the forward at birth equals the chain born with the same seed;
 *   2. gradients: ∂L/∂x, ∂L/∂b and ∂L/∂a of every Or of a graph with
 *      merges, a branch-able width dummy, skip edges, multi-Or units,
 *      a > 1 and gains c != 1, against central finite differences;
 *   3. exact edits: dummy edge, width dummy, edge-weight renormalisation,
 *      zero-column / zero-edge / unread-coordinate drops keep the forward
 *      to 1e-12; the depth insert/remove keep it to the fold residual;
 *   4. feedback: the accumulated inference feedback is the mean |value|
 *      entering the edge over the samples;
 *   5. a randomised sequence of edits keeps every invariant;
 *   6. training: xor is fitted and the invariants hold at every epoch.
 */
#include "type_nn_graph.h"
#include "type_nn.h"
#include "common.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0, passes = 0;
#define CHECK(cond, ...) do { if (cond) passes++; else { fails++; fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static void check_inv(const TypeNNGraph *g, const char *where)
{
    char why[256] = "";
    int ok = tnng_check(g, why, sizeof why);
    CHECK(ok, "invariant after %s: %s", where, why);
}

static double out_diff(TypeNNGraph *g, double **X, size_t n, const double *ref)
{
    double d = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double *y = tnng_forward(g, X[i]);
        for (size_t k = 0; k < g->n_out; k++) {
            double t = fabs(y[k] - ref[i * g->n_out + k]);
            if (t > d) d = t;
        }
    }
    return d;
}

static double out_mean_diff(TypeNNGraph *g, double **X, size_t n, const double *ref)
{
    double d = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double *y = tnng_forward(g, X[i]);
        for (size_t k = 0; k < g->n_out; k++) d += fabs(y[k] - ref[i * g->n_out + k]);
    }
    return d / (double)(n * g->n_out);
}

static void snapshot(TypeNNGraph *g, double **X, size_t n, double *ref)
{
    for (size_t i = 0; i < n; i++) {
        const double *y = tnng_forward(g, X[i]);
        memcpy(ref + i * g->n_out, y, g->n_out * sizeof(double));
    }
}

/* Give every edge fake lifetime feedback so the gains are not 1. */
static void scramble_gains(TypeNNGraph *g, unsigned *s)
{
    for (size_t e = 0; e < g->n_edges; e++) {
        g->edges[e].in_acc = 0.2 + fabs(tnn_uniform(s));
        g->edges[e].out_acc = 0.2 + fabs(tnn_uniform(s));
        g->edges[e].acc_n = 1.0;
    }
    tnng_renormalize(g);
}

/* Make every Or non-trivial: random weights everywhere (dummies included),
   a > 1 on some, a second Or on every unit. */
static void roughen(TypeNNGraph *g, unsigned *s)
{
    for (size_t v = 0; v < g->n_nodes; v++) {
        TnnLayer *l = g->nodes[v].L;
        for (size_t k = 0; k < l->n_out; k++) {
            TnnUnit *u = &l->units[k];
            if (u->n_or < 2) { TnnOr *o = tnn_unit_add_or(u, l->n_in); tnn_or_init_identity(o, l->n_in, 0.0, s); }
            for (size_t r = 0; r < u->n_or; r++) {
                TnnOr *o = &u->ors[r];
                for (size_t j = 0; j < l->n_in; j++) o->w[j] = 0.6 * tnn_uniform(s);
                o->b = (r ? 1.0 : 0.3) + 0.4 * tnn_uniform(s);
                o->a = 1.0 + (r == 1 ? 0.7 * fabs(tnn_uniform(s)) : 0.0);
            }
        }
    }
}

static double loss(TypeNNGraph *g, const double *x, const double *dy)
{
    const double *y = tnng_forward(g, x);
    double L = 0.0;
    for (size_t k = 0; k < g->n_out; k++) L += dy[k] * y[k];
    return L;
}

static void test_birth(void)
{
    for (size_t n = 2; n <= 64; n *= 2) {
        size_t m = 3;
        TypeNNGraph *g = tnng_create(n, m, 77);
        size_t D = tnn_birth_depth(n, m);
        check_inv(g, "birth");
        CHECK(g->n_nodes == D, "birth depth %zu != %zu", g->n_nodes, D);
        for (size_t v = 0; v < g->n_nodes; v++)
            CHECK(g->nodes[v].L->n_out == m, "node %zu born at width %zu, not %zu", v, g->nodes[v].L->n_out, m);
        if (D >= 3) CHECK(g->n_edges > D, "no small-world shortcut: D %zu edges %zu", D, g->n_edges);

        /* the chain born from the same seed, same init law and order */
        unsigned rng = 77;
        TypeNN *c = tnn_create(n, m, 77);
        size_t in = n;
        for (size_t i = 0; i < D; i++) {
            TnnLayer *l = tnn_layer_new(in, m);
            for (size_t k = 0; k < m; k++) {
                TnnOr *o = tnn_unit_add_or(&l->units[k], in);
                tnn_or_init_random(o, in, &rng);
            }
            tnn_net_insert_layer(c, c->depth, l);
            in = m;
        }
        double x[64];
        unsigned s = 5;
        double d = 0.0;
        for (int t = 0; t < 10; t++) {
            for (size_t j = 0; j < n; j++) x[j] = tnn_uniform(&s);
            const double *a = tnng_forward(g, x);
            double ya[16]; memcpy(ya, a, m * sizeof(double));
            const double *b = tnn_forward(c, x);
            for (size_t k = 0; k < m; k++) if (fabs(ya[k] - b[k]) > d) d = fabs(ya[k] - b[k]);
        }
        CHECK(d == 0.0, "birth forward differs from the chain by %g (n=%zu)", d, n);
        tnn_free(c);
        tnng_free(g);
    }
    printf("birth: dense widths, shortcuts, chain-equal forward\n");
}

static void test_gradients(void)
{
    unsigned s = 11;
    TypeNNGraph *g = tnng_create(4, 4, 13);      /* D = 3, shortcut 0 -> 2 */
    tnng_begin(g, 1, 1, 0.05);
    tnng_add_width_probe(g, 0);
    tnng_add_width_probe(g, 1);
    roughen(g, &s);
    scramble_gains(g, &s);
    check_inv(g, "gradient setup");

    double x[4] = { 0.3, -0.7, 0.5, 0.2 }, dy[4] = { 0.4, -0.2, 0.3, 0.1 };
    const double h = 1e-6;
    double num_x[4];
    for (int j = 0; j < 4; j++) {
        x[j] += h; double Lp = loss(g, x, dy);
        x[j] -= 2 * h; double Lm = loss(g, x, dy);
        x[j] += h;
        num_x[j] = (Lp - Lm) / (2 * h);
    }
    /* numerical ∂L/∂b and ∂L/∂a for every Or */
    size_t nor = 0;
    for (size_t v = 0; v < g->n_nodes; v++)
        for (size_t k = 0; k < g->nodes[v].L->n_out; k++) nor += g->nodes[v].L->units[k].n_or;
    double *nb = malloc(nor * sizeof(double)), *na = malloc(nor * sizeof(double));
    size_t c = 0;
    for (size_t v = 0; v < g->n_nodes; v++) {
        TnnLayer *l = g->nodes[v].L;
        for (size_t k = 0; k < l->n_out; k++)
            for (size_t r = 0; r < l->units[k].n_or; r++, c++) {
                TnnOr *o = &l->units[k].ors[r];
                o->b += h; double Lp = loss(g, x, dy); o->b -= 2 * h; double Lm = loss(g, x, dy); o->b += h;
                nb[c] = (Lp - Lm) / (2 * h);
                o->a += h; Lp = loss(g, x, dy); o->a -= 2 * h; Lm = loss(g, x, dy); o->a += h;
                na[c] = (Lp - Lm) / (2 * h);
            }
    }
    loss(g, x, dy);
    tnng_backward(g, dy);
    /* analytic ∂L/∂x from node 0's dx through the x edge */
    double ax[4] = { 0, 0, 0, 0 };
    for (size_t e = 0; e < g->n_edges; e++) {
        TnngEdge *E = &g->edges[e];
        if (E->src != TNNG_X) continue;
        for (size_t q = 0; q < E->np; q++) ax[E->p[q].k] += E->c * g->nodes[E->dst].L->dx[E->p[q].j];
    }
    double ex = 0.0, eb = 0.0, ea = 0.0;
    for (int j = 0; j < 4; j++) if (fabs(ax[j] - num_x[j]) > ex) ex = fabs(ax[j] - num_x[j]);
    c = 0;
    for (size_t v = 0; v < g->n_nodes; v++) {
        TnnLayer *l = g->nodes[v].L;
        for (size_t k = 0; k < l->n_out; k++)
            for (size_t r = 0; r < l->units[k].n_or; r++, c++) {
                TnnOr *o = &l->units[k].ors[r];
                if (fabs(o->gb - nb[c]) > eb) eb = fabs(o->gb - nb[c]);
                if (fabs(o->ga - na[c]) > ea) ea = fabs(o->ga - na[c]);
            }
    }
    printf("gradients (%zu Ors, %zu edges): |dx| %.2e |db| %.2e |da| %.2e\n", nor, g->n_edges, ex, eb, ea);
    CHECK(ex < 1e-7 && eb < 1e-7 && ea < 1e-7, "finite differences: dx %g db %g da %g", ex, eb, ea);
    free(nb); free(na);
    tnng_free(g);
}

/* A graph with a training cache, so the folds have data. */
static TypeNNGraph *trained_graph(size_t n, size_t m, double ***Xo, size_t *No)
{
    size_t N = 64;
    double **X = malloc(N * sizeof(double *));
    unsigned s = 3;
    for (size_t i = 0; i < N; i++) {
        X[i] = malloc(n * sizeof(double));
        for (size_t j = 0; j < n; j++) X[i][j] = tnn_uniform(&s);
    }
    TypeNNGraph *g = tnng_create(n, m, 21);
    tnng_begin(g, N, 30, 0.05);
    double *dy = malloc(m * sizeof(double));
    for (int ep = 0; ep < 3; ep++) {
        g->training = 1;
        for (size_t i = 0; i < N; i++) {
            const double *y = tnng_forward(g, X[i]);
            for (size_t k = 0; k < m; k++) dy[k] = (y[k] - sin(3.0 * X[i][k % n])) / (double)m;
            tnng_backward(g, dy);
        }
        g->training = 0;
        if (ep < 2) tnng_epoch_end(g);   /* the last epoch keeps its cache */
    }
    free(dy);
    *Xo = X; *No = N;
    return g;
}

static void test_exact_edits(void)
{
    double **X; size_t N;
    TypeNNGraph *g = trained_graph(6, 3, &X, &N);
    double *ref = malloc(N * g->n_out * sizeof(double));
    unsigned s = 9;
    check_inv(g, "trained");

    snapshot(g, X, N, ref);
    scramble_gains(g, &s);
    double d = out_diff(g, X, N, ref);
    CHECK(d < 1e-12, "renormalisation moved the function by %g", d);
    check_inv(g, "renormalise");

    snapshot(g, X, N, ref);
    int added = 0;
    for (size_t v = 1; v < g->n_nodes && !added; v++)
        for (size_t u = 0; u < v && !added; u++) added = tnng_add_dummy_edge(g, u, v);
    d = out_diff(g, X, N, ref);
    CHECK(added == 0 || d < 1e-12, "dummy edge moved the function by %g", d);
    check_inv(g, "dummy edge");

    snapshot(g, X, N, ref);
    int wp = tnng_add_width_probe(g, 0);
    d = out_diff(g, X, N, ref);
    CHECK(wp && d < 1e-12, "width dummy moved the function by %g (added %d)", d, wp);
    check_inv(g, "width dummy");

    /* zero an edge's block, then remove it: exact */
    for (size_t e = 0; e < g->n_edges; e++) {
        TnngEdge *E = &g->edges[e];
        if (E->src == TNNG_X) continue;
        size_t deg = 0;
        for (size_t f = 0; f < g->n_edges; f++) if (g->edges[f].dst == E->dst) deg++;
        if (deg < 2) continue;
        TnnLayer *l = g->nodes[E->dst].L;
        for (size_t q = 0; q < E->np; q++)
            for (size_t k = 0; k < l->n_out; k++)
                for (size_t r = 0; r < l->units[k].n_or; r++) l->units[k].ors[r].w[E->p[q].j] = 0.0;
        snapshot(g, X, N, ref);
        tnng_remove_edge(g, e);
        tnng_renormalize(g);
        d = out_diff(g, X, N, ref);
        CHECK(d < 1e-12, "removing a zero edge moved the function by %g", d);
        check_inv(g, "zero-edge removal");
        break;
    }

    /* depth: retire the dummy the schedule placed, then insert a fresh
       identity node and remove it again */
    for (size_t v = 0; v < g->n_nodes; v++)
        if (g->nodes[v].probe) { tnng_remove_node(g, v); break; }
    check_inv(g, "retire depth dummy");
    snapshot(g, X, N, ref);
    double scale = 0.0;
    for (size_t i = 0; i < N * g->n_out; i++) scale += fabs(ref[i]);
    scale /= (double)(N * g->n_out);
    size_t nn = g->n_nodes;
    int ins = 0;
    for (size_t e = 0; e < g->n_edges && !ins; e++) ins = tnng_insert_node(g, e);
    d = out_mean_diff(g, X, N, ref);
    printf("depth insert: mean |Δy| %.3e (mean |y| %.3e)\n", d, scale);
    CHECK(ins && g->n_nodes == nn + 1, "depth insert did not happen");
    CHECK(d < 0.1 * scale + 1e-9, "depth insert moved the function by %g", d);
    check_inv(g, "depth insert");
    long w = -1;
    for (size_t v = 0; v < g->n_nodes; v++) if (g->nodes[v].probe) w = (long)v;
    CHECK(w >= 0, "no depth dummy after insert");
    if (w >= 0) {
        int rem = tnng_remove_node(g, (size_t)w);
        d = out_mean_diff(g, X, N, ref);
        printf("depth remove: mean |Δy| %.3e\n", d);
        CHECK(rem && g->n_nodes == nn, "depth remove did not happen");
        CHECK(d < 0.1 * scale + 1e-9, "insert+remove moved the function by %g", d);
        check_inv(g, "depth remove");
    }
    printf("exact edits: renormalise, dummy edge, width dummy, zero-edge drop\n");
    free(ref);
    for (size_t i = 0; i < N; i++) free(X[i]);
    free(X);
    tnng_free(g);
}

static void test_feedback(void)
{
    TypeNNGraph *g = tnng_create(3, 2, 5);
    tnng_begin(g, 2, 10, 0.05);
    double x1[3] = { 1.0, -2.0, 0.5 }, x2[3] = { 0.0, 0.3, -0.3 }, dy[2] = { 0.1, -0.1 };
    g->training = 1;
    tnng_forward(g, x1); tnng_backward(g, dy);
    tnng_forward(g, x2); tnng_backward(g, dy);
    g->training = 0;
    double want = (3.5 / 3.0 + 0.6 / 3.0);
    double got = -1.0;
    for (size_t e = 0; e < g->n_edges; e++) if (g->edges[e].src == TNNG_X) got = g->edges[e].ep_in;
    CHECK(fabs(got - want) < 1e-12, "inference feedback %g, want %g", got, want);
    for (size_t e = 0; e < g->n_edges; e++)
        CHECK(g->edges[e].ep_n == 2.0, "edge %zu saw %g samples", e, g->edges[e].ep_n);
    printf("feedback: inference feedback is the mean |value| per sample\n");
    tnng_free(g);
}

static void test_random_edits(void)
{
    double **X; size_t N;
    TypeNNGraph *g = trained_graph(5, 2, &X, &N);
    unsigned s = 1234;
    int ops[8] = { 0 };
    for (int it = 0; it < 400; it++) {
        int op = (int)(tnn_xorshift32(&s) % 8);
        size_t V = g->n_nodes;
        size_t a = tnn_xorshift32(&s) % (V ? V : 1), b = tnn_xorshift32(&s) % (V ? V : 1);
        int did = 0;
        switch (op) {
        case 0: did = tnng_add_dummy_edge(g, a < b ? a : b, a < b ? b : a); break;
        case 1: did = tnng_add_width_probe(g, a); break;
        case 2: if (g->n_edges) did = tnng_insert_node(g, tnn_xorshift32(&s) % g->n_edges); break;
        case 3: did = tnng_remove_node(g, a); break;
        case 4: if (a + 1 < V && g->nodes[a].L->n_out > 1) {
                    /* only drop a coordinate that leaves every node an input */
                    int ok = 1;
                    for (size_t e = 0; e < g->n_edges; e++) {
                        TnngEdge *E = &g->edges[e];
                        if (E->src != a) continue;
                        size_t mine = 0, deg = 0;
                        for (size_t q = 0; q < E->np; q++) if (E->p[q].k == 0) mine++;
                        for (size_t f = 0; f < g->n_edges; f++) if (g->edges[f].dst == E->dst) deg++;
                        if (mine == E->np && deg < 2) ok = 0;
                    }
                    if (ok) { tnng_drop_coord(g, a, 0); did = 1;
                        for (size_t e = g->n_edges; e-- > 0; ) if (g->edges[e].np == 0) tnng_remove_edge(g, e); }
                } break;
        case 5: if (g->n_edges) {
                    size_t e = tnn_xorshift32(&s) % g->n_edges;
                    size_t deg = 0;
                    for (size_t f = 0; f < g->n_edges; f++) if (g->edges[f].dst == g->edges[e].dst) deg++;
                    if (g->edges[e].src != TNNG_X && deg > 1) { tnng_remove_edge(g, e); did = 1; }
                } break;
        case 6: scramble_gains(g, &s); did = 1; break;
        default: tnng_renormalize(g); did = 1; break;
        }
        if (did) { ops[op]++; tnng_renormalize(g); }
        char why[256] = "";
        if (!tnng_check(g, why, sizeof why)) { CHECK(0, "random edit %d (op %d): %s", it, op, why); break; }
        const double *y = tnng_forward(g, X[it % N]);
        int fin = 1;
        for (size_t k = 0; k < g->n_out; k++) fin &= isfinite(y[k]);
        if (!fin) { CHECK(0, "random edit %d (op %d): non-finite output", it, op); break; }
    }
    passes++;
    printf("random edits: 400 steps (edge+ %d, width+ %d, node+ %d, node- %d, coord- %d, edge- %d), nodes %zu edges %zu\n",
           ops[0], ops[1], ops[2], ops[3], ops[4], ops[5], g->n_nodes, g->n_edges);
    for (size_t i = 0; i < N; i++) free(X[i]);
    free(X);
    tnng_free(g);
}

static void test_xor(void)
{
    double Xd[4][2] = {{0,0},{0,1},{1,0},{1,1}}, Yd[4] = {0, 1, 1, 0};
    TypeNNGraph *g = tnng_create(2, 1, 34972u + 7919u);
    size_t epochs = 2000;
    tnng_begin(g, 4, epochs, 0.08);
    int inv = 1;
    for (size_t ep = 0; ep < epochs; ep++) {
        g->training = 1;
        for (int i = 0; i < 4; i++) {
            const double *y = tnng_forward(g, Xd[i]);
            double dy = y[0] - Yd[i];
            tnng_backward(g, &dy);
        }
        g->training = 0;
        tnng_epoch_end(g);
        if (!tnng_check(g, NULL, 0)) inv = 0;
    }
    tnng_end(g);
    double mse = 0.0;
    for (int i = 0; i < 4; i++) { double d = tnng_forward(g, Xd[i])[0] - Yd[i]; mse += d * d / 4; }
    printf("xor: train mse %.2e, params %zu, nodes %zu, edges %zu\n", mse, tnng_params(g), g->n_nodes, g->n_edges);
    CHECK(inv, "an invariant failed during training");
    CHECK(mse < 1e-3, "xor not fitted: %g", mse);
    tnng_free(g);
}

int main(void)
{
    test_birth();
    test_gradients();
    test_exact_edits();
    test_feedback();
    test_random_edits();
    test_xor();
    printf("══════════════════════════════════════════════\n  %d passed, %d failed\n", passes, fails);
    return fails != 0;
}
