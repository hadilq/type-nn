/*
 * test_type_nn_overfit.c — the frozen type-nn-overfit, checked against its definition.
 *
 *   forward     z_k = sign(A_k) ln(1+|A_k|),  A_k = Π_r Or_{k,r}^{a_{k,r}}
 *   backward    every gradient against central finite differences
 *   surgery     probes are exact identities; the depth fold helps
 *   schedule    grow early, prune late; invariants after training
 */
#include "type_nn_overfit.h"
#include "type_nn_overfit_scale.h"
#include "common.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_pass, g_fail;
#define CHECK(cond, ...) do { if (cond) g_pass++; else { g_fail++; \
    printf("  FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

/* ── helpers ───────────────────────────────────────────────────────── */

/* A net with the birth structure but no probes: begin with lr = 0 makes
   probes exact identities, then strip them. */
static void strip_probes(TypeNNOverfit *net)
{
    for (size_t i = 0; i < net->depth; i++) {
        TnnoLayer *l = net->L[i];
        for (size_t k = 0; k < l->n_out; k++) {
            TnnoUnit *u = &l->units[k];
            for (size_t r = u->n_or; r-- > 0; )
                if (u->ors[r].probe) tnno_unit_drop_or(u, r);
        }
        for (size_t k = l->n_out; k-- > 0; )
            if (l->units[k].probe) {
                tnno_layer_drop_unit(l, k);
                tnno_layer_drop_input(net->L[i + 1], k);
            }
    }
}

static TypeNNOverfit *make_rich_net(size_t in, size_t out, unsigned seed, unsigned *rng)
{
    TypeNNOverfit *net = tnno_create(in, out, seed);
    tnno_begin(net, 10, 10, 0.0);     /* lr 0: gradients only, no steps */
    strip_probes(net);
    /* add a second and third Or to every unit, with a > 1 */
    for (size_t i = 0; i < net->depth; i++) {
        TnnoLayer *l = net->L[i];
        for (size_t k = 0; k < l->n_out; k++) {
            for (int extra = 0; extra < 2; extra++) {
                TnnoOr *o = tnno_unit_add_or(&l->units[k], l->n_in);
                tnno_or_init_random(o, l->n_in, rng);
                o->b += 0.7 * tnn_uniform(rng);
                o->a = 1.0 + 0.8 * fabs(tnn_uniform(rng));
            }
        }
    }
    return net;
}

static double loss_at(TypeNNOverfit *net, const double *x, const double *t)
{
    const double *y = tnno_forward(net, x);
    double s = 0.0;
    for (size_t k = 0; k < net->n_out; k++) s += (y[k] - t[k]) * (y[k] - t[k]);
    return s / (2.0 * (double)net->n_out);
}

static void grads_at(TypeNNOverfit *net, const double *x, const double *t)
{
    const double *y = tnno_forward(net, x);
    double dy[16];
    for (size_t k = 0; k < net->n_out; k++) dy[k] = (y[k] - t[k]) / (double)net->n_out;
    tnno_backward(net, dy);
}

static double rel_err(double a, double b)
{
    /* mixed tolerance: relative for large gradients, absolute near 0 */
    return fabs(a - b) / (fabs(a) + fabs(b) + 1e-4);
}

/* ── tests ─────────────────────────────────────────────────────────── */

static void test_birth_depth(void)
{
    printf("── birth depth = round(ln(1 + n m)) ──\n");
    CHECK(tnno_birth_depth(2, 1) == 1, "xor %zu", tnno_birth_depth(2, 1));
    CHECK(tnno_birth_depth(4, 3) == 3, "iris %zu", tnno_birth_depth(4, 3));
    CHECK(tnno_birth_depth(13, 3) == 4, "wine %zu", tnno_birth_depth(13, 3));
    CHECK(tnno_birth_depth(30, 1) == 3, "wdbc %zu", tnno_birth_depth(30, 1));
    CHECK(tnno_birth_depth(10, 1) == 2, "diabetes %zu", tnno_birth_depth(10, 1));
    CHECK(tnno_birth_depth(34, 1) == 4, "ionosphere %zu", tnno_birth_depth(34, 1));
    CHECK(tnno_birth_depth(1, 1) == 1, "never zero");
}

static void test_forward_formula(void)
{
    printf("── forward is the partition function of the typed product ──\n");
    unsigned rng = 7;
    TypeNNOverfit *net = make_rich_net(3, 2, 11, &rng);
    double x[3] = { 0.3, -1.2, 0.8 };
    const double *y = tnno_forward(net, x);
    double out[2] = { y[0], y[1] };
    /* independent re-evaluation with pow, layer by layer */
    double cur[16], nxt[16];
    memcpy(cur, x, sizeof(x));
    for (size_t i = 0; i < net->depth; i++) {
        TnnoLayer *l = net->L[i];
        for (size_t k = 0; k < l->n_out; k++) {
            double A = 1.0;
            for (size_t r = 0; r < l->units[k].n_or; r++) {
                TnnoOr *o = &l->units[k].ors[r];
                double v = o->b;
                for (size_t j = 0; j < l->n_in; j++) v += o->w[j] * cur[j];
                A *= copysign(pow(fabs(v), o->a), v);
            }
            nxt[k] = copysign(log(1.0 + fabs(A)), A);
        }
        memcpy(cur, nxt, l->n_out * sizeof(double));
    }
    CHECK(rel_err(out[0], cur[0]) < 1e-12 && rel_err(out[1], cur[1]) < 1e-12,
          "log-space forward %g %g vs pow %g %g", out[0], out[1], cur[0], cur[1]);
    CHECK(net->L[0]->n_out == 2 && net->L[1]->n_in == 2,
          "output of one layer is the input of the next");
    tnno_free(net);
}

static void test_gradients(void)
{
    printf("── backward matches central finite differences ──\n");
    unsigned rng = 99;
    for (int trial = 0; trial < 3; trial++) {
        TypeNNOverfit *net = make_rich_net(4, 3, 21 + trial, &rng);
        double x[4], t[3];
        for (int j = 0; j < 4; j++) x[j] = tnn_uniform(&rng) * 1.5;
        for (int k = 0; k < 3; k++) t[k] = tnn_uniform(&rng);
        grads_at(net, x, t);
        double worst_x = 0.0, worst_b = 0.0, worst_a = 0.0, worst_w = 0.0;
        const double h = 1e-6;
        /* ∂L/∂x through the whole stack */
        for (int j = 0; j < 4; j++) {
            double xp[4], xm[4];
            memcpy(xp, x, sizeof(x)); memcpy(xm, x, sizeof(x));
            xp[j] += h; xm[j] -= h;
            double fd = (loss_at(net, xp, t) - loss_at(net, xm, t)) / (2 * h);
            double e = rel_err(net->L[0]->dx[j], fd);
            if (e > worst_x) worst_x = e;
        }
        /* ∂L/∂b, ∂L/∂a, ∂L/∂w_0 of every Or (analytic values were recorded
           by the single backward above, before anything moved) */
        grads_at(net, x, t);
        for (size_t i = 0; i < net->depth; i++) {
            TnnoLayer *l = net->L[i];
            for (size_t k = 0; k < l->n_out; k++) {
                for (size_t r = 0; r < l->units[k].n_or; r++) {
                    TnnoOr *o = &l->units[k].ors[r];
                    double gb = o->gb, ga = o->ga;
                    double gw = gb * l->x[0];
                    double b0 = o->b, a0 = o->a, w0 = o->w[0];
                    o->b = b0 + h; double lp = loss_at(net, x, t);
                    o->b = b0 - h; double lm = loss_at(net, x, t);
                    o->b = b0;
                    double e = rel_err(gb, (lp - lm) / (2 * h));
                    if (e > worst_b) worst_b = e;
                    o->a = a0 + h; lp = loss_at(net, x, t);
                    o->a = a0 - h; lm = loss_at(net, x, t);
                    o->a = a0;
                    e = rel_err(ga, (lp - lm) / (2 * h));
                    if (e > worst_a) worst_a = e;
                    o->w[0] = w0 + h; lp = loss_at(net, x, t);
                    o->w[0] = w0 - h; lm = loss_at(net, x, t);
                    o->w[0] = w0;
                    e = rel_err(gw, (lp - lm) / (2 * h));
                    if (e > worst_w) worst_w = e;
                    grads_at(net, x, t);   /* restore caches */
                }
            }
        }
        CHECK(worst_x < 1e-5, "dL/dx rel err %g", worst_x);
        CHECK(worst_b < 1e-5, "dL/db rel err %g", worst_b);
        CHECK(worst_a < 1e-5, "dL/da rel err %g", worst_a);
        CHECK(worst_w < 1e-5, "dL/dw rel err %g", worst_w);
        tnno_free(net);
    }
}

static void test_zero_factor(void)
{
    printf("── a factor exactly at 0: cofactor slope (a = 1), 0 (a > 1) ──\n");
    unsigned rng = 5;
    TypeNNOverfit *net = make_rich_net(2, 1, 3, &rng);
    double x[2] = { 0.4, -0.9 }, t[1] = { 0.3 };
    /* put the first Or of the first unit of layer 0 exactly at 0 */
    TnnoLayer *l0 = net->L[0];
    TnnoOr *o = &l0->units[0].ors[0];
    o->a = 1.0;
    o->b = -(o->w[0] * x[0] + o->w[1] * x[1]);
    grads_at(net, x, t);
    CHECK(o->o == 0.0, "Or is exactly zero (%g)", o->o);
    const double h = 1e-7;
    double b0 = o->b;
    o->b = b0 + h; double lp = loss_at(net, x, t);
    o->b = b0 - h; double lm = loss_at(net, x, t);
    o->b = b0;
    grads_at(net, x, t);
    CHECK(rel_err(o->gb, (lp - lm) / (2 * h)) < 1e-5, "a=1 slope %g vs fd %g",
          o->gb, (lp - lm) / (2 * h));
    CHECK(isfinite(net->L[0]->dx[0]) && isfinite(net->L[0]->dx[1]), "dx finite");
    o->a = 2.0;
    grads_at(net, x, t);
    CHECK(o->gb == 0.0 && o->ga == 0.0, "a>1 at zero: gb %g ga %g", o->gb, o->ga);
    tnno_free(net);
}

static void test_probes_exact(void)
{
    printf("── probes are exact identities (Or: 0-weight column, And: ×1) ──\n");
    TypeNNOverfit *a = tnno_create(5, 2, 17);
    TypeNNOverfit *b = tnno_create(5, 2, 17);
    tnno_begin(a, 10, 10, 0.0);       /* probes, noise amplitude lr = 0 */
    tnno_begin(b, 10, 10, 0.0);
    strip_probes(b);
    size_t pa = tnno_params(a), pb = tnno_params(b);
    CHECK(pa > pb, "probes exist (%zu > %zu params)", pa, pb);
    double worst = 0.0;
    unsigned rng = 1;
    for (int s = 0; s < 20; s++) {
        double x[5];
        for (int j = 0; j < 5; j++) x[j] = 2.0 * tnn_uniform(&rng);
        const double *ya = tnno_forward(a, x);
        double y0 = ya[0], y1 = ya[1];
        const double *yb = tnno_forward(b, x);
        worst = fmax(worst, fmax(fabs(y0 - yb[0]), fabs(y1 - yb[1])));
    }
    CHECK(worst == 0.0, "outputs differ by %g", worst);
    tnno_free(a);
    tnno_free(b);
}

static void test_dev(void)
{
    printf("── distance from identity ──\n");
    TnnoLayer *l = tnno_layer_new(3, 3);
    unsigned rng = 1;
    for (size_t k = 0; k < 3; k++) {
        TnnoOr *o = tnno_unit_add_or(&l->units[k], 3);
        o->w[k] = 1.0; o->b = 0.0; o->a = 1.0;         /* carrier */
        TnnoOr *p = tnno_unit_add_or(&l->units[k], 3);
        tnno_or_init_identity(p, 3, 0.0, &rng);          /* ×1 */
        CHECK(tnno_dev_or(p, 3) == 0.0, "identity Or dev 0");
    }
    CHECK(tnno_dev_layer(l) == 0.0, "identity layer dev %g", tnno_dev_layer(l));
    l->units[1].ors[0].w[2] = 0.5;
    CHECK(tnno_dev_layer(l) > 0.0, "moved layer dev > 0");
    CHECK(tnno_dev_column(l, 0) > 0.0, "carrier column is not a dummy column");
    TnnoLayer *r = tnno_layer_new(3, 2);
    CHECK(isinf(tnno_dev_layer(r)), "non-square layer is never identity");
    tnno_layer_free(l);
    tnno_layer_free(r);
}

static void test_schedule_threshold(void)
{
    printf("── schedule: grow early, prune late; dynamic threshold ──\n");
    CHECK(tnno_phase_at(0.0) == TNNO_GROW && tnno_phase_at(0.33) == TNNO_GROW, "grow");
    CHECK(tnno_phase_at(0.34) == TNNO_FIT && tnno_phase_at(0.66) == TNNO_FIT, "fit");
    CHECK(tnno_phase_at(0.67) == TNNO_PRUNE && tnno_phase_at(1.0) == TNNO_PRUNE, "prune");
    TypeNNOverfit *net = tnno_create(2, 1, 1);
    net->lr = 0.01;
    net->n_train = 100;
    CHECK(fabs(tnno_threshold_band(net) - tnno_threshold_up(net, 100)) < 1e-15,
          "band = up at one epoch of age");
    double r = tnno_threshold_up(net, 1600) / tnno_threshold_up(net, 100);
    CHECK(fabs(r - 8.0) < 1e-12, "θ ∝ age^{3/4} (%g)", r);
    /* noise displacement lr·√T falls behind θ; drift lr·T overtakes it */
    CHECK(0.01 * sqrt(1600.0) < tnno_threshold_up(net, 1600) &&
          0.01 * 1600.0 > tnno_threshold_up(net, 1600), "θ between noise and drift");
    tnno_free(net);
}

static void test_residual_gate(void)
{
    printf("── growth stops once the residual is explained ──\n");
    TypeNNOverfit *net = tnno_create(2, 1, 1);
    net->n_train = 4;
    net->epoch_base = 0.25;             /* Var(t) of XOR */
    net->epoch_loss = 0.25 / 4 + 1e-9;
    CHECK(tnno_residual_unexplained(net), "above Var(t)/N grows");
    net->epoch_loss = 0.25 / 4 - 1e-9;
    CHECK(!tnno_residual_unexplained(net), "below Var(t)/N stops");
    tnno_free(net);
}

static void test_depth_fold(void)
{
    printf("── depth probe: the fold makes insertion closer to identity ──\n");
    /* Train briefly so the junction statistics exist, then compare the
       scaled insert against a naive insert of the same identity layer. */
    unsigned rng = 3;
    size_t N = 64;
    double X[64][3], Y[64][2];
    for (size_t i = 0; i < N; i++) {
        for (int j = 0; j < 3; j++) X[i][j] = 1.5 * tnn_uniform(&rng);
        Y[i][0] = X[i][0] * X[i][1] > 0; Y[i][1] = 1.0 - Y[i][0];
    }
    TypeNNOverfit *a = tnno_create(3, 2, 9);
    tnno_begin(a, N, 30, 0.0);
    a->training = 1;                      /* lr 0: statistics and residual only */
    for (size_t i = 0; i < N; i++) {
        const double *y = tnno_forward(a, X[i]);
        double dy[2] = { (y[0] - Y[i][0]) / 2.0, (y[1] - Y[i][1]) / 2.0 };
        tnno_backward(a, dy);
    }
    a->training = 0;
    double before[64][2];
    for (size_t i = 0; i < N; i++) {
        const double *y = tnno_forward(a, X[i]);
        before[i][0] = y[0]; before[i][1] = y[1];
    }
    /* naive copy: insert an identity layer with no fold */
    TypeNNOverfit *b = tnno_create(3, 2, 9);
    tnno_begin(b, N, 30, 0.0);
    TnnoLayer *id = tnno_layer_new(3, 3);
    for (size_t k = 0; k < 3; k++) {
        TnnoOr *o = tnno_unit_add_or(&id->units[k], 3);
        o->w[k] = 1.0;
    }
    tnno_net_insert_layer(b, 0, id);
    /* scaled insert through the grow step */
    size_t d0 = a->depth;
    tnno_scale_epoch(a);
    CHECK(a->depth == d0 + 1, "depth probe inserted (%zu → %zu)", d0, a->depth);
    double err_fold = 0.0, err_naive = 0.0;
    for (size_t i = 0; i < N; i++) {
        const double *y = tnno_forward(a, X[i]);
        err_fold += fabs(y[0] - before[i][0]) + fabs(y[1] - before[i][1]);
        y = tnno_forward(b, X[i]);
        err_naive += fabs(y[0] - before[i][0]) + fabs(y[1] - before[i][1]);
    }
    CHECK(err_fold < err_naive, "fold %g < naive %g", err_fold / N, err_naive / N);
    tnno_free(a);
    tnno_free(b);
}

static void train_loop(TypeNNOverfit *net, double **X, double **Y, size_t n, size_t epochs, double lr)
{
    size_t out = net->n_out;
    double dy[8];
    tnno_begin(net, n, epochs, lr);
    for (size_t ep = 0; ep < epochs; ep++) {
        net->training = 1;
        for (size_t s = 0; s < n; s++) {
            const double *y = tnno_forward(net, X[s]);
            for (size_t k = 0; k < out; k++) dy[k] = (y[k] - Y[s][k]) / (double)out;
            tnno_backward(net, dy);
        }
        net->training = 0;
        tnno_epoch_end(net);
    }
    tnno_end(net);
}

static void test_xor(void)
{
    printf("── XOR fits ──\n");
    double Xd[4][2] = {{0,0},{0,1},{1,0},{1,1}}, Yd[4][1] = {{0},{1},{1},{0}};
    double *X[4], *Y[4];
    for (int i = 0; i < 4; i++) { X[i] = Xd[i]; Y[i] = Yd[i]; }
    TypeNNOverfit *net = tnno_create(2, 1, 34972);
    train_loop(net, X, Y, 4, 2000, 0.08);
    double mse = 0.0;
    int ok = 0;
    for (int i = 0; i < 4; i++) {
        double y = tnno_forward(net, X[i])[0];
        mse += (y - Yd[i][0]) * (y - Yd[i][0]) / 4.0;
        ok += (y >= 0.5) == (Yd[i][0] >= 0.5);
        printf("    [%g %g] -> %.6f\n", Xd[i][0], Xd[i][1], y);
    }
    CHECK(ok == 4 && mse < 1e-3, "xor mse %g acc %d/4", mse, ok);
    tnno_free(net);
}

static void test_invariants(void)
{
    printf("── after training: no probes left, structure is well formed ──\n");
    unsigned rng = 77;
    size_t N = 80, in = 6, out = 2;
    double **X = malloc(N * sizeof(double *)), **Y = malloc(N * sizeof(double *));
    for (size_t i = 0; i < N; i++) {
        X[i] = malloc(in * sizeof(double)); Y[i] = malloc(out * sizeof(double));
        for (size_t j = 0; j < in; j++) X[i][j] = tnn_uniform(&rng);
        Y[i][0] = X[i][0] * X[i][1] + 0.3 > 0.3;
        Y[i][1] = 1.0 - Y[i][0];
    }
    TypeNNOverfit *net = tnno_create(in, out, 5);
    train_loop(net, X, Y, N, 60, 0.05);
    int probes = 0, empty_unit = 0, mismatch = 0, nonfinite = 0;
    for (size_t i = 0; i < net->depth; i++) {
        TnnoLayer *l = net->L[i];
        probes += l->probe;
        if (i + 1 < net->depth && l->n_out != net->L[i + 1]->n_in) mismatch++;
        if (l->n_out == 0) empty_unit++;
        for (size_t k = 0; k < l->n_out; k++) {
            probes += l->units[k].probe;
            if (l->units[k].n_or == 0) empty_unit++;
            for (size_t r = 0; r < l->units[k].n_or; r++) {
                TnnoOr *o = &l->units[k].ors[r];
                probes += o->probe;
                if (!isfinite(o->b) || !isfinite(o->a) || o->a < 1.0) nonfinite++;
            }
        }
    }
    CHECK(net->L[0]->n_in == in, "input width fixed");
    CHECK(net->L[net->depth - 1]->n_out == out, "task width fixed");
    CHECK(probes == 0, "%d probes left", probes);
    CHECK(empty_unit == 0, "every And keeps an Or, every layer an output");
    CHECK(mismatch == 0, "widths chain");
    CHECK(nonfinite == 0, "finite parameters, a >= 1");
    CHECK(net->phase == TNNO_DONE, "phase done");
    size_t p = 0;
    for (size_t i = 0; i < net->depth; i++)
        for (size_t k = 0; k < net->L[i]->n_out; k++)
            p += net->L[i]->units[k].n_or * (net->L[i]->n_in + 2);
    CHECK(p == tnno_params(net), "params = Σ (n_in + 2) per Or");
    printf("    depth %zu (birth %zu), or +%u/-%u, and +%u/-%u, layer +%u/-%u\n",
           net->depth, net->init_depth, net->or_add, net->or_drop,
           net->and_add, net->and_drop, net->layer_add, net->layer_drop);
    for (size_t i = 0; i < N; i++) { free(X[i]); free(Y[i]); }
    free(X); free(Y);
    tnno_free(net);
}

int main(void)
{
    test_birth_depth();
    test_forward_formula();
    test_gradients();
    test_zero_factor();
    test_probes_exact();
    test_dev();
    test_schedule_threshold();
    test_residual_gate();
    test_depth_fold();
    test_xor();
    test_invariants();
    printf("══════════════════════════════════════════════\n");
    printf("  %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
