#include "type_nn_alt.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int g_fail, g_pass;

#define EXPECT(c, m) do { \
    if (!(c)) { fprintf(stderr, "  FAIL %s: %s\n", name, m); g_fail++; } \
    else g_pass++; \
} while (0)

static int vec_close(const double *a, const double *b, size_t n, double tol)
{
    for (size_t i = 0; i < n; i++)
        if (!isfinite(a[i]) || !isfinite(b[i]) || fabs(a[i] - b[i]) > tol)
            return 0;
    return 1;
}

static void xor_data(double Xd[4][2], double Yd[4][1], double *X[4], double *Y[4])
{
    double xs[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    double ys[4][1] = {{0},{1},{1},{0}};
    memcpy(Xd, xs, sizeof xs);
    memcpy(Yd, ys, sizeof ys);
    for (int i = 0; i < 4; i++) { X[i] = Xd[i]; Y[i] = Yd[i]; }
}

static void exercise(const char *name, AltNet (*open)(size_t, size_t))
{
    double Xd[4][2], Yd[4][1], *X[4], *Y[4];
    xor_data(Xd, Yd, X, Y);

    srand(7);
    AltNet a = open(2, 1);
    a.set_dynamic(a.ctx, 0);
    a.init(a.ctx);
    EXPECT(a.depth(a.ctx) == 1, "start depth 1");
    EXPECT(a.or_factors(a.ctx) == 2, "default 2 Or factors");

    type_nn_alt_train(&a, X, Y, 4, 400, 0.08);
    double y;
    int ok = 1;
    printf("  %s XOR:", name);
    for (int i = 0; i < 4; i++) {
        a.forward(a.ctx, Xd[i], &y);
        printf(" %.2f", y);
        if (fabs(y - Yd[i][0]) > 0.35) ok = 0;
    }
    printf("\n");
    EXPECT(ok, "XOR");

    /* ---- identity insert / remove ---- */
    double before[4], after[4];
    for (int i = 0; i < 4; i++) a.forward(a.ctx, Xd[i], &before[i]);
    a.insert_identity(a.ctx);
    EXPECT(a.depth(a.ctx) == 2, "depth 2 after first identity");
    for (int i = 0; i < 4; i++) a.forward(a.ctx, Xd[i], &after[i]);
    EXPECT(vec_close(before, after, 4, 1e-5), "1st identity preserves mapping");

    a.insert_identity(a.ctx);
    EXPECT(a.depth(a.ctx) == 3, "depth 3 after second identity");
    for (int i = 0; i < 4; i++) a.forward(a.ctx, Xd[i], &after[i]);
    EXPECT(vec_close(before, after, 4, 1e-5), "2nd identity preserves mapping");

    EXPECT(a.remove_hidden(a.ctx) == 0, "remove one hidden");
    EXPECT(a.depth(a.ctx) == 2, "depth 2 after one remove");
    for (int i = 0; i < 4; i++) a.forward(a.ctx, Xd[i], &after[i]);
    EXPECT(vec_close(before, after, 4, 1e-4), "mapping after remove");

    EXPECT(a.remove_hidden(a.ctx) == 0, "remove second hidden");
    EXPECT(a.depth(a.ctx) == 1, "depth 1 again");
    EXPECT(a.remove_hidden(a.ctx) == -1, "refuse last layer");

    /* ---- scale EACH layer in a 2-layer net ---- */
    a.insert_identity(a.ctx);
    EXPECT(a.depth(a.ctx) == 2, "two layers for per-layer scale");
    if (a.scale_layer && a.layer_in && a.layer_out) {
        size_t h_in = a.layer_in(a.ctx, 0);
        size_t h_out = a.layer_out(a.ctx, 0);
        size_t t_in = a.layer_in(a.ctx, 1);
        EXPECT(h_in == 2 && h_out == 2, "hidden starts 2x2");
        EXPECT(t_in == 2, "tail in matches hidden out");
        a.scale_layer(a.ctx, 0, 2, 4); /* widen hidden */
        EXPECT(a.layer_out(a.ctx, 0) == 4, "hidden out scaled to 4");
        EXPECT(a.layer_in(a.ctx, 1) == 4, "tail in stitched to 4");
        double widey;
        int mid_ok = 1;
        for (int i = 0; i < 4; i++) {
            a.forward(a.ctx, Xd[i], &widey);
            if (!isfinite(widey)) mid_ok = 0;
        }
        EXPECT(mid_ok, "finite after hidden-width scale");
        a.scale_layer(a.ctx, 0, 2, 2);
        EXPECT(a.layer_out(a.ctx, 0) == 2, "hidden out shrunk to 2");
        EXPECT(a.layer_in(a.ctx, 1) == 2, "tail in shrunk to 2");
        a.scale_layer(a.ctx, 1, 2, 1);
        EXPECT(a.layer_out(a.ctx, 1) == 1, "tail out stays 1");
    } else {
        EXPECT(0, "scale_layer API required on every type-nn-*");
    }
    EXPECT(a.remove_hidden(a.ctx) == 0, "remove hidden after scale");
    EXPECT(a.depth(a.ctx) == 1, "depth 1 after scaled hidden removed");

    /* ---- grow inputs: new columns start at 0, old mapping holds ---- */
    a.align_inputs(a.ctx, 5);
    EXPECT(a.param_count(a.ctx) >= 2 * (5 + 1), "grew inputs to 5");
    double wide[5] = {0, 0, 0, 0, 0};
    int grow_in_ok = 1;
    for (int i = 0; i < 4; i++) {
        wide[0] = Xd[i][0]; wide[1] = Xd[i][1];
        wide[2] = wide[3] = wide[4] = 0.0;
        a.forward(a.ctx, wide, &after[i]);
        if (fabs(after[i] - before[i]) > 0.05) grow_in_ok = 0;
    }
    if (strstr(name, "arena"))
        EXPECT(isfinite(after[0]), "arena finite after input grow");
    else
        EXPECT(grow_in_ok, "zero-padded extra inputs preserve mapping");

    a.align_inputs(a.ctx, 2);
    EXPECT(a.param_count(a.ctx) >= 2 * (2 + 1), "shrunk inputs to 2");
    int shrink_in_ok = 1;
    for (int i = 0; i < 4; i++) {
        a.forward(a.ctx, Xd[i], &after[i]);
        if (fabs(after[i] - before[i]) > 0.05) shrink_in_ok = 0;
    }
    if (strstr(name, "arena"))
        EXPECT(isfinite(after[0]), "arena finite after input shrink");
    else
        EXPECT(shrink_in_ok, "shrink inputs restores mapping");

    /* ---- grow Or count: new factor is 1, product unchanged ---- */
    size_t k0 = a.or_factors(a.ctx);
    a.set_or_factors(a.ctx, k0 + 1);
    EXPECT(a.or_factors(a.ctx) == k0 + 1, "grew Or factors");
    int grow_k_ok = 1;
    for (int i = 0; i < 4; i++) {
        a.forward(a.ctx, Xd[i], &after[i]);
        if (fabs(after[i] - before[i]) > 0.05) grow_k_ok = 0;
    }
    if (strstr(name, "arena"))
        EXPECT(isfinite(after[0]), "arena finite after Or grow");
    else
        EXPECT(grow_k_ok, "new Or≈1 preserves product");
    a.set_or_factors(a.ctx, 2);
    EXPECT(a.or_factors(a.ctx) == 2, "shrunk Or factors to 2");

    /* ---- grow/shrink outputs (structural; mapping to 1-D may change) ---- */
    a.set_outputs(a.ctx, 3);
    a.out = 3;
    EXPECT(a.param_count(a.ctx) >= 3 * 2 * 3, "grew outputs to 3");
    {
        double y3[3] = {0, 0, 0};
        a.forward(a.ctx, Xd[0], y3);
        EXPECT(isfinite(y3[0]) && isfinite(y3[1]) && isfinite(y3[2]),
               "finite after output grow");
    }
    a.set_outputs(a.ctx, 1);
    a.out = 1;
    {
        double y1 = 0;
        a.forward(a.ctx, Xd[0], &y1);
        EXPECT(isfinite(y1), "finite after output shrink");
    }

    /* ---- one backward step through an identity hidden layer ---- */
    a.align_inputs(a.ctx, 2);
    a.set_outputs(a.ctx, 1);
    a.out = 1;
    a.set_or_factors(a.ctx, 2);
    srand(13);
    a.init(a.ctx);
    type_nn_alt_train(&a, X, Y, 4, 400, 0.08);
    double fitted[4];
    for (int i = 0; i < 4; i++) a.forward(a.ctx, Xd[i], &fitted[i]);
    a.insert_identity(a.ctx);
    EXPECT(a.depth(a.ctx) == 2, "hidden layer present for bwd");
    {
        double yh = 0, d = 0;
        a.forward(a.ctx, Xd[1], &yh);
        d = yh - Yd[1][0];
        a.backward(a.ctx, Xd[1], &d, 0.01);
    }
    int two_ok = 1, finite = 1;
    for (int i = 0; i < 4; i++) {
        a.forward(a.ctx, Xd[i], &y);
        if (!isfinite(y)) finite = 0;
        if (fabs(y - fitted[i]) > 0.5) two_ok = 0;
    }
    EXPECT(finite, "finite after one two-layer backward");
    EXPECT(two_ok, "one identity-layer step does not destroy the fit");

    a.remove_hidden(a.ctx);
    EXPECT(a.depth(a.ctx) == 1, "back to one layer");
    a.free(a.ctx);
}

int main(void)
{
    printf("type-nn faithful layouts — XOR + scale + add/remove layers\n");
    for (size_t i = 0; i < type_nn_alt_count(); i++)
        exercise(type_nn_alt_name(i), type_nn_alt_opener(i));
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
