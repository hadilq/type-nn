#include "type_nn_alt.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static int g_fail, g_pass;

#define EXPECT(c, m) do { \
    if (!(c)) { fprintf(stderr, "  FAIL %s:%s\n", name, m); g_fail++; } \
    else g_pass++; \
} while (0)

static void exercise(const char *name, AltNet (*open)(size_t, size_t))
{
    double Xd[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    double Yd[4][1] = {{0},{1},{1},{0}};
    double *X[4], *Y[4];
    for (int i = 0; i < 4; i++) { X[i] = Xd[i]; Y[i] = Yd[i]; }

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

    double before[4], after[4];
    for (int i = 0; i < 4; i++) a.forward(a.ctx, Xd[i], &before[i]);
    a.insert_identity(a.ctx);
    EXPECT(a.depth(a.ctx) == 2, "depth 2 after identity");
    int same = 1;
    for (int i = 0; i < 4; i++) {
        a.forward(a.ctx, Xd[i], &after[i]);
        if (fabs(after[i] - before[i]) > 1e-5) same = 0;
    }
    EXPECT(same, "identity insert preserves XOR mapping");
    EXPECT(a.remove_hidden(a.ctx) == 0, "remove hidden");
    EXPECT(a.depth(a.ctx) == 1, "depth back to 1");
    EXPECT(a.remove_hidden(a.ctx) == -1, "refuse last layer");

    a.align_inputs(a.ctx, 5);
    EXPECT(a.param_count(a.ctx) >= 2 * (5 + 1), "grew inputs");
    a.set_outputs(a.ctx, 3);
    a.out = 3;
    EXPECT(a.param_count(a.ctx) >= 3 * 2 * 6 - 20, "grew outputs");
    a.set_or_factors(a.ctx, 3);
    EXPECT(a.or_factors(a.ctx) == 3, "3 Or factors");
    a.set_or_factors(a.ctx, 2);
    EXPECT(a.or_factors(a.ctx) == 2, "shrunk to 2 Ors");
    a.align_inputs(a.ctx, 2);
    a.set_outputs(a.ctx, 1);
    a.out = 1;

    double z[1] = {0};
    double xx[2] = {0.2, 0.3};
    a.forward(a.ctx, xx, z);
    EXPECT(isfinite(z[0]), "finite after reshape");
    a.free(a.ctx);
}

int main(void)
{
    printf("type-nn alternate layouts — XOR + scale + layer ops\n");
    for (size_t i = 0; i < type_nn_alt_count(); i++)
        exercise(type_nn_alt_name(i), type_nn_alt_opener(i));
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
