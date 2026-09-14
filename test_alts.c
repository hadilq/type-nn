#include "type_nn_alt.h"
#include "type_nn_cmlp.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int g_fail, g_pass;

#define EXPECT(c, m) do { \
    if (!(c)) { fprintf(stderr, "  FAIL %s: %s\n", name, m); g_fail++; } \
    else g_pass++; \
} while (0)

int main(void)
{
    const char *name = "c-mlp";
    printf("c-mlp baseline — XOR smoke\n");
    double Xd[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    double Yd[4][1] = {{0},{1},{1},{0}};
    double *X[4], *Y[4];
    for (int i = 0; i < 4; i++) { X[i] = Xd[i]; Y[i] = Yd[i]; }

    srand(7);
    AltNet a = type_nn_cmlp_open(2, 1);
    EXPECT(a.depth(a.ctx) == 2, "c-mlp is depth 2 at birth");
    a.init(a.ctx);
    type_nn_alt_train(&a, X, Y, 4, 400, 0.08);
    int ok = 1;
    printf("  c-mlp XOR:");
    for (int i = 0; i < 4; i++) {
        double y = 0;
        a.forward(a.ctx, Xd[i], &y);
        printf(" %.2f", y);
        if (fabs(y - Yd[i][0]) > 0.35) ok = 0;
    }
    printf("\n");
    EXPECT(ok, "XOR");
    EXPECT(a.param_count(a.ctx) == 33, "2-8-1 param count");
    a.free(a.ctx);
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
