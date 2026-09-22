/* test_cmlp.c — the board baseline. */
#include "c_mlp.h"
#include "common.h"

#include <math.h>
#include <stdio.h>

static int g_pass, g_fail;
#define CHECK(cond, ...) do { if (cond) g_pass++; else { g_fail++; \
    printf("  FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static double loss(CMlp *m, double X[][2], double Y[][1], int n)
{
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        double y = cmlp_forward(m, X[i])[0];
        s += (y - Y[i][0]) * (y - Y[i][0]);
    }
    return s / n;
}

int main(void)
{
    double X[4][2] = {{0,0},{0,1},{1,0},{1,1}}, Y[4][1] = {{0},{1},{1},{0}};

    printf("── c-mlp: shape and parameter count ──\n");
    CMlp *m = cmlp_create(2, 1, 34972);
    CHECK(cmlp_hidden(m) == 8, "H = 8 for in <= 4");
    CHECK(cmlp_params(m) == 8 * 3 + 1 * 9, "params %zu", cmlp_params(m));
    CMlp *w = cmlp_create(30, 1, 1);
    CHECK(cmlp_hidden(w) == 16 && cmlp_params(w) == 16 * 31 + 17, "H = 16 for in > 4");
    cmlp_free(w);

    printf("── c-mlp: one small step descends ──\n");
    cmlp_begin(m, 4, 1, 1e-3);
    double l0 = loss(m, X, Y, 4);
    for (int i = 0; i < 4; i++) {
        double y = cmlp_forward(m, X[i])[0];
        double dy = y - Y[i][0];
        cmlp_backward(m, &dy);
    }
    CHECK(loss(m, X, Y, 4) < l0, "loss %g -> %g", l0, loss(m, X, Y, 4));
    cmlp_free(m);

    printf("── c-mlp: XOR fits ──\n");
    m = cmlp_create(2, 1, 34972);
    cmlp_begin(m, 4, 2000, 0.08);
    for (int ep = 0; ep < 2000; ep++)
        for (int i = 0; i < 4; i++) {
            double dy = cmlp_forward(m, X[i])[0] - Y[i][0];
            cmlp_backward(m, &dy);
        }
    double l = loss(m, X, Y, 4);
    printf("    c-mlp XOR: %.3f %.3f %.3f %.3f\n", cmlp_forward(m, X[0])[0],
           cmlp_forward(m, X[1])[0], cmlp_forward(m, X[2])[0], cmlp_forward(m, X[3])[0]);
    CHECK(l < 1e-3, "xor mse %g", l);
    cmlp_free(m);

    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
