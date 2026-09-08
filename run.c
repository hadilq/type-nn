/*
 * main.c – Demonstrates the linked-Node neural network.
 *
 * Demo 1: XOR problem  (2 → 4 → 1)
 * Demo 2: sin(x) regression (1 → 16 → 16 → 1) over [-π, π]
 */

#include "vortex.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ────────────────────────────────────────────
   Demo 1: XOR
   ──────────────────────────────────────────── */
static void demo_xor(void)
{
    printf("\n╔══════════════════════════════════╗\n");
    printf("║          DEMO 1 – XOR            ║\n");
    printf("╚══════════════════════════════════╝\n\n");

    /* Network: 2 → 4 (ReLU) → 1 (Sigmoid) */
    Network *net = network_create(2, 1);
    network_init_weights(net);

    /* XOR data */
    double X_data[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    double Y_data[4][1] = {{0},  {1},  {1},  {0}};

    double *X[4], *Y[4];
    for (int i = 0; i < 4; i++) { X[i] = X_data[i]; Y[i] = Y_data[i]; }

    printf("Training XOR for 5000 epochs\n\n");
    network_train(net, X, Y, 4, 5000);

    printf("\nInference after training:\n");
    printf("  Input       Expected   Predicted\n");
    printf("  ─────────────────────────────────\n");
    double pred[1];
    for (int i = 0; i < 4; i++) {
        network_predict(net, X_data[i], 2, pred, 1);
        printf("  [%.0f, %.0f]  →  %.0f       →  %.4f\n",
               X_data[i][0], X_data[i][1], Y_data[i][0], pred[0]);
    }

    network_free(net);
}

/* ────────────────────────────────────────────
   Demo 2: sin(x) regression
   ──────────────────────────────────────────── */
static void demo_sin(void)
{
    printf("\n╔══════════════════════════════════╗\n");
    printf("║       DEMO 2 – sin(x) fit        ║\n");
    printf("╚══════════════════════════════════╝\n\n");

    /* Network: 1 → 16 (ReLU) → 16 (ReLU) → 1 (Linear) */
    Network *net = network_create(1, 1);
    network_init_weights(net);

    const size_t N = 100;
    double **X = (double **)malloc(N * sizeof(double *));
    double **Y = (double **)malloc(N * sizeof(double *));
    for (size_t i = 0; i < N; i++) {
        X[i] = (double *)malloc(sizeof(double));
        Y[i] = (double *)malloc(sizeof(double));
        double t = -M_PI + 2.0 * M_PI * i / (N - 1);
        X[i][0] = t / M_PI;          /* normalise input to [-1,1] */
        Y[i][0] = sin(t);
    }

    printf("Training 1→16(ReLU)→16(ReLU)→1(Linear) on sin(x)\n\n");
    network_train(net, X, Y, N, 3000);

    printf("\nInference comparison (every 10th sample):\n");
    printf("  x/π       sin(x)     predicted   error\n");
    printf("  ──────────────────────────────────────────\n");
    double pred[1];
    for (size_t i = 0; i < N; i += 10) {
        double t = -M_PI + 2.0 * M_PI * i / (N - 1);
        network_predict(net, X[i], 1, pred, 1);
        double err = fabs(sin(t) - pred[0]);
        printf("  %+.4f    %+.4f     %+.4f      %.4f\n",
               X[i][0], sin(t), pred[0], err);
    }

    for (size_t i = 0; i < N; i++) { free(X[i]); free(Y[i]); }
    free(X); free(Y);
    network_free(net);
}

/* ────────────────────────────────────────────
   Demo 3: print internal Node structure
   ──────────────────────────────────────────── */
static void demo_nodes(void)
{
    printf("\n╔══════════════════════════════════════════════╗\n");
    printf("║   DEMO 3 – Inspect Linked Node structure     ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");

    Network *net = network_create(2, 2);
    network_init_weights(net);

    printf("Layer 0 weights W[3×2] (traversed via Node links):\n");
    Layer *l0 = net->head;
    and_print(l0->and_row, "W0");

    printf("\nLayer 1 weights W[1×3]:\n");
    and_print(l0->next->and_row, "W1");

    network_free(net);
}

/* ────────────────────────────────────────────
   main
   ──────────────────────────────────────────── */
int main(void)
{
    demo_xor();
    demo_sin();
    demo_nodes();
    return 0;
}


