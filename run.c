/*
 * run.c – Demonstrations of the linked-node AND-OR network.
 *
 * Demo 1: XOR
 * Demo 2: sin(x) regression
 * Demo 3: inspect the linked structure + dynamic scaling
 */
#include "type_nn.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static void demo_xor(void)
{
    printf("\n╔══════════════════════════════════╗\n");
    printf("║          DEMO 1 – XOR            ║\n");
    printf("╚══════════════════════════════════╝\n\n");

    srand(34972);
    Network *net = network_create(2, 1);
    network_set_dynamic(net, 1);
    network_set_learning_rate(net, 0.08);
    network_init_weights(net);

    double X_data[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    double Y_data[4][1] = {{0},  {1},  {1},  {0}};
    double *X[4], *Y[4];
    for (int i = 0; i < 4; i++) { X[i] = X_data[i]; Y[i] = Y_data[i]; }

    printf("Training XOR (dynamic AND-OR net) for 1500 epochs\n\n");
    network_train(net, X, Y, 4, 1500);

    printf("\nInference after training (depth = %zu):\n", network_depth(net));
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

static void demo_sin(void)
{
    printf("\n╔══════════════════════════════════╗\n");
    printf("║       DEMO 2 – sin(x) fit        ║\n");
    printf("╚══════════════════════════════════╝\n\n");

    srand(34972);
    Network *net = network_create(1, 1);
    network_add_layer(net, 1, 1);
    network_set_dynamic(net, 1);
    network_set_learning_rate(net, 0.02);
    network_init_weights(net);

    const size_t N = 40;
    double **X = (double **)malloc(N * sizeof(double *));
    double **Y = (double **)malloc(N * sizeof(double *));
    for (size_t i = 0; i < N; i++) {
        X[i] = (double *)malloc(sizeof(double));
        Y[i] = (double *)malloc(sizeof(double));
        double t = -M_PI + 2.0 * M_PI * i / (N - 1);
        X[i][0] = t / M_PI;
        Y[i][0] = sin(t);
    }

    printf("Training dynamic net on sin(x), %zu samples\n\n", N);
    network_train(net, X, Y, N, 400);

    printf("\nInference comparison (every 8th sample), depth = %zu:\n",
           network_depth(net));
    printf("  x/π       sin(x)     predicted   error\n");
    printf("  ──────────────────────────────────────────\n");
    double pred[1];
    for (size_t i = 0; i < N; i += 8) {
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

static void demo_nodes(void)
{
    printf("\n╔══════════════════════════════════════════════╗\n");
    printf("║   DEMO 3 – Inspect + scale linked structure  ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");

    Network *net = network_create(2, 2);
    network_init_weights(net);

    printf("Initial 2→2 layer:\n");
    and_print(net->head->and_row, "W0");

    printf("\nGrow outputs 2 → 3:\n");
    layer_set_outputs(net->head, 3);
    printf("  and_count = %zu\n", and_count(net->head->and_row));

    printf("\nGrow inputs 2 → 4:\n");
    layer_align_inputs(net->head, 4);
    printf("  in_size = %zu, weights on first OR = %zu\n",
           net->head->in_size,
           weight_count(net->head->and_row->or_row->weight));

    printf("\nInsert identity hidden layer:\n");
    network_insert_identity(net, net->head);
    printf("  depth = %zu\n", network_depth(net));

    printf("\nRemove the hidden layer:\n");
    network_remove_layer(net, net->head);
    printf("  depth = %zu\n", network_depth(net));

    network_free(net);
}

int main(void)
{
    demo_xor();
    demo_sin();
    demo_nodes();
    return 0;
}
