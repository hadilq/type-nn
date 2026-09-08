/*
 * test_vortex.c – unit + scenario tests for the dynamic AND-OR network.
 *
 * Exit status is the number of failed assertions (0 = all passed).
 */
#include "vortex.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static int g_fail = 0;
static int g_pass = 0;

#define EXPECT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL  %s:%d  %s\n", __FILE__, __LINE__, msg); \
        g_fail++; \
    } else { \
        g_pass++; \
    } \
} while (0)

#define EXPECT_NEAR(a, b, eps, msg) do { \
    double _va = (a), _vb = (b); \
    if (fabs(_va - _vb) > (eps)) { \
        fprintf(stderr, "  FAIL  %s:%d  %s  (got %.6f expected %.6f)\n", \
                __FILE__, __LINE__, msg, _va, _vb); \
        g_fail++; \
    } else { \
        g_pass++; \
    } \
} while (0)

static void section(const char *title)
{
    printf("\n── %s ──\n", title);
}

/* Manual product: two ORs with known weights. */
static void test_forward_known(void)
{
    section("forward product of linear units");
    Network *net = network_create(2, 1);
    network_set_dynamic(net, 0);
    /* Force deterministic weights:
         or0 = 1*x0 + 0*x1 + 0     = x0
         or1 = 0*x0 + 1*x1 + 0     = x1
         and = x0 * x1
    */
    AndNode *a = net->head->and_row;
    OrNode *o0 = a->or_row;
    OrNode *o1 = o0->right;
    EXPECT(o0 && o1, "default layer has two OR factors");
    o0->bias.value = 0;
    o1->bias.value = 0;
    WeightNode *w = o0->weight;
    while (w) {
        w->value = (w->right_index == 0) ? 1.0 : 0.0;
        w = w->right;
    }
    w = o1->weight;
    while (w) {
        w->value = (w->right_index == 1) ? 1.0 : 0.0;
        w = w->right;
    }

    double x[2] = {3.0, 4.0};
    double y[1] = {0};
    network_predict(net, x, 2, y, 1);
    EXPECT_NEAR(y[0], 12.0, 1e-9, "3 * 4 = 12");

    x[0] = 0; x[1] = 5;
    network_predict(net, x, 2, y, 1);
    EXPECT_NEAR(y[0], 0.0, 1e-9, "0 * 5 = 0");
    network_free(net);
}

static double mse_on(Network *net, double **X, double **Y, size_t n, size_t in, size_t out)
{
    double acc = 0.0;
    double *pred = (double *)calloc(out, sizeof(double));
    for (size_t i = 0; i < n; i++) {
        network_predict(net, X[i], in, pred, out);
        for (size_t k = 0; k < out; k++) {
            double d = pred[k] - Y[i][k];
            acc += d * d;
        }
    }
    free(pred);
    return acc / (double)n;
}

static void test_backward_decreases_loss(void)
{
    section("one SGD step reduces squared error on a linear-ish target");
    srand(1);
    Network *net = network_create(2, 1);
    network_set_dynamic(net, 0);
    network_set_learning_rate(net, 0.05);
    network_init_weights(net);

    double Xd[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    double Yd[4][1] = {{0},{0.3},{0.3},{0.6}};
    double *X[4], *Y[4];
    for (int i = 0; i < 4; i++) { X[i] = Xd[i]; Y[i] = Yd[i]; }

    double before = mse_on(net, X, Y, 4, 2, 1);
    network_train(net, X, Y, 4, 40);
    double after = mse_on(net, X, Y, 4, 2, 1);
    printf("  mse %.4f -> %.4f\n", before, after);
    EXPECT(after < before, "loss should drop after 40 epochs");
    network_free(net);
}

static void test_xor(void)
{
    section("XOR is representable by a product of two linear units");
    srand(7);
    Network *net = network_create(2, 1);
    network_set_dynamic(net, 0);
    network_set_learning_rate(net, 0.08);
    network_init_weights(net);

    double Xd[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    double Yd[4][1] = {{0},{1},{1},{0}};
    double *X[4], *Y[4];
    for (int i = 0; i < 4; i++) { X[i] = Xd[i]; Y[i] = Yd[i]; }

    network_train(net, X, Y, 4, 300);
    double pred[1];
    int ok = 1;
    printf("  XOR inference:\n");
    for (int i = 0; i < 4; i++) {
        network_predict(net, Xd[i], 2, pred, 1);
        double want = Yd[i][0];
        printf("    [%.0f %.0f] -> %.3f  (want %.0f)\n",
               Xd[i][0], Xd[i][1], pred[0], want);
        if (fabs(pred[0] - want) > 0.35) ok = 0;
    }
    EXPECT(ok, "XOR predictions within 0.35 of targets");
    network_free(net);
}

static void test_grow_inputs(void)
{
    section("layer input width grows when a wider vector is forwarded");
    Network *net = network_create(1, 1);
    network_set_dynamic(net, 1);
    network_init_weights(net);
    EXPECT(net->head->in_size == 1, "starts with in_size 1");
    EXPECT(weight_count(net->head->and_row->or_row->weight) == 1,
           "one weight per OR at birth");

    double x3[3] = {0.2, 0.4, 0.6};
    double y[1];
    network_predict(net, x3, 3, y, 1);

    EXPECT(net->head->in_size == 3, "in_size grew to 3");
    EXPECT(weight_count(net->head->and_row->or_row->weight) == 3,
           "weights aligned to 3 inputs");
    network_free(net);
}

static void test_grow_outputs(void)
{
    section("layer_set_outputs grows and shrinks And rows");
    Network *net = network_create(2, 1);
    EXPECT(and_count(net->head->and_row) == 1, "one output at birth");
    layer_set_outputs(net->head, 4);
    EXPECT(and_count(net->head->and_row) == 4, "grew to 4 outputs");
    EXPECT(net->head->out_size == 4, "out_size field updated");
    layer_set_outputs(net->head, 2);
    EXPECT(and_count(net->head->and_row) == 2, "shrunk to 2 outputs");
    network_free(net);
}

static void test_align_inputs_shrink(void)
{
    section("layer_align_inputs can shrink feature width");
    Network *net = network_create(4, 1);
    EXPECT(weight_count(net->head->and_row->or_row->weight) == 4, "4 weights");
    layer_align_inputs(net->head, 2);
    EXPECT(net->head->in_size == 2, "in_size is 2");
    EXPECT(weight_count(net->head->and_row->or_row->weight) == 2, "2 weights remain");
    network_free(net);
}

static void test_insert_remove_layer(void)
{
    section("explicit insert / remove of identity hidden layers");
    Network *net = network_create(3, 2);
    network_init_weights(net);
    EXPECT(network_depth(net) == 1, "starts at depth 1");

    Layer *hid = network_insert_identity(net, net->tail);
    EXPECT(hid != NULL, "insert returned a layer");
    EXPECT(network_depth(net) == 2, "depth is 2 after insert");
    EXPECT(net->head == hid, "identity sits at the head");
    EXPECT(hid->in_size == 3 && hid->out_size == 3, "identity is square 3x3");
    EXPECT(net->tail->in_size == 3, "tail still consumes 3");

    /* Identity must not change the function much: compare before/after
       on a frozen tail by predicting through both. */
    double x[3] = {0.5, -0.25, 0.75};
    double y[2];
    network_predict(net, x, 3, y, 2);
    EXPECT(isfinite(y[0]) && isfinite(y[1]), "prediction stays finite");

    int rc = network_remove_layer(net, hid);
    EXPECT(rc == 0, "remove succeeded");
    EXPECT(network_depth(net) == 1, "depth back to 1");
    EXPECT(net->head == net->tail, "single remaining layer");

    rc = network_remove_layer(net, net->tail);
    EXPECT(rc == -1, "refuses to drop the last / output layer");
    network_free(net);
}

static void test_or_growth_during_backprop(void)
{
    section("backprop can append extra OR factors when residual is large");
    srand(3);
    Network *net = network_create(2, 1);
    network_set_dynamic(net, 1);
    network_set_learning_rate(net, 0.05);
    network_init_weights(net);

    size_t before = or_count(net->head->and_row->or_row);
    double Xd[1][2] = {{0.9, 0.9}};
    double Yd[1][1] = {{1.8}};   /* far enough to stress residual */
    double *X[1] = {Xd[0]};
    double *Y[1] = {Yd[0]};
    network_train(net, X, Y, 1, 8);

    size_t after = or_count(net->head->and_row->or_row);
    printf("  OR factors %zu -> %zu\n", before, after);
    EXPECT(after >= before, "OR count did not shrink below start from a hard target");
    EXPECT(after <= net->max_or, "OR count respects max_or");
    network_free(net);
}

static void test_auto_layer_insert(void)
{
    section("large incoming gradient can insert a hidden layer");
    srand(11);
    Network *net = network_create(2, 1);
    network_set_dynamic(net, 1);
    net->max_depth = 4;
    network_set_learning_rate(net, 0.05);
    network_init_weights(net);

    size_t d0 = network_depth(net);
    /* A target far outside the initial range produces a large residual. */
    double Xd[2][2] = {{1,1},{-1,-1}};
    double Yd[2][1] = {{3},{-3}};
    double *X[2] = {Xd[0], Xd[1]};
    double *Y[2] = {Yd[0], Yd[1]};
    network_train(net, X, Y, 2, 25);

    printf("  depth %zu -> %zu\n", d0, network_depth(net));
    EXPECT(network_depth(net) >= d0, "depth never went below start");
    EXPECT(network_depth(net) <= net->max_depth, "depth respects max_depth");
    EXPECT(net->head && net->tail, "head/tail still valid");
    EXPECT(net->head->prev == NULL, "head has no prev");
    EXPECT(net->tail->next == NULL, "tail has no next");

    /* Walk the chain and check prev/next consistency + size handshake. */
    Layer *l = net->head;
    size_t walked = 0;
    while (l) {
        if (l->next) {
            EXPECT(l->next->prev == l, "next->prev points back");
            EXPECT(l->next->in_size == l->out_size,
                   "adjacent layers agree on width");
        }
        walked++;
        l = l->next;
    }
    EXPECT(walked == network_depth(net), "walked length equals depth");
    network_free(net);
}

static void test_multi_layer_stack(void)
{
    section("manually stacked 2-4-1 network trains a tiny regression");
    srand(19);
    Network *net = network_create(2, 4);
    network_add_layer(net, 4, 1);
    network_set_dynamic(net, 0);
    network_set_learning_rate(net, 0.03);
    network_init_weights(net);
    EXPECT(network_depth(net) == 2, "two layers");
    EXPECT(net->head->out_size == 4 && net->tail->in_size == 4, "4-wide hidden");

    double Xd[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    double Yd[4][1] = {{0},{0.5},{0.5},{1.0}};
    double *X[4], *Y[4];
    for (int i = 0; i < 4; i++) { X[i] = Xd[i]; Y[i] = Yd[i]; }

    double before = mse_on(net, X, Y, 4, 2, 1);
    network_train(net, X, Y, 4, 60);
    double after = mse_on(net, X, Y, 4, 2, 1);
    printf("  stacked mse %.4f -> %.4f\n", before, after);
    EXPECT(after < before + 1e-6, "stacked net does not explode loss");
    network_free(net);
}

static void test_predict_buffer_and_free(void)
{
    section("predict writes the requested number of outputs and frees cleanly");
    Network *net = network_create(3, 2);
    network_init_weights(net);
    double x[3] = {0.1, 0.2, 0.3};
    double y[2] = {999, 999};
    network_predict(net, x, 3, y, 2);
    EXPECT(isfinite(y[0]) && isfinite(y[1]), "both outputs finite");
    EXPECT(y[0] != 999 && y[1] != 999, "buffer overwritten");
    network_free(net);
}

static void test_identity_insert_preserves_output(void)
{
    section("identity hidden layer preserves the tail mapping");
    srand(23);
    Network *net = network_create(2, 1);
    network_set_dynamic(net, 0);
    network_init_weights(net);

    double x[2] = {0.3, -0.7};
    double before[1], after[1];
    network_predict(net, x, 2, before, 1);

    network_insert_identity(net, net->tail);
    EXPECT(network_depth(net) == 2, "depth 2 after identity insert");
    network_predict(net, x, 2, after, 1);
    EXPECT_NEAR(after[0], before[0], 1e-9, "identity insert is function-preserving");
    network_free(net);
}

int main(void)
{
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║          Vortex dynamic-NN test suite        ║\n");
    printf("╚══════════════════════════════════════════════╝\n");

    test_forward_known();
    test_backward_decreases_loss();
    test_xor();
    test_grow_inputs();
    test_grow_outputs();
    test_align_inputs_shrink();
    test_insert_remove_layer();
    test_or_growth_during_backprop();
    test_auto_layer_insert();
    test_multi_layer_stack();
    test_predict_buffer_and_free();
    test_identity_insert_preserves_output();

    printf("\n══════════════════════════════════════════════\n");
    printf("  %d passed, %d failed\n", g_pass, g_fail);
    printf("══════════════════════════════════════════════\n");
    return g_fail ? 1 : 0;
}
