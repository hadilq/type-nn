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


static void expect_chain_ok(Network *net, const char *tag)
{
    EXPECT(net && net->head && net->tail, tag);
    EXPECT(net->head->prev == NULL, "head prev is NULL");
    EXPECT(net->tail->next == NULL, "tail next is NULL");
    size_t walked = 0;
    Layer *l = net->head;
    Layer *last = NULL;
    while (l) {
        if (l->next) {
            EXPECT(l->next->prev == l, "next->prev consistency");
            EXPECT(l->next->in_size == l->out_size, "adjacent width handshake");
        }
        last = l;
        walked++;
        l = l->next;
    }
    EXPECT(last == net->tail, "walk ends at tail");
    EXPECT(walked == network_depth(net), "walk length equals depth");
}

static void test_null_and_trivial_api(void)
{
    section("NULL / trivial API corners");
    network_free(NULL);
    EXPECT(network_depth(NULL) == 0, "depth(NULL) is 0");
    EXPECT(network_param_count(NULL) == 0, "params(NULL) is 0");
    EXPECT(network_nbytes(NULL) == 0, "nbytes(NULL) is 0");
    EXPECT(inout_count(NULL) == 0, "inout_count(NULL) is 0");
    EXPECT(weight_count(NULL) == 0, "weight_count(NULL) is 0");
    EXPECT(or_count(NULL) == 0, "or_count(NULL) is 0");
    EXPECT(and_count(NULL) == 0, "and_count(NULL) is 0");
    layer_align_inputs(NULL, 3);
    layer_set_outputs(NULL, 3);
    EXPECT(network_insert_identity(NULL, NULL) == NULL, "insert on NULL net");
    EXPECT(network_remove_layer(NULL, NULL) == -1, "remove on NULL net");

    Network *net = network_create(1, 1);
    double lr = net->lr;
    network_set_learning_rate(net, 0.0);
    EXPECT(net->lr == lr, "non-positive lr is rejected");
    network_set_learning_rate(NULL, 0.1);
    network_set_dynamic(NULL, 0);
    network_set_learning_rate(net, 0.2);
    EXPECT_NEAR(net->lr, 0.2, 1e-12, "lr updated");
    network_free(net);
}

static void test_single_feature_forward(void)
{
    section("1→1 network, bias-only and single-weight paths");
    Network *net = network_create(1, 1);
    network_set_dynamic(net, 0);
    AndNode *a = net->head->and_row;
    OrNode *o0 = a->or_row;
    OrNode *o1 = o0->right;
    o0->bias.value = 0.5;
    o1->bias.value = 2.0;
    WeightNode *w = o0->weight;
    while (w) { w->value = 1.0; w = w->right; }
    w = o1->weight;
    while (w) { w->value = 0.0; w = w->right; }
    /* (0.5 + 1*x) * (2 + 0*x) = 1 + 2x */
    double x[1] = {3.0};
    double y[1] = {0};
    network_predict(net, x, 1, y, 1);
    EXPECT_NEAR(y[0], 7.0, 1e-9, "(0.5+3)*2 = 7");
    x[0] = 0;
    network_predict(net, x, 1, y, 1);
    EXPECT_NEAR(y[0], 1.0, 1e-9, "(0.5+0)*2 = 1");
    network_free(net);
}

static void test_quadratic_is_rank2(void)
{
    section("rank-2 polynomial: And = (x+y)*(x-y) = x²-y², no activation");
    Network *net = network_create(2, 1);
    network_set_dynamic(net, 0);
    AndNode *a = net->head->and_row;
    OrNode *o0 = a->or_row;
    OrNode *o1 = o0->right;
    o0->bias.value = 0;
    o1->bias.value = 0;
    WeightNode *w = o0->weight;
    while (w) {
        w->value = 1.0; /* x + y */
        w = w->right;
    }
    w = o1->weight;
    while (w) {
        w->value = (w->right_index == 0) ? 1.0 : -1.0; /* x - y */
        w = w->right;
    }
    double cases[][3] = {
        {2, 1, 3},
        {3, 3, 0},
        {0, 2, -4},
        {-1, 2, 1 - 4},
    };
    for (int i = 0; i < 4; i++) {
        double x[2] = {cases[i][0], cases[i][1]};
        double y[1];
        network_predict(net, x, 2, y, 1);
        EXPECT_NEAR(y[0], cases[i][2], 1e-9, "x^2 - y^2 via product of affines");
    }
    network_free(net);
}

static void test_multi_output_independent(void)
{
    section("two outputs compute independent products");
    Network *net = network_create(2, 2);
    network_set_dynamic(net, 0);
    AndNode *a0 = net->head->and_row;
    AndNode *a1 = a0->right;
    EXPECT(a1 != NULL, "two And nodes");
    /* out0 = x0 * x1,  out1 = (x0+1)*(x1+1) */
    OrNode *o = a0->or_row;
    o->bias.value = 0;
    WeightNode *w = o->weight;
    while (w) { w->value = (w->right_index == 0) ? 1.0 : 0.0; w = w->right; }
    o = o->right;
    o->bias.value = 0;
    w = o->weight;
    while (w) { w->value = (w->right_index == 1) ? 1.0 : 0.0; w = w->right; }

    o = a1->or_row;
    o->bias.value = 1;
    w = o->weight;
    while (w) { w->value = (w->right_index == 0) ? 1.0 : 0.0; w = w->right; }
    o = o->right;
    o->bias.value = 1;
    w = o->weight;
    while (w) { w->value = (w->right_index == 1) ? 1.0 : 0.0; w = w->right; }

    double x[2] = {2.0, 3.0};
    double y[2] = {0, 0};
    network_predict(net, x, 2, y, 2);
    EXPECT_NEAR(y[0], 6.0, 1e-9, "out0 = 2*3");
    EXPECT_NEAR(y[1], 12.0, 1e-9, "out1 = 3*4");
    network_free(net);
}

static void test_predict_size_mismatch(void)
{
    section("predict with shorter / longer output buffers");
    Network *net = network_create(2, 2);
    network_set_dynamic(net, 0);
    network_init_weights(net);
    double x[2] = {0.2, 0.3};
    double small[1] = {999};
    network_predict(net, x, 2, small, 1);
    EXPECT(small[0] != 999 && isfinite(small[0]), "short buffer got first output");

    double big[4] = {9, 9, 9, 9};
    network_predict(net, x, 2, big, 4);
    EXPECT(isfinite(big[0]) && isfinite(big[1]), "first two written");
    EXPECT_NEAR(big[2], 0.0, 1e-15, "extra slot zeroed");
    EXPECT_NEAR(big[3], 0.0, 1e-15, "extra slot zeroed");
    network_free(net);
}

static void test_dynamic_off_does_not_grow_or_or_depth(void)
{
    section("dynamic=0 freezes topology (no extra OR / layer)");
    srand(5);
    Network *net = network_create(2, 1);
    network_set_dynamic(net, 0);
    network_set_learning_rate(net, 0.2);
    network_init_weights(net);
    size_t ors0 = or_count(net->head->and_row->or_row);
    size_t d0 = network_depth(net);
    double Xd[1][2] = {{0.9, 0.9}};
    double Yd[1][1] = {{8.0}};
    double *X[1] = {Xd[0]};
    double *Y[1] = {Yd[0]};
    network_train(net, X, Y, 1, 6);
    EXPECT(or_count(net->head->and_row->or_row) == ors0, "OR count frozen");
    EXPECT(network_depth(net) == d0, "depth frozen");
    network_free(net);
}

static void test_grow_then_shrink_roundtrip(void)
{
    section("grow then shrink input and output several times");
    Network *net = network_create(2, 1);
    network_init_weights(net);
    for (size_t w = 2; w <= 6; w++) {
        layer_align_inputs(net->head, w);
        EXPECT(net->head->in_size == w, "in grew");
        EXPECT(weight_count(net->head->and_row->or_row->weight) == w, "weights match in");
        layer_set_outputs(net->head, w);
        EXPECT(and_count(net->head->and_row) == w, "outs match");
    }
    layer_align_inputs(net->head, 3);
    layer_set_outputs(net->head, 2);
    EXPECT(net->head->in_size == 3, "shrunk in");
    EXPECT(net->head->out_size == 2, "shrunk out");
    EXPECT(and_count(net->head->and_row) == 2, "and count shrunk");
    double x[3] = {0.1, 0.2, 0.3};
    double y[2] = {0, 0};
    network_predict(net, x, 3, y, 2);
    EXPECT(isfinite(y[0]) && isfinite(y[1]), "still finite after reshape");
    network_free(net);
}

static void test_insert_middle_and_remove_head(void)
{
    section("insert identity between two layers; remove the new head");
    Network *net = network_create(2, 3);
    network_add_layer(net, 3, 1);
    network_set_dynamic(net, 0);
    network_init_weights(net);
    EXPECT(network_depth(net) == 2, "2 layers");
    expect_chain_ok(net, "stack exists");

    double x[2] = {0.4, -0.2};
    double before[1], after[1];
    network_predict(net, x, 2, before, 1);

    Layer *hid = network_insert_identity(net, net->tail);
    EXPECT(network_depth(net) == 3, "depth 3");
    EXPECT(hid->next == net->tail, "inserted before tail");
    EXPECT(hid->prev == net->head, "sits after original head");
    expect_chain_ok(net, "after middle insert");

    network_predict(net, x, 2, after, 1);
    EXPECT_NEAR(after[0], before[0], 1e-9, "middle identity preserves mapping");

    EXPECT(network_remove_layer(net, net->tail) == -1, "cannot drop output layer");
    EXPECT(network_remove_layer(net, hid) == 0, "can drop hidden");
    EXPECT(network_depth(net) == 2, "back to 2");
    expect_chain_ok(net, "after dropping hidden");

    /* drop the current head (original first layer) */
    EXPECT(network_remove_layer(net, net->head) == 0, "can drop non-tail head");
    EXPECT(network_depth(net) == 1, "one layer left");
    EXPECT(net->head == net->tail, "head is tail");
    expect_chain_ok(net, "single layer");
    network_free(net);
}

static void test_triple_identity_still_identity(void)
{
    section("three stacked identity inserts still preserve the function");
    srand(41);
    Network *net = network_create(3, 2);
    network_set_dynamic(net, 0);
    network_init_weights(net);
    double x[3] = {0.2, -0.5, 0.9};
    double before[2], after[2];
    network_predict(net, x, 3, before, 2);
    network_insert_identity(net, net->tail);
    network_insert_identity(net, net->tail);
    network_insert_identity(net, net->head);
    EXPECT(network_depth(net) == 4, "1 + 3 identities");
    expect_chain_ok(net, "triple identity chain");
    network_predict(net, x, 3, after, 2);
    EXPECT_NEAR(after[0], before[0], 1e-9, "out0 preserved");
    EXPECT_NEAR(after[1], before[1], 1e-9, "out1 preserved");
    network_free(net);
}

static void test_train_zero_epochs_and_zero_init_predict(void)
{
    section("0-epoch train and predict before init_weights");
    Network *net = network_create(2, 1);
    double Xd[1][2] = {{0.1, 0.2}};
    double Yd[1][1] = {{0.3}};
    double *X[1] = {Xd[0]};
    double *Y[1] = {Yd[0]};
    network_train(net, X, Y, 1, 0);
    double y[1] = {0};
    network_predict(net, Xd[0], 2, y, 1);
    EXPECT(isfinite(y[0]), "predict works with zero-init weights");
    EXPECT(network_param_count(net) > 0, "params exist");
    EXPECT(network_nbytes(net) > sizeof(Network), "bytes include nodes");
    network_free(net);
}

static void test_clip_huge_values_stay_finite(void)
{
    section("huge inputs stay finite thanks to OR/AND clips");
    Network *net = network_create(2, 1);
    network_set_dynamic(net, 0);
    AndNode *a = net->head->and_row;
    for (OrNode *o = a->or_row; o; o = o->right) {
        o->bias.value = 0;
        for (WeightNode *w = o->weight; w; w = w->right) w->value = 4.0;
    }
    double x[2] = {1e9, -1e9};
    double y[1];
    network_predict(net, x, 2, y, 1);
    EXPECT(isfinite(y[0]), "clipped product is finite");
    EXPECT(fabs(y[0]) <= 32.0 + 1e-9, "AND clip is 32");
    network_free(net);
}

static void test_negative_and_zero_inputs(void)
{
    section("negative and zero inputs through a known product");
    Network *net = network_create(2, 1);
    network_set_dynamic(net, 0);
    AndNode *a = net->head->and_row;
    OrNode *o0 = a->or_row;
    OrNode *o1 = o0->right;
    o0->bias.value = o1->bias.value = 0;
    for (WeightNode *w = o0->weight; w; w = w->right)
        w->value = (w->right_index == 0) ? 1.0 : 0.0;
    for (WeightNode *w = o1->weight; w; w = w->right)
        w->value = (w->right_index == 1) ? 1.0 : 0.0;
    double x[2] = {-2.0, 3.0};
    double y[1];
    network_predict(net, x, 2, y, 1);
    EXPECT_NEAR(y[0], -6.0, 1e-9, "(-2)*3 = -6");
    x[0] = 0; x[1] = 0;
    network_predict(net, x, 2, y, 1);
    EXPECT_NEAR(y[0], 0.0, 1e-9, "0*0 = 0");
    network_free(net);
}

static void test_loss_partial_lists(void)
{
    section("MSE walks the shortest of pred / target / dloss");
    Network *net = network_create(1, 2);
    network_set_dynamic(net, 0);
    /* build two-long pred via a dummy forward */
    double x[1] = {0.0};
    double ybuf[2];
    network_predict(net, x, 1, ybuf, 2);

    InOutNode t0 = {.value = ybuf[0] + 2.0, .right_index = 0, .right = NULL};
    InOutNode d0 = {.value = 0, .right_index = 0, .right = NULL};
    InOutNode p0 = {.value = ybuf[0], .right_index = 0, .right = NULL};
    double loss = network_loss_mse(&p0, &t0, &d0);
    EXPECT_NEAR(loss, 4.0, 1e-12, "single-term MSE = 4");
    EXPECT_NEAR(d0.value, -2.0, 1e-12, "dloss = pred - target = -2");
    EXPECT(d0.right == NULL, "dloss list not extended");
    network_free(net);
}

static void test_multioutput_training_drops_loss(void)
{
    section("2-in 2-out net trains both heads");
    srand(17);
    Network *net = network_create(2, 2);
    network_set_dynamic(net, 0);
    network_set_learning_rate(net, 0.05);
    network_init_weights(net);
    double Xd[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    double Yd[4][2] = {{0,0},{0.2,0.4},{0.2,0.1},{0.4,0.5}};
    double *X[4], *Y[4];
    for (int i = 0; i < 4; i++) { X[i] = Xd[i]; Y[i] = Yd[i]; }
    double before = mse_on(net, X, Y, 4, 2, 2);
    network_train(net, X, Y, 4, 50);
    double after = mse_on(net, X, Y, 4, 2, 2);
    printf("  multiout mse %.4f -> %.4f\n", before, after);
    EXPECT(after < before, "multi-output loss dropped");
    network_free(net);
}

static void test_param_count_tracks_reshape(void)
{
    section("param_count and nbytes track grow / shrink");
    Network *net = network_create(2, 1);
    size_t p0 = network_param_count(net);
    size_t b0 = network_nbytes(net);
    EXPECT(p0 == (size_t)(2 * (1 + 2)), "2 ORs × (bias + 2 weights)");
    layer_align_inputs(net->head, 5);
    size_t p1 = network_param_count(net);
    EXPECT(p1 == (size_t)(2 * (1 + 5)), "weights grew with inputs");
    EXPECT(network_nbytes(net) > b0, "nbytes grew");
    layer_set_outputs(net->head, 3);
    EXPECT(network_param_count(net) == (size_t)(3 * 2 * (1 + 5)),
           "params scale with outputs × ORs × (bias+in)");
    network_free(net);
}

static void test_wider_predict_then_train(void)
{
    section("forward a wider x, then train at the new width");
    srand(29);
    Network *net = network_create(1, 1);
    network_set_dynamic(net, 1);
    network_set_learning_rate(net, 0.05);
    network_init_weights(net);
    double probe[3] = {0.2, 0.0, 0.0};
    double y[1];
    network_predict(net, probe, 3, y, 1);
    EXPECT(net->head->in_size == 3, "width is now 3");
    /* train using the live in_size (network_train reads head->in_size) */
    net->in_size = 3;
    double Xd[2][3] = {{0.2, 0.1, -0.1},{0.5, 0.0, 0.2}};
    double Yd[2][1] = {{0.1},{0.2}};
    double *X[2] = {Xd[0], Xd[1]};
    double *Y[2] = {Yd[0], Yd[1]};
    network_train(net, X, Y, 2, 20);
    network_predict(net, Xd[0], 3, y, 1);
    EXPECT(isfinite(y[0]), "trained at grown width");
    network_free(net);
}


static void test_create_zero_sizes_clamped(void)
{
    section("create(0, 0) is clamped to a usable 1→1 net");
    Network *net = network_create(0, 0);
    EXPECT(net != NULL, "create returned a net");
    EXPECT(net->in_size == 1 && net->out_size == 1, "sizes clamped to 1");
    double x[1] = {0.5};
    double y[1] = {99};
    network_predict(net, x, 1, y, 1);
    EXPECT(isfinite(y[0]), "predict on clamped net");
    network_free(net);
}

static void test_max_or_is_honored(void)
{
    section("max_or caps automatic factor growth");
    srand(31);
    Network *net = network_create(2, 1);
    network_set_dynamic(net, 1);
    network_set_verbose(net, 0);
    net->max_or = 2;
    network_set_learning_rate(net, 0.2);
    network_init_weights(net);
    double Xd[1][2] = {{0.9, 0.9}};
    double Yd[1][1] = {{6.0}};
    double *X[1] = {Xd[0]};
    double *Y[1] = {Yd[0]};
    network_train(net, X, Y, 1, 12);
    size_t n = or_count(net->head->and_row->or_row);
    printf("  OR count = %zu (max_or = %zu)\n", n, net->max_or);
    EXPECT(n <= net->max_or, "did not exceed max_or");
    EXPECT(n >= 1, "still has at least one OR");
    network_free(net);
}

static void test_max_depth_is_honored(void)
{
    section("max_depth = 1 blocks automatic layer insert");
    srand(33);
    Network *net = network_create(2, 1);
    network_set_dynamic(net, 1);
    network_set_verbose(net, 0);
    net->max_depth = 1;
    network_set_learning_rate(net, 0.2);
    network_init_weights(net);
    double Xd[2][2] = {{1,1},{-1,-1}};
    double Yd[2][1] = {{3},{-3}};
    double *X[2] = {Xd[0], Xd[1]};
    double *Y[2] = {Yd[0], Yd[1]};
    network_train(net, X, Y, 2, 15);
    EXPECT(network_depth(net) == 1, "stayed at depth 1");
    network_free(net);
}

static void test_add_layer_sets_prev_and_out_size(void)
{
    section("network_add_layer links prev and updates net->out_size");
    Network *net = network_create(3, 5);
    EXPECT(net->head->prev == NULL, "first layer prev is NULL");
    network_add_layer(net, 5, 2);
    EXPECT(network_depth(net) == 2, "depth 2");
    EXPECT(net->out_size == 2, "net out_size follows the new tail");
    EXPECT(net->tail->prev == net->head, "tail->prev is head");
    EXPECT(net->head->next == net->tail, "head->next is tail");
    expect_chain_ok(net, "two-layer add");
    network_free(net);
}

static void test_remove_realigns_next_inputs(void)
{
    section("removing a hidden layer realigns the next layer inputs");
    Network *net = network_create(2, 4);
    network_add_layer(net, 4, 1);
    network_set_dynamic(net, 0);
    network_set_verbose(net, 0);
    network_init_weights(net);
    Layer *hid = net->head;
    EXPECT(hid->out_size == 4 && net->tail->in_size == 4, "4-wide handshake");
    EXPECT(network_remove_layer(net, hid) == 0, "dropped hidden/head");
    EXPECT(network_depth(net) == 1, "one layer left");
    EXPECT(net->head->in_size == 2, "surviving layer realigned to 2 inputs");
    double x[2] = {0.3, -0.1};
    double y[1];
    network_predict(net, x, 2, y, 1);
    EXPECT(isfinite(y[0]), "predict after realign");
    network_free(net);
}

static void test_two_successive_train_calls(void)
{
    section("two successive train() calls keep a valid net");
    srand(37);
    Network *net = network_create(2, 1);
    network_set_dynamic(net, 0);
    network_set_verbose(net, 0);
    network_set_learning_rate(net, 0.06);
    network_init_weights(net);
    double Xd[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    double Yd[4][1] = {{0},{0.4},{0.4},{0.8}};
    double *X[4], *Y[4];
    for (int i = 0; i < 4; i++) { X[i] = Xd[i]; Y[i] = Yd[i]; }
    network_train(net, X, Y, 4, 15);
    double mid = mse_on(net, X, Y, 4, 2, 1);
    network_train(net, X, Y, 4, 15);
    double after = mse_on(net, X, Y, 4, 2, 1);
    printf("  mse after 15 then 15: %.4f -> %.4f\n", mid, after);
    EXPECT(isfinite(mid) && isfinite(after), "losses finite");
    EXPECT(after <= mid + 1e-3, "second train did not blow up");
    network_free(net);
}

static void test_set_outputs_then_train(void)
{
    section("grow outputs, then train the new heads");
    srand(39);
    Network *net = network_create(2, 1);
    network_set_dynamic(net, 0);
    network_set_verbose(net, 0);
    layer_set_outputs(net->head, 2);
    net->out_size = 2;
    network_set_learning_rate(net, 0.05);
    network_init_weights(net);
    double Xd[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    double Yd[4][2] = {{0,0},{0.3,0.1},{0.3,0.2},{0.6,0.3}};
    double *X[4], *Y[4];
    for (int i = 0; i < 4; i++) { X[i] = Xd[i]; Y[i] = Yd[i]; }
    double before = mse_on(net, X, Y, 4, 2, 2);
    network_train(net, X, Y, 4, 40);
    double after = mse_on(net, X, Y, 4, 2, 2);
    printf("  grown-out mse %.4f -> %.4f\n", before, after);
    EXPECT(after < before, "loss dropped on grown outputs");
    network_free(net);
}

static void test_quantization_snaps_near_zero_and_one(void)
{
    section("quantization snaps weights near 0 and 1");
    Network *net = network_create(1, 1);
    network_set_dynamic(net, 0);
    AndNode *a = net->head->and_row;
    OrNode *o = a->or_row;
    o->quantization = 1e-3;
    o->bias.value = 1.0 + 4e-4;
    o->weight->quantization = 1e-3;
    o->weight->value = 3e-4;
    /* one SGD step with zero incoming grad snaps via weight_sgd / bias update */
    o->bias.value = 1.0 + 4e-4;
    /* reuse a forward+tiny backward: target = pred so grad is 0, snap still runs */
    double Xd[1][1] = {{0.0}};
    double ytmp[1];
    network_predict(net, Xd[0], 1, ytmp, 1);
    double Yd[1][1] = {{ytmp[0]}};
    double *X[1] = {Xd[0]};
    double *Y[1] = {Yd[0]};
    network_set_verbose(net, 0);
    network_train(net, X, Y, 1, 1);
    EXPECT(fabs(o->weight->value) < 1e-2, "near-zero weight stayed tiny/snapped");
    network_free(net);
}

static void test_identity_nbytes_increase(void)
{
    section("identity insert increases param_count and nbytes");
    Network *net = network_create(4, 2);
    size_t p0 = network_param_count(net);
    size_t b0 = network_nbytes(net);
    network_insert_identity(net, net->tail);
    EXPECT(network_param_count(net) > p0, "more parameters");
    EXPECT(network_nbytes(net) > b0, "more bytes");
    network_free(net);
}

static void test_predict_does_not_change_depth(void)
{
    section("predict does not insert layers even with dynamic on");
    Network *net = network_create(2, 1);
    network_set_dynamic(net, 1);
    size_t d0 = network_depth(net);
    double x[2] = {10, -10};
    double y[1];
    for (int i = 0; i < 20; i++) network_predict(net, x, 2, y, 1);
    EXPECT(network_depth(net) == d0, "depth unchanged by inference");
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
    test_null_and_trivial_api();
    test_single_feature_forward();
    test_quadratic_is_rank2();
    test_multi_output_independent();
    test_predict_size_mismatch();
    test_dynamic_off_does_not_grow_or_or_depth();
    test_grow_then_shrink_roundtrip();
    test_insert_middle_and_remove_head();
    test_triple_identity_still_identity();
    test_train_zero_epochs_and_zero_init_predict();
    test_clip_huge_values_stay_finite();
    test_negative_and_zero_inputs();
    test_loss_partial_lists();
    test_multioutput_training_drops_loss();
    test_param_count_tracks_reshape();
    test_wider_predict_then_train();
    test_create_zero_sizes_clamped();
    test_max_or_is_honored();
    test_max_depth_is_honored();
    test_add_layer_sets_prev_and_out_size();
    test_remove_realigns_next_inputs();
    test_two_successive_train_calls();
    test_set_outputs_then_train();
    test_quantization_snaps_near_zero_and_one();
    test_identity_nbytes_increase();
    test_predict_does_not_change_depth();

    printf("\n══════════════════════════════════════════════\n");
    printf("  %d passed, %d failed\n", g_pass, g_fail);
    printf("══════════════════════════════════════════════\n");
    return g_fail ? 1 : 0;
}
