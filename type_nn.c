#define _GNU_SOURCE
#include "type_nn.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#ifdef TYPE_NN_DEBUG
#define DBG(...) printf(__VA_ARGS__)
#else
#define DBG(...) ((void)0)
#endif

#define DEFAULT_OR_FACTORS  2
#define DEFAULT_MAX_AND     4
#define DEFAULT_LR          0.05
#define DEFAULT_MAX_DEPTH   6
#define DEFAULT_MAX_OR      8
#define DEFAULT_QUANT       1e-7

static double frand(void)
{
    return (double)rand() / (double)RAND_MAX;
}

static double small_rand(void)
{
    return (frand() * 2.0 - 1.0) * 0.15;
}

static double snap_quant(double v, double q)
{
    if (q <= 0) return v;
    if (fabs(v) < q) return 0.0;
    if (fabs(v - 1.0) < q) return 1.0;
    return v;
}

/* ════════════════════════════════════════════
   Counts
   ════════════════════════════════════════════ */

size_t inout_count(const InOutNode *n)
{
    size_t c = 0;
    while (n) { c++; n = n->right; }
    return c;
}

size_t weight_count(const WeightNode *n)
{
    size_t c = 0;
    while (n) { c++; n = n->right; }
    return c;
}

size_t or_count(const OrNode *n)
{
    size_t c = 0;
    while (n) { c++; n = n->right; }
    return c;
}

size_t and_count(const AndNode *n)
{
    size_t c = 0;
    while (n) { c++; n = n->right; }
    return c;
}

size_t network_depth(const Network *net)
{
    return net ? net->depth : 0;
}

size_t network_param_count(const Network *net)
{
    if (!net) return 0;
    size_t n = 0;
    for (const Layer *l = net->head; l; l = l->next) {
        for (const AndNode *a = l->and_row; a; a = a->right) {
            for (const OrNode *o = a->or_row; o; o = o->right) {
                n += 1; /* bias */
                n += weight_count(o->weight);
            }
        }
    }
    return n;
}

size_t network_nbytes(const Network *net)
{
    if (!net) return 0;
    size_t bytes = sizeof(Network);
    for (const Layer *l = net->head; l; l = l->next) {
        bytes += sizeof(Layer);
        for (const InOutNode *n = l->in; n; n = n->right) bytes += sizeof(InOutNode);
        for (const InOutNode *n = l->out; n; n = n->right) bytes += sizeof(InOutNode);
        for (const InOutNode *n = l->din; n; n = n->right) bytes += sizeof(InOutNode);
        for (const AndNode *a = l->and_row; a; a = a->right) {
            bytes += sizeof(AndNode);
            for (const OrNode *o = a->or_row; o; o = o->right) {
                bytes += sizeof(OrNode);
                for (const WeightNode *w = o->weight; w; w = w->right)
                    bytes += sizeof(WeightNode);
            }
        }
    }
    return bytes;
}

/* ════════════════════════════════════════════
   Weight list
   ════════════════════════════════════════════ */

static WeightNode *weight_create(size_t size, double quant)
{
    if (!size) return NULL;
    WeightNode *head = NULL, *prev = NULL;
    for (size_t i = 0; i < size; i++) {
        WeightNode *n = (WeightNode *)calloc(1, sizeof(WeightNode));
        n->right_index = i;
        n->quantization = quant;
        if (!head) head = n;
        if (prev) prev->right = n;
        prev = n;
    }
    return head;
}

static void weight_init(WeightNode *node, double value, double quantization)
{
    while (node) {
        node->value = value;
        node->quantization = quantization;
        node = node->right;
    }
}

static void weight_free(WeightNode *node)
{
    while (node) {
        WeightNode *tmp = node->right;
        free(node);
        node = tmp;
    }
}

/* Insert or update a weight at `index`. List stays sorted. */
static WeightNode *weight_upsert(WeightNode *head, size_t index,
                                 double value, double quant)
{
    WeightNode *prev = NULL, *cur = head;
    while (cur && cur->right_index < index) {
        prev = cur;
        cur = cur->right;
    }
    if (cur && cur->right_index == index) {
        cur->value = value;
        cur->quantization = quant;
        return head;
    }
    WeightNode *n = (WeightNode *)calloc(1, sizeof(WeightNode));
    n->right_index = index;
    n->value = value;
    n->quantization = quant;
    n->right = cur;
    if (prev) prev->right = n;
    else head = n;
    return head;
}

/* Make sure every index in [0, in_size) exists. */
static WeightNode *weight_align_size(WeightNode *head, size_t in_size,
                                     double fill, double quant)
{
    for (size_t i = 0; i < in_size; i++) {
        WeightNode *cur = head;
        int found = 0;
        while (cur) {
            if (cur->right_index == i) { found = 1; break; }
            cur = cur->right;
        }
        if (!found) head = weight_upsert(head, i, fill, quant);
    }
    /* drop indices >= in_size */
    WeightNode dummy = {0};
    dummy.right = head;
    WeightNode *prev = &dummy;
    while (prev->right) {
        if (prev->right->right_index >= in_size) {
            WeightNode *dead = prev->right;
            prev->right = dead->right;
            free(dead);
        } else {
            prev = prev->right;
        }
    }
    return dummy.right;
}

static WeightNode *weight_prune(WeightNode *head, double thresh, size_t keep_min)
{
    size_t n = weight_count(head);
    WeightNode dummy = {0};
    dummy.right = head;
    WeightNode *prev = &dummy;
    while (prev->right) {
        WeightNode *cur = prev->right;
        if (n > keep_min && fabs(cur->value) < thresh) {
            prev->right = cur->right;
            free(cur);
            n--;
        } else {
            prev = cur;
        }
    }
    return dummy.right;
}

void weight_print(const WeightNode *node, const char *label)
{
    if (!node) {
        printf("%s Weight row: (empty)\n", label ? label : "");
        return;
    }
    printf("%s Weight row: ", label ? label : "");
    const WeightNode *row = node;
    while (row) {
        printf("(%zu, %8.4f), ", row->right_index, row->value);
        row = row->right;
    }
    printf("\n");
}

/* ════════════════════════════════════════════
   Or list
   ════════════════════════════════════════════ */

static OrNode *or_create(size_t in_size, size_t pol_size, double quant)
{
    if (!pol_size) return NULL;
    OrNode *head = NULL, *prev = NULL;
    for (size_t i = 0; i < pol_size; i++) {
        OrNode *n = (OrNode *)calloc(1, sizeof(OrNode));
        n->right_index = i;
        n->quantization = quant;
        n->weight = weight_create(in_size, quant);
        if (!head) head = n;
        if (prev) prev->right = n;
        prev = n;
    }
    return head;
}

static void or_init_random(OrNode *node, double quantization)
{
    while (node) {
        node->quantization = quantization;
        node->bias.value = small_rand();
        WeightNode *w = node->weight;
        while (w) {
            w->value = small_rand();
            w->quantization = quantization;
            w = w->right;
        }
        node = node->right;
    }
}

static void or_init_identity(OrNode *node, size_t feature, double quantization)
{
    /* first OR: w[feature]=1, rest 0, bias 0
       remaining ORs: all w=0, bias=1  → product equals x[feature] */
    if (!node) return;
    node->quantization = quantization;
    node->bias.value = 0.0;
    WeightNode *w = node->weight;
    while (w) {
        w->value = (w->right_index == feature) ? 1.0 : 0.0;
        w->quantization = quantization;
        w = w->right;
    }
    OrNode *rest = node->right;
    while (rest) {
        rest->quantization = quantization;
        rest->bias.value = 1.0;
        w = rest->weight;
        while (w) {
            w->value = 0.0;
            w->quantization = quantization;
            w = w->right;
        }
        rest = rest->right;
    }
}

static void or_free(OrNode *node)
{
    while (node) {
        OrNode *tmp = node->right;
        weight_free(node->weight);
        free(node);
        node = tmp;
    }
}

static void or_align_inputs(OrNode *node, size_t in_size, double fill, double quant)
{
    while (node) {
        node->weight = weight_align_size(node->weight, in_size, fill, quant);
        node = node->right;
    }
}

static OrNode *or_append_unit(OrNode *head, size_t in_size, double quant)
{
    /* New factor ≈ 1 so the product is unchanged. */
    OrNode *n = (OrNode *)calloc(1, sizeof(OrNode));
    n->quantization = quant;
    n->bias.value = 1.0;
    n->weight = weight_create(in_size, quant);
    weight_init(n->weight, 0.0, quant);
    if (!head) {
        n->right_index = 0;
        return n;
    }
    OrNode *tail = head;
    while (tail->right) tail = tail->right;
    n->right_index = tail->right_index + 1;
    tail->right = n;
    return head;
}

static OrNode *or_drop_at(OrNode *head, OrNode *target)
{
    if (!head || !target) return head;
    if (head == target) {
        OrNode *n = head->right;
        weight_free(head->weight);
        free(head);
        return n;
    }
    OrNode *prev = head;
    while (prev->right && prev->right != target) prev = prev->right;
    if (prev->right == target) {
        prev->right = target->right;
        weight_free(target->weight);
        free(target);
    }
    return head;
}

void or_print(const OrNode *node, const char *label)
{
    if (!node) return;
    printf("    %s Or row:\n", label ? label : "");
    const OrNode *row = node;
    while (row) {
        char buf[64];
        snprintf(buf, sizeof(buf), "      index %zu value %.4f bias %.4f",
                 row->right_index, row->value, row->bias.value);
        weight_print(row->weight, buf);
        row = row->right;
    }
}

/* ════════════════════════════════════════════
   And list
   ════════════════════════════════════════════ */

static AndNode *and_create(size_t in_size, size_t out_size, size_t pol_size, double quant)
{
    if (!out_size) return NULL;
    AndNode *head = NULL, *prev = NULL;
    for (size_t i = 0; i < out_size; i++) {
        AndNode *n = (AndNode *)calloc(1, sizeof(AndNode));
        n->right_index = i;
        n->quantization = quant;
        n->or_row = or_create(in_size, pol_size, quant);
        if (!head) head = n;
        if (prev) prev->right = n;
        prev = n;
    }
    return head;
}

static void and_init_random(AndNode *node, double quantization)
{
    while (node) {
        node->quantization = quantization;
        or_init_random(node->or_row, quantization);
        node = node->right;
    }
}

static void and_free(AndNode *node)
{
    while (node) {
        AndNode *tmp = node->right;
        or_free(node->or_row);
        free(node);
        node = tmp;
    }
}

static AndNode *and_append(AndNode *head, size_t in_size, size_t pol, double quant)
{
    AndNode *n = (AndNode *)calloc(1, sizeof(AndNode));
    n->quantization = quant;
    n->or_row = or_create(in_size, pol, quant);
    or_init_random(n->or_row, quant);
    if (!head) {
        n->right_index = 0;
        return n;
    }
    AndNode *tail = head;
    while (tail->right) tail = tail->right;
    n->right_index = tail->right_index + 1;
    tail->right = n;
    return head;
}

static AndNode *and_append_ones(AndNode *head, size_t in_size, size_t index,
                                size_t pol, double quant)
{
    /* Dummy And ≡ 1: every Or is (b=1, W=0). y ← y × 1. */
    AndNode *n = (AndNode *)calloc(1, sizeof(AndNode));
    n->quantization = quant;
    n->right_index = index;
    n->or_row = or_create(in_size, pol, quant);
    OrNode *o = n->or_row;
    while (o) {
        o->bias.value = 1.0;
        WeightNode *w = o->weight;
        while (w) { w->value = 0.0; w = w->right; }
        o = o->right;
    }
    if (!head) return n;
    AndNode *tail = head;
    while (tail->right) tail = tail->right;
    tail->right = n;
    return head;
}

static AndNode *and_drop_at(AndNode *head, AndNode *target)
{
    if (!head || !target) return head;
    if (head == target) {
        AndNode *n = head->right;
        or_free(head->or_row);
        free(head);
        return n;
    }
    AndNode *prev = head;
    while (prev->right && prev->right != target) prev = prev->right;
    if (prev->right == target) {
        prev->right = target->right;
        or_free(target->or_row);
        free(target);
    }
    return head;
}

static int and_is_ones(const AndNode *node)
{
    if (!node || !node->or_row) return 0;
    const OrNode *o = node->or_row;
    while (o) {
        if (fabs(o->bias.value - 1.0) > 0.2) return 0;
        const WeightNode *w = o->weight;
        while (w) {
            if (fabs(w->value) > 0.2) return 0;
            w = w->right;
        }
        o = o->right;
    }
    return 1;
}

static AndNode *and_trim(AndNode *head, size_t out_size)
{
    if (!head) return NULL;
    if (out_size == 0) {
        and_free(head);
        return NULL;
    }
    AndNode *cur = head;
    size_t i = 0;
    while (cur && i + 1 < out_size) {
        cur = cur->right;
        i++;
    }
    if (cur) {
        and_free(cur->right);
        cur->right = NULL;
    }
    return head;
}

void and_print(const AndNode *node, const char *label)
{
    if (!node) {
        printf("  %s And row: (empty)\n", label ? label : "");
        return;
    }
    printf("  %s And row:\n", label ? label : "");
    const AndNode *row = node;
    while (row) {
        char buf[32];
        snprintf(buf, sizeof(buf), "index %zu", row->right_index);
        or_print(row->or_row, buf);
        row = row->right;
    }
}

/* ════════════════════════════════════════════
   InOut list
   ════════════════════════════════════════════ */

static InOutNode *in_out_create(size_t size)
{
    if (!size) return NULL;
    InOutNode *head = NULL, *prev = NULL;
    for (size_t i = 0; i < size; i++) {
        InOutNode *n = (InOutNode *)calloc(1, sizeof(InOutNode));
        n->right_index = i;
        if (!head) head = n;
        if (prev) prev->right = n;
        prev = n;
    }
    return head;
}

static void in_out_free(InOutNode *node)
{
    while (node) {
        InOutNode *tmp = node->right;
        free(node);
        node = tmp;
    }
}

static InOutNode *in_out_from_array(const double *x, size_t n)
{
    InOutNode *head = in_out_create(n);
    InOutNode *cur = head;
    for (size_t i = 0; i < n && cur; i++) {
        cur->value = x[i];
        cur->right_index = i;
        cur = cur->right;
    }
    return head;
}

static void in_out_to_array(const InOutNode *node, double *out, size_t n)
{
    for (size_t i = 0; i < n; i++) out[i] = 0.0;
    while (node) {
        if (node->right_index < n) out[node->right_index] = node->value;
        node = node->right;
    }
}

static InOutNode *in_out_clone_values(const InOutNode *src)
{
    if (!src) return NULL;
    InOutNode *head = NULL, *prev = NULL;
    while (src) {
        InOutNode *n = (InOutNode *)calloc(1, sizeof(InOutNode));
        n->right_index = src->right_index;
        n->value = src->value;
        n->grad = src->grad;
        if (!head) head = n;
        if (prev) prev->right = n;
        prev = n;
        src = src->right;
    }
    return head;
}

static InOutNode *inout_add(InOutNode *head, size_t index, double value)
{
    InOutNode *prev = NULL, *cur = head;
    while (cur && cur->right_index < index) {
        prev = cur;
        cur = cur->right;
    }
    if (cur && cur->right_index == index) {
        cur->value += value;
        return head;
    }
    InOutNode *n = (InOutNode *)calloc(1, sizeof(InOutNode));
    n->right_index = index;
    n->value = value;
    n->right = cur;
    if (prev) prev->right = n;
    else head = n;
    return head;
}

void in_out_print(const InOutNode *node, const char *label)
{
    printf("[ InOut row: %s ]: ", label ? label : "");
    const InOutNode *row = node;
    while (row) {
        printf("(%zu, %8.4f), ", row->right_index, row->value);
        row = row->right;
    }
    printf("\n");
}

/* ════════════════════════════════════════════
   Layer
   ════════════════════════════════════════════ */

static Layer *layer_create(size_t in, size_t out, double quant)
{
    Layer *l = (Layer *)calloc(1, sizeof(Layer));
    l->in_size  = in;
    l->out_size = out;
    l->and_row  = and_create(in, out, DEFAULT_OR_FACTORS, quant);
    l->in       = in_out_create(in);
    l->out      = in_out_create(out);
    l->din      = in_out_create(in);
    return l;
}

static void layer_init_weights(Layer *l)
{
    if (!l) return;
    and_init_random(l->and_row, l->and_row ? l->and_row->quantization : DEFAULT_QUANT);
}

static void layer_free(Layer *l)
{
    if (!l) return;
    and_free(l->and_row);
    in_out_free(l->in);
    in_out_free(l->out);
    in_out_free(l->din);
    free(l);
}

void layer_align_inputs(Layer *l, size_t in_size)
{
    if (!l) return;
    AndNode *a = l->and_row;
    while (a) {
        or_align_inputs(a->or_row, in_size, 0.0, a->quantization);
        a = a->right;
    }
    in_out_free(l->in);
    l->in = in_out_create(in_size);
    in_out_free(l->din);
    l->din = in_out_create(in_size);
    l->in_size = in_size;
}

void layer_set_outputs(Layer *l, size_t out_size)
{
    if (!l) return;
    size_t cur = and_count(l->and_row);
    double q = l->and_row ? l->and_row->quantization : DEFAULT_QUANT;
    while (cur < out_size) {
        l->and_row = and_append(l->and_row, l->in_size, DEFAULT_OR_FACTORS, q);
        cur++;
    }
    if (cur > out_size) l->and_row = and_trim(l->and_row, out_size);
    in_out_free(l->out);
    l->out = in_out_create(out_size);
    l->out_size = out_size;
}

/* ════════════════════════════════════════════
   Network lifecycle
   ════════════════════════════════════════════ */

void network_add_layer(Network *net, size_t in, size_t out)
{
    Layer *l = layer_create(in, out, net->quantization);
    if (!net->head) {
        net->head = net->tail = l;
    } else {
        l->prev = net->tail;
        net->tail->next = l;
        net->tail = l;
    }
    net->out_size = out;
    net->depth++;
}

Network *network_create(size_t in, size_t out)
{
    Network *net = (Network *)calloc(1, sizeof(Network));
    if (in == 0) in = 1;
    if (out == 0) out = 1;
    net->in_size = in;
    net->out_size = out;
    net->lr = DEFAULT_LR;
    net->dynamic = 1;
    net->max_depth = DEFAULT_MAX_DEPTH;
    net->max_or = DEFAULT_MAX_OR;
    net->quantization = DEFAULT_QUANT;
    net->verbose = 1;
    network_add_layer(net, in, out);
    return net;
}

void network_init_weights(Network *net)
{
    Layer *l = net->head;
    while (l) {
        layer_init_weights(l);
        l = l->next;
    }
}

void network_free(Network *net)
{
    if (!net) return;
    Layer *l = net->head;
    while (l) {
        Layer *tmp = l->next;
        layer_free(l);
        l = tmp;
    }
    free(net);
}

void network_set_learning_rate(Network *net, double lr)
{
    if (net && lr > 0.0) net->lr = lr;
}

void network_set_dynamic(Network *net, int enabled)
{
    if (net) net->dynamic = enabled ? 1 : 0;
}

void network_set_layer_probe(Network *net, int enabled)
{
    if (net) net->layer_probe = enabled ? 1 : 0;
}

void network_set_verbose(Network *net, int enabled)
{
    if (net) net->verbose = enabled ? 1 : 0;
}

static void layer_make_identity(Layer *l)
{
    /* Map x -> x componentwise. Requires square layer. */
    size_t n = l->in_size < l->out_size ? l->in_size : l->out_size;
    AndNode *a = l->and_row;
    size_t i = 0;
    while (a && i < n) {
        or_init_identity(a->or_row, i, a->quantization);
        a = a->right;
        i++;
    }
}

Layer *network_insert_identity(Network *net, Layer *at)
{
    if (!net) return NULL;
    if (!at) at = net->tail;
    size_t dim = at->in_size;
    Layer *l = layer_create(dim, dim, net->quantization);
    layer_make_identity(l);

    l->next = at;
    l->prev = at->prev;
    if (at->prev) at->prev->next = l;
    at->prev = l;
    if (net->head == at) net->head = l;
    net->depth++;
    return l;
}

int network_remove_layer(Network *net, Layer *node)
{
    if (!net || !node) return -1;
    if (net->head == net->tail) return -1;          /* keep at least one */
    if (node == net->tail) return -1;               /* keep output layer */
    /* Neighbours must agree on width. */
    if (node->next && node->next->in_size != node->in_size) {
        /* Skip the removed layer: the next one now consumes its inputs. */
        layer_align_inputs(node->next, node->in_size);
        node->next->in_size = node->in_size;
    }
    Layer *next = node->next;
    Layer *prev = node->prev;
    if (next) next->prev = prev;
    if (prev) prev->next = next;
    if (node == net->head) net->head = next;
    if (node == net->tail) net->tail = prev;
    layer_free(node);
    net->depth--;
    return 0;
}

/* ════════════════════════════════════════════
   Forward
   ════════════════════════════════════════════ */

static double or_forward(OrNode *node, const InOutNode *x)
{
    double sum = node->bias.value;
    const WeightNode *w = node->weight;
    const InOutNode *xi = x;
    while (w && xi) {
        if (w->right_index < xi->right_index) {
            w = w->right;
        } else if (xi->right_index < w->right_index) {
            xi = xi->right;
        } else {
            sum += w->value * xi->value;
            w = w->right;
            xi = xi->right;
        }
    }
    node->value = sum;
    return sum;
}

static double and_forward(AndNode *node, const InOutNode *x)
{
    OrNode *or_row = node->or_row;
    double prod = 1.0;
    while (or_row) {
        prod *= or_forward(or_row, x);
        or_row = or_row->right;
    }
    node->value = prod;
    return prod;
}

/* y = tanh(z). For |z|>20, |tanh|=1 to machine precision and 1-y²=0. */
static double stable_tanh(double z)
{
    if (isnan(z)) return 0.0;
    if (!isfinite(z)) return (z > 0.0) ? 1.0 : -1.0;
    if (z > 20.0) return 1.0;
    if (z < -20.0) return -1.0;
    return tanh(z);
}

static void layer_forward(Layer *l, const InOutNode *x)
{
    in_out_free(l->in);
    l->in = in_out_clone_values(x);
    l->in_size = inout_count(l->in);

    /* Align weights to whatever features showed up. */
    AndNode *a = l->and_row;
    while (a) {
        or_align_inputs(a->or_row, l->in_size, 0.0, a->quantization);
        a = a->right;
    }

    in_out_free(l->out);
    l->out = NULL;
    a = l->and_row;
    while (a) {
        and_forward(a, l->in);
        a = a->right;
    }
    /* Product of Ands that share an index. Dummy And ≡ 1. */
    a = l->and_row;
    InOutNode *tail = NULL;
    size_t count = 0;
    while (a) {
        InOutNode *slot = l->out;
        while (slot && slot->right_index != a->right_index) slot = slot->right;
        if (!slot) {
            slot = (InOutNode *)calloc(1, sizeof(InOutNode));
            slot->right_index = a->right_index;
            slot->value = 1.0;
            if (!l->out) l->out = slot;
            else tail->right = slot;
            tail = slot;
            count++;
        }
        slot->value *= a->value;
        a = a->right;
    }
    /* Tail readout only: y = tanh(Π And). Hidden layers stay the raw
       product so an identity insert is still x ↦ x. Dummy And ≡ 1 is
       unchanged by this map (tanh is applied after the product). */
    if (!l->next) {
        InOutNode *slot = l->out;
        while (slot) {
            slot->value = stable_tanh(slot->value);
            slot = slot->right;
        }
    }
    l->out_size = count;
}

InOutNode *network_forward(Network *net, const InOutNode *input, size_t input_size)
{
    (void)input_size;
    const InOutNode *x = input;
    Layer *l = net->head;
    if (l && inout_count(x) != l->in_size) {
        layer_align_inputs(l, inout_count(x));
        net->in_size = l->in_size;
    }
    while (l) {
        layer_forward(l, x);
        x = l->out;
        if (l->next && l->out_size != l->next->in_size)
            layer_align_inputs(l->next, l->out_size);
        l = l->next;
    }
    return net->tail->out;
}

void network_predict(Network *net, double *x, size_t in,
                     double *out_buf, size_t out)
{
    InOutNode *x_mat = in_out_from_array(x, in);
    InOutNode *pred = network_forward(net, x_mat, in);
    in_out_to_array(pred, out_buf, out);
    in_out_free(x_mat);
}

/* ════════════════════════════════════════════
   Loss
   ════════════════════════════════════════════ */

double network_loss_mse(const InOutNode *pred, const InOutNode *target, InOutNode *dloss)
{
    double loss = 0.0;
    size_t n = 0;
    const InOutNode *p = pred;
    const InOutNode *t = target;
    InOutNode *dl = dloss;
    while (p && t) {
        double diff = p->value - t->value;
        loss += diff * diff;
        if (dl) {
            dl->value = diff;
            dl->right_index = p->right_index;
            dl = dl->right;
        }
        p = p->right;
        t = t->right;
        n++;
    }
    if (n > 0) loss /= (double)n;
    return loss;
}

/* ════════════════════════════════════════════
   Backward
   ════════════════════════════════════════════ */

static void compute_accums(AndNode *node)
{
    /* accum_k = Π_{j≠k} or_j  so  ∂and/∂or_k = accum_k */
    size_t n = or_count(node->or_row);
    if (n == 0) return;

    double *vals = (double *)calloc(n, sizeof(double));
    OrNode *or_row = node->or_row;
    for (size_t i = 0; i < n; i++) {
        vals[i] = or_row->value;
        or_row = or_row->right;
    }

    double prefix = 1.0;
    or_row = node->or_row;
    for (size_t i = 0; i < n; i++) {
        or_row->accum = prefix;
        prefix *= vals[i];
        or_row = or_row->right;
    }
    double suffix = 1.0;
    /* walk backwards via array, then write */
    or_row = node->or_row;
    OrNode **order = (OrNode **)malloc(n * sizeof(OrNode *));
    for (size_t i = 0; i < n; i++) {
        order[i] = or_row;
        or_row = or_row->right;
    }
    for (size_t k = n; k-- > 0; ) {
        order[k]->accum *= suffix;
        suffix *= vals[k];
    }
    free(order);
    free(vals);
}

static void weight_sgd(WeightNode *w, double grad, double lr)
{
    w->grad = grad;
    w->value = snap_quant(w->value - lr * grad, w->quantization);
}

static int or_is_ones(const OrNode *node)
{
    /* reserved / identity-for-product: bias≈1 and all weights≈0 */
    if (fabs(node->bias.value - 1.0) > 0.2) return 0;
    const WeightNode *w = node->weight;
    while (w) {
        if (fabs(w->value) > 0.2) return 0;
        w = w->right;
    }
    return 1;
}

static int or_is_dead(const OrNode *node)
{
    /* product killer: everything ~ 0 */
    if (fabs(node->bias.value) > 0.05) return 0;
    const WeightNode *w = node->weight;
    while (w) {
        if (fabs(w->value) > 0.05) return 0;
        w = w->right;
    }
    return 1;
}

static void or_backward(OrNode *node, const InOutNode *x,
                        double d_or, double lr)
{
    node->grad = d_or;
    node->bias.grad = d_or;
    node->bias.value = snap_quant(node->bias.value - lr * d_or, node->quantization);

    WeightNode *w = node->weight;
    const InOutNode *xi = x;
    while (w && xi) {
        if (w->right_index < xi->right_index) {
            /* no matching input this step – still regularize slightly */
            weight_sgd(w, 0.0, lr);
            w = w->right;
        } else if (xi->right_index < w->right_index) {
            xi = xi->right;
        } else {
            weight_sgd(w, d_or * xi->value, lr);
            w = w->right;
            xi = xi->right;
        }
    }
    while (w) {
        weight_sgd(w, 0.0, lr);
        w = w->right;
    }
}

/* Keep exactly one identity Or (b=1, W=0) in the product. */
static void or_ensure_dummy(AndNode *node, size_t in,
                            size_t max_or, unsigned *or_add)
{
    size_t n = 0, n_ones = 0;
    OrNode *o = node->or_row;
    while (o) {
        n++;
        if (or_is_ones(o)) n_ones++;
        o = o->right;
    }
    if (n_ones == 0 && n < max_or) {
        node->or_row = or_append_unit(node->or_row, in, node->quantization);
        if (or_add) (*or_add)++;
    }
}

static void or_prune_identities(AndNode *node, unsigned *or_drop)
{
    size_t n = or_count(node->or_row);
    size_t n_ones = 0;
    OrNode *o = node->or_row;
    while (o && n > 1) {
        OrNode *next = o->right;
        int ones = or_is_ones(o);
        if (or_is_dead(o) || (ones && n_ones > 0)) {
            node->or_row = or_drop_at(node->or_row, o);
            n--;
            if (or_drop) (*or_drop)++;
        } else if (ones) {
            n_ones++;
        }
        o = next;
    }
}

static bool and_backward(AndNode *node, const InOutNode *x,
                         double d_and, double lr,
                         int dynamic, size_t max_or,
                         unsigned *or_add, unsigned *or_drop)
{
    if (dynamic)
        or_ensure_dummy(node, inout_count(x), max_or, or_add);

    compute_accums(node);
    OrNode *or_row = node->or_row;
    while (or_row) {
        double d_or = d_and * or_row->accum;
        or_backward(or_row, x, d_or, lr);
        or_row = or_row->right;
    }

    if (dynamic)
        or_prune_identities(node, or_drop);
    return true;
}

static void and_ensure_dummy(Layer *l, size_t index, unsigned *and_add)
{
    size_t n = 0, n_ones = 0;
    AndNode *a = l->and_row;
    while (a) {
        if (a->right_index == index) {
            n++;
            if (and_is_ones(a)) n_ones++;
        }
        a = a->right;
    }
    if (n_ones == 0 && n < DEFAULT_MAX_AND) {
        double q = l->and_row ? l->and_row->quantization : DEFAULT_QUANT;
        l->and_row = and_append_ones(l->and_row, l->in_size, index,
                                     DEFAULT_OR_FACTORS, q);
        if (and_add) (*and_add)++;
    }
}

static void and_prune_identities(Layer *l, size_t index, unsigned *and_drop)
{
    size_t n = 0, n_ones = 0;
    AndNode *a = l->and_row;
    while (a) {
        if (a->right_index == index) n++;
        a = a->right;
    }
    a = l->and_row;
    while (a && n > 1) {
        AndNode *next = a->right;
        if (a->right_index == index) {
            int ones = and_is_ones(a);
            /* Identity And = all Ors ≈ (1,0). Keep one. Drop extras. */
            if (ones && n_ones > 0) {
                l->and_row = and_drop_at(l->and_row, a);
                n--;
                if (and_drop) (*and_drop)++;
            } else if (ones) {
                n_ones++;
            }
        }
        a = next;
    }
}

static bool layer_backward(Layer *l, const InOutNode *dloss, double lr,
                           int dynamic, size_t max_or,
                           unsigned *or_add, unsigned *or_drop,
                           unsigned *and_add, unsigned *and_drop)
{
    const InOutNode *d = dloss;
    AndNode *a = l->and_row;
    bool fitted = false;

    /* Zip And nodes against dloss by index; grow outputs if dloss is longer. */
    while (d) {
        AndNode *match = l->and_row;
        AndNode *found = NULL;
        while (match) {
            if (match->right_index == d->right_index) { found = match; break; }
            match = match->right;
        }
        if (!found) {
            /* Grow an output slot as identity, not a random product jump. */
            double q = l->and_row ? l->and_row->quantization : DEFAULT_QUANT;
            l->and_row = and_append_ones(l->and_row, l->in_size, d->right_index,
                                         DEFAULT_OR_FACTORS, q);
            found = l->and_row;
            while (found->right) found = found->right;
            if (and_add) (*and_add)++;
        }
        if (dynamic)
            and_ensure_dummy(l, d->right_index, and_add);
        /* Hidden: y = Π And, ∂y/∂And_r = Π_{q≠r} And_q.
           Tail:   y = tanh(Π And), ∂y/∂And_r = (1-y²) Π_{q≠r} And_q. */
        double prod = 1.0;
        AndNode *t = l->and_row;
        while (t) {
            if (t->right_index == d->right_index) prod *= t->value;
            t = t->right;
        }
        double dprod = d->value;
        if (!l->next) {
            double y = stable_tanh(prod);
            dprod *= (1.0 - y * y);
            if (!isfinite(dprod)) dprod = 0.0;
        }
        t = l->and_row;
        while (t) {
            if (t->right_index == d->right_index) {
                double others = 0.0;
                if (isfinite(prod) && fabs(t->value) > 1e-12)
                    others = prod / t->value;
                if (!isfinite(others)) others = 0.0;
                t->grad = dprod * others;
                fitted = and_backward(t, l->in, t->grad, lr, dynamic,
                                      max_or, or_add, or_drop) || fitted;
            }
            t = t->right;
        }
        if (dynamic)
            and_prune_identities(l, d->right_index, and_drop);
        d = d->right;
    }
    {
        size_t uniq = 0;
        AndNode *t = l->and_row;
        while (t) {
            int seen = 0;
            AndNode *u = l->and_row;
            while (u != t) {
                if (u->right_index == t->right_index) { seen = 1; break; }
                u = u->right;
            }
            if (!seen) uniq++;
            t = t->right;
        }
        l->out_size = uniq;
    }

    /* Input-side gradient: din_j = Σ_i Σ_k dloss_i * accum_k * w_{k,j} */
    in_out_free(l->din);
    l->din = NULL;
    const InOutNode *xin = l->in;
    while (xin) {
        double sum = 0.0;
        d = dloss;
        a = l->and_row;
        while (a) {
            double da = 0.0;
            da = a->grad;
            OrNode *o = a->or_row;
            while (o) {
                const WeightNode *w = o->weight;
                while (w) {
                    if (w->right_index == xin->right_index)
                        sum += da * o->accum * w->value;
                    w = w->right;
                }
                o = o->right;
            }
            a = a->right;
        }
        l->din = inout_add(l->din, xin->right_index, sum);
        xin = xin->right;
    }
    if (!l->din) l->din = in_out_create(l->in_size);

    /* Optional prune of near-zero weights (keep at least one). */
    if (dynamic) {
        a = l->and_row;
        while (a) {
            OrNode *o = a->or_row;
            while (o) {
                o->weight = weight_prune(o->weight, o->quantization * 10.0, 1);
                o = o->right;
            }
            a = a->right;
        }
    }
    return fitted;
}

void network_backward(Network *net, const InOutNode *dloss)
{
    const InOutNode *din = dloss;
    Layer *l = net->tail;
    while (l) {
        layer_backward(l, din, net->lr, net->dynamic, net->max_or,
                       &net->or_add, &net->or_drop, &net->and_add, &net->and_drop);
        din = l->din;

        /* Removal is explicit (network_remove_layer). Doing it here while
           `din` still aliases the layer we would free is a use-after-free. */
        l = l->prev;
    }

    if (net->layer_probe && net->dynamic && net->depth < net->max_depth
        && net->tail && net->depth == 1) {
        /* At most one automatic hidden layer: only from a single-layer net,
           and only when the input-side gradient is large. */
        double mag = 0.0;
        const InOutNode *p = net->tail->din;
        while (p) { mag += fabs(p->value); p = p->right; }
        if (mag > 3.0) {
            network_insert_identity(net, net->tail);
            net->layer_add++;
        }
    }
}

/* ════════════════════════════════════════════
   Training loop
   ════════════════════════════════════════════ */

void network_train(Network *net,
                   double **X, double **Y,
                   size_t n_samples,
                   size_t epochs)
{
    size_t in = net->head ? net->head->in_size : net->in_size;
    size_t out = net->tail ? net->tail->out_size : net->out_size;
    net->in_size = in;
    net->out_size = out;
    InOutNode *dloss = in_out_create(out);

    if (net->dynamic) {
        for (Layer *l = net->head; l; l = l->next) {
            AndNode *a = l->and_row;
            while (a) {
                or_ensure_dummy(a, l->in_size, net->max_or, NULL);
                and_ensure_dummy(l, a->right_index, NULL);
                a = a->right;
            }
        }
    }

    for (size_t ep = 0; ep < epochs; ep++) {
        double total_loss = 0.0;
        for (size_t s = 0; s < n_samples; s++) {
            InOutNode *x_mat = in_out_from_array(X[s], in);
            InOutNode *y_mat = in_out_from_array(Y[s], out);

            /* Growing output side if the target is longer than the net. */
            size_t ycnt = inout_count(y_mat);
            if (ycnt > net->out_size) {
                layer_set_outputs(net->tail, ycnt);
                net->out_size = ycnt;
                out = ycnt;
                in_out_free(dloss);
                dloss = in_out_create(out);
            }

            InOutNode *pred = network_forward(net, x_mat, in);
            total_loss += network_loss_mse(pred, y_mat, dloss);
            network_backward(net, dloss);

            in_out_free(x_mat);
            in_out_free(y_mat);
        }

        if (net->verbose && ((ep + 1) % 100 == 0 || ep == 0 || ep + 1 == epochs)) {
            printf("Epoch %5zu / %zu  |  MSE loss = %.6f  |  depth = %zu"
                   "  |  or +%u/-%u  and +%u/-%u  layer +%u/-%u\n",
                   ep + 1, epochs, total_loss / (double)n_samples, net->depth,
                   net->or_add, net->or_drop, net->and_add, net->and_drop,
                   net->layer_add, net->layer_drop);
        }
    }
    in_out_free(dloss);
}
