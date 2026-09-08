
#define _GNU_SOURCE
#include "vortex.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>

/* ════════════════════════════════════════════
   Weight
   ════════════════════════════════════════════ */

static WeightNode *weight_create(size_t size)
{
    if (!size) return NULL;
    WeightNode *node = (WeightNode *)calloc(1, sizeof(WeightNode)), *head = node;
    for (size_t i = 0; i < size; i++) {
        node->right_index = i;
        if (i != size - 1) {
            node->right = (WeightNode *)calloc(1, sizeof(WeightNode));
        }
        node = node->right;
    }
    return head;
}

void weight_init(WeightNode *node, double value, double quantization)
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

static void weight_push(WeightNode *node, WeightNode *target)
{
    if (node && target) {
        target->right = node->right;
        node->right = target;
    }
}

static void weight_drop(WeightNode *node)
{
    if (!node || !node->right) return;
    WeightNode *tmp = node->right;
    node->right = node->right->right;
    free(tmp);
}

void weight_print(const WeightNode *node, const char *label) {
    if (!node) return;
    printf("%s Weight row: ", label);
    const WeightNode *row = node;
    while (row) {
        printf("(%zu, %8.4f), ", row->right_index, row->value);
        row = row->right;
    }
    printf("\n");
}

/* ════════════════════════════════════════════
   Or
   ════════════════════════════════════════════ */

static OrNode *or_create(size_t in_size, size_t pol_size)
{
    if (!pol_size) return NULL;
    OrNode *node = (OrNode *)calloc(1, sizeof(OrNode)), *head = node;
    for (size_t i = 0; i < pol_size; i++) {
        node->right_index = i;
        node->weight = weight_create(in_size);
        if (i != pol_size - 1) {
            node->right = (OrNode *)calloc(1, sizeof(OrNode));
        }
        node = node->right;
    }
    return head;
}

void or_init_weights(OrNode *node, double weight, double bias, double quantization)
{
    while (node) {
        weight_init(node->weight, weight, quantization);
        node->bias.value = bias;
        node->quantization = quantization;
        node = node->right;
    }
}

static void or_push(OrNode *node, OrNode *target)
{
    if (node && target) {
        target->right = node->right;
        node->right = target;
    }
}

static void or_remove(OrNode *node)
{
    if (node) {
        OrNode *right = node->right;
        if (right) {
            node->right = right->right;
            free(right);
        }
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

void or_print(const OrNode *node, const char *label) {
    if (!node) return;
    printf("    %s Or row:\n", label);
    const OrNode *row = node;
    while (row) {
        char *string;
        asprintf(&string, "      index: %zu, value: ", row->right_index);
        weight_print(row->weight, string);
        free(string);
        row = row->right;
    }
    printf("\n");
}

/* ════════════════════════════════════════════
   And
   ════════════════════════════════════════════ */

void and_print(const AndNode *node, const char *label) {
    if (!node) return;
    printf("  %s And row:\n", label);
    const AndNode *row = node;
    while (row) {
        char *string;
        asprintf(&string, "index: %zu, value: ", row->right_index);
        or_print(row->or_row, string);
        free(string);
        row = row->right;
    }
}

static AndNode *and_create(size_t in_size, size_t out_size, size_t pol_size)
{
    if (!out_size) return NULL;
    AndNode *node = (AndNode *)calloc(1, sizeof(AndNode)), *head = node;
    node->right_index = 100;
    for (size_t i = 0; i < out_size; i++) {
        node->right_index = i;
        node->or_row = or_create(in_size, pol_size);
        if (i != out_size - 1) {
            node->right = (AndNode *)calloc(1, sizeof(AndNode));
        }
        and_print(node, "and_create");
        node = node->right;
    }
    return head;
}

void and_init_weights(AndNode *node, double quantization)
{
    while (node) {
        node->quantization = quantization;
        or_init_weights(node->or_row, 1, 0, quantization);
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

static void and_push(AndNode *node, AndNode *target)
{
    if (node && target) {
        target->right = node->right;
        node->right = target;
    }
}

static void and_drop(AndNode *node)
{
    if (!node || !node->right) return;
    AndNode *tmp = node->right;
    node->right = node->right->right;
    free(tmp);
}

/* ════════════════════════════════════════════
   InOut
   ════════════════════════════════════════════ */

static InOutNode *in_out_create(size_t size)
{
    printf("[ Here : %s, size: %zu ]\n", "1.1", size);
    InOutNode *node = (InOutNode *)calloc(1, sizeof(InOutNode)), *head = node;
    printf("[ Here : %s ]\n", "1.2");
    for (size_t i = 0; i < size; i++) {
        printf("[ Here : %s, i: %zu ]\n", "1.3", i);
        node->right_index = i;
        if (i != size - 1) {
            node->right = (InOutNode *)calloc(1, sizeof(InOutNode));
        }
        node = node->right;
    }
    printf("[ Here : %s, head: %zu ]\n", "1.4", head);
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

void in_out_print(const InOutNode *node, const char *label) {
    printf("[ InOut row: %s ]: ", label);
    const InOutNode *row = node;
    while (row) {
        printf("(%zu, %8.4f), ", row->right_index, row->value);
        row = row->right;
    }
    printf("\n");
}

static InOutNode *in_out_insert(InOutNode *node)
{
    in_out_print(node, "[Here : 7.1.1.11.1]");
    if (!node) {
        printf("[ Here : %s ]\n", "7.1.1.11.2");
        node = (InOutNode *)calloc(1, sizeof(InOutNode));
    } else {
        printf("[ Here : %s ]\n", "7.1.1.11.3.1");
        InOutNode *tmp = node;
        node = (InOutNode *)calloc(1, sizeof(InOutNode));
        node->right = tmp->right;
        tmp->right = node;
    }
    return node;
}

static void in_out_push(InOutNode *node, InOutNode *target)
{
    if (node && target) {
        target->right = node->right;
        node->right = target;
    }
}

static void in_out_drop(InOutNode *node)
{
    if (!node || !node->right) return;
    InOutNode *tmp = node->right;
    node->right = node->right->right;
    free(tmp);
}

static InOutNode *in_out_update_with(InOutNode *node, const InOutNode *target, size_t* size) {
    printf("[ Here : %s ]\n", "7.1.1.0");
    InOutNode *head = node, *old_node = node;
    const InOutNode *target_node = target;
    size_t count = 0;
    printf("[ Here : %s, size: %zu, old right: %zu, target right: %zu ]\n", "7.1.1.1", *size, old_node->right_index, target_node->right_index);
    // ZIP IT
    assert(old_node);
    assert(target_node);
    while (old_node && target_node && old_node->right_index < target_node->right_index) {
        printf("[ Here : %s, count: %zu ]\n", "7.1.1.2", count);
        InOutNode *tmp = head;
        head = head->right;
        free(tmp);
        old_node = head;
        if (!old_node) {
            head = (InOutNode *)calloc(1, sizeof(InOutNode));
            head->right_index = target_node->right_index;
            head->value = target_node->value;
            head->grad = target_node->grad;
            old_node = head;
            target_node = target_node->right;
            count += 1;
        }
    }
    printf("[ Here : %s ]\n", "7.1.1.3");
    if (old_node && target_node && target_node->right_index < old_node->right_index) {
        printf("[ Here : %s, count: %zu ]\n", "7.1.1.4", count);
        InOutNode *tmp = head;
        head = (InOutNode *)calloc(1, sizeof(InOutNode));
        head->right = tmp;
        head->right_index = target_node->right_index;
        head->value = target_node->value;
        head->grad = target_node->grad;
        old_node = head;
        target_node = target_node->right;
        count += 1;
    }
    while(target_node && old_node) {
        printf("[ Here : %s, count: %zu ]\n", "7.1.1.5", count);
        if (target_node->right && old_node->right) {
            printf("[ Here : %s, count: %zu, old right: %zu, target right: %zu ]\n", "7.1.1.6", count, old_node->right->right_index, target_node->right->right_index);
            if (old_node->right->right_index < target_node->right->right_index) {
                printf("[ Here : %s, count:%zu ]\n", "7.1.1.7", count);
                InOutNode *tmp = in_out_insert(old_node);
                tmp->right_index = target_node->right->right_index;
                tmp->value = target_node->right->value;
                tmp->grad = target_node->right->grad;
                old_node = old_node->right;
                target_node = target_node->right;
                count += 1;
            } else if (target_node->right->right_index < old_node->right->right_index) {
                printf("[ Here : %s, count: %zu ]\n", "7.1.1.8", count);
                in_out_drop(old_node);
            } else {
                printf("[ Here : %s, count: %zu ]\n", "7.1.1.9", count);
                old_node->value = target_node->value;
                old_node = old_node->right;
                target_node = target_node->right;
                count += 1;
            }
        } else if (old_node->right && !target_node->right) {
            printf("[ Here : %s, count: %zu ]\n", "7.1.1.12", count);
            in_out_drop(old_node);
        } else if (!old_node->right && target_node->right) {
            printf("[ Here : %s, count: %zu ]\n", "7.1.1.11", count);
            InOutNode *tmp = in_out_insert(old_node);
            tmp->right_index = target_node->right->right_index;
            tmp->value = target_node->right->value;
            tmp->grad = target_node->right->grad;
            old_node = old_node->right;
            target_node = target_node->right;
            count += 1;
            assert(old_node && target_node);
        } else {
            printf("[ Here : %s, count: %zu ]\n", "7.1.1.12", count);
            old_node->right_index = target_node->right_index;
            old_node->value = target_node->value;
            old_node->grad = target_node->grad;
            old_node = old_node->right;
            target_node = target_node->right;
            count += 1;
        }
    }
    assert(!old_node);
    assert(!target_node);
    printf("[ Here : %s, %zu ]\n", "7.1.1.13", count);
    *size = count;
    return head;
}

/* ════════════════════════════════════════════
   Layer
   ════════════════════════════════════════════ */

static Layer *layer_create(size_t in, size_t out)
{
    Layer *l = (Layer *)calloc(1, sizeof(Layer));
    l->in_size   = in;
    l->out_size  = out;
    l->and_row   = and_create(in, out, 2);
    l->in        = in_out_create(in);
    l->out       = in_out_create(out);
    l->din       = in_out_create(in);
    printf("[ Here : %s, l: %zu, in: %zu, out: %zu, din: %zu ]\n", "8.2.2.1", l, l->in, l->out, l->din);
    return l;
}

void layer_init_weights(Layer *l)
{
    while (l) {
        // quantization is 0.0000001
        and_init_weights(l->and_row, 0.0000001);
        l = l->next;
    }
}

static void layer_insert(Layer *node, Layer * insert)
{
    if (node && insert) {
        Layer *prev = node->prev;
        if (prev) {
            prev->next = insert;
            insert->prev = prev;
        }
        node->prev = insert;
        insert->next = node;
    }
}

static void layer_drop(Network *net, Layer *node)
{
    if (!node || net->head == net->tail) return;
    Layer *next = node->next;
    Layer *prev = node->prev;
    if (next) {
        next->prev = prev;
    }
    if (prev) {
        prev->next = next;
    }
    if (node == net->head) {
        net->head = next;
    }
    if (node == net->tail) {
        net->tail = prev;
    }
    free(node);
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

/* ════════════════════════════════════════════
   Network
   ════════════════════════════════════════════ */

void network_add_layer(Network *net, size_t in, size_t out)
{
    Layer *l = layer_create(in, out);
    if (!net->head) {
        net->head = net->tail = l;
    } else {
        net->tail->next = l;
        net->tail = l;
    }
    net->depth++;
}

Network *network_create(size_t in, size_t out)
{
    Network *net = (Network *)calloc(1, sizeof(Network));
    net->in_size = in;
    net->out_size = out;
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
    Layer *l = net->head;
    while (l) {
        Layer *tmp = l->next;
        layer_free(l);
        l = tmp;
    }
    free(net);
}

/* ════════════════════════════════════════════
   Forward pass (inference)
   ════════════════════════════════════════════ */

static double or_forward(OrNode *node, const InOutNode *x)
{
    const InOutNode *x_node = x;
    WeightNode *weight = node->weight;
    double sum = node->bias.value;
    printf("[ Here : %s, sum: %f ]\n", "7.1.7.1.1", sum);
    assert(weight);
    assert(x_node);
    while (weight && x_node && weight->right_index < x_node->right_index) {
        printf("[ Here : %s ]\n", "7.1.7.1.2");
        WeightNode *tmp = node->weight;
        node->weight = node->weight->right;
        free(tmp);
        weight = node->weight;
        if (!weight) {
            node->weight = weight_create(1);
            weight_init(tmp, 1, node->weight->quantization);
            node->weight->right = tmp;
            node->weight->right_index = x_node->right_index;
            sum += x_node->value;
            weight = node->weight;
            x_node = x_node->right;
        }
    }
    if (weight && x_node && x_node->right_index < weight->right_index) {
        printf("[ Here : %s ]\n", "7.1.7.1.3");
        WeightNode *tmp = node->weight;
        node->weight = weight_create(1);
        weight_init(tmp, 1, node->weight->quantization);
        node->weight->right = tmp;
        node->weight->right_index = x_node->right_index;
        sum += x_node->value;
        weight = node->weight;
        x_node = x_node->right;
    }
    while (weight && x_node) {
        printf("[ Here : %s ]\n", "7.1.7.1.4");
        if (x_node->right && weight->right) {
            printf("[ Here : %s ]\n", "7.1.7.1.5");
            if (weight->right->right_index < x_node->right->right_index) {
                printf("[ Here : %s ]\n", "7.1.7.1.6");
                WeightNode *tmp = weight_create(1);
                weight_init(tmp, 1, weight->right->quantization);
                weight_push(weight->right, tmp);
                tmp->right_index = x_node->right->right_index;
                sum += x_node->right->value;
                weight = weight->right;
                x_node = x_node->right;
            } else if (x_node->right->right_index < weight->right->right_index) {
                printf("[ Here : %s ]\n", "7.1.7.1.7");
                weight_drop(weight);
            } else {
                printf("[ Here : %s ]\n", "7.1.7.1.8");
                assert(weight->right_index == x_node->right_index);
                sum += weight->right->value * x_node->right->value;
                weight = weight->right;
                x_node = x_node->right;
            }
        } else if (weight->right && !x_node->right) {
            printf("[ Here : %s ]\n", "7.1.7.1.9");
            weight_drop(weight);
        } else if(!weight->right && x_node->right) {
            printf("[ Here : %s ]\n", "7.1.7.1.12");
            WeightNode *tmp = weight_create(1);
            weight_init(tmp, 0, weight->quantization);
            weight_push(weight, tmp);
            tmp->right_index = x_node->right_index;
            sum += x_node->value; // weight->value is one
            weight = weight->right;
            x_node = x_node->right;
            assert(weight && x_node);
        } else {
            printf("[ Here : %s ]\n", "7.1.7.1.13");
            sum += weight->value * x_node->value;
            assert(weight->right_index == x_node->right_index);
            weight = weight->right;
            x_node = x_node->right;
        }
    }
    assert(!weight);
    assert(!x_node);
    printf("[ Here : %s, sum: %f ]\n", "7.1.7.1.14", sum);
    node->value = sum;
    return sum;
}

static double and_forward(AndNode *node, const InOutNode *x)
{
    OrNode *or_row = node->or_row;
    double prod = 1;
    while (or_row) {
        or_print(or_row, "7.1.7.1");
        double or_value = or_forward(or_row, x);
        assert(or_row->weight);
        prod *= or_value;
        printf("[ Here : %s, prod: %f, or_value: %f ]\n", "7.1.7.3", prod, or_value);
        or_row = or_row->right;
    }
    node->value = prod;
    return prod;
}

/*
 * Single-layer forward:
 *   z[i] = Σ_j W[i,j] * x[j]  +  b[i]
 *   a[i] = act(z[i])
 *
 * Input  x : (in  × 1) Matrix
 * Output a : (out × 1) Matrix  (already stored in layer)
 */
static void layer_forward(Layer *l, const InOutNode *x)
{
    size_t in_size = 0;
    in_out_print(l->in, "Here : 7.1.1, l->in");
    in_out_print(x, "Here : 7.1.1, x");
    l->in = in_out_update_with(l->in, x, &in_size);
    assert(l->in);
    printf("[ Here : %s, in_size: %zu ]\n", "7.1.2", in_size);
    l->in_size = in_size;
    AndNode *and_row = l->and_row;
    InOutNode *out = l->out;
    assert(out);
    assert(and_row);
    printf("[ Here : %s ]\n", "7.1.3");
    while (out && and_row && out->right_index < and_row->right_index) {
        printf("[ Here : %s ]\n", "7.1.4");
        InOutNode *tmp = l->out;
        l->out = l->out->right;
        free(tmp);
        out = l->out;
    }
    if (out && and_row && and_row->right_index < out->right_index) {
        printf("[ Here : %s ]\n", "7.1.5");
        InOutNode *tmp = l->out;
        l->out = in_out_create(1);
        l->out->right = tmp;
        l->out->right_index = and_row->right_index;
        l->out->value = and_forward(and_row, x);
        out = l->out;
        and_row = and_row->right;
    }
    while (and_row && out) {
        printf("[ Here : %s ]\n", "7.1.6");
        if (and_row->right && out->right) {
            printf("[ Here : %s ]\n", "7.1.7");
            if (out->right->right_index < and_row->right->right_index) {
                printf("[ Here : %s ]\n", "7.1.8");
                InOutNode *tmp = in_out_create(1);
                in_out_push(out->right, tmp);
                tmp->right_index = and_row->right->right_index;
                out->value = and_forward(and_row, x);
                out = out->right;
                and_row = and_row->right;
            } else if (and_row->right->right_index < out->right->right_index) {
                printf("[ Here : %s ]\n", "7.1.9");
                in_out_drop(out);
            } else {
                printf("[ Here : %s ]\n", "7.1.10");
                assert(out->right_index == and_row->right_index);
                out->value = and_forward(and_row, x);
                out = out->right;
                and_row = and_row->right;
            }
        } else if (out->right && !and_row->right) {
            printf("[ Here : %s ]\n", "7.1.11");
            in_out_drop(out);
        } else if(!out->right && and_row->right) {
            printf("[ Here : %s ]\n", "7.1.12");
            InOutNode *tmp = in_out_create(1);
            in_out_push(out, tmp);
            tmp->right_index = and_row->right_index;
            out->value = and_forward(and_row, x);
            out = out->right;
            and_row = and_row->right;
            assert(out && and_row);
        } else {
            printf("[ Here : %s ]\n", "7.1.13");
            out->value = and_forward(and_row, x);
            assert(out->right_index == and_row->right_index);
            out->value = and_forward(and_row, x);
            out = out->right;
            and_row = and_row->right;
        }
    }
    assert(!and_row);
    assert(!out);
    and_print(l->and_row, "7.1.14, l->and_row");
    in_out_print(l->out, "7.1.14, l->out ");
    printf("[ Here : %s ]\n", "7.1.14");
}

/* Returns pointer to the last layer's activation matrix. */
InOutNode *network_forward(Network *net, const InOutNode *input, size_t input_size)
{
    const InOutNode *x = input;
    Layer *l = net->head;
    printf("[ Here : %s, input_size: %zu ]\n", "7.1", input_size);
    while (l) {
        layer_forward(l, x);
        assert(l->out);
        in_out_print(l->out, "Here 7.2");
        x = l->out;
        l = l->next;
    }
    return net->tail->out;
}

/* ════════════════════════════════════════════
   Loss: Mean Squared Error
   ════════════════════════════════════════════ */

/*
 * L = (1/n) Σ (pred[i] - target[i])²
 * dL/d(pred[i]) = (2/n)(pred[i] - target[i])
 *
 * dloss must be pre-allocated (same shape as pred).
 * Returns the scalar loss value.
 */
double network_loss_mse(const InOutNode *pred, const InOutNode *target, InOutNode *dloss)
{
    double loss = 0.0;
    const InOutNode *cur_p = pred;
    const InOutNode *cur_t = target;
    InOutNode *cur_dl = dloss;
    while(cur_p && cur_t && cur_dl) {
        double diff = cur_p->value - cur_t->value;
        loss += diff * diff;
        cur_dl->value = diff;
        cur_p = cur_p->right;
        cur_t = cur_t->right;
        cur_dl = cur_dl->right;
    }
    return loss;
}

/* ════════════════════════════════════════════
   Backward pass
   ════════════════════════════════════════════ */

static double compute_accums(
    const AndNode *node, const InOutNode *x, const InOutNode *d_node, size_t *count
) {
    printf("[ Here : %s, d_node index: %zu, d_node value %f ]\n", "8.1.13.1.1", d_node->right_index, d_node->value);
    OrNode *or_row = node->or_row;
    *count = 0;
    while (or_row) {
        *count += 1;
        or_row = or_row->right;
    }
    printf("[ Here : %s, count: %zu ]\n", "8.1.13.1.2", *count);
    double accums_or[*count];
    double or_values[*count];
    or_row = node->or_row;
    for (size_t i = 0; i < *count; i++) {
        or_values[i] = or_row->value;
        if (i) {
            printf("[ Here : %s, i: %zu, or_value[i - 1]: %zu ]\n", "8.1.13.1.4", i, or_values[i - 1]);
            accums_or[i] = accums_or[i - 1] * or_values[i - 1];
        } else {
            printf("[ Here : %s, i: %zu ]\n", "8.1.13.1.5", i);
            accums_or[0] = 1;
        }
        or_row = or_row->right;
    }
    double prod = 1;
    for (int i = *count - 1; i >= 0; i--) {
        if (i) {
            accums_or[i] = accums_or[i - 1] * prod;
            printf("[ Here : %s, i: %zu, prod: %zu ]\n", "8.1.13.1.6", i, prod);
        } else {
            printf("[ Here : %s, i: %zu, prod: %zu ]\n", "8.1.13.1.7", i, prod);
            accums_or[0] = prod;
        }
        prod *= or_values[i];
        or_row = or_row->right;
    }
    or_row = node->or_row;
    for (size_t i = 0; i < *count; i++) {
        printf("[ Here : %s, i: %zu, accums_or[i]: %f ]\n", "8.1.13.1.6", i, accums_or[i]);
        or_row->accum = accums_or[i];
        or_row = or_row->right;
    }

    // Compute norm
    const InOutNode *x_node = x;
    double norm = 0;
    for (size_t i = 0; i < *count; i++) {
        // For bias elements
        double elem = d_node->value * accums_or[i] ;
        printf("[ Here : %s, elem: %f ]\n", "8.1.13.1.7", elem);
        norm += elem * elem;
        while (x_node) {
            // For weight elements
            elem = d_node->value * accums_or[i] * x_node->value;
            norm += elem * elem;
            x_node = x_node->right;
        }
    }
    printf("[ Here : %s, norm: %f ]\n", "8.1.13.1.8", norm);
    return norm;
}

static void weight_backward(
    WeightNode *node, const InOutNode *x_node, double c_accum, double rate
) {
    double grad = c_accum * x_node->value;
    node->grad = grad;
    node->value += -rate * grad;
    printf("[ Here : %s, c_accum: %f, rate: %f, node->value: %f ]\n", "8.1.13.3.4.1", c_accum, rate, node->value);
    printf("[ Here : %s, node->value: %f ]\n", "8.1.13.3.4.2", node->value);
    if (fabs(node->value - 1) < node->quantization) {
        node->value = 1;
    }
    printf("[ Here : %s ]\n", "8.1.13.3.4.4");
}

static void or_backward(
    OrNode *node, const InOutNode *x,
    double c_accum, double rate, bool *is_fitted, bool *is_reserved
) {
    printf("[ Here : %s, c_accum: %f, rate: %f, node->bias.value: %f ]\n", "8.1.13.3.1", c_accum, rate, node->bias.value);
    node->bias.grad = c_accum;
    node->bias.value += -rate * c_accum;

    printf("[ Here : %s, node->bias.value: %f ]\n", "8.1.13.3.2", node->bias.value);
    if (fabs(node->bias.value - 1) < node->quantization) {
        printf("[ Here : %s ]\n", "8.1.13.3.3");
        node->bias.value = 1;
    }

    *is_reserved = node->bias.value == 1;
    *is_fitted = node->bias.value != 0;
    WeightNode *weight = node->weight;
    const InOutNode *x_node = x;
    in_out_print(x, "8.1.13.3.4, x");
    weight_print(weight, "8.1.13.3.4, weight");
    while (weight && x_node) {
        printf("[ Here : %s, weight->right_index: %zu, x_node->right_index: %zu ]\n", "8.1.13.3.4", weight->right_index, x_node->right_index);
        assert(weight->right_index == x_node->right_index);
        weight_backward(weight, x_node, c_accum, rate);
        assert(weight->right_index == x_node->right_index);
        *is_reserved = *is_reserved && weight->value == 0.0;
        *is_fitted = *is_fitted || weight->value != 1.0;
        weight = weight->right;
        x_node = x_node->right;
    }
    // printf("[ Here : %s, !x_node: %zu, weight: %zu, weight->value: %f, !weight->right: %zu, weight->right_index: %zu ]\n", "8.1.13.3.5", !x_node, weight, weight->value, !weight->right, weight->right_index);
    printf("[ Here : %s, x_node: %zu, weight: %zu ]\n", "8.1.13.3.5", x_node, weight);
    assert(!x_node);
    assert(!weight);
    // assert(weight->value == 0);
    // assert(!weight->right);
    // assert(!x_node && !weight);
}

static bool and_backward(
    AndNode *node, const InOutNode *x, const InOutNode *d_node
) {
    printf("[ Here : %s, d_node index: %zu, d_node value %f ]\n", "8.1.13.1", d_node->right_index, d_node->value);
    size_t in_size = 0;
    double norm = compute_accums(node, x, d_node, &in_size);
    printf("[ Here : %s, norm: %f ]\n", "8.1.13.2", norm);
    double rate;
    if (norm) {
        rate = d_node->value / norm / 2;
    } else {
        rate = rand();
    }

    size_t i = 0;
    OrNode *or_row = node->or_row;
    OrNode *prev = or_row;
    bool is_reserved = false;
    bool has_reserved = false;
    bool is_fitted = false;
    bool is_or_fitted = false;
    while (or_row) {
        printf("[ Here : %s, or_row: (%zu, %f), accum: %f, d_node: %f ]\n", "8.1.13.3", or_row->right_index, or_row->value, or_row->accum, d_node->value);
        or_backward(
            or_row, x, or_row->accum * d_node->value, rate, &is_or_fitted, &is_reserved
        );
        if (or_row->right && is_reserved && i > 1) {
            or_remove(prev);
            or_row = prev->right;
        } else {
            prev = or_row;
            or_row = or_row->right;
        }
        has_reserved |= is_reserved;
        is_fitted |= is_or_fitted;
        i += 1;
    }
    if (!has_reserved && false) {
        printf("[ Here : %s, in_size: %zu, %f ]\n", "8.1.13.4", in_size);
        OrNode *tmp = or_create(in_size, 2);
        or_init_weights(tmp, 0, 1, prev->quantization);
        or_push(prev, tmp);
    }
    printf("[ Here : %s, is_fitted: %f ]\n", "8.1.13.5", is_fitted);
    return is_fitted;
}

/*
 * Given dL/da (da) propagates gradients through one layer:
 *
 *   dz[i]   = da[i] * act'(z[i])
 *   dW[i,j] = dz[i] * x[j]            (accumulated)
 *   db[i]   = dz[i]                    (accumulated)
 *   dx[j]  += Σ_i W[i,j] * dz[i]      (passed to previous layer)
 *
 * x = the input that was fed into this layer during forward pass.
 */
static bool layer_backward(Layer *l, const InOutNode *dloss)
{
    in_out_print(dloss, "Here 8.1.0, dloss");
    const InOutNode *d_node = dloss;
    AndNode *and_row = l->and_row;
    size_t count = 0;
    bool is_fitted = false;
    printf("[ Here : %s, and index: %zu, d index: %zu ]\n", "8.1.1", and_row->right_index, d_node->right_index);
    and_print(l->and_row, "8.1.1");
    // ZIP IT
    assert(and_row);
    assert(d_node);
    while (and_row && d_node && and_row->right_index < d_node->right_index) {
        printf("[ Here : %s, and index: %zu, d index: %zu ]\n", "8.1.2", and_row->right_index, d_node->right_index);
        AndNode *tmp = l->and_row;
        l->and_row = l->and_row->right;
        free(tmp);
        and_row = l->and_row;
        if (!and_row) {
            // create one and_row for one output
            l->and_row = and_create(l->in_size, 1, 2);
            and_init_weights(l->and_row, l->and_row->quantization);
            l->and_row->right_index = d_node->right_index;
            is_fitted |= and_backward(tmp, l->in, d_node);
            and_row = l->and_row;
            d_node = d_node->right;
            count += 1;
        }
    }
    if (and_row && d_node && d_node->right_index < and_row->right_index) {
        printf("[ Here : %s, and index: %zu, d index: %zu, l->in_size: %f ]\n", "8.1.3", and_row->right_index, d_node->right_index, l->in_size);
        AndNode *tmp = l->and_row;
        // create one and_row for one output
        l->and_row = and_create(l->in_size, 1, 2);
        and_init_weights(l->and_row, l->and_row->quantization);
        l->and_row->right = tmp;
        l->and_row->right_index = d_node->right_index;
        is_fitted |= and_backward(tmp, l->in, d_node);
        and_row = l->and_row;
        d_node = d_node->right;
        count += 1;
    }
    while (and_row && d_node) {
        and_print(and_row, "8.1.3.1");
        printf("[ Here : %s, and index: %zu ]\n", "8.1.3.1", and_row->right_index);
        if (d_node->right && and_row->right) {
            printf("[ Here : %s, and index: %zu, d index: %zu ]\n", "8.1.4", and_row->right->right_index, d_node->right->right_index);
            if (and_row->right->right_index < d_node->right->right_index) {
                printf("[ Here : %s, l->in_size: %f ]\n", "8.1.5", l->in_size);
                // create one and_row for one output
                AndNode *tmp = and_create(l->in_size, 1, 2);
                printf("[ Here : %s ]\n", "8.1.6");
                and_init_weights(tmp, and_row->right->quantization);
                printf("[ Here : %s ]\n", "8.1.7");
                and_push(and_row->right, tmp);
                printf("[ Here : %s ]\n", "8.1.8");
                tmp->right_index = d_node->right->right_index;
                printf("[ Here : %s ]\n", "8.1.9");
                is_fitted |= and_backward(tmp, l->in, d_node);
                printf("[ Here : %s ]\n", "8.1.10");
                and_row = and_row->right;
                d_node = d_node->right;
                count += 1;
            } else if (d_node->right->right_index < and_row->right->right_index) {
                printf("[ Here : %s ]\n", "8.1.11");
                and_drop(and_row);
                printf("[ Here : %s ]\n", "8.1.12");
            } else {
                printf("[ Here : %s ]\n", "8.1.13");
                assert(and_row->right_index == d_node->right_index);
                is_fitted |= and_backward(and_row, l->in, d_node);
                printf("[ Here : %s ]\n", "8.1.14");
                and_row = and_row->right;
                d_node = d_node->right;
                count += 1;
            }
        } else if (and_row->right && !d_node->right) {
            printf("[ Here : %s ]\n", "8.1.17");
            and_drop(and_row);
        } else if (!and_row->right && d_node->right) {
            printf("[ Here : %s, l->in_size: %f ]\n", "8.1.19", l->in_size);
            // create one and_row for one output
            AndNode *tmp = and_create(l->in_size, 1, 2);
            and_init_weights(tmp, and_row->quantization);
            and_push(and_row, tmp);
            tmp->right_index = d_node->right->right_index;
            is_fitted |= and_backward(tmp, l->in, d_node);
            printf("[ Here : %s ]\n", "8.1.24");
            and_row = and_row->right;
            d_node = d_node->right;
            count += 1;
            assert(and_row && d_node);
        } else {
            is_fitted |= and_backward(and_row, l->in, d_node);
            printf("[ Here : %s ]\n", "8.1.24");
            assert(and_row->right_index == d_node->right_index);
            and_row = and_row->right;
            d_node = d_node->right;
            count += 1;
        }
    }
    printf("[ Here : %s ]\n", "8.1.25");
    assert(l->and_row);
    in_out_print(dloss, "8.1.25");
    in_out_print(l->in, "8.1.25");
    and_print(l->and_row, "8.1.25");
    printf("[ Here : %s ]\n", "8.1.25");
    l->out_size = count;

    InOutNode *in = l->in;
    printf("[ Here : %s, l->in: %zu ]\n", "8.1.26", l->in);
    in_out_free(l->din);
    l->din = NULL;
    InOutNode *din  = NULL;
    printf("[ Here : %s ]\n", "8.1.27");
    bool reserved_in_calculated = false;
    double reserved_dloss = 0;
    in_out_print(in, "8.1.27, in");
    assert(in);
    while (in) {
        printf("[ Here : %s ]\n", "8.1.28");
        d_node = dloss;
        and_row = l->and_row;
        double sum = 0;
        double useful_in = 0;
        while (d_node && and_row) {
            printf("[ Here : %s ]\n", "8.1.29");
            assert(d_node->right_index == and_row->right_index);
            double or_sum = 0;
            double or_sum_reserved = 0;
            OrNode *or_row = and_row->or_row;
            while (or_row) {
                printf("[ Here : %s, in->right_index: %zu ]\n", "8.1.30", in->right_index);
                WeightNode *weight = or_row->weight;
                bool collided = false;
                weight_print(weight, "8.1.30");
                while (weight) {
                    printf("[ Here : %s ]\n", "8.1.31");
                    if (in->right_index == weight->right_index) {
                        printf("[ Here : %s, weight->right_index: %zu, weight->value: %f ]\n", "8.1.33", weight->right_index, weight->value);
                        useful_in += fabs(weight->value);
                        or_sum += or_row->accum * weight->value;
                        collided = true;
                        break;
                    }
                    if (!reserved_in_calculated && !weight->right) {
                        printf("[ Here : %s ]\n", "8.1.34");
                        or_sum_reserved += or_row->accum;
                    }
                    weight = weight->right;
                }
                assert(collided);
                or_row = or_row->right;
            }
            sum += d_node->value * or_sum;
            printf("[ Here : %s ]\n", "8.1.35");
            if (!reserved_in_calculated) {
                printf("[ Here : %s, d_node: %f, or_sum_reserved: %f ]\n", "8.1.36", d_node->value, or_sum_reserved);
                reserved_dloss += d_node->value * or_sum_reserved;
            }
          
            d_node = d_node->right;
            and_row = and_row->right;
        }
        printf("[ Here : %s ]\n", "8.1.37");
        reserved_in_calculated = true;
        if (in->right) {
            printf("[ Here : %s, useful_in: %f ]\n", "8.1.38", useful_in);
            if (useful_in > 0.5) {
                printf("[ Here : %s ]\n", "8.1.39");
                din = in_out_insert(din);
                if (!l->din) {
                    l->din = din;
                }
                din->value = sum;
                din->right_index = in->right_index;
            }
        } else {
            printf("[ Here : %s, reserved_dloss: %f ]\n", "8.1.40", reserved_dloss);
            // The last input
            if (reserved_dloss > 0.25 || !l->din) {
                printf("[ Here : %s ]\n", "8.1.41");
                din = in_out_insert(din);
                if (!l->din) {
                    l->din = din;
                }
                din->value = reserved_dloss;
                din->right_index = in->right_index + 1;
            }
        }
        in = in->right;
    }
    assert(l->din);

    printf("[ Here : %s, l->in: %zu ]\n", "8.1.42", l->in);
    in_out_print(l->din, "8.1.42, l->din");
    return is_fitted;
}

/*
 * Full backward pass through all layers.
 * dloss: gradient of loss w.r.t. network output (shape = out × 1).
 */
void network_backward(Network *net, const InOutNode *dloss)
{
    const InOutNode *din = dloss;
    Layer *l = net->tail;
    assert(l);
    while (l) {
        printf("[ Here : %s, in_size: %zu, out_size: %zu, l->in: %zu ]\n", "8.1", l->in_size, l->out_size, l->in);
        bool is_fitted = layer_backward(l, din);
        printf("[ Here : %s, is_fitted: %d, l->in ]\n", "8.2", is_fitted, l->in);
        if (!is_fitted && false) { // disabled
            printf("[ Here : %s ]\n", "8.2.1");
            layer_drop(net, l);
            continue;
        }
        din = l->din;
        printf("[ Here : %s, l->in_size: %zu ]\n", "8.2.2", l->in_size);
        Layer *dummy_layer = layer_create(l->in_size, l->in_size);
        printf("[ Here : %s, is_fitted: %d, dummy_layer: %zu, in: %zu, out: %zu, din: %zu ]\n", "8.2.3", is_fitted, dummy_layer, dummy_layer->in, dummy_layer->out, dummy_layer->din);
        layer_init_weights(dummy_layer);
        printf("[ Here : %s, is_fitted: %d, dummy in: %zu, l->in: %zu ]\n", "8.3", is_fitted, dummy_layer->in, l->in);
        size_t in_size = 0;
        dummy_layer->in = in_out_update_with(dummy_layer->in, l->in, &in_size);
        assert(dummy_layer->in);
        printf("[ Here : %s, is_fitted: %d, l->in_size: %f ]\n", "8.4", is_fitted, l->in_size);
        assert(in_size == l->in_size);
        printf("[ Here : %s, is_fitted: %d ]\n", "8.5", is_fitted);
        is_fitted = layer_backward(dummy_layer, l->din);
        printf("[ Here : %s, is_fitted: %d ]\n", "8.7", is_fitted);
        if (is_fitted) {
            layer_insert(l, dummy_layer);
            printf("[ Here : %s, is_fitted: %d ]\n", "8.8", is_fitted);
            if (l == net->head) {
                printf("[ Here : %s, is_fitted: %d ]\n", "8.9", is_fitted);
                net->head = dummy_layer;
            }
            l->in_size = in_size;
            l = dummy_layer->prev;
        } else {
            l = l->prev;
        }
    }
    assert(net->head);
    assert(net->tail);
}

/* ════════════════════════════════════════════
   High-level training loop
   ════════════════════════════════════════════ */

/*
 * network_train – stochastic gradient descent, one sample at a time.
 *
 * X        : array of n_samples input  vectors of length `in`
 * Y        : array of n_samples output vectors of length `out`
 * n_samples: number of training samples
 * in / out : dimensionality
 * epochs   : number of full passes over data
 * lr       : learning rate
 */
void network_train(Network *net,
                   double **X, double **Y,
                   size_t n_samples,
                   size_t epochs)
{
    srand(34972);
    size_t in = net->in_size, out = net->out_size;
    printf("[ Here : %s, in: %zu ]\n", "1", net->in_size);
    InOutNode *x_mat = in_out_create(net->in_size);
    printf("[ Here : %s, out: %zu ]\n", "2", net->out_size);
    InOutNode *y_mat = in_out_create(net->out_size);
    printf("[ Here : %s, out: %zu ]\n", "3", net->out_size);
    InOutNode *dloss = in_out_create(net->out_size);
    printf("[ Here : %s ]\n", "4");

    for (size_t ep = 0; ep < epochs; ep++) {
        double total_loss = 0.0;

        for (size_t s = 0; s < n_samples; s++) {
            /* load sample into node matrices */
            InOutNode *node = x_mat;
            for (size_t i = 0; i < in;  i++) {
                printf("[ Here : %s, s: %zu, i: %zu, X[s][i]: %f ]\n", "5", s, i, X[s][i]);
                node->value = X[s][i];
                node = node->right;
            }
            in_out_print(node, "input");
            node = y_mat;
            for (size_t i = 0; i < out; i++) {
                printf("[ Here : %s, s: %zu, i: %zu, Y[s][i]: %f ]\n", "6", i, Y[s][i]);
                node->value = Y[s][i];
                node = node->right;
            }
            in_out_print(node, "output");
            printf("[ Here : %s ]\n", "7");

            /* ── forward ── */
            InOutNode *pred = network_forward(net, x_mat, in);
            in_out_print(pred, "pred");

            /* ── loss ── */
            total_loss += network_loss_mse(pred, y_mat, dloss);
            printf("[ Here : %s, total_loss: %f ]\n", "8", total_loss);

            /* ── backward ── */
            network_backward(net, dloss);
            printf("[ Here : %s\n", "9");
        }

        if ((ep + 1) % 100 == 0 || ep == 0) {
            printf("Epoch %5d / %d  |  MSE loss = %.6f\n",
                   ep + 1, epochs, total_loss / n_samples);
        }
    }

    in_out_free(x_mat);
    in_out_free(y_mat);
    in_out_free(dloss);
}

/* ════════════════════════════════════════════
   Single-sample inference helper
   ════════════════════════════════════════════ */

void network_predict(Network *net, double *x, size_t in,
                     double *out_buf, size_t out)
{
    InOutNode *x_mat = in_out_create(in);
    InOutNode *node = x_mat;
    for (size_t i = 0; i < in; i++) {
        node->value = x[i];
        node = node->right;
    }

    InOutNode *pred = network_forward(net, x_mat, in);
    node = pred;
    for (size_t i = 0; i < out; i++) {
        out_buf[i] = node->value;
        node = node->right;
    }

    in_out_free(x_mat);
}

