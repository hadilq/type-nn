#ifndef TYPE_NN_H
#define TYPE_NN_H

#define _USE_MATH_DEFINES

#include <stddef.h>
#include <stdbool.h>

/* ─────────────────────────────────────────────
   BiasNode
   ───────────────────────────────────────────── */
typedef struct BiasNode {
    double value;           /* forward-pass value    */
    double grad;            /* ∂L/∂value (backprop)  */
    double m, v;            /* Adam first / second moment */
} BiasNode;

/* ─────────────────────────────────────────────
   WeightNode  – sparse weight list, sorted by right_index
   ───────────────────────────────────────────── */
typedef struct WeightNode {
    double value;             /* forward-pass value    */
    double grad;              /* ∂L/∂value (backprop)  */
    double m, v;              /* Adam first / second moment */
    double quantization;      /* snap / prune threshold */
    size_t right_index;       /* feature index         */
    struct WeightNode *right; /* next weight           */
} WeightNode;

/* ─────────────────────────────────────────────
   OrNode  – linear unit: bias + Σ w_i x_i
   ───────────────────────────────────────────── */
typedef struct OrNode {
    double value;              /* forward-pass value              */
    double grad;               /* ∂L/∂value (backprop)            */
    double quantization;       /* quantization level              */
    double accum;              /* Π of sibling Or values          */
    struct WeightNode *weight; /* sparse weights                  */
    struct BiasNode bias;
    size_t right_index;
    int    cool_left;          /* samples this Or still claims as “busy” */
    double cut;                /* tail snap after BP (annealed in orcool) */
    struct OrNode *right;
} OrNode;

/* ─────────────────────────────────────────────
   AndNode  – product of OrNodes (one output unit)
   ───────────────────────────────────────────── */
typedef struct AndNode {
    double value;
    double grad;
    double quantization;
    double expn;               /* assembly index a_{i,r} > 0; z = Π A^a */
    double expn_grad;          /* ∂L/∂a_{i,r} */
    double expn_m, expn_v;     /* Adam moments on the assembly index */
    double and_gate;           /* ∂λ̃/∂λ for log-space And product */
    struct OrNode *or_row;
    size_t right_index;
    int probe_cool;
    struct AndNode *right;
} AndNode;

/* ─────────────────────────────────────────────
   InOutNode  – sparse activation / gradient list
   ───────────────────────────────────────────── */
typedef struct InOutNode {
    double value;
    double grad;
    size_t right_index;
    struct InOutNode *right;
} InOutNode;

/* ─────────────────────────────────────────────
   Layer
   ───────────────────────────────────────────── */
typedef struct Layer {
    size_t       in_size;
    size_t       out_size;
    AndNode      *and_row;
    InOutNode    *in;
    InOutNode    *out;
    InOutNode    *din;
    /* Per-head tail scale τ_i. Learned from ∂L/∂τ on ln models
       (no √d). Hidden layers leave these unused. */
    double      *tau;
    double      *tau_m;
    double      *tau_v;
    double      *tau_g;
    size_t       tau_n;
    struct Layer *next;
    struct Layer *prev;
} Layer;

/* ─────────────────────────────────────────────
   Network
   ───────────────────────────────────────────── */
typedef struct {
    size_t in_size;
    size_t out_size;
    Layer  *head;
    Layer  *tail;
    size_t  depth;
    double  lr;
    int     dynamic;     /* Or / And probes */
    int     layer_probe; /* 0 = no automatic layer insert (original) */
    size_t  max_depth;
    size_t  max_or;
    double  quantization;
    int     verbose;
    unsigned or_add, or_drop;
    unsigned and_add, and_drop;
    unsigned layer_add, layer_drop;
    int      orcool;           /* prefer Or training: freeze dummy And while Ors cool */
    unsigned orcool_step;
    unsigned orcool_span;      /* steps over which cool/cut anneal; 0 → n*epochs */
    unsigned andpol;           /* And-preference bits; see network_set_andpol */
    unsigned lnpol;            /* type-nn-ln recipe bits; see type_nn_ln.h */
    unsigned layerpol;         /* hidden-layer insert/drop; type_nn_layer.h */
    unsigned growpol;          /* Or/And spawn gates; type_nn_grow.h */
    double   last_dloss_l1;    /* ||y-t||_1 of the sample just backwarded */
    unsigned long adam_t;      /* Adam step index (1-based while training) */
    double   adam_b1p, adam_b2p;
    double   sched_grow;     /* u < grow: scale Or/And/Layer up */
    double   sched_cut;      /* u > cut: refuse spawn, prune */
} Network;

#define TNN_AP_BUDGET 2u
#define TNN_AP_STUCK  4u
#define TNN_AP_ANDTAU 64u
#define TNN_AP_TENS   512u
#define TNN_AP_TSGD   1024u
#define TNN_AP_LOG    2048u

/* ── print helpers ── */
void and_print(const AndNode *node, const char *label);
void in_out_print(const InOutNode *node, const char *label);
void weight_print(const WeightNode *node, const char *label);
void or_print(const OrNode *node, const char *label);

/* ── Network lifecycle ── */
Network *network_create(size_t in, size_t out);
void     network_init_weights(Network *net);
void     network_free(Network *net);
void     network_add_layer(Network *net, size_t in, size_t out);
void     network_set_learning_rate(Network *net, double lr);
void     network_set_dynamic(Network *net, int enabled);
void     network_set_layer_probe(Network *net, int enabled);
void     network_set_verbose(Network *net, int enabled);
void     network_set_orcool(Network *net, int enabled);
void     network_set_orcool_span(Network *net, unsigned span);
void     network_set_andpol(Network *net, const char *name);
void     network_set_layerpol(Network *net, const char *name);
size_t   network_depth(const Network *net);
double   network_progress(const Network *net); /* u in [0,1] */

/* Insert an identity hidden layer in front of `at` (NULL = before tail). */
Layer   *network_insert_identity(Network *net, Layer *at);
/* Remove a hidden layer. Returns 0 on success, -1 if refused. */
int      network_remove_layer(Network *net, Layer *node);

/* Grow / shrink the sparse structure of a layer to match sizes. */
void     layer_align_inputs(Layer *l, size_t in_size);
void     layer_set_outputs(Layer *l, size_t out_size);

/* ── Forward / inference ── */
InOutNode *network_forward(Network *net, const InOutNode *input, size_t input_size);
void       network_predict(Network *net, double *x, size_t in, double *out_buf, size_t out);

/* ── Backward / training ── */
double   network_loss_mse(const InOutNode *pred, const InOutNode *target, InOutNode *dloss);
void     network_backward(Network *net, const InOutNode *dloss);

void     network_train(Network *net,
                       double **X, double **Y,
                       size_t n_samples,
                       size_t epochs);

/* Count helpers (useful for tests). */
size_t inout_count(const InOutNode *n);
size_t weight_count(const WeightNode *n);
size_t or_count(const OrNode *n);
size_t and_count(const AndNode *n);
size_t network_param_count(const Network *net);
size_t network_nbytes(const Network *net);

#endif /* TYPE_NN_H */
