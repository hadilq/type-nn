#ifndef NN_H
#define NN_H

#define _USE_MATH_DEFINES

#include <stddef.h>

/* ─────────────────────────────────────────────
   BiasNode
   ───────────────────────────────────────────── */
typedef struct BiasNode {
    double value;           /* forward-pass value    */
    double grad;            /* ∂L/∂value (backprop)  */
} BiasNode;

/* ─────────────────────────────────────────────
   WeightNode
   ───────────────────────────────────────────── */
typedef struct WeightNode {
    double value;             /* forward-pass value    */
    double grad;              /* ∂L/∂value (backprop)  */
    double quantization;      /* quantization level    */
    size_t right_index;       /* sums up with right    */
    struct WeightNode *right; /* right element         */
} WeightNode;

/* ─────────────────────────────────────────────
   OrNode
   ───────────────────────────────────────────── */
typedef struct OrNode {
    double value;              /* forward-pass value              */
    double grad;               /* ∂L/∂value (backprop)            */
    double quantization;       /* quantization level              */
    double accum;            /* accumulated and terms except one  */
    struct WeightNode *weight; /* right element                   */
    struct BiasNode bias;      /* right element                   */
    size_t right_index;        /* sums up with right              */
    struct OrNode *right;      /* right element                   */
} OrNode;

/* ─────────────────────────────────────────────
   AndNode
   ───────────────────────────────────────────── */
typedef struct AndNode {
    double value;          /* forward-pass value                */
    double grad;           /* ∂L/∂value (backprop)              */
    double quantization;   /* quantization level                */
    struct OrNode *or_row; /* right element                     */
    size_t right_index;    /* sums up with right                */
    struct AndNode *right; /* right element                     */
} AndNode;

/* ─────────────────────────────────────────────
   InOutNode
   ───────────────────────────────────────────── */
typedef struct InOutNode {
    double value;            /* forward-pass value    */
    double grad;             /* ∂L/∂value (backprop)  */
    size_t right_index;      /* sums up with right    */
    struct InOutNode *right; /* right element         */
} InOutNode;

/* ─────────────────────────────────────────────
   Layer
   ───────────────────────────────────────────── */
typedef struct Layer {
    size_t       in_size;
    size_t       out_size;
    AndNode      *and_row; /* and row           */
    InOutNode    *in;      /* in                */
    InOutNode    *out;     /* out               */
    InOutNode    *din;     /* ∂L/∂(prev layer a)    */
    struct Layer *next;
    struct Layer *prev;
} Layer;

/* ─────────────────────────────────────────────
   Network
   ───────────────────────────────────────────── */
typedef struct {
    size_t in_size;
    size_t out_size;
    Layer  *head;       /* first layer           */
    Layer  *tail;       /* last  layer           */
    size_t  depth;
} Network;

/* ── AndNode helpers ── */
void    and_print(const AndNode *node, const char *label);

/* ── Network lifecycle ── */
Network *network_create(size_t in, size_t out);
void     network_init_weights(Network *net);
void     network_predict(Network *net, double *x, size_t in, double *out_buf, size_t out);
void     network_free(Network *net);

/* ── Forward / inference ── */
InOutNode  *network_forward(Network *net, const InOutNode *input, size_t input_size);

/* ── Backward / training ── */
double   network_loss_mse(const InOutNode *pred, const InOutNode *target, InOutNode *dloss);
void     network_backward(Network *net, const InOutNode *dloss);

/* ── Convenience ── */
void     network_train(Network *net,
                       double **X, double **Y,
                       size_t n_samples, 
                       size_t epochs);
void     network_predict(Network *net, double *x, size_t in, double *out_buf, size_t out);

#endif /* NN_H */
