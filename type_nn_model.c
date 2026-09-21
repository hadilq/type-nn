#include "type_nn_model.h"
#include "type_nn_grow.h"
#include "type_nn_layer.h"
#include "type_nn_ln.h"

void type_nn_model_apply(Network *net)
{
    if (!net) return;
    net->dynamic = 1;
    tnn_ln_apply(net, "ln-v2");
    tnn_ln_apply(net, "adam");
    /* Pulse: dummy-Or And-scale, dummy-width Or-scale, signal cut.
       Ldepth births floor(1+ln(n m)) typed products of width m
       — never n→n — and inserts a typed layer between any two
       on a loud residual. */
    tnn_grow_apply(net, "scale-pulse");
    tnn_layer_apply(net, "Ldepth");
    /* Grow the first 30%, fit through mid, drop only in the last 30%. */
    net->sched_grow = 0.30;
    net->sched_cut  = 0.70;
}
