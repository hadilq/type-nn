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
    tnn_grow_apply(net, "scale-pulse");
    tnn_layer_apply(net, "Ldepth");
    /* Grow the first 30%, fit through mid, drop only in the last 30%. */
    net->sched_grow = 0.30;
    net->sched_cut  = 0.70;
}
