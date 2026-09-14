#include "type_nn_winner.h"
#include "type_nn_grow.h"
#include "type_nn_layer.h"
#include "type_nn_ln.h"

void type_nn_winner_seed(Network *net) { (void)net; }

void type_nn_winner_apply(Network *net)
{
    if (!net) return;
    net->dynamic = 1;
    tnn_ln_apply(net, "ln-v2");
    tnn_grow_apply(net, "scale-sched-bal");
    tnn_layer_apply(net, "Lsched");
}
