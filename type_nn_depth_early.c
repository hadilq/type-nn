#include "type_nn_depth_early.h"
#include "type_nn_layer.h"

void type_nn_depth_early_apply(Network *net)
{
    tnn_layer_apply(net, "depth-early");
}
