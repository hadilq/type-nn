#include "type_nn_depth_hold.h"
#include "type_nn_layer.h"

void type_nn_depth_hold_apply(Network *net)
{
    tnn_layer_apply(net, "depth-hold");
}
