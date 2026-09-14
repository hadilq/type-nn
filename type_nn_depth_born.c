#include "type_nn_depth_born.h"
#include "type_nn_layer.h"

void type_nn_depth_born_apply(Network *net)
{
    tnn_layer_apply(net, "depth-born");
}
