#include "type_nn_scale_ej_born.h"
#include "type_nn_grow.h"
#include "type_nn_depth_born.h"

void type_nn_scale_ej_born_apply(Network *net)
{
    tnn_grow_apply(net, "scale-ej");
    type_nn_depth_born_apply(net);
}
