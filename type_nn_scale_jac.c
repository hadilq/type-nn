#include "type_nn_scale_jac.h"
#include "type_nn_grow.h"

void type_nn_scale_jac_apply(Network *net)
{
    tnn_grow_apply(net, "scale-jac");
}
