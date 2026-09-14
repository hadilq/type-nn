#include "type_nn_scale_jac_early.h"
#include "type_nn_scale_jac.h"
#include "type_nn_depth_early.h"

void type_nn_scale_jac_early_apply(Network *net)
{
    type_nn_scale_jac_apply(net);
    type_nn_depth_early_apply(net);
}
