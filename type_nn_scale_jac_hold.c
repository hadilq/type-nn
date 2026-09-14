#include "type_nn_scale_jac_hold.h"
#include "type_nn_scale_jac.h"
#include "type_nn_depth_hold.h"

void type_nn_scale_jac_hold_apply(Network *net)
{
    type_nn_scale_jac_apply(net);
    type_nn_depth_hold_apply(net);
}
