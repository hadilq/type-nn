#include "type_nn_scale_energy_hold.h"
#include "type_nn_scale_energy.h"
#include "type_nn_depth_hold.h"

void type_nn_scale_energy_hold_apply(Network *net)
{
    type_nn_scale_energy_apply(net);
    type_nn_depth_hold_apply(net);
}
