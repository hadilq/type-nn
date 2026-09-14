#include "type_nn_scale_mix_hold.h"
#include "type_nn_scale_mix.h"
#include "type_nn_depth_hold.h"

void type_nn_scale_mix_hold_apply(Network *net)
{
    type_nn_scale_mix_apply(net);
    type_nn_depth_hold_apply(net);
}
