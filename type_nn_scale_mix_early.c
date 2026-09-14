#include "type_nn_scale_mix_early.h"
#include "type_nn_scale_mix.h"
#include "type_nn_depth_early.h"

void type_nn_scale_mix_early_apply(Network *net)
{
    type_nn_scale_mix_apply(net);
    type_nn_depth_early_apply(net);
}
