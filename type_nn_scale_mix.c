#include "type_nn_scale_mix.h"
#include "type_nn_grow.h"

void type_nn_scale_mix_apply(Network *net)
{
    tnn_grow_apply(net, "scale-mix");
}
