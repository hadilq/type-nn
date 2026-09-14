#include "type_nn_scale_energy.h"
#include "type_nn_grow.h"

void type_nn_scale_energy_apply(Network *net)
{
    tnn_grow_apply(net, "scale-energy");
}
