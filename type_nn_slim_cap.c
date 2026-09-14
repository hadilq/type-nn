#include "type_nn_slim_cap.h"
#include "type_nn_grow.h"
void type_nn_slim_cap_apply(Network *net) { tnn_grow_apply(net, "slim-cap"); }
