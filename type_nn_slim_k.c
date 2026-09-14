#include "type_nn_slim_k.h"
#include "type_nn_grow.h"
void type_nn_slim_k_apply(Network *net) { tnn_grow_apply(net, "slim-k"); }
