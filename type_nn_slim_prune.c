#include "type_nn_slim_prune.h"
#include "type_nn_grow.h"
void type_nn_slim_prune_apply(Network *net) { tnn_grow_apply(net, "slim-prune"); }
