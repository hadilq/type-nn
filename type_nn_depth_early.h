#ifndef TYPE_NN_DEPTH_EARLY_H
#define TYPE_NN_DEPTH_EARLY_H

#include "type_nn.h"

/*
 * depth-early: identity hidden at the first train step, wide like c-mlp,
 * drop blocked until late in the schedule. Combine with a scale recipe:
 *   network_set_andpol(net, "scale-mix+depth-early");
 */
void type_nn_depth_early_apply(Network *net);

#endif
