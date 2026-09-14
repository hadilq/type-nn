#ifndef TYPE_NN_DEPTH_HOLD_H
#define TYPE_NN_DEPTH_HOLD_H

#include "type_nn.h"

/*
 * depth-hold: insert on a large residual without waiting for Or/And
 * caps. Drop only after 65% of the train span, and only if the hidden
 * is identity with ||dW|| ≈ 0. Never drop back to depth 1.
 */
void type_nn_depth_hold_apply(Network *net);

#endif
