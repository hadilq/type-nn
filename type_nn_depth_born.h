#ifndef TYPE_NN_DEPTH_BORN_H
#define TYPE_NN_DEPTH_BORN_H

#include "type_nn.h"

/*
 * depth-born: hidden exists before weight init (random, MLP-shaped).
 * Kept for the whole run. This is the "start deep like c-mlp" control.
 */
void type_nn_depth_born_apply(Network *net);

#endif
