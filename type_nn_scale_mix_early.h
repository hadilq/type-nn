#ifndef TYPE_NN_SCALE_MIX_EARLY_H
#define TYPE_NN_SCALE_MIX_EARLY_H

#include "type_nn.h"

/* scale-mix + depth-early: energy Or, jac And, identity hidden at step 0. */
void type_nn_scale_mix_early_apply(Network *net);

#endif
