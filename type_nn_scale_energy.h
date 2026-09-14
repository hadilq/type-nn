#ifndef TYPE_NN_SCALE_ENERGY_H
#define TYPE_NN_SCALE_ENERGY_H

#include "type_nn.h"

/* scale-energy: spawn dummy Or/And iff |parent| > κ · Σ|live child grad|. */
void type_nn_scale_energy_apply(Network *net);

#endif
