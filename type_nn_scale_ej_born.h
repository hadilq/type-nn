#ifndef TYPE_NN_SCALE_EJ_BORN_H
#define TYPE_NN_SCALE_EJ_BORN_H

#include "type_nn.h"

/*
 * All three scale gates (energy AND jac on Or and And) plus a random
 * hidden born before init and kept. Closest type-nn cousin of c-mlp.
 */
void type_nn_scale_ej_born_apply(Network *net);

#endif
