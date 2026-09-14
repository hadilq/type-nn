#ifndef TYPE_NN_SCALE_JAC_H
#define TYPE_NN_SCALE_JAC_H

#include "type_nn.h"

/* scale-jac: spawn iff residual large and live |∂L/∂W| is small. */
void type_nn_scale_jac_apply(Network *net);

#endif
