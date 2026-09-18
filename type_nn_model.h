#ifndef TYPE_NN_MODEL_H
#define TYPE_NN_MODEL_H

#include "type_nn.h"

/*
 * The board type-nn. Faithful recipe for the three scaling problems:
 *
 *   Or     grow the previous layer's output by one dummy coordinate;
 *          the current layer pairs it with a dummy weight. Drop a
 *          dummy coordinate late.
 *   And    keep one dummy Or (×1) per product. If back-prop gives it
 *          weight, append a new dummy Or. If two dummy Ors remain,
 *          drop one.
 *   Depth  birth floor(1+ln(n m)) typed-product layers. Insert a
 *          layer between whichever pair is loudest. Drop an identity
 *          hidden late.
 *
 * Every layer emits y = sign(z) ln(1+|z|/τ). Grow early, drop late.
 */
void type_nn_model_apply(Network *net);

#endif
