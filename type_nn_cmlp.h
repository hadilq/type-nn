#ifndef TYPE_NN_CMLP_H
#define TYPE_NN_CMLP_H

#include "type_nn_alt.h"

/*
 * c-mlp — Linear(in, H) → ReLU → Linear(H, out)
 *   H = 8 if in <= 4 else 16
 * Full-batch-equivalent Adam with per-sample lr * 0.15.
 * Board baseline. Not a type-nn product net.
 */
AltNet type_nn_cmlp_open(size_t in, size_t out);

#endif
