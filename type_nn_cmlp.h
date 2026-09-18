#ifndef TYPE_NN_CMLP_H
#define TYPE_NN_CMLP_H

#include "type_nn_alt.h"

/*
 * c-mlp — Linear(in, H) → ReLU → Linear(H, out) → ln tail
 *   H = 8 if in <= 4 else 16
 * Per-sample Adam, same (β1, β2, ε) and task lr as type-nn.
 * Tail y = sign(z) ln(1+|z|/τ) so MSE is on the type-nn coordinate.
 * Board baseline. Not a type-nn product net.
 */
AltNet type_nn_cmlp_open(size_t in, size_t out);

#endif
