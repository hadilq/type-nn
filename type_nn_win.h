#ifndef TYPE_NN_WIN_H
#define TYPE_NN_WIN_H

#include "type_nn_alt.h"

/*
 * Dynamic Type-NN. Shape comes from back-prop energy, not a
 * per-file recipe.
 *
 *   Or_{i,r} = clip_O(b_{i,r} + Σ_j W_{i,r,j} x_j)
 *   And_i    = clip_A(Π_r Or_{i,r})
 *
 * Starts as one product layer (k = 2). After a short warmup, at most
 * one add / drop / widen / extra-factor per settle window.
 */

AltNet type_nn_win_open(size_t in, size_t out);

#endif
