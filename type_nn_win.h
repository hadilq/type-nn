#ifndef TYPE_NN_WIN_H
#define TYPE_NN_WIN_H

#include "type_nn_alt.h"

/*
 * Standalone Type-NN winner (A+D+E+K).
 *
 * Does not compile type_nn_par.c / over / site / opt.
 * Public entry: type_nn_win_open.
 *
 *   Or_{i,r} = clip_O(b_{i,r} + Σ_j W_{i,r,j} x_j)
 *   And_i    = clip_A(Π_r Or_{i,r})
 *
 * Dynamic policy (in < 4 stays one product layer):
 *   tick 16: insert k=1 basis before a product on raw x,
 *            insert k=1 readout after a product tail, lock depth
 *   prune |W| < 0.03
 *   drop an Or row only if k > 2 and ||W||² is tiny
 *   grow the k=1 basis width from 2 while ge is high (cap 8, or 12 if in≥20)
 */

AltNet type_nn_win_open(size_t in, size_t out);

#endif
