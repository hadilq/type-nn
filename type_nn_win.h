#ifndef TYPE_NN_WIN_H
#define TYPE_NN_WIN_H

#include "type_nn_alt.h"

/*
 * type-nn-win = F + H + site-local refuse.
 *   F    revert a depth insert that does not drop EMA
 *   H    grow k=1 width toward rank(in); stop when ge goes flat
 *   rank unused column rank is spent before adding depth
 *   no CE (G: Ands are not logits)
 */

AltNet type_nn_win_open(size_t in, size_t out);
AltNet type_nn_A_open(size_t in, size_t out);
AltNet type_nn_B_open(size_t in, size_t out);
AltNet type_nn_C_open(size_t in, size_t out);
AltNet type_nn_D_open(size_t in, size_t out);
AltNet type_nn_E_open(size_t in, size_t out);
AltNet type_nn_F_open(size_t in, size_t out);
AltNet type_nn_G_open(size_t in, size_t out);
AltNet type_nn_H_open(size_t in, size_t out);
AltNet type_nn_I_open(size_t in, size_t out);

#endif
