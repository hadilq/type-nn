#ifndef TYPE_NN_LAYER_H
#define TYPE_NN_LAYER_H

#include "type_nn.h"

/*
 * Hidden-layer insert / drop. Same dummy rule as Or and And:
 *
 *   1. Insert is an identity map, so the function does not jump.
 *   2. Back-prop is allowed to move that identity.
 *   3. After back-prop, drop only if the layer is identity AND
 *      ||∂L/∂W||_1 ≈ 0 (the Jacobian does not want it to move).
 *
 * No wall-clock.
 *
 * Iteration 3 gate — capacity exhausted (Lcap):
 *
 *   insert iff
 *     (a) tail residual ||y−t||_1 is still large
 *     (b) every output head is at its And cap and every live And
 *         is at its Or cap, with no dummy Or/And left to leave 1
 *     (c) ||∂L/∂W||_1 on the tail is small vs the residual
 *         (moving W will not cut the residual)
 *
 * Lfull drops (c). Lstuck drops (b). Those are controls so the
 * board can say which clause did the work.
 *
 * Combine with a numeric recipe via '+':  ln-v2w+Lcap
 */

#define TNN_LP_GRAD    1u
#define TNN_LP_RESID   2u
#define TNN_LP_RATIO   4u
#define TNN_LP_DUMMY   8u
#define TNN_LP_DROP    16u
#define TNN_LP_WIDE    32u
#define TNN_LP_NARROW  64u
#define TNN_LP_MULTI   128u
#define TNN_LP_CAP     256u    /* tail Or/And lists are full */
#define TNN_LP_STUCK   512u    /* ||dW|| small vs residual   */
#define TNN_LP_REFUSE  1024u   /* Or and And dummies refused to leave 1 */
#define TNN_LP_KEEP_DW  1e-3   /* identity hidden with tinier ||dW|| drops */

#define TNN_LP_GRAD_T    3.0
#define TNN_LP_RESID_T   1.0
#define TNN_LP_RATIO_K   2.0
#define TNN_LP_STUCK_K   0.25  /* ||dW||_1 < K · ||y−t||_1  */
#define TNN_LP_ID_BAND   0.2
#define TNN_LP_MAX_AND   4     /* matches type_nn.c DEFAULT_MAX_AND */

int      tnn_layer_apply(Network *net, const char *name);
int      tnn_layer_on(const Network *net);
int      tnn_layer_is_identity(const Layer *l);
int      tnn_layer_at_cap(const Layer *l, size_t max_or);
double   tnn_layer_l1(const InOutNode *n);
double   tnn_layer_dw_l1(const Layer *l);
void     tnn_layer_step(Network *net);

#endif
