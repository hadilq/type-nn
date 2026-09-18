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
#define TNN_LP_EARLY   2048u   /* insert a hidden as soon as training starts */
#define TNN_LP_HOLD    4096u   /* refuse drop until late in the schedule */
#define TNN_LP_BORN    8192u   /* hidden exists before init (random, MLP-like) */
#define TNN_LP_WIN     16384u  /* winner: 6-wide born hidden */
#define TNN_LP_SCHED   32768u  /* early insert / late drop from u */
#define TNN_LP_PHASE   65536u  /* grow/cut/shrink depth with the Or/And clock */
#define TNN_LP_LINEAR  131072u /* hidden stays degree-1 (typed linear map) */
#define TNN_LP_COMPOSE 262144u /* insert when grow asked for depth */
#define TNN_LP_DEPTH   524288u /* grow: sparse identity; shrink: drop it */
#define TNN_LP_KEEP_DW  1e-3   /* identity hidden with tinier ||dW|| drops */
#define TNN_LP_HOLD_DW  1e-5   /* HOLD: only drop a truly frozen identity */
#define TNN_LP_HOLD_U   0.65   /* HOLD: no drop while step/span < 0.65 */

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
/* Insert the first hidden now (BORN: before init; EARLY: after init).
   DEPTH: stack to 1+ln(n m) sparse identities after tail init. */
void     tnn_layer_birth(Network *net);
/* Floor(1 + ln(n m)), at least 1. n = in dim, m = out dim. */
int      tnn_layer_init_depth(size_t in, size_t out);
/* Or-scale: dummy output coordinate on a hidden layer, paired with
   dummy incoming weights on the next layer. Grow early, drop late. */
void     tnn_layer_width_step(Network *net);

#endif
