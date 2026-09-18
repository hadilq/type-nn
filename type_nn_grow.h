#ifndef TYPE_NN_GROW_H
#define TYPE_NN_GROW_H

#include "type_nn.h"

/*
 * Back-prop-only Or / And spawn. No max_or, no max_and.
 *
 * The only occupancy rule is the dummy rule: at most one identity
 * Or per And, at most one identity And per output. Live factors are
 * unbounded. A new dummy is born only when the gate says the current
 * Jacobian cannot explain the incoming residual.
 *
 * Iteration 4 (count caps retired on the scale path):
 *   scale-keep    always replace a missing dummy
 *   scale-off     never spawn
 *   scale-resid   |incoming| > T
 *   scale-ratio   |incoming| > κ · max|child grad|
 *   scale-dead    children still and incoming large
 *   scale-or      ratio on Or only
 *   scale-and     ratio on And only
 *   scale-refuse  ratio + layer insert when probes refused
 *
 * Iteration 5 (width from BP statistics only):
 *   scale-energy  |parent| > κ · Σ|live child grad|
 *   scale-jac     |parent| > T and Σ|∂L/∂W|_live < κ_j · |parent|
 *   scale-slack   |parent| > T and every live child is specialized
 *   scale-sign    |parent| > T and live child grads fight / miss sign
 *   scale-mix     energy on Or, jac on And
 *   scale-ej      energy AND jac on both axes
 *   scale-layer   scale-ej + Ljac (refuse + stuck + residual)
 *
 * Combine: ln-v2w+scale-energy
 */

#define TNN_G_OR_KEEP    1u
#define TNN_G_OR_RESID   2u
#define TNN_G_OR_RATIO   4u
#define TNN_G_OR_DEAD    8u
#define TNN_G_AND_KEEP   16u
#define TNN_G_AND_RESID  32u
#define TNN_G_AND_RATIO  64u
#define TNN_G_AND_DEAD   128u
#define TNN_G_REFUSE     256u
#define TNN_G_OR_ENERGY  512u
#define TNN_G_AND_ENERGY 1024u
#define TNN_G_OR_JAC     2048u
#define TNN_G_AND_JAC    4096u
#define TNN_G_OR_SLACK   8192u
#define TNN_G_AND_SLACK  16384u
#define TNN_G_OR_SIGN    32768u
#define TNN_G_AND_SIGN   65536u
#define TNN_G_CAP        131072u  /* max_or stays TNN_G_CAP_OR */
#define TNN_G_PRUNE      262144u  /* drop |w| < TNN_G_PRUNE_T */
#define TNN_G_TOPK       524288u  /* keep TNN_G_TOPK_N largest |w| */
#define TNN_G_SCHED      1048576u /* early grow / late cut from u */
#define TNN_G_PHASE      2097152u /* three-phase Or/And: grow, cut, shrink */
#define TNN_G_SIGNAL     4194304u /* prune / identity band from forward signal */
#define TNN_G_NORM       8388608u /* gates on mean Jacobian / RMS residual */
#define TNN_G_ASSEMBLE   16777216u /* raise a before adding an And */
#define TNN_G_COMPOSE    33554432u /* extra degree asks for a layer instead */
#define TNN_G_FREE       67108864u /* no count cap; spawn from BP + signal */
#define TNN_G_CLOCK      134217728u /* residual-EMA grow/cut (wine/wdbc) */
#define TNN_G_WIDTH      268435456u /* Or-scale: grow/drop previous-layer width */

#define TNN_G_PRUNE_T  0.05
#define TNN_G_TOPK_N   3
#define TNN_G_CAP_OR   2

/* Phase of the schedule: 0 grow, 1 cut, 2 shrink. */
int    tnn_grow_phase(const Network *net);
double tnn_grow_signal_or(const OrNode *o);
double tnn_grow_signal_and(const AndNode *a);
/* Weight / identity threshold. Stronger signal → lower thresh (keep). */
double tnn_grow_thresh(const Network *net, double signal);

#define TNN_G_T      0.30
#define TNN_G_K      2.0
#define TNN_G_DEAD   1e-4
#define TNN_G_JAC_K  0.25
#define TNN_G_SPEC   0.50

void tnn_grow_bind(Network *net);
int  tnn_grow_apply(Network *net, const char *name);
int  tnn_grow_on(const Network *net);
int  tnn_grow_want_or(AndNode *a, double d_and);
int  tnn_grow_want_and(const Layer *l, size_t index, double d_z);
int  tnn_grow_probes_refused(const Layer *l, size_t index);

#endif
