#ifndef TYPE_NN_SCALE_SCHED_H
#define TYPE_NN_SCALE_SCHED_H
#include "type_nn.h"
void type_nn_scale_sched_apply(Network *net);
void type_nn_scale_sched_tight_apply(Network *net);
void type_nn_scale_sched_wide_apply(Network *net);
#endif
