#include "type_nn_scale_sched.h"
#include "type_nn_grow.h"
#include "type_nn_layer.h"
#include "type_nn_ln.h"

static void sched_base(Network *net, const char *g)
{
    net->dynamic = 1;
    tnn_ln_apply(net, "ln-v2");
    tnn_grow_apply(net, g);
    tnn_layer_apply(net, "Lsched");
}

void type_nn_scale_sched_apply(Network *net)
{
    sched_base(net, "scale-sched");
}

void type_nn_scale_sched_tight_apply(Network *net)
{
    sched_base(net, "scale-sched-tight");
}

void type_nn_scale_sched_wide_apply(Network *net)
{
    sched_base(net, "scale-sched-wide");
}
