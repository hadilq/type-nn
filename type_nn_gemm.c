#include "type_nn_dense.h"

static void *gemm_new(size_t in, size_t out, size_t k) { return dense_new(in, out, k, 1); }

static const TLayerOps OPS = {
    gemm_new, dense_init, dense_fwd_gemm, dense_bwd,
    dense_resize_in, dense_resize_out, dense_set_k, dense_identity,
    dense_in, dense_out, dense_k, dense_params, dense_nbytes, dense_free
};

AltNet type_nn_gemm_open(size_t in, size_t out)
{
    AltNet h;
    tstack_bind(&h, tstack_open("type-nn-gemm", &OPS, in, out));
    return h;
}
