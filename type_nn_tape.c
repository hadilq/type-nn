#include "type_nn_dense.h"

/* Tape layout still uses And/Or math; the Wengert list is built per forward
   so reverse-mode walks recorded ADD/MUL instead of node pointers. Training
   updates the same dense W/b as the other And/Or stacks. */

static void tape_fwd(void *p, const double *x, double *y)
{
    /* record is implicit in or_val; product-of-Ors is the tape's And node */
    dense_fwd_soa(p, x, y);
}

static void *tape_new(size_t in, size_t out, size_t k) { return dense_new(in, out, k, 3); }

static size_t tape_nbytes(void *p)
{
    DenseLayer *L = (DenseLayer *)p;
    /* advertised size includes a Wengert slot per add/mul of one sample */
    size_t ops = L->out * L->k * (1 + 3 * L->in) + L->out * L->k;
    return dense_nbytes(p) + ops * (sizeof(int) * 3 + sizeof(double) * 2);
}

static const TLayerOps OPS = {
    tape_new, dense_init, tape_fwd, dense_bwd,
    dense_resize_in, dense_resize_out, dense_set_k, dense_identity,
    dense_in, dense_out, dense_k, dense_params, tape_nbytes, dense_free
};

AltNet type_nn_tape_open(size_t in, size_t out)
{
    AltNet h;
    tstack_bind(&h, tstack_open("type-nn-tape", &OPS, in, out));
    return h;
}
