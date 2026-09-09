#ifndef TYPE_NN_ALT_H
#define TYPE_NN_ALT_H

#include <stddef.h>

/*
 * Alternate layouts of the Type Mechanics AND/OR net.
 * type_nn.h / type_nn.c are not compiled into these objects.
 *
 *   Or_{i,k} = b_{i,k} + Σ_j W[i,k,j] x_j     (sum-type)
 *   And_i    = Π_k Or_{i,k}                    (product-type)
 *
 * Every layout supports:
 *   - multiple And/Or layers
 *   - grow / shrink input width and output width
 *   - grow / shrink the number of Or factors
 *   - insert an identity hidden layer / remove a hidden layer
 */

typedef struct AltNet AltNet;

struct AltNet {
    const char *impl;
    void       *ctx;
    size_t      in, out;

    void   (*init)(void *ctx);
    void   (*forward)(void *ctx, const double *x, double *y);
    void   (*backward)(void *ctx, const double *x, const double *dy, double lr);

    void   (*align_inputs)(void *ctx, size_t in);
    void   (*set_outputs)(void *ctx, size_t out);
    void   (*set_or_factors)(void *ctx, size_t k);
    void   (*insert_identity)(void *ctx);
    int    (*remove_hidden)(void *ctx);
    void   (*set_dynamic)(void *ctx, int on);

    size_t (*depth)(void *ctx);
    size_t (*or_factors)(void *ctx);
    size_t (*param_count)(void *ctx);
    size_t (*nbytes)(void *ctx);
    void   (*free)(void *ctx);
    /* optional: scale layer idx and stitch neighbours (NULL if unsupported) */
    void   (*scale_layer)(void *ctx, size_t idx, size_t in, size_t out);
    size_t (*layer_in)(void *ctx, size_t idx);
    size_t (*layer_out)(void *ctx, size_t idx);
    size_t (*layer_k)(void *ctx, size_t idx);
    size_t (*n_add)(void *ctx);
    size_t (*n_drop)(void *ctx);
};

#define TNN_SNAP_MAX 8
typedef struct {
    size_t depth;
    size_t n_and[TNN_SNAP_MAX]; /* product-type count (And / out) */
    size_t n_or[TNN_SNAP_MAX];  /* sum-type count     (Or = out*k) */
} TnnSnap;

void   type_nn_alt_snap(const AltNet *a, TnnSnap *s);
void   type_nn_alt_dyn_score(const TnnSnap *before, const TnnSnap *after,
                             double *dyn_scale, int *dyn_depth);


AltNet type_nn_arena_open(size_t in, size_t out);
AltNet type_nn_soa_open(size_t in, size_t out);
AltNet type_nn_gemm_open(size_t in, size_t out);
AltNet type_nn_csr_open(size_t in, size_t out);
AltNet type_nn_hotcold_open(size_t in, size_t out);
AltNet type_nn_q8_open(size_t in, size_t out);
AltNet type_nn_tape_open(size_t in, size_t out);
AltNet type_nn_opt_q8_open(size_t in, size_t out);
AltNet type_nn_opt_open(size_t in, size_t out);
AltNet type_nn_over_open(size_t in, size_t out);
AltNet type_nn_bp_open(size_t in, size_t out);
AltNet type_nn_mom_open(size_t in, size_t out);
AltNet type_nn_adam_open(size_t in, size_t out);
AltNet type_nn_bpgemm_open(size_t in, size_t out);
AltNet type_nn_dyn_open(size_t in, size_t out);
AltNet type_nn_dyn_sgd_open(size_t in, size_t out);
AltNet type_nn_dyn_adam_open(size_t in, size_t out);
AltNet type_nn_proj_open(size_t in, size_t out);
AltNet type_nn_proj_dyn_open(size_t in, size_t out);
AltNet type_nn_proj2_open(size_t in, size_t out);
AltNet type_nn_static_open(size_t in, size_t out);
AltNet type_nn_proj2_dyn_open(size_t in, size_t out);
AltNet type_nn_bpdyn_open(size_t in, size_t out);
AltNet type_nn_bpgap_open(size_t in, size_t out);
AltNet type_nn_bpcurv_open(size_t in, size_t out);
AltNet type_nn_bpcombo_open(size_t in, size_t out);
AltNet type_nn_bpcube_open(size_t in, size_t out);
AltNet type_nn_bpwide_open(size_t in, size_t out);
AltNet type_nn_bpsite_open(size_t in, size_t out);
AltNet type_nn_bpearly_open(size_t in, size_t out);
AltNet type_nn_bpdeep_open(size_t in, size_t out);
AltNet type_nn_idi_open(size_t in, size_t out);
AltNet type_nn_idfact_open(size_t in, size_t out);
AltNet type_nn_idn_open(size_t in, size_t out);
AltNet type_nn_idtgt_open(size_t in, size_t out);
AltNet type_nn_idema_open(size_t in, size_t out);
AltNet type_nn_idmax_open(size_t in, size_t out);
AltNet type_nn_typefact_open(size_t in, size_t out);
AltNet type_nn_adapt_open(size_t in, size_t out);
AltNet type_nn_init2_open(size_t in, size_t out);
AltNet type_nn_init2p_open(size_t in, size_t out);
AltNet type_nn_one_open(size_t in, size_t out);
AltNet type_nn_lin_open(size_t in, size_t out);
AltNet type_nn_dyn_l_open(size_t in, size_t out);
AltNet type_nn_dyn_w_open(size_t in, size_t out);
AltNet type_nn_dyn_k_open(size_t in, size_t out);

void   type_nn_alt_train(AltNet *a, double **X, double **Y,
                         size_t n, size_t epochs, double lr);
size_t type_nn_alt_count(void);
const char *type_nn_alt_name(size_t i);
AltNet (*type_nn_alt_opener(size_t i))(size_t, size_t);

#endif
