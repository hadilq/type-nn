#ifndef TYPE_NN_ALT_H
#define TYPE_NN_ALT_H

#include <stddef.h>

/*
 * Thin trainer used by the c-mlp baseline. The type-nn-A..I / win
 * layouts were removed from the board; their code is gone.
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
    void   (*scale_layer)(void *ctx, size_t idx, size_t in, size_t out);
    size_t (*layer_in)(void *ctx, size_t idx);
    size_t (*layer_out)(void *ctx, size_t idx);
    size_t (*layer_k)(void *ctx, size_t idx);
    size_t (*n_add)(void *ctx);
    size_t (*n_drop)(void *ctx);
    void   (*set_corpus)(void *ctx, size_t n);
    size_t (*or_add)(void *ctx);
    size_t (*or_drop)(void *ctx);
    size_t (*and_add)(void *ctx);
    size_t (*and_drop)(void *ctx);
};

#define TNN_SNAP_MAX 8
typedef struct {
    size_t depth;
    size_t n_and[TNN_SNAP_MAX];
    size_t n_or[TNN_SNAP_MAX];
} TnnSnap;

void   type_nn_alt_snap(const AltNet *a, TnnSnap *s);
void   type_nn_alt_dyn_score(const TnnSnap *before, const TnnSnap *after,
                             double *dyn_scale, int *dyn_depth);
void   type_nn_alt_train(AltNet *a, double **X, double **Y,
                         size_t n, size_t epochs, double lr);

size_t type_nn_alt_count(void);
const char *type_nn_alt_name(size_t i);
AltNet (*type_nn_alt_opener(size_t i))(size_t, size_t);

#endif
