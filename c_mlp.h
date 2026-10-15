#ifndef C_MLP_H
#define C_MLP_H

#include <stddef.h>

/*
 * c-mlp — the board baseline.
 *
 *   u = W2 · ReLU(W1 x + b1) + b2          (hidden width H)
 *   y = F(u) = sign(u) ln(1 + |u|)         (same readout as type-nn)
 *
 *   H = 8 if in <= 4 else 16, and never below the output width.
 *
 * H is the baseline's fixed definition; it is not tuned here and
 * type-nn does not read it. Init is torch.nn.Linear's
 * U(-1/sqrt(fan_in), +1/sqrt(fan_in)). Training is per-sample Adam from
 * common.h, identical to type-nn.
 */

typedef struct CMlp CMlp;

CMlp         *cmlp_create(size_t in, size_t out, unsigned seed);
/* The same baseline with a given hidden width (c-mlp-matched on the
   board: the widest MLP whose params do not exceed graph-type-nn's). */
CMlp         *cmlp_create_hidden(size_t in, size_t out, size_t hid, unsigned seed);
void          cmlp_free(CMlp *m);
void          cmlp_begin(CMlp *m, size_t n_train, size_t epochs, double lr);
const double *cmlp_forward(CMlp *m, const double *x);
void          cmlp_backward(CMlp *m, const double *dy);
size_t        cmlp_params(const CMlp *m);
size_t        cmlp_hidden(const CMlp *m);

#endif
