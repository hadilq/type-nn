#ifndef TYPE_NN_LN_H
#define TYPE_NN_LN_H

#include "type_nn.h"

/*
 * type-nn-ln — typed product + log tail, isolated from the tanh net.
 *
 * Indices: output head i, clause r, linear factor t, feature j.
 *
 *   Or_{i,r,t}  = b_{i,r,t} + Σ_j W_{i,r,t,j} x_j
 *   Õ_{i,r,t}   = clip(Or_{i,r,t}, ±C)     if LN_ORCLIP and Or is live
 *                 Or_{i,r,t}               otherwise
 *
 * Log-space And (LN_LOGZ; default on every ln-* except ln-naive):
 *
 *   σ_{i,r}     = Π_t sign(Õ_{i,r,t})      (sign(0) kills the product)
 *   λ_{i,r}     = Σ_t log max(|Õ_{i,r,t}|, ε)
 *   λ̃_{i,r}    = clip(λ_{i,r}, −L, L)
 *   γ_{i,r}     = 1_{|λ_{i,r}| < L}        (∂λ̃/∂λ)
 *   A_{i,r}     = σ_{i,r} exp(λ̃_{i,r})
 *
 * Type generating function:
 *
 *   S_i         = Π_r sign(A_{i,r})
 *   ℓ_i         = Σ_r a_{i,r} log max(|A_{i,r}|, ε)
 *   ℓ̃_i        = clip(ℓ_i, −L, L)
 *   g_i         = 1_{|ℓ_i| < L}            (∂ℓ̃/∂ℓ)
 *   z_i         = S_i exp(ℓ̃_i)
 *
 * Tail (last layer only; hidden layers stay z_i):
 *
 *   τ_i         = √d                       (d = tail input arity)
 *                 √d · Σ_{r live} |a_{i,r}|  if LN_ATAU and that sum > 1
 *   y_i         = sign(z_i) ln(1 + |z_i|/τ_i)
 *                 / ln 2                     if LN_Y01  (|z|=τ ⇒ |y|=1)
 *
 * Dummy And ≡ 1 ⇒ log 1 = 0 and 1^{a} ≡ 1, so ∂z/∂a = 0.
 * a_{i,r} = 1 recovers the untyped product.
 *
 * Jacobians used in back-prop (A ≠ 0, Õ ≠ 0):
 *
 *   ∂L/∂y_i              = y_i − t_i
 *   ∂y_i/∂z_i            = 1 / (τ_i + |z_i|)          [ / ln 2 if LN_Y01 ]
 *   ∂z_i/∂A_{i,r}        = z_i · g_i · a_{i,r} / A_{i,r}
 *   ∂z_i/∂a_{i,r}        = z_i · g_i · log max(|A|, ε)
 *   ∂A_{i,r}/∂Õ_{i,r,t}  = A_{i,r} · γ_{i,r} / Õ_{i,r,t}
 *   ∂Õ/∂Or               = 1_{|Or| < C}               if LN_ORCLIP, else 1
 *   ∂Or/∂W_{…,j}         = x_j
 *   ∂Or/∂b               = 1
 *
 * Without LN_LOGZ the same z is Π_r sign(A)|A|^a and both gates are 1
 * (until a coordinate overflows).
 */

#define LN_LOGZ    1u     /* And and z in log-space with hard cap */
#define LN_APOS    2u     /* a ≥ 0; snap ~0 */
#define LN_ORCLIP  4u     /* clip live Or to ±LN_OR_CAP */
#define LN_ASMALL  8u     /* a born at 1/√d */
#define LN_ALR     16u    /* η_a = η/10 */
#define LN_Y01     32u    /* y /= ln 2  so |z|=τ ⇒ |y|=1 */
#define LN_ATAU    64u    /* τ = √d · Σ|a| on live Ands */
#define LN_NAIVE   128u   /* keep product in linear space */
#define LN_AMAX    256u   /* clip a to [0, LN_A_CAP] (needs APOS) */
#define LN_WIDE    512u   /* max_or = clamp(round(2√d), 8, 16) */

#define LN_ELL_CAP 20.0   /* |ℓ̃| < 20,  e^{20} ≈ 4.85e8 */
#define LN_LOG_EPS 1e-12
#define LN_OR_CAP  4.0
#define LN_A_CAP   3.0    /* a ∈ [0, 3] when LN_AMAX */

typedef struct {
    double z;      /* signed typed product */
    double ell;    /* raw Σ a log|A| */
    double ell_s;  /* clipped ℓ */
    double gate;   /* ∂ℓ̃/∂ℓ = 1_{|ℓ|<L}; 1 if no logz */
} TnnLnZ;

void     tnn_ln_bind(Network *net);
int      tnn_ln_apply(Network *net, const char *name);
int      tnn_ln_on(void);
int      tnn_ln_bit(unsigned bit);
double   tnn_ln_signed_pow(double base, double p);
double   tnn_ln_and_term(const AndNode *a);
double   tnn_ln_and_eval(AndNode *a);          /* sets a->value, a->and_gate */
TnnLnZ   tnn_ln_eval(const AndNode *row, size_t index);
double   tnn_ln_tau(const Layer *l, size_t index, double tau_sqrt_d);
double   tnn_ln_readout(double z, double tau);
double   tnn_ln_dydz(double z, double tau);
double   tnn_ln_lr_a(double lr);
void     tnn_ln_init_expn(Network *net);
double   tnn_ln_or_clip(double or_value, int is_dummy);
int      tnn_ln_or_clip_gate(double or_value, int is_dummy);
void     tnn_ln_step_expn(AndNode *a, double dL_dz, double z, double gate, double lr);
int      tnn_ln_zero_expn(const AndNode *a);
int      tnn_ln_and_is_dummy(const AndNode *a);
void     tnn_ln_and_accums(AndNode *a);        /* Or.accum = ∂A/∂Õ */

#endif
