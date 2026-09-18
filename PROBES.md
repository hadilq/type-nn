# Product, probes, back-prop

An And is a product of Ors. A layer is a product of Ands, typed by the
assembly index. There is no sum of Ands.

Or, And, and Depth use the same dummy rule:

1. Before back-prop, the product list contains exactly one identity.
2. Back-prop is allowed to move that identity.
3. After back-prop, every factor whose weights are below threshold
   (identity again) is dropped, except one dummy that always stays
   while we are still allowed to grow.

## Forward

    Or_t = b_t + W_t · x
    A    = Π_t Or_t^{a}
    z    = sign(A) ln(1 + |A|/τ)     # every layer, not just the tail

    ∂A / ∂Or_t = Π_{u≠t} Or_u
    ∂z / ∂A    = 1 / (τ + |A|)
    ∂L / ∂y    = (y − t) / n_out

Dummy Or is born (b,W)=(1,0) so Or ≡ 1.
Dummy And is a product of dummy Ors, so And ≡ 1.
Dummy incoming coordinate is born with weight 0 on the next layer.

The log-space rewrite (`type_nn_ln.c`) is the same product with

    λ = Σ log max(|Or|, ε),   A = σ exp(clip(λ, ±20))
    ℓ = Σ a log max(|A|, ε),  Z = S exp(clip(ℓ, ±20))
    z = sign(Z) ln(1 + |Z|/τ)

so a long list cannot overflow. That exponential *is* the partition
function of the layer.

## Board recipes

    type-nn      ln on every layer; Or / And / Depth dummy rule
    c-mlp        Linear-ReLU-Linear + ln tail, not a product net

Internal gate names (`scale-energy`, `scale-jac`, `Lresid`, …) still
exist in `type_nn_grow.c` / `type_nn_layer.c` so tests can isolate one
predicate. They are not board models.
