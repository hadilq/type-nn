# Probes and back-prop

An And is only a product of Ors. There is no sum of Ands.

Or and And use the same dummy rule:

1. Before back-prop, the product list contains exactly one identity.
2. Back-prop is allowed to move that identity.
3. After back-prop, every factor whose weights are below threshold
   (identity again) is dropped, except one dummy that always stays.

## Original type-nn (no clip)

    Or_t = b_t + W_t · x
    And  = Π_t Or_t
    z_i  = Π_r And_{i,r}
    τ    = √d                      # d = tail input arity
    y_i  = tanh(z_i / τ)           # tail only; hidden layers stay z

    ∂And / ∂Or_t = Π_{u≠t} Or_u
    ∂z   / ∂And_r = Π_{q≠r} And_q
    ∂y   / ∂z     = (1 − y²) / τ
    ∂L   / ∂y     = y − t

Dummy Or is born (b,W)=(1,0) so Or ≡ 1.
Dummy And is a product of dummy Ors, so And ≡ 1.
tanh is applied after the product, so a dummy ×1 does not change z.

τ = √d puts every dataset on the same coordinate: a 2-D XOR and a
34-D radar file are not raw products of different length.

If |z/τ| > 20, y = sign(z) and 1−y² = 0.

## type-nn-* (clip + tanh)

    Or_t = clip(b_t + W_t · x, ±4)
    And  = clip(Π_t Or_t, ±32)
    y_i  = tanh(Π_r And_{i,r})

## Counters

    or_add  / or_drop     dummy Or injected / extra identity dropped
    and_add / and_drop    dummy And injected / extra identity dropped
    layer_add / layer_drop  identity layer (type-nn-* only)

Original type-nn sets layer_probe = 0.

## type-nn-orcool

Same product as original. Each Or has `cool_left` (BP cooldown) and
`cut` (tail snap after BP). Both anneal with `u = step/span`:

    cool_len = 48(1-u) + 4u
    cut      = 1e-6(1-u) + 0.2 u

A live Or resets `cool_left = cool_len`. Dummy And SGD is skipped while
any Or on that output is cooling. Ors train first; Ands open later.

## And-preference models (`./bench_type_nn TASK name`)

    budget     at most one live And + one dummy And per output
    stuck      dummy And only if dummy Ors are idle and live Ors are full
    timescale  dummy And SGD / ensure only every 32 steps
    asym       And identity band 0.05 (stays dummy unless a weight > 0.05)
    gres       dummy And SGD iff |∂L/∂And_d| > 2 max|∂L/∂Or_d|
    andtau     tail τ = √d · n_And (extra clauses do not inflate z)
    degree     new dummy And is born with one Or (degree 0→1, then Ors grow)
    soft       dummy And SGD iff |∂L/∂And_d| > |∂L/∂Or_d| + 0.15 n_And
    combo      kitchen-sink (too many constraints; kept for the record)
    ta         32-tick spawn AND dummy SGD + andtau
    tag        ta + gres
    tap        32-tick spawn only; dummy And trains every sample + andtau
    tas        32-tick dummy SGD only; spawn when a dummy left 1 + andtau
    next       tap + budget + max_or = clamp(round(2√d), 8, 16)
    ln         original type-nn algebra; tail tanh → log readout below.

## type-nn-ln

Same And/Or product as original type-nn. Indices: output head \(i\),
clause \(r\), linear factor \(t\), feature \(j\). Hidden layers stay \(z_i\).

    Or_{i,r,t} = b_{i,r,t} + Σ_j W_{i,r,t,j} x_j
    And_{i,r}  = Π_t Or_{i,r,t}
    a_{i,r}    born at 1
    z_i        = Π_r And_{i,r}^{a_{i,r}}      # sign(A)|A|^a so z_i ∈ ℝ
    τ          = √d                           # d = tail input arity
    y_i        = sign(z_i) ln(1 + |z_i|/τ)    # real completion of ln(1+z_i/τ)

    ∂L / ∂y_i                = y_i − t_i
    ∂y_i / ∂z_i              = 1 / (τ + |z_i|)
    ∂z_i / ∂And_{i,r}        = z_i a_{i,r} / And_{i,r}
    ∂z_i / ∂a_{i,r}          = z_i log |And_{i,r}|
    ∂And_{i,r} / ∂Or_{i,r,t} = Π_{s≠t} Or_{i,r,s}
    ∂Or_{i,r,t} / ∂W_{i,r,t,j} = x_j
    ∂Or_{i,r,t} / ∂b_{i,r,t}   = 1

And^{a} := sign(And)|And|^{a} keeps z_i real when And < 0.
Dummy And ≡ 1 ⇒ 1^{a} ≡ 1 and ∂z/∂a = z log 1 = 0, so dummy
exponents do not train. a_{i,r}=1 recovers the old product.
ln(1+z_i/τ) is only real for z_i > −τ; the abs is that
completion. For z_i ≥ 0 the two agree. y_i is C¹ at 0.

τ = √d · max(1, n_live_And). Dummy identity Ands are not clauses.

`make bench` rewrites BOARD.txt from the table it prints.
