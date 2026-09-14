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

## type-nn-ln  (implementation: type_nn_ln.c / type_nn_ln.h)

Indices: output head \(i\), clause \(r\), linear factor \(t\), feature \(j\).
Hidden layers stay \(z_i\). The default recipe (`ln`, `ln-logz`) evaluates
both products in log-space so a long And list cannot overflow.

    Or_{i,r,t}   = b_{i,r,t} + Σ_j W_{i,r,t,j} x_j
    Õ_{i,r,t}    = clip(Or, ±4)                 # LN_ORCLIP, live Or only
    σ_{i,r}      = Π_t sign(Õ_{i,r,t})
    λ_{i,r}      = Σ_t log max(|Õ_{i,r,t}|, ε)
    λ̃_{i,r}     = clip(λ_{i,r}, −L, L)         # L = 20
    γ_{i,r}      = 1_{|λ_{i,r}| < L}
    A_{i,r}      = σ_{i,r} exp(λ̃_{i,r})
    S_i          = Π_r sign(A_{i,r})
    ℓ_i          = Σ_r a_{i,r} log max(|A_{i,r}|, ε)
    ℓ̃_i         = clip(ℓ_i, −L, L)
    g_i          = 1_{|ℓ_i| < L}
    z_i          = S_i exp(ℓ̃_i)
    τ_i          = √d                           # d = tail input arity
    y_i          = sign(z_i) ln(1 + |z_i|/τ_i)

`ln-naive` skips the log-space rewrite and uses \(A=\Pi Õ\),
\(z=\Pi \mathrm{sign}(A)|A|^a\) directly.

    ∂L / ∂y_i                 = y_i − t_i
    ∂y_i / ∂z_i               = 1 / (τ_i + |z_i|)
    ∂z_i / ∂A_{i,r}           = z_i · g_i · a_{i,r} / A_{i,r}
    ∂z_i / ∂a_{i,r}           = z_i · g_i · log max(|A_{i,r}|, ε)
    ∂A_{i,r} / ∂Õ_{i,r,t}     = A_{i,r} · γ_{i,r} / Õ_{i,r,t}
    ∂Õ / ∂Or                  = 1_{|Or| < 4}            # LN_ORCLIP
    ∂Or / ∂W_{i,r,t,j}        = x_j
    ∂Or / ∂b                  = 1

The hard clip is not differentiable at \(\pm L\); the code takes the
almost-everywhere derivative \(0\) on the wall, \(1\) inside. Dummy
And \(\equiv 1\) contributes \(\log 1 = 0\) and \(\partial z/\partial a = 0\).
\(a_{i,r}=1\) recovers the untyped product. \(y_i\) is \(C^1\) at \(0\).

Recipes (`./bench_type_nn TASK name`), each bit isolated on top of LOGZ:

    ln / ln-logz   log-space And and z, no extra policy
    ln-naive       linear product (control)
    ln-orclip      + clip live Or to ±4
    ln-alr         + η_a = η/10
    ln-apos        + a ≥ 0, prune a≈0
    ln-asmall      + a born at 1/√d
    ln-y01         + y /= ln 2  so |z|=τ ⇒ |y|=1
    ln-atau        + τ = √d · Σ|a| on live Ands
    ln-stuck       + dummy And waits for full/idle Ors
    ln-tas         + 32-tick dummy-And SGD + andtau
    ln-budget      + one live And + one dummy per output
    ln-next        + tap + budget + andtau
    ln-stable      logz + orclip
    ln-clock       logz + orclip + tas
    ln-best        kitchen sink (usually worse than its parts)

Iteration 2, combinations of the bits that actually moved a hard task:

    ln-v2          a>0 assembly + η_a=η/10
    ln-aw          a≥0 + max_or = clamp(2√d, 8, 16)
    ln-v2w         v2 + wide Or          <- best all-round
    ln-v2t         v2 + tas clock        <- best ionosphere / iris
    ln-v3          v2t + a∈[0,3] + wide Or
    ln-cap         a∈[0,3]
    ln-v2c         v2 + a∈[0,3]
    ln-v2a         v2 + τ = √d · Σ|a|
    ln-adam        logz + assembly a>0 + Adam on W,b,a
    ln-v2w-adam    v2w + Adam   (plus-form: ln-v2w+adam)

## Or / And spawn (type_nn_grow.c)

No `max_or`, no `max_and` on any `scale-*` recipe. Dummy rule only.

    scale-keep     always replace a missing dummy
    scale-resid    |incoming| > T
    scale-ratio    |incoming| > κ · max|child grad|
    scale-dead     children still, incoming large
    scale-or       ratio on Or only
    scale-and      ratio on And only
    scale-energy   |parent| > κ · Σ|live child grad|
    scale-jac      |parent| > T and Σ|∂L/∂W|_live < κ_j · |parent|
    scale-slack    |parent| > T and every live child is specialized
    scale-sign     |parent| > T and live grads fight or miss sign
    scale-mix      energy on Or, jac on And
    scale-ej       energy AND jac on both axes
    scale-layer    scale-ej + Ljac

T = 0.30, κ = 2, κ_j = 0.25, specialized band = 0.5.
Combine: `ln-v2w+scale-mix`.

## Hidden-layer insert / drop (type_nn_layer.c)

Same dummy rule as Or and And. No wall-clock.

    insert  identity map in front of the tail (function unchanged)
    train   back-prop may move the identity
    drop    hidden is identity AND ||∂L/∂W||_1 ≈ 0

Never stack a second identity: the live hidden has to leave id first.

    Lgrad     insert if ||∂L/∂x||_1 > 3          (legacy one-shot)
    Lresid    insert if ||y−t||_1 > 1
    Lratio    insert if ||∂L/∂x||_1 > 2 ||∂L/∂W||_1
    Ldummy    keep exactly one identity hidden
    Ldrop     only the drop half
    Lwide     Lgrad, hidden width = 2·in
    Lnarrow   Lgrad, hidden width = max(out, ⌈in/2⌉)
    Lboth     Lgrad + drop
    Lcap      residual large AND Or/And lists full AND ||dW|| small vs residual
    Lfull     residual large AND lists full
    Lstuck    residual large AND ||dW|| small

Combine: `ln-v2w+Lcap`. Legacy gates stay at max_depth 2.

Iteration 6 — early depth, keep the layer:

    depth-early / Learly   identity hidden at the first train step,
                           square, drop blocked until u≥0.65, never
                           drop back to depth 1
    depth-hold  / Lhold    insert on a large residual (no Or/And cap
                           wait); same late drop rule
    depth-born  / Lborn    hidden exists before init_weights (random,
                           width 8/16 like c-mlp) and is kept

Combine with a scale probe: `scale-mix+depth-early`,
`scale-energy+depth-hold`, `scale-ej+depth-born`.

`make bench` rewrites BOARD.txt from the table it prints. The live
board is only those scale + depth models plus c-mlp. A–I / win /
torch rows were removed with their code. See AUDIT.md.

