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
    y_i  = tanh(z_i)          # tail only; hidden layers stay z

    ∂And / ∂Or_t = Π_{u≠t} Or_u
    ∂z   / ∂And_r = Π_{q≠r} And_q
    ∂y   / ∂z     = 1 − y²
    ∂L   / ∂y     = y − t

Dummy Or is born (b,W)=(1,0) so Or ≡ 1.
Dummy And is a product of dummy Ors, so And ≡ 1.
tanh is applied after the product, so a dummy ×1 does not change z.

If |z| > 20, y = sign(z) and 1−y² = 0 (gradient stops at overflow).

## type-nn-* (clip + tanh)

    Or_t = clip(b_t + W_t · x, ±4)
    And  = clip(Π_t Or_t, ±32)
    y_i  = tanh(Π_r And_{i,r})

## Counters

    or_add  / or_drop     dummy Or injected / extra identity dropped
    and_add / and_drop    dummy And injected / extra identity dropped
    layer_add / layer_drop  identity layer (type-nn-* only)

Original type-nn sets layer_probe = 0.
`make bench` rewrites BOARD.txt from the table it prints.
