# type-nn

C And-Or net. An And is a product of Ors. Dummy Or / dummy And are
identity factors (`×1`). After back-prop, extra identities drop.

The current learned recipe is **type-nn-next**:

- dummy And trains every sample (it may leave `1`)
- a *new* dummy And is spawned only every 32 steps
- at most one live And + one dummy And per output
- tail `y = tanh(z / τ)` with `τ = √d · n_live_And`
- `max_or = clamp(round(2√d), 8, 16)` so wide inputs get more
  linear factors, not more clauses

Original `type-nn` is the same algebra with no And-preference policy.

**type-nn-ln** is that net with a type exponent on every And and a
log tail (hidden layers stay the raw typed product):

    Or_{i,r,t} = b_{i,r,t} + Σ_j W_{i,r,t,j} x_j
    And_{i,r}  = Π_t Or_{i,r,t}
    z_i        = Π_r And_{i,r}^{a_{i,r}}     # sign(A)|A|^a, a born at 1
    y_i        = sign(z_i) ln(1 + |z_i|/√d)  # tail only

`a_{i,r}` is trained by SGD. Dummy And ≡ 1 ⇒ `1^a ≡ 1` and
`∂z/∂a = 0`. Derivations: [PROBES.md](PROBES.md).

type-nn-B / A–I are dense-layout experiments (clip + layer probe).

Numbers: [BOARD.txt](BOARD.txt).

## Fair split

xorshift32 Fisher-Yates, seed 34972, 70/30. Same cut for every impl,
including torch. `make bench` rewrites BOARD.txt.

    ./bench_type_nn iris next
    ./bench_type_nn iris ln
    make bench
    make test
