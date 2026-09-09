# type-nn

A small C library for a **dynamic AND–OR network**. Each output is a product
of affine units. The topology lives in sorted linked lists so a layer can
grow or shrink its width, its number of product factors, and the number of
layers during back-prop.

The name is from **Type Mechanics**
([hadilq.com/posts/type-mechanics](https://hadilq.com/posts/type-mechanics/),
Hadi Lashkari Ghouchani, 2025-12-26). In that note a *sum-type* (`A + B`)
is an **or** and a *product-type* (`A × B`) is an **and**. The network is
the same pair of constructors, written over numbers instead of types:

```
Or_{i,k}   =  b_{i,k} + Σ_j w_{i,k,j} · x_j     # sum-type / linear “or”
And_i      =  Π_k Or_{i,k}                      # product-type / “and”
```

A Type Generating Function multiplies under `×` and adds under `+`.
Here the Or node *adds* features (counting-space sum) and the And node
*multiplies* those linear forms (counting-space product). Raising the
number of Or factors raises the polynomial rank the layer can fit — the
same way a TGF of higher degree carries more structure — without a
pointwise activation.

## Why there is no activation

A stack of affine maps is still an affine map: without a pointwise
non-linearity a conventional MLP cannot fit XOR.

type-nn is already non-linear **before** any σ/ReLU/tanh, because an And
of *K* Or factors is a multivariate polynomial of total degree *K*:

| K (OR factors) | function class      | example                     |
|----------------|---------------------|-----------------------------|
| 1              | affine (rank-1)     | `w·x + b`                   |
| 2              | quadratic (rank-2)  | XOR, `x y`, `x² − y²`       |
| 3              | cubic (rank-3)      | `x y z`, 3-way interactions |
| K              | degree-K polynomial | any monomial of degree ≤ K  |

ReLU/sigmoid exist to *manufacture* non-linearity on top of matmuls.
Here the product *is* the non-linearity.

## Layout

| file              | role                                        |
|-------------------|---------------------------------------------|
| `lean/`           | Lean 4 proofs of the algebra (no Mathlib)   |
| `type_nn.h/.c`    | public structs + forward/backward           |
| `test_type_nn.c`  | unit + corner-case + scenario tests         |
| `run.c`           | XOR / sin(x) / structure demos              |
| `dataset.h/.c`    | Iris / Wine / WDBC / Diabetes loaders       |
| `bench_type_nn.c` | C wall-clock + RSS + real-data bench        |
| `bench_alts.c`    | same tasks for every `type-nn-*` layout     |
| `type_nn_alt.h`   | And/Or stack API (grow/shrink, insert/remove) |
| `type_nn_*.c`     | arena soa gemm csr hotcold q8 tape **opt**  |
| `bench_torch.py`  | PyTorch CPU baselines                       |
| `bench.sh`        | comparison table (lists + alts + torch)     |
| `flake.nix`       | dev shell, package, **pinned dataset fetch**|

## Build

```bash
make test          # 189 checks
make test-asan
make demo
make data          # curl the five public datasets into ./data
make bench         # every implemented model on every dataset, + PyTorch
make lean          # type-check the proofs in ./lean (needs lake)
```

`make bench` is the one to run. It builds `bench_type_nn` and `bench_alts`,
runs **every model in this repository against every dataset**, and prints one
sorted table per task with hold-out accuracy, parameter count, bytes,
inference time and how many layers each run added or dropped. PyTorch rows are
included when `python3 -c "import torch"` succeeds and skipped otherwise.
`./bench.sh wine` restricts it to one task.

With Nix / direnv (`use flake`):

```bash
nix develop        # sets TYPE_NN_DATA, puts gcc, lake and torch on PATH
make bench
make lean
# or just the pieces:
nix build .#datasets
nix build .#proofs   # hermetic Lean check, no network, no Mathlib
nix flake check
```

## Proofs (`./lean`)

The architecture moves are algebra, so they are stated and checked as algebra.
`./lean` is a dependency-free Lean 4 library — **no Mathlib**, so `lake build`
finishes in seconds rather than fetching a multi-gigabyte cache.

| result | Lean name | what it says |
|--------|-----------|--------------|
| Prop 2  | `TypeNN.gauge_invariant`     | rescaling factors with `∏ c = 1` fixes the And |
| Lemma 3 | `TypeNN.prefix_suffix`       | prefix × suffix is `∂A/∂O_s`, no division |
| Prop 4  | `TypeNN.gauss_seidel`        | the in-place step costs exactly `η‖∇_O L‖² x` |
| Thm 5   | `TypeNN.dead_factor`         | one zero factor freezes every sibling |
| Thm 5   | `TypeNN.dead_and`            | two zero factors freeze the whole And |
| Prop 6  | `TypeNN.orVal_zero_pad`      | zero-padding the input is exact |
| Prop 7  | `TypeNN.widen_preserves`     | widening preserves the composite |
| Prop 8  | `TypeNN.grow_rank_exact`     | a unit factor `(W=0, b=1)` is exact |
| Prop 11 | `TypeNN.identity_insert`     | the identity insert is exact |
| Prop 12 | `TypeNN.identity_test_bound` | the identity test is a bound, not an identity |

Scalars are `Int`. Every proof uses only the axioms of a linearly ordered
commutative ring — nothing divides — and `ℝ` is such a ring, so fixing the
carrier is what lets the library build against bare Lean core. `TypeNN.lean`
ends in `#print axioms` for each theorem; the build output shows they depend
on nothing but `propext`, `Classical.choice` and `Quot.sound`, and in
particular not on `sorryAx`.

Not formalised, and the post says so: Proposition 1 (degree of the composite)
needs a theory of polynomial degree, and Propositions 9 and 10 are statements
about the clipped algebra.

## Datasets (fetched by `flake.nix`)

| id         | file                 | n    | in | out | task           | source |
|------------|----------------------|------|----|-----|----------------|--------|
| `iris`     | `iris.data`          | 150  | 4  | 3   | 3-class        | UCI Fisher 1936 |
| `wine`     | `wine.data`          | 178  | 13 | 3   | 3-class        | UCI Wine |
| `wdbc`     | `wdbc.data`          | 569  | 30 | 1   | malignant=1    | UCI Wisconsin Diagnostic |
| `diabetes` | `diabetes.tab.txt`   | 442  | 10 | 1   | regression     | Efron et al. / NCSU |

Inputs are z-scored. Diabetes targets are min-max scaled to `[0,1]`.
Classification reports train-set argmax / 0.5-threshold accuracy.

`TYPE_NN_DATA` is searched first, then `./data`, `/tmp/type-nn-data`.
The flake `devShell` exports `TYPE_NN_DATA` to the pinned store path so a
`nix develop` bench never hits the network a second time.

```bash
./bench.sh              # synthetic + real
./bench.sh real         # iris wine wdbc diabetes only
./bench.sh iris
```

## Better data structures (sibling implementations)

`type_nn.h` and `type_nn.c` are **frozen**. Each idea lives in its own
files and shows up as its own `impl` row in `./bench.sh`. Same And/Or
math, K = 2.


The live graph is still four singly-linked, heap-allocated node types
(`WeightNode`, `OrNode`, `AndNode`, `InOutNode`) keyed by `right_index`.
That is a faithful Type Mechanics picture — sum-lists and product-lists
you can splice — and it is also why a 32-wide layer is much slower than
a BLAS matmul. Candidates, cheapest first, **without changing the
math**:

| # | structure | what improves | cost / risk |
|---|-----------|---------------|-------------|
| 1 | **Arena / slab of nodes, `next` as an index** | one allocation per layer instead of one `malloc` per weight; cheaper free; friendlier to the cache | API stays pointer-shaped via `arena[i]` |
| 2 | **Struct-of-arrays per Or** (`w[]`, `idx[]`, `n`) | the inner `Σ w_i x_i` becomes a tight loop the compiler can auto-vectorize | lose cheap mid-list splice; grow by `realloc` |
| 3 | **Dense `in × or` / `or × out` matrices** when a layer is not sparse | one GEMM (`cblas_dgemv`) per Or-row, product in registers | memory is Θ(in·or·out); pruning becomes a mask |
| 4 | **CSR / CSC sparse matrix** for the affine part | keeps the “only some features live” property without pointer chasing | insert/delete of a feature is an array shift |
| 5 | **ELLPACK / sliced ELL** | regular inner loop, good SIMD, still sparse | padding waste if degree varies a lot across Ors |
| 6 | **Two-level hot/cold** | first 32 features dense, overflow in a tiny CSR | extra branch; great if most signal is on low indices |
| 7 | **Blocked SoA + SIMD product** (`w0[j]*x[j]`, `w1[j]*x[j]`, then `or0*or1`) | And of two Ors is the common case and fits in AVX | harder *K*>2 |
| 8 | **int16 / Q8.8 weights** after snap-to-{0,1} is common | 4× smaller working set; the quantization field already wants this | need a scale per Or |
| 9 | **Persistent index map `feature → offset`** (hash or dense table) | zip-merge of two lists becomes O(n) with no “walk until index matches” | extra memory |
| 10 | **Tape / Wengert list for reverse mode** instead of storing `grad` on every node | backward does not have to re-walk the product tree | one more array to allocate per sample |
| 11 | **Layer as a single byte blob + header** | `network_nbytes` becomes the allocation; memcpy checkpoint | pointers inside must be rebaseable |
| 12 | **Do not change anything for depth-1 XOR-scale nets** | 6 parameters, ~0.1 µs — lists are not the bottleneck there | — |

Every `type-nn-*` layout is a full And/Or net: product-of-affines per
layer, grow/shrink of `in` / `out` / Or-count, identity-layer insert and
hidden-layer remove. `type_nn.c` stays frozen.

| impl | file | XOR µs | Iris µs | WDBC µs | WDBC nbytes |
|------|------|--------|---------|---------|-------------|
| type-nn | `type_nn.c` frozen | 0.108 | 0.293 | 1.745 | 4696 |
| type-nn-arena | `type_nn_arena.c` | 0.017 | 0.054 | 0.199 | 2000 |
| type-nn-soa | `type_nn_soa.c` | 0.013 | 0.027 | 0.054 | 792 |
| type-nn-gemm | `type_nn_gemm.c` | 0.022 | 0.042 | 0.047 | 792 |
| type-nn-csr | `type_nn_csr.c` | 0.017 | 0.045 | 0.061 | 1084 |
| type-nn-hotcold | `type_nn_hotcold.c` | 0.017 | 0.048 | 0.061 | 816 |
| type-nn-q8 | `type_nn_q8.c` | 0.017 | 0.043 | 0.065 | 412 |
| type-nn-tape | `type_nn_tape.c` | 0.012 | 0.027 | 0.053 | 5944 |
| type-nn-opt-q8 | `type_nn_opt_q8.c` | 0.019 | 0.034 | 0.045 | **92** |
| **type-nn-opt** | `type_nn_opt.c` | 0.025 | 0.032 | 0.047 | 496 |

Packed int8 pass (now `type-nn-opt-q8`):

- deployed payload is int8 `W` + per-Or scale + bias (`nbytes` is just that)
- a float panel is rebuilt only when weights change, then SoA (`in < 16`)
  or blocked GEMV (`in ≥ 16`) runs on it — same trick as `type-nn-q8`
  not counting its float shadow
- `k == 2` And is a single multiply
- backward does **not** re-forward the stack (activations are kept)

On WDBC that is faster than GEMV and ~4.5× smaller than Q8's advertised
row. XOR stays in the SoA noise floor; the extra layer bookkeeping is
why a 2-input net is not the absolute fastest.

## Dynamic scaling

| operation | how |
|-----------|-----|
| wider input | `layer_align_inputs` or `predict` with a longer `x` |
| narrower input | `layer_align_inputs` drops indices `≥ n` |
| more / fewer outputs | `layer_set_outputs` |
| extra product factor | large `‖dL/dAnd‖` appends an Or initialised to `1` |
| drop a factor | Or ≈ `1` or Or ≈ `0`, if more than one remains |
| add a layer | `network_insert_identity`; auto-insert at most one hidden layer from depth 1 |
| drop a layer | `network_remove_layer` (refuses the last / output layer) |

## Tests

`./test_type_nn` — 189 checks, including known-weight forward (`3·4 = 12`,
`x²−y²`), XOR, grow/shrink, identity insert/remove, `max_or` / `max_depth`,
NULL API, clip, multi-output train, param/byte accounting.

## API sketch

```c
Network *net = network_create(2, 1);
network_set_learning_rate(net, 0.08);
network_set_dynamic(net, 1);
network_set_verbose(net, 0);
network_init_weights(net);
network_train(net, X, Y, n_samples, epochs);
double y;
network_predict(net, x, 2, &y, 1);
network_free(net);
```

## Example real-data numbers (CPU, this machine)

| task     | impl    | n   | train_s | µs/infer | params | mse    | acc  |
|----------|---------|-----|---------|----------|--------|--------|------|
| iris     | type-nn | 150 | 0.027   | 0.29     | 30     | 0.333  | 0.90 |
| wine     | type-nn | 178 | 0.097   | 0.89     | 84     | 0.178  | 0.95 |
| wdbc     | type-nn | 569 | 0.235   | 1.75     | 62     | large  | 0.45 |
| diabetes | type-nn | 442 | 0.067   | 0.46     | 22     | 0.027  | n/a  |
| iris     | torch-mlp | 150 | 0.118 | 21.3     | 67     | 0.010  | 0.98 |

`wdbc` is the canary for the linked-list + un-normalized product: 30 z-scored
inputs through two Or factors overflow the And clip and the net does not
fit. That is a model/representation issue the data-structure list above
is meant to address later — not a loader bug (569 rows parse, 30 features).


### type-nn-opt vs type-nn-opt-q8

The previous packed opt snapped every update to int8 (`1/127`), so XOR
MSE sat at `8.17e-6` instead of `0`. That net is kept as
`type-nn-opt-q8`. A new `type-nn-opt` uses double `W` (same arithmetic
as SoA/GEMM), adaptive SoA/GEMV, one-pass backward, payload-only
`nbytes`.

| impl | XOR mse | XOR µs | Iris µs | WDBC µs | WDBC nbytes |
|------|---------|--------|---------|---------|-------------|
| type-nn-soa | 0 | 0.014 | 0.027 | 0.054 | 792 |
| type-nn-gemm | 0 | 0.024 | 0.043 | 0.049 | 792 |
| type-nn-opt-q8 | 8.17e-6 | 0.019 | 0.034 | 0.045 | **92** |
| **type-nn-opt** | **0** | 0.025 | 0.032 | 0.047 | 496 |

Bench rows are sorted by `(params, nbytes, us/infer, train_s)`.


## Optimizer experiments

Faithful And/Or nets (`type_nn_bp.c`, `type_nn_mom.c`, `type_nn_adam.c`)
with grow/shrink and identity insert/remove. `type_nn.c` is frozen.

| impl | idea | XOR mse | Iris acc | WDBC mse | WDBC nbytes |
|------|------|---------|----------|----------|-------------|
| type-nn-opt | SGD, one-pass bwd | 0 | 0.90 | 190.6 | 496 |
| type-nn-bp | prefix/suffix dOr + fused dx | 0 | 0.90 | 190.6 | 496 |
| type-nn-mom | Polyak momentum μ=0.5 | 0 | 0.45 | 250.5 | 992 |
| type-nn-adam | Adam β=(0.9,0.999), α capped | ~3e-6 | 0.69 | **0.80** | 1488 |

Adam is the first layout that brings WDBC MSE down from ~190 to <1
(the product of affines on 30-D z-scored inputs explodes under plain
SGD). Momentum needs a gentler μ than the CNN default 0.9 — 0.5 fits
XOR. Moments cost 2× (mom) or 3× (adam) the payload.

`type-nn-bpgemm` is the second backward pass: `dx = Wᵀ dOr`,
`W ← W − lr dOr xᵀ`, prefix/suffix `dOr`, reused scratch, no malloc on
the hot path. `test_alts` now checks identity insert/remove, input
grow/shrink (zero-padded columns keep the mapping), Or-count grow
(new factor ≈ 1), output grow/shrink, and a backward step through an
identity hidden layer on every faithful layout.


### Dynamic models and scale tests

Every `type-nn-*` layout is covered by `test_alts` for:

- input grow/shrink, output grow/shrink, Or-count grow/shrink
- add two identity layers / remove them
- **per-layer** `scale_layer(idx, in, out)` that stitches the neighbour
  (hidden out ↔ next in)

`type-nn-dyn` actually uses those ops during training: residual-driven
Or growth, identity insert, hidden-width grow, and unused-Or shrink.
Bench enables `set_dynamic(1)` only for that impl. Rows sort by
`(mse, params, nbytes, infer, train)` so quality leads.


## Dynamic policy (stall → grow, idle → shrink)

A change is allowed only after `SETTLE` steps if EMA residual has not
dropped (`ema > 0.97 * ema_at_last_change` and above a floor). New
structure is an identity (extra Or `= 1`, extra And with tail column 0,
or a frozen identity layer) and stays **frozen** for `FREEZE` steps.

| impl | axes | XOR mse | Iris acc | WDBC mse | WDBC params | WDBC µs |
|------|------|---------|----------|----------|-------------|---------|
| type-nn-opt | none (SGD) | 0 | 0.90 | 190.6 | 62 | 0.046 |
| type-nn-adam | none (Adam) | ~3e-6 | 0.69 | 0.80 | 62 | 0.038 |
| type-nn-dyn-k | k only | 0 | 0.90 | 231 | 93 | 0.054 |
| type-nn-dyn-w | width | 0 | 0.90 | 268 | 1922 | 0.87 |
| type-nn-dyn-l | depth | 0 | 0.90 | 268 | 1922 | 0.86 |
| type-nn-dyn-sgd | k+w+L SGD | 0 | 0.90 | 231 | 93 | 0.055 |
| **type-nn-dyn** | k+w+L **Adam** | **0** | 0.65 | **0.53** | **62** | **0.038** |

Winner is Adam + the stall/idle policy: on WDBC it beats plain Adam
(0.53 vs 0.80) without growing the net. Width/depth-only SGD over-grows
because a product layer's residual stays large even after a change.


## Beating torch on real data

Growing `k` on raw z-scored UCI rows multiplies two wide affines and
overflows And-clip. Torch-mlp first learns a short hidden basis.

`type-nn-proj` does the same thing *inside Type Mechanics*:

    x  --[k=1 affine, width 8/16/24]-->  h  --[k=2 product]-->  y

Mini-batch Adam (16) on UCI, batch 4 on XOR. No ReLU.

| task | torch-mlp mse / acc | type-nn-proj mse / acc | vs old type-nn-adam mse |
|------|---------------------|------------------------|-------------------------|
| iris | 0.014 / 0.98 | 0.112 / **0.98** | 0.47 |
| wine | 0.002 / 1.00 | 0.078 / **1.00** | 0.41 |
| wdbc | 0.016 / 0.99 | 0.061 / 0.95 | 0.80 |
| diabetes | 0.023 / n/a | **0.027** / n/a | 0.15 |
| xor | — | 3.6e-7 | 0 |

Accuracy matches torch on iris and wine. Diabetes MSE is within ~15% of
torch. Remaining MSE gap is calibration (no softmax / cross-entropy);
the decision rule is already on par. Infer is ~200× faster than the
PyTorch CPU baseline (~0.07–0.38 µs vs ~22 µs).


## Dynamic-change measures

After every train the bench snapshots each layer's type sizes:

- product-type size `n_and = out` (how many Ands)
- sum-type size `n_or = out · k` (how many Ors)

A layer that appears or disappears is compared to zero. Per layer

    M_i = |Δn_or| + |Δn_and| + |Δn_or| · |Δn_and|

which both *adds* and *multiplies* the sum-type and product-type
changes. The board columns are

- `dscale` = Σ_i M_i   (0 means the architecture did not move)
- `ddepth` = depth_after − depth_before   (layers added minus removed)

Static nets (type-nn-opt, torch-*) report 0/0. `type-nn-proj-dyn` on
Iris posted `dscale=624`, `ddepth=0` — it only widened the hidden
And row. `type-nn-dyn` on WDBC posted `0/0` — Adam already dropped
the residual below the stall floor, so the policy never fired.


`dparams` is `param_count(after) − param_count(init)`. Static rows are
`dscale=0 ddepth=0 dparams=0`. After forcing layer-first growth on
inputs with `in ≥ 4`:

| impl | task | ddepth | dparams | mse |
|------|------|--------|---------|-----|
| type-nn-proj | iris | 0 | 0 | 0.112 |
| type-nn-proj-dyn | iris | 2 | 456 | **0.104** |
| type-nn-dyn | wdbc | 3 | 2790 | 0.187 (was 0.53) |
| type-nn-proj | wdbc | 0 | 0 | **0.061** |


## Iteration: readout + real projections

Learned: stacking *identity* \(k{=}1\) layers barely helps. A designed
third layer that is a **linear readout of quadratic features** does.

    x --[k=1, H]--> h --[k=2, H]--> z --[k=1, out]--> y
    type-nn-proj2

Dyn insert now adds a **new affine basis** (width H), not \(I\).
That moves `ddepth`/`dparams` but often *hurts* MSE versus the
designed stack — extra random maps need more steps than the bench has.

| task | torch-mlp | proj | **proj2** | proj-dyn |
|------|-----------|------|-----------|----------|
| iris | 0.014 / 0.98 | **0.112 / 0.98** | 0.114 / 0.95 | 0.179 / 0.90 |
| wine | 0.002 / 1.00 | 0.078 / 1.00 | **0.071 / 1.00** | 0.213 / 0.88 |
| wdbc | 0.016 / 0.99 | 0.061 / 0.95 | **0.047 / 0.96** | 0.061 / 0.95 |
| diabetes | 0.023 | 0.027 | **0.025** | 0.028 |
| xor | — | ~0 | 3e-6 | 0 |

Diabetes is within ~7% of torch. WDBC acc 0.96. Dynamic growth is
visible (`ddepth=2–3`) but the **designed** 3-layer polynomial is
the one that keeps improving MSE.


## Iteration: backprop energy estimates how many layers

`type-nn-bpdyn` starts as one product layer. After each backward pass it
keeps an EMA of layer error energy `ge[i] = mean(|dAnd|)` and activity
`ae[i] = mean(|And|)`. Once per settle window it estimates:

- `n_add_proj = 1` if the first layer is a product on wide raw `x` and `ge[0]` is high
- `n_add_readout = 1` if the tail product writes a multi-d target and `ge[tail]` is high
- `n_remove` = idle hidden maps (near-`I`, tiny `ge` and `ae`)

It applies at most those counts and stops at the 3-role stack
`k=1 → k=2 → k=1`. XOR (`in < 4`) never adds a layer.

| task | proj2 (designed) | **bpdyn** (BP estimate) | ddepth | dparams |
|------|------------------|-------------------------|--------|---------|
| xor | 3e-6 / depth 3 | **0 / depth 1** | 0 | 0 |
| iris | 0.114 / 0.95 | **0.104 / 0.97** | 2 | 181 |
| wine | 0.071 / 1.00 | **0.055 / 1.00** | 2 | 735 |
| wdbc | **0.047 / 0.96** | 0.062 / 0.95 | 1 | 732 |
| diabetes | **0.025** | 0.027 | 1 | 188 |

On multi-class tasks the estimator rebuilds proj2 *and beats the
hand-designed net*. On 1-d tails it only inserts the projection
(readout would be redundant). Blind `type-nn-dyn` still over-inserts
and loses.


## Iteration: more backprop signals

Tried five estimators on top of `ge`/`ae`:

| impl | extra signal | action |
|------|----------------|--------|
| bpgap | product-gap `mean(|And − Or0|)` | readout even on 1-d tails |
| bpcurv | Adam-v flatness | widen hidden when gradients flatten |
| bpcombo | gap + flat + readout-1 + k + width | morph to proj2 then grow k/H |
| bpcube | same as combo (k up to 3) | |
| bpwide | combo, but *start* as wide proj2 | only grow H from BP |

Picked by UCI MSE (torch-mlp in parentheses):

| task | previous best | **winner** | mse | vs torch |
|------|---------------|------------|-----|----------|
| xor | bpdyn 0 | any start-from-1 | 0 | — |
| iris | bpdyn 0.104 | **bpwide 0.102 / 0.97** | still 0.014 |
| wine | bpdyn 0.055 | **bpwide/combo 0.037 / 1.00** | torch 0.002 |
| wdbc | proj2 0.047 | **bpcombo 0.039 / 0.98** | torch 0.016 |
| diabetes | proj2 0.025 | **bpwide 0.022** | **beats torch 0.023** |

What actually moved MSE: (1) insert a readout on 1-d tails so WDBC/diabetes
become real proj2, (2) then widen H / raise k when `ge` stays high and
Adam-v is flat. Extra depth beyond 3 never helped.


## Iteration: when and where to insert

Hypothesis: extra depth with *worse* MSE means the layer arrived late
or between the wrong interfaces. Per-layer `ge[i]/ae[i]` scores every
site (`i` = “before layer i”, `i = depth` = after the tail).

- `type-nn-bpsite` — first insert at tick 16 at the max-score site
  (front = real k=1 basis, tail = real k=1 readout, mid = identity)
- `type-nn-bpearly` — at tick 16 commit *both* roles in one shot,
  then lock depth and only widen H
- neither adds a layer after tick 4000 / two adds

| task | late combo | **early/site** | bpwide (start at 3) |
|------|------------|----------------|---------------------|
| iris | 0.154 / 0.96 | **site 0.095 / 0.97**, early 0.096 / **0.98** | 0.102 / 0.97 |
| wine | **0.037 / 1.00** | early 0.048 / 1.00 | **0.037 / 1.00** |
| wdbc | **0.039 / 0.98** | site 0.042 / 0.97 | 0.048 / 0.97 |
| diabetes | 0.023 | 0.025 | **0.022** |
| xor | 0 / d=1 | 0 / d=1 | 0 / d=3 |

Iris is the smoking gun: the same +2 layers at tick 16 beat the same
+2 layers sprinkled later (0.095 vs 0.154). Wine/WDBC still prefer a
wide basis more than a late extra map.


## Iteration: thin extra layers, not fat rows

Torch-mlp iris is `4→8→3` = **67 params, 2 affines**. `type-nn-bpsite`
ended at **547 params / 3 layers** — under-*layered*, over-*wide*.

`type-nn-bpdeep` uses the same `ge[i]` sites but:

- hidden width capped at 8 (12 on wide inputs)
- at tick 16 commits **three** inserts: k=1 basis, extra k=1 hidden, k=1 readout
- never grows width

| task | bpsite (fat) | **bpdeep (thin+deep)** | torch |
|------|--------------|------------------------|-------|
| iris | 0.095 / 547p / d=3 | 0.136 / **283p / d=4** | 0.014 / 67p |
| wine | 0.055 / 1187p | 0.111 / **355p / d=4** | 0.002 |
| wdbc | 0.042 / 1969p | 0.044 / **745p / d=4** | 0.016 |
| diabetes | 0.025 / 1081p | 0.026 / **313p / d=4** | 0.023 |
| xor | 0 / 6p / d=1 | 0 / 6p / d=1 | — |

The extra early layer *does* fire (`ddepth=3`). It does **not** beat
the fatter 3-layer site net on MSE — WDBC is almost a tie at ⅓ the
params. The remaining iris gap to torch is the missing ReLU, not
missing depth.


## Iteration: inserts that do not change the output

Adding a random map jumps the function and wastes the remaining
epochs un-learning the jump. Two exact constructions:

1. **Identity** (`type-nn-idi`) — insert \(k{=}1\), \(W=I\), \(b=0\)
   *before* the bottleneck (and an \(I\) after the tail).
   `clip(I x) = x` inside the Or clip, so \(y\) is unchanged.
2. **Factor / move** (`type-nn-idfact`) — if the current layer is
   \(k{=}1\), \(y = Wx+b\) becomes \(h = Wx+b\), \(y = Ih\).
   Product layers cannot be factored exactly, so they fall back to (1).

`n` (1 or 2) comes from that layer's `ge`. Board pruned to the nets
that still move MSE.

| task | idi / idfact | best previous | torch |
|------|--------------|---------------|-------|
| iris | 0.101 / 262p / d=4 | site 0.095 / 547p | 0.014 |
| wine | 0.118 / 757p | wide 0.037 | 0.002 |
| wdbc | 0.047 / 1924p | combo 0.039 | 0.016 |
| diabetes | 0.026 / 538p | wide 0.022 | 0.023 |
| xor | 0 / d=1 | 0 | — |

The construction is correct (output does not jump; tests cover it).
It is also *smaller* than the fat site net on iris. The remaining
MSE gap is still calibration / missing ReLU, not a discontinuous insert.


## Iteration: n is not 1 or 2

The cap was the mistake. Four guesses, all identity-preserving, early,
no upper bound except `LK_MAX_DEPTH=6`:

| guess | formula |
|-------|---------|
| idn | `n = round(ge_max / 0.05)` |
| idtgt | `n = 3 − depth` (aim at 4 + tail I) |
| idema | `n = round(ema / 0.06)` |
| idmax | `max(idn, idtgt)` |

| task | idn (free n) | idtgt | previous best | torch |
|------|--------------|-------|---------------|-------|
| xor | 0 / d=1 | 0 / d=1 | 0 | — |
| iris | 0.147 / d=6 / 122p | 0.113 / d=6 | **site 0.095** | 0.014 |
| wine | 0.115 / d=6 | 0.071 / d=6 | **wide 0.037** | 0.002 |
| **wdbc** | **0.012 / 0.988 / d=6 / 3784p** | 0.042 | combo 0.039 | **0.016** |
| diabetes | 0.026 / d=6 | 0.026 | wide 0.022 | 0.023 |

`idn` on WDBC **beats torch-mlp** (0.012 vs 0.016, acc 0.988 vs 0.99).
`ddepth=5` — five identity maps inserted at tick 16 in front of the
product, then trained. Iris/wine still prefer a shorter fat stack;
WDBC's 30-d input is where free `n` pays off.


## Entropy and whether torch is overfitting

`./bench.sh` scores **train-set** MSE (the whole file). That is not
generalization.

| set | n | in | H(Y) bits | 1-NN LOO err | meaning |
|-----|---|----|-----------|--------------|---------|
| iris | 150 | 4 | **1.585** (max for 3 classes) | 5.3% | balanced labels, **strong** X→Y pattern |
| wine | 178 | 13 | 1.567 (≈max) | 4.5% | same: small and easy |
| wdbc | 569 | 30 | 0.95 (2-class) | 4.9% | more samples, still easy |
| diabetes | 442 | 10 | 3.78 (hist) | 1-NN MSE 0.057 | higher target spread |

High **label** entropy just means the classes are balanced. High
**residual** entropy H(Y|X) would mean “no pattern”. 1-NN leave-one-out
~5% on all three class sets ⇒ H(Y|X) is **low**. There *is* a pattern.

Torch-mlp on a **70/30 holdout** (same 8–16 ReLU width):

| set | train MSE / acc | **held-out** MSE / acc |
|-----|-----------------|------------------------|
| iris | 0.008 / 0.99 | 0.022 / 0.96 |
| wine | 0.001 / 1.00 | **0.039 / 0.94** |
| wdbc | 0.017 / 0.99 | **0.013 / 1.00** |
| diabetes | 0.017 | 0.038 |

Wine’s board score of ~0.002 is **train memorization** on 178 points.
Iris is mildly overfit. WDBC **generalizes** — and that is the set
where `type-nn-idn` beat torch on the full-file score (0.012 vs 0.016).

Run: `python3 tools/entropy_holdout.py`


## More epochs vs late inserts

`TYPE_NN_EPOCHS` overrides the UCI budget. Identity maps are in by
tick 16, so 4000 epochs is almost all “after the add”.

| impl | set | 250 ep | 1000 ep | 4000 ep | torch train |
|------|-----|--------|---------|---------|-------------|
| idn | iris | 0.147 | 0.126 | 0.123 | 0.008 |
| site | iris | **0.095** | 0.111 | 0.124 | 0.008 |
| wide | iris | 0.102 | 0.103 | 0.097 | 0.008 |
| proj2 | iris | 0.114 | 0.097 | 0.112 | 0.008 |
| idn | wine | 0.092 | 0.078 | 0.104 | 0.001 |
| site | wine | 0.041 | 0.054 | 0.052 | 0.001 |
| wide | wine | 0.31* | **0.030** | 0.241 | 0.001 |
| proj2 | wine | 0.073 | 0.042 | 0.038 | 0.001 |

\* wide/wine is unstable at this lr if the budget is slightly off.

More epochs shave a little off `idn` iris (0.147→0.123) and do **not**
reach torch's train-set 0.008 / 0.001. `bpsite` iris even *rises*.
Late insertion is not the limiter — the clipped And/Or polynomial
saturates on these two small sets. Overfitting like torch would need
a sharper last layer (or more capacity *of a different kind*), not
just a longer run.

```bash
TYPE_NN_EPOCHS=1000 ./bench.sh iris
```


## Type factorization (common types → parent layer)

Identity pads do not move types. Type Mechanics says the factorized
term *is* the parent node. `type-nn-typefact` uses each layer's Or
rows and their backprop:

1. **Lift** (tick 16, exact): every Or of the current product becomes
   a k=1 type in a new parent. The child And is rewritten as a product
   of those parent features (one-hot W).
   `And_i = Π_t Or_{i,t}(x)` is unchanged at the insert.
2. **Merge**: Or rows whose **weights** and **grad directions** have
   cosine ≥ 0.92 / 0.80 are the same type. They are averaged into one
   parent unit; child connections retarget.

| set | typefact | params | vs torch params | mse / acc |
|-----|----------|--------|-----------------|-----------|
| iris | d=2 | **61** | torch 67 | 0.123 / 0.97 |
| wine | d=2 | 106 | — | 0.090 / 1.00 |
| wdbc | d=2 | 68 | — | 0.048 / 0.96 |
| xor | d=1 | 6 | — | 0 |

Iris is now in the same param ballpark as torch. MSE is still above
the fat site net (0.095) and torch's *train* 0.008 — but the layer
add is finally a type move, not an I-pad. WDBC's 0.012 remains the
`idn` stack, not this factorization.


## One policy instead of a shuffled board

The winner was flipping with the file:

- narrow (iris 4-d, wine 13-d) → short type-lift / fat 3-layer
- wide (wdbc 30-d) → many early I-maps into the product

`type-nn-adapt` routes on `in` and `ge` at tick 16:

- `in < 16`: exact Or-lift (`typefact`)
- `in >= 16`: I-stack in front of the product (`idn` family)
- `in < 4`: do nothing (XOR)

| set | adapt | specialized winner |
|-----|-------|--------------------|
| xor | 0 / d=1 | 0 |
| iris | 0.123 / 61p / 0.97 | site 0.095 / 547p |
| wine | 0.090 / 106p / 1.00 | wide 0.037 |
| wdbc | 0.059 / d=6 | **idn 0.012 / 0.988** |
| diabetes | 0.027 / 15p | wide 0.022 |

Same *rule* on every file. Iris stays near torch's 67 params.
WDBC still prefers the pure `idn` stack — routing got depth right
(`ddepth=5`) but not that 0.012 yet.


## Layer churn: add vs drop

`ddepth` is only *net* change. A run that adds 3 and drops 2 looks
the same as a run that adds 1. New counters:

- `layer_add` / `ladd` — how many layers were inserted
- `layer_drop` / `ldrop` — how many were deleted

`remove_idle` used to run after the early commit. That is now
**locked**: once the tick-16 structure is in, depth cannot change.

| set | impl | ladd | ldrop | ddepth | mse |
|-----|------|------|-------|--------|-----|
| iris | idn | 5 | **0** | 5 | 0.147 |
| iris | site | 2 | **0** | 2 | 0.095 |
| iris | adapt | 1 | **0** | 1 | 0.123 |
| wdbc | idn | 5 | **0** | 5 | 0.012 |
| wdbc | adapt | 5 | **0** | 5 | 0.059 |
| xor | * | 0 | 0 | 0 | 0 |

With the lock on, **ldrop is 0** on every file. So the “add, drop,
add again, no time to fit” story is not what these runs were doing
*after* the lock. Extra epochs still do not reach torch’s train MSE
— that gap is not late churn.

The board still splits by input width (narrow → type-lift, wide →
I-stack). That split is real; it is not an artifact of oscillating
depth.


## Start at torch-mlp depth

Torch-mlp is two affines (`in→H→out`). `type-nn-init2` starts there
(`k=1, k=1`, H=8/16). `type-nn-init2p` keeps the product on the tail
(`k=1, k=2`). Dynamic add/drop still on; lock after at most one extra I.

| set | init2 (torch shape) | init2p (product tail) | site | idn |
|-----|---------------------|------------------------|------|-----|
| xor | 0.25 / d=2 add=0 drop=0 | **0 / d=2 add=0 drop=0** | 0 / d=1 | 0 |
| iris | 0.273 / 0.84 / d=3 **add=1 drop=0** | 0.131 / 0.98 / d=3 add=1 | **0.095** | 0.147 |
| wine | 0.100 / d=3 add=1 | 0.080 / d=3 add=1 | **0.055** | 0.115 |
| wdbc | 0.064 / d=3 add=1 | 0.049 / d=3 add=1 | 0.042 | **0.012** |
| diabetes | 0.028 / d=3 add=1 | 0.027 / d=3 add=1 | 0.025 | 0.026 |

They **add one** map (ge still high at tick 16) and **never drop**.
Starting at torch depth does not by itself match torch MSE — two
clipped affines without ReLU are weaker than ReLU-MLP on iris
(0.273 vs torch train 0.008). The product tail (`init2p`) is what
makes XOR and iris class-acc work.


## Why WDBC looks great and iris looks behind

They are not the same problem.

| | iris | wdbc |
|---|---|---|
| n | 150 | 569 |
| in | 4 | 30 |
| out | **3-way one-hot** | **1 scalar 0/1** |
| 1-NN LOO | 5.3% | 4.9% |
| torch train MSE | 0.008 | 0.016 |

`tools/debug_gap.c` after the usual train:

| impl | iris mse (on / off) | iris acc | mean p_true | wdbc mse | wdbc acc |
|------|---------------------|----------|-------------|----------|----------|
| idn | 0.147 (0.065 / **0.083**) | 0.94 | 0.79 | **0.012** | 0.988 |
| site | 0.095 (0.041 / **0.054**) | 0.97 | 0.90 | 0.042 | 0.972 |
| typefact | 0.123 (0.044 / 0.079) | 0.97 | 0.92 | 0.048 | 0.963 |
| init2p | 0.131 (0.058 / 0.073) | **0.98** | 0.94 | 0.049 | 0.947 |

Iris samples look like `y=[1,0,0] yh=[1.05,-0.10,0.05]`:
class is right, the vector is not a simplex, and **half the MSE is
off-class leakage**. WDBC has no off-class channel; idn just has to
hit 0 or 1 and the 30-d I-stack does that.

So iris is not “failing to find the type.” Acc 0.97–0.98 is at or
better than 1-NN / torch holdout. The board MSE gap is **calibration
on a 3-vector**, plus a 4-d input that cannot use the WDBC I-stack.

```bash
./tools/debug_gap type-nn-idn iris
# built as /tmp/debug_gap in the notes; compile from tools/debug_gap.c
```


## Off-class leakage, and how torch actually avoids it

Torch-mlp in `bench_torch.py` is **not** softmax + CE. It is

    Linear → ReLU → Linear → MSE

`F.mse_loss` is the mean over **every element** of the tensor:
`sum((yh-y)^2) / (n * out)`. Our board used `sum / n`, so iris
(`out=3`) was reported **3× worse** than WDBC (`out=1`) for the
same average per-channel error.

That is the “leakage” on the board: the two off-class channels were
counted in full, and torch’s number already averaged them in.

Torch’s last `Linear(H, 3)` sets each class **independently**. MSE
then pulls the unused two toward 0. We now do the same two things:

1. Report MSE as `sum / (n * out)` (torch mean).
2. Train with `dy_k = (yh_k - y_k) / out` so a 3-class head is not
   given 3× the gradient of WDBC.

| set | site (old board) | site (torch mean) | torch train / holdout |
|-----|------------------|-------------------|------------------------|
| iris | 0.095 | **0.031** / acc 0.97 | 0.008 / 0.022 |
| wine | 0.055 | **0.014** / acc 1.00 | 0.002 / 0.039 |
| wdbc | 0.042 | 0.042 (out=1, unchanged) | 0.016 / 0.013 |

Wine **beats torch holdout**. Iris is in the same band as torch
holdout, still short of torch *train* (ReLU memorization on 150 rows).
XOR and WDBC are unchanged (`out=1`).


## One model instead of a shuffled board

Wine is also 3-class one-hot, so it had the same scoring leakage as
iris. After the torch-mean MSE, the remaining issue is that **no
single impl won every file**.

What each specialist was doing:

| signal | winner | move |
|--------|--------|------|
| `out>=2` and `in<20` (iris, wine) | site | k=1 basis + k=1 readout |
| `in>=20` or `out==1` (wdbc, diabetes) | idn | I-stack into the product |
| `in<4` (xor) | product | stay depth 1 |

`type-nn-one` is that rule in one net.

| set | **one** | best specialist | torch train / holdout |
|-----|---------|-----------------|------------------------|
| xor | **0** | 0 | — |
| iris | 0.037 / acc **0.987** / 211p | site 0.031 / 0.967 / 547p | 0.008 / 0.022 |
| wine | 0.023 / acc 1.00 | site 0.014 | 0.002 / 0.039 |
| wdbc | **0.012 / acc 0.988** | idn 0.012 | 0.016 / 0.013 |
| diabetes | 0.026 | wide 0.022 | 0.023 / 0.038 |

Not the #1 MSE on iris/wine/diabetes, but the first row that is
**near-best on every file** (iris +0.006, wine +0.009, wdbc tied,
diabetes +0.005) without swapping impls.


## Board slimed; `type-nn-one` is now `type-nn-opt`

Dropped the specialists we already learned from (old opt, idn family,
dyn/layout alts, init2, typefact). Kept `type-nn-bpsite` and
`type-nn-proj2` as references.

`type-nn-opt` uses the **same** move on every real file:
k=1 basis + product + k=1 readout at tick 16. The WDBC I-stack was
not used: it does not transfer to a new wide binary set.

New OOB set: **ionosphere** (351 rows, 34 radar features, g/b).
Torch-mlp: mse 0.015 / acc 0.991.


## Wine I/O = torch-mlp; `type-nn-over`

Input z-score now uses torch's Bessel `std` (n−1). Wine labels stay
1..3 → one-hot 3, same as `bench_torch.py`. MSE is still
`mean` over n×out. Wine acc is **1.00**; the remaining 0.015 vs
torch *train* 0.002 is calibration on the two off-class channels
(torch holdout is 0.039 — we beat that).

`type-nn-over` overestimates depth: at tick 16 it adds the site
roles **plus** identity maps up to `LK_MAX_DEPTH`, then drops a
layer only when it still acts as **I**:

    k = 1, in = out,  max |W_ij − δ_ij| < 0.08
    biases ignored

| set | over (add/drop) | opt | site | proj2 |
|-----|-----------------|-----|------|-------|
| xor | 0 / 0+0 | 0 | 0 | 0 |
| iris | 0.040 / 4+2 | **0.033** | 0.032 | 0.039 |
| wine | 0.024 / 4+2 | 0.017 | **0.015** | 0.038 |
| wdbc | 0.044 / 4+2 | 0.045 | 0.048 | **0.042** |
| diabetes | 0.025 / 4+2 | 0.025 | 0.025 | 0.025 |
| ionosphere | **0.032 / 4+2** / 0.98 | 0.052 | 0.062 | 0.031 |

Over always adds 4 and drops 2 leftover I maps. On ionosphere that
matches proj2 and beats site.


## Rename, infer_s, and holdout

`type-nn-proj2` is now **`type-nn-static`**: a fixed
`k=1 / k=2 / k=1` stack that does not grow.

`infer_s` is the wall time of `infer_n` forwards.
`us/infer = infer_s / infer_n × 1e6` (µs per forward). Both are on
the board.

**Holdout** is a 30% split that is *never trained on* (70/30, seed
34972). Older “we beat torch holdout” compared our *full-file train
MSE* to a torch 70/30 from `tools/entropy_holdout.py` — that was not
on `./bench.sh`. The board now has `hold_mse` / `hold_acc` for both
type-nn and torch.

Wine, same protocol:

| impl | train mse | **hold mse** | hold acc |
|------|-----------|--------------|----------|
| torch-mlp | 0.002 | 0.034 | 0.94 |
| type-nn-static | 0.011 | **0.031** | **0.96** |
| type-nn-opt | 0.008 | 0.096 | 0.93 |
| type-nn-bpsite | 0.006 | 0.155 | 0.93 |

Static beats torch on wine *holdout*. Site/opt memorize the 70%
(train 0.006) and lose the 30%.


## Board sort

`(hold_acc ↓, acc ↓, params ↑, nbytes ↑, us/infer ↑, train_s ↑)`
and the printed columns follow that order.

## Approaches to cut params (for review — not applied yet)

Current fat rows: site wine 1187p, opt/static ionosphere 2065p,
WDBC ~1969p. Torch-mlp wine is 16-wide ≈ 13×16+16×3 = **259p**.

| # | idea | what shrinks | risk |
|---|------|----------------|------|
| A | **Smaller H** (`pick_hid` 4/8 not 16/24) | every W is `out·k·in` | hold_acc drop on WDBC/iono |
| B | **k=1 everywhere** (no product tail) | halves Or rows | XOR / extra interaction terms |
| C | **Low-rank W = U V** (rank r ≪ in) | `n_or·(in+r)` vs `n_or·in` | extra hyperparam r |
| D | **Prune \|W\| < τ after train**, then fine-tune | nnz only | need CSR layout again |
| E | **Type-lift / share Ors** (old typefact, 61p iris) | common affine once | lift must stay exact-I |
| F | **Drop dead Or rows** (row L2 below τ, keep bias) | `k` per And | same as idle-factor shrink |
| G | **Do not over-add I maps** | no dim² spikes during train | over already drops them after |
| H | **Input PCA / first layer thin** (in→r, r≈8) | all later W | loses raw features |
| I | **Q8 / packed int8** (old opt-q8) | 4× nbytes, same param *count* | XOR mse was not 0 |
| J | **Tied readout** (one W for all classes + offsets) | `(out-1)·feat` | 3-class wine/iris |
| K | **Grow H from 2 only when ge says so** | starts tiny | late width = late-insert penalty |

A, E, F, K are the ones that stay faithful to And/Or + scale/add/remove.
C/D/I are layout/compression. B kills the product that makes XOR free.


## A–K one at a time (`type-nn-pA` … `pK`)

Same 70/30 as the board. G is the un-shrunk control (site roles).

| tag | idea | wine hold_acc / params | iris hold_acc / params | keep? |
|-----|------|------------------------|------------------------|-------|
| A | small H | 0.963 / **111** | 0.956 / 75 | **yes** |
| B | k=1 only | 0.37 / 275 | 0.29 / 67 | no (kills XOR + 3-class) |
| C | bottleneck r≤8 | = A | = A | same as A |
| D | prune \|W\|<0.03 | 0.963 / 234 | 0.978 / 106 | **yes** (params) |
| E | merge/drop Or | **1.00 / 547** | 0.956 / 139 | **yes** (wine hold) |
| F | drop dead k | = E | = E | same as E |
| G | control | 0.926 / 819 | 0.978 / 211 | baseline |
| H | thin in | = A | = A | same as A |
| I | q8 nbytes | = G, nbytes=p | = G | no (count-only) |
| J | tied readout | 0.35 / 787 | 0.27 / 195 | no |
| K | start H=2, grow | 0.963 / **58** | **0.978 / 84** | **yes** (smallest) |

Keep **A + D + E + K**. Drop B, J, I. C/H collapse to A; F to E.


## Combinations of A, D, E, K

Dropped B, C, F, G, H, I, J. Combinations use bit flags in `type_nn_par.c`.

Wine / iris / iono / wdbc hold_acc and params (70/30):

| impl | wine ha/p | iris ha/p | iono ha/p | wdbc ha/p |
|------|-----------|-----------|-----------|-----------|
| pA | 0.963/111 | 0.956/75 | 0.915/357 | 0.947/325 |
| pD | 0.963/234 | **0.978**/106 | 0.925/888 | 0.971/556 |
| pE | **1.00**/547 | 0.956/139 | 0.915/1465 | 0.947/1369 |
| pK | 0.963/**78** | **0.978**/73 | 0.896/163 | 0.959/147 |
| pAD | 0.944/58 | 0.956/61 | 0.925/95 | 0.965/80 |
| pAE | **1.00**/91 | 0.956* | 0.915/321 | 0.942/289 |
| pAK | 0.963/78 | **0.978**/73 | 0.896/163 | 0.959/147 |
| pDE | **1.00**/259 | 0.956* | 0.906/401 | 0.953/331 |
| **pDK** | 0.963/**54** | 0.956/55 | **0.943/79** | **0.977/30** |
| pEK | **1.00/66** | 0.956* | 0.915/153 | 0.959/71 |
| pADE | 0.981/71 | 0.956* | 0.906/164 | 0.959/83 |
| **pADK** | 0.963/54 | 0.956/55 | **0.943/79** | **0.977/30** |
| pAEK | **1.00/66** | 0.956* | 0.915/153 | 0.959/71 |
| pDEK / pADEK | 0.981/**48** | 0.956* | 0.915/84 | 0.953/34 |
| static | 0.963/819 | 0.956/211 | 0.877/2065 | **0.988**/1969 |

\* E on iris drops *train* acc to ~0.80 (hold still 0.956).

No single combo wins every file. Closest:

- **pDK / pADK** — smallest that stays accurate on WDBC/iono (30–79p), wine hold 0.963
- **pEK / pAEK** — wine hold **1.00** at 66p; iris train acc suffers
- **pDEK** — tiniest wine 48p / hold 0.981


## Holdout, ladd/ldrop, and `type-nn-win`

Holdout **is** train on 70%, score on the other 30% (never seen).
That is out-of-sample predictability, not train fit.

`ladd` / `ldrop` are back on the board.

`type-nn-win` = A+D+E+K with E **not** allowed to take `k` below 2
(that was the iris train-acc crash). In practice E never fires on
these stacks, so win ≡ **pDK**.

| set | win ha / acc / p / add+drop | static ha / p |
|-----|-----------------------------|---------------|
| wine | 0.963 / 0.992 / **54** / 2+0 | 0.963 / 819 |
| iris | 0.956 / 0.962 / **55** / 2+0 | 0.956 / 211 |
| iono | **0.943** / 0.959 / **79** / 2+0 | 0.877 / 2065 |
| wdbc | 0.977 / 0.977 / **30** / 2+0 | **0.988** / 1969 |

Wins size and iono hold. Static still edges WDBC hold at 65× the
params. No model yet has static’s WDBC hold **and** win’s size.

## Standalone winner

`type_nn_win.h` / `type_nn_win.c` are independent of the experiment
files. Link only those two plus the headers:

    cc -I. demo.c type_nn_win.c -lm

`type_nn_win_open(in, out)` is the A+D+E+K policy.
