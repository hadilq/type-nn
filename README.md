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
make data          # curl the four public datasets into ./data
make bench         # type-nn vs PyTorch (needs python3 + torch)
```

With Nix / direnv (`use flake`):

```bash
nix develop        # sets TYPE_NN_DATA to the flake-fetched datasets
make bench
# or just the datasets package:
nix build .#datasets
```

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
