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
