# Vortex

A small C library for a **dynamic AND-OR network**: each output unit is a
product of affine units. The topology is stored as sorted linked lists so a
layer can grow or shrink its width, its number of product factors, and the
number of layers themselves during back-prop.

```
And_i     =  Π_k  Or_{i,k}
Or_{i,k}  =  b_{i,k} + Σ_j  w_{i,k,j} · x_j
```

## Why there is no activation

A stack of affine maps is still an affine map: without a pointwise
non-linearity a conventional MLP cannot fit XOR, or any other function that
is not a hyperplane.

Vortex is already non-linear **before** any σ/ReLU/tanh is applied, because
an And node of *K* Or factors is a multivariate polynomial of total degree
*K*:

| K (OR factors) | function class                         | example                         |
|----------------|----------------------------------------|---------------------------------|
| 1              | affine (rank-1)                        | `w·x + b`                       |
| 2              | quadratic (rank-2)                     | XOR, `x y`, `x² − y²`           |
| 3              | cubic (rank-3)                         | `x y z`, 3-way interactions     |
| K              | degree-K polynomial                    | any monomial of degree ≤ K      |

So the missing activation is not an omission. ReLU/sigmoid exist to *manufacture*
non-linearity on top of matrix multiplies. Here the product *is* the
non-linearity, and raising the number of Or factors raises the polynomial
rank the layer can fit. A single 2-factor layer fits XOR to MSE `0` without
a hidden ReLU layer.

Identity hidden layers (`(1·x_i)·(1)`) can still be stacked when extra depth
is useful; they do not change the function until training moves the weights.

## Layout

| file              | role                                              |
|-------------------|---------------------------------------------------|
| `vortex.h`        | public structs + API                              |
| `vortex.c`        | forward, backward, dynamic reshape                |
| `test_vortex.c`   | unit + corner-case + scenario tests               |
| `run.c`           | XOR / sin(x) / structure demos                    |
| `bench_vortex.c`  | C wall-clock + RSS bench                          |
| `bench_torch.py`  | PyTorch CPU baselines (MLP and poly-linear)       |
| `bench.sh`        | runs both and prints a comparison table           |
| `Makefile`        | `test`, `test-asan`, `demo`, `bench`              |

## Build

```bash
make test          # compile + run the suite
make test-asan     # same, with ASan + UBSan
make demo          # XOR + sin(x) + structure walk
make bench         # Vortex vs PyTorch (needs python3 + torch)
```

Needs a C11 compiler and libm. The PyTorch bench needs `python3` with
`torch` installed. A Nix flake is included if you use direnv/nix.

## Dynamic scaling

| operation | how |
|-----------|-----|
| wider input | `layer_align_inputs(l, n)` or just `network_predict` with a longer `x` |
| narrower input | `layer_align_inputs(l, n)` drops indices `≥ n` |
| more / fewer outputs | `layer_set_outputs(l, n)` |
| extra product factor | large `‖dL/dAnd‖` appends an OR initialised to `1` (product unchanged) |
| drop a factor | OR ≈ `1` (bias 1, weights 0) or OR ≈ `0` is removed if >1 remain |
| add a layer | `network_insert_identity(net, at)`; back-prop also inserts one identity hidden layer when the net is still depth 1 and `‖din‖` is large |
| drop a layer | `network_remove_layer(net, l)` (refuses the last / output layer). Not done mid-backward — that used to free the `din` list still in use. |

Identity layers are initialised so

```
out_i = (1 · x_i + 0) · (0 · x + 1) = x_i
```

and therefore inserting one does not change the function the net computes.

## Tests

`./test_vortex` covers the original scenarios plus corner cases:

- known-weight forward (`3·4 = 12`, `x²−y²`, independent multi-output)
- NULL / trivial API, rejected learning rates, 0-epoch train
- predict with short and long output buffers
- `dynamic=0` freezes OR-count and depth
- grow then shrink input/output several times
- identity insert in the middle of a stack; drop head vs tail
- three stacked identities remain function-preserving
- clip of huge inputs stays finite
- negative and zero inputs
- MSE on a truncated list
- multi-output training drops loss
- `param_count` / `nbytes` track reshape
- train after a wider `predict` grew the input side
- XOR, loss drop, auto layer insert, stacked 2-4-1
- `create(0,0)` clamps to 1→1
- `max_or` / `max_depth` caps
- `add_layer` prev/next and `net->out_size`
- remove-head realigns the surviving layer
- two successive `train()` calls
- grow outputs then train the new heads
- quantization snap on near-0 weights
- identity insert increases `param_count` / `nbytes`
- `predict` never inserts layers

## Benchmarks

`make bench` (or `./bench.sh`) times **training** and **single-sample
inference** on CPU against two PyTorch implementations, which together are
the relevant “best available” baselines for this size of problem:

| impl        | what it is |
|-------------|------------|
| `vortex`    | this library (linked-list product-of-affines, SGD) |
| `torch-mlp` | `Linear + ReLU` MLP trained with Adam — the production default |
| `torch-poly`| `Linear` on explicit degree-2 features — same function class as a 2-OR Vortex layer |

Tasks:

| task         | data                          | Vortex shape    |
|--------------|-------------------------------|-----------------|
| `xor`        | 4 points                      | 2 → 1           |
| `quadratic`  | 64 samples of `x y + 0.25 x`  | 2 → 1           |
| `mlp32x16x8` | 256 samples, 32-in 8-out      | 32 → 16 → 8     |

Inference is one sample at a time on a single CPU thread so the APIs match.
PyTorch would pull ahead on a large *batched* GPU workload; that is a
different regime than a few-dozen-parameter polynomial net.

Memory: Vortex reports the live node heap (`network_nbytes`) plus process
RSS. PyTorch RSS includes the interpreter, dispatcher and MKL/OpenMP
runtime, so RSS is not an apples-to-apples parameter-memory number — use
`params` / `nbytes` for that.

```bash
./bench.sh              # all tasks
./bench.sh xor          # one task
```

Example, CPU, 1 thread (this machine):

| task        | impl        | train_s | µs/infer | rss_kb | params | mse     |
|-------------|-------------|---------|----------|--------|--------|---------|
| xor         | vortex      | 0.0006  | 0.11     | 1604   | 6      | 0       |
| xor         | torch-mlp   | 0.205   | 22.4     | 645172 | 33     | ~0      |
| xor         | torch-poly  | 0.131   | 41.4     | 646884 | 7      | ~0      |
| quadratic   | vortex      | 0.0028  | 0.12     | 1604   | 6      | 0       |
| quadratic   | torch-mlp   | 0.081   | 21.9     | 645932 | 65     | 2e-4    |
| quadratic   | torch-poly  | 0.065   | 39.7     | 646884 | 7      | ~0      |
| mlp32x16x8  | vortex      | 0.868   | 20.2     | 1860   | 1328   | 0.235   |
| mlp32x16x8  | torch-mlp   | 0.063   | 22.8     | 646740 | 664    | 0.019   |
| mlp32x16x8  | torch-poly  | 0.594   | 3542     | 647140 | 4496   | 2e-4    |

Reading that table:

- On **XOR / rank-2 polynomials** Vortex is the right tool: 6 parameters, ~0.1 µs/sample, exact fit, ~400× less RSS than a PyTorch process.
- On a **wide dense MLP** PyTorch + MKL wins training (GEMM vs walking linked lists) and fits the random map better because ReLU MLPs and explicit polynomial features are a larger function class than two product-of-affine layers. That is expected — Vortex is not trying to be a GEMM engine.
- Compare `params` / Vortex `nbytes` for model footprint. PyTorch `rss_kb` is dominated by the runtime, not the 664 floats.

## API sketch

```c
Network *net = network_create(2, 1);
network_set_learning_rate(net, 0.08);
network_set_dynamic(net, 1);          /* default on */
network_set_verbose(net, 0);          /* mute epoch logs */
network_init_weights(net);
network_train(net, X, Y, n_samples, epochs);

double y;
network_predict(net, x, 2, &y, 1);

network_insert_identity(net, net->tail);
network_remove_layer(net, net->head);
network_free(net);
```

`network_param_count` / `network_nbytes` walk the live lists so tests and
the bench can see how reshape changes the footprint.

## What was broken in the original draft

The first version never finished training: `compute_accums` walked an
`OrNode *` that was already `NULL` (segfault), `or_forward` compared *next*
indices but accumulated *next* values (every pair of features double-counted
the last one and skipped the first), several `free`s were followed by
`weight_init(tmp, …)`, `layer_drop` / `and_drop` / `or_remove` leaked the
inner lists, `prev` links were never set, and the zip-merge on `InOutNode`
could not keep two sparse lists aligned.

Those paths are rewritten:

- forward is a straight index-zip and does **not** mutate the graph
- accumulators are computed with an explicit prefix/suffix product
- SGD is `θ ← clip(θ − lr · ∂L/∂θ)` with optional snap-to-{0,1}
- grow / shrink happens in dedicated helpers that tests call directly and
  that back-prop can trigger when `network_set_dynamic(net, 1)` is on
