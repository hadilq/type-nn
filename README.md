# Vortex

A small C library for a **dynamic AND-OR network**: each output unit is a
product of linear units (no pointwise activation). The topology is stored as
sorted linked lists so a layer can grow or shrink its width, its number of
product factors, and the number of layers themselves during back-prop.

```
And_i  =  Π_k  Or_{i,k}
Or_{i,k}  =  b_{i,k} + Σ_j  w_{i,k,j} · x_j
```

That product-of-affine-forms is already a multivariate polynomial, which is
why a single layer can fit XOR (a quadratic) without a ReLU/sigmoid.

## Layout

| file            | role                                      |
|-----------------|-------------------------------------------|
| `vortex.h`      | public structs + API                      |
| `vortex.c`      | forward, backward, dynamic reshape        |
| `test_vortex.c` | unit + scenario tests                     |
| `run.c`         | XOR / sin(x) / structure demos            |
| `Makefile`      | `make test`, `make test-asan`, `make demo`|

## Build

```bash
make test          # compile + run the suite
make test-asan     # same, with ASan + UBSan
make demo          # XOR + sin(x) + structure walk
```

Needs a C11 compiler and libm. A Nix flake is included if you use direnv/nix.

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

`./test_vortex` covers:

1. Known-weight forward (`3 * 4 = 12`)
2. Loss drops on a tiny regression
3. XOR trains to the right quadrant
4. Input width grows on a longer vector
5. Output width grows and shrinks
6. Input width shrinks and drops weights
7. Explicit layer insert + remove + link integrity
8. OR-factor growth under a large residual
9. Automatic layer insert respects `max_depth`
10. A hand-stacked 2-4-1 net trains without blowing up
11. `network_predict` fills the caller buffer
12. Identity insert is function-preserving

## API sketch

```c
Network *net = network_create(2, 1);
network_set_learning_rate(net, 0.08);
network_set_dynamic(net, 1);          /* default on */
network_init_weights(net);
network_train(net, X, Y, n_samples, epochs);

double y;
network_predict(net, x, 2, &y, 1);

network_insert_identity(net, net->tail);
network_remove_layer(net, net->head);
network_free(net);
```
