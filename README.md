# type-nn

A C library for a dynamic AND–OR network. Each output is a product of
affine units:

```
Or_{i,k}  =  b_{i,k} + Σ_j w_{i,k,j} · x_j
And_i     =  Π_k Or_{i,k}
```

The product is the non-linearity. There is no ReLU. `type-nn-win` starts
as one rank-2 product layer and then **adds, drops, or resizes** using
only signals that back-prop already computes.

Numbers for the fair hold-out live in [`BOARD.txt`](BOARD.txt).

## Dynamic rule (what back-prop is allowed to change)

Each layer already computes two scalars on the backward pass:

- `ge[i]` — EMA of mean `|dAnd|` arriving at layer `i` (error energy)
- `ae[i]` — EMA of mean `|And|` leaving layer `i` (activity)

and the net keeps `ema`, the EMA of output energy: mean `|dY|` when
`out = 1`, class margin `max(0, runner-up − winner + 1)` when `out > 1`
(`|dY|` goes quiet once train acc is 1.0; margin does not).

At most one move per settle window, in this order:

1. Drop a hidden `k=1` map with `W≈I` and tiny `ge`, `ae`.
2. Drop a dead extra Or (`k>2`, `‖W‖²` tiny).
3. If a `k=1` map has unused column rank (`out < in`) and `ge` is still
   falling, widen it by 1. Do not add depth while rank remains.
4. If energy is stalled — and, once a hidden map exists, the *slope* of
   `ema` is flat — insert at the highest-scoring interface. Snapshot
   weights; revert the insert if `ema` does not drop.

Site score is only `ge` / `ae`:

- front of a product: `2·ge` (raw coordinates through a product)
- after a product tail: `1.5·ge`
- between layers: `0.5·(ge_up+ge_down)·ge_up/(ae_down+ε)`

What we tried and rejected as *the* rule: CE on Ands (they are not
logits), a forced readout before a basis (hurts wine), per-layer `lr∝ge`
and residual-aligned new columns (hurt wide binary files). Those stay
in `type-nn-A`…`I` for comparison.

## Fair split

Every impl, C and Python, uses the same 70/30:

- xorshift32 Fisher–Yates, seed `34972` (`dataset_perm()`)
- computed **before** weight init
- epoch order in `type_nn_alt_train` uses the same generator

`c-mlp` is Linear→ReLU→Linear + Adam in C with torch-mlp's widths
(`H = 8` if `in≤4` else `16`). Compare quality **and** wall time to
`c-mlp`. `torch-mlp` is the same model in Python; its microseconds
include the interpreter.

## Build

```bash
make test          # frozen type_nn.c + type-nn-win
make data          # UCI files into ./data (already shipped here)
TYPE_NN_DATA=./data ./bench.sh real
```

`BOARD.txt` is the last full table from that command.
