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

## Dynamic rule

No per-file recipe, no `tick == 16`, no locked `k=1 / k=2 / k=1` stack.

After a short warmup (enough samples for an EMA to exist), at most one
move per settle window:

| signal | move |
|--------|------|
| hidden layer is `k=1`, `W≈I`, tiny `ge` and `ae` | drop that layer |
| extra Or factor with `‖W‖²` tiny and `k>2` | drop that factor |
| residual EMA stalled above a floor | consider a grow/add |
| a `k=1` layer has the highest `ge`, and the last width grow lowered `ge` | widen that layer by 1 |
| a product (`k≥2`) still carries error | add an Or factor (`≈1`, so the And does not jump) |
| interface `i` has the highest site score | insert a layer there |

Site score uses only that layer's error energy `ge` and activity `ae`:

- in front of a product: `2·ge` (raw coordinates through a product)
- after a product tail: `1.5·ge` (needs a linear mix of features)
- between layers: `0.5·(ge_up + ge_down)·ge_up / (ae_down + ε)`

A new front/tail map is a fresh `k=1` affine whose width starts at
`max(2, out)` and is then grown by `ge`, never past `in` (a linear map
wider than its input is redundant). A mid-stack insert is identity, so
the function does not jump. If an insert does not move the residual
EMA, further inserts are refused.

XOR is not special-cased. It may pick up one basis layer if the first
settle window still looks stalled; width then stops at `out == in`.

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
