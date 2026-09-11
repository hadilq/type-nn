# type-nn-B

A C And-Or network that grows and shrinks itself from back-prop.
The published controller is **type-nn-B**: it waits for a flat residual
slope before adding a layer, then uses the same per-layer error energy
to widen, add an Or factor, or drop dead maps.

There is no ReLU. The product is the non-linearity.

Numbers: [BOARD.txt](BOARD.txt). Probe calculus: [PROBES.md](PROBES.md). Why original type-nn hits 0 XOR MSE and type-nn-* does not: [XOR.md](XOR.md).

## What this run achieved

On the same 70/30 split (xorshift32, seed 34972):

- WDBC hold-out **0.971** with 118 parameters, against c-mlp / torch-mlp
  at 0.942 / 0.947 with 513 parameters. More accuracy in a quarter of
  the weights, trained and inferred in C.
- Iris hold-out **0.978** (c-mlp / torch 0.956) with 50 parameters
  instead of 67.
- Diabetes hold MSE **0.032** vs 0.048 (c-mlp) and 0.037 (torch),
  41 parameters vs 193.
- Depth is a function of the residual, not a hyper-parameter:
  iris / wine / diabetes take one extra map; wdbc / ionosphere take
  three. The slope test is what lets those counts differ.
- Train and infer beat c-mlp on every real file. torch-mlp times
  include the Python runtime; they are listed for completeness.

Honest gaps, same split:

- XOR. A fixed tau=0.05 treated four rows as permanently stalled
  (MSE 0.25, depth 5). tau = 0.05 n/(n+48) gives 0.0038 on XOR;
  B adds one basis and lands at MSE 0.003.
- Wine. 0.944 (51/54) vs c-mlp 0.981 (53/54). Both confuse cultivar
  3 with 2; B misses two hold-out bottles, c-mlp misses one.
- Ionosphere. 0.906 vs c-mlp 0.925, but B's hold MSE is better
  (0.096 vs 0.134) and it uses 91 parameters against 577.

B is not the smallest net we trained, and it is not the fastest
type-nn variant. It is the one whose add / remove decisions track
the data.

## Three models

### 1. type-nn-B

Layer L with k Or factors, input x in R^n, output y in R^m:

    Or_{i,t} = clip( b_{i,t} + W_{i,t} · x , ±4 )     t = 1..k
    y_i      = clip( Π_t Or_{i,t} , ±32 )

A stack is composition f = L_{d-1} ∘ … ∘ L_0. The published net
starts as one layer with k=2 (a pair of affines, multiplied).
k=1 is an affine map.

Loss (same trainer for every C model):

    L = (1 / 2o) Σ_i (y_i − y*_i)^2
    ∂L/∂y_i = (y_i − y*_i) / o

Backward, one layer. Let g_i = ∂L/∂y_i. The product rule on the And:

    ∂y_i / ∂Or_{i,t} = Π_{u≠t} Or_{i,u}

then

    ∂L/∂W_{i,t,j} = g_i · (∂y_i/∂Or_{i,t}) · x_j
    ∂L/∂x_j       = Σ_{i,t} g_i · (∂y_i/∂Or_{i,t}) · W_{i,t,j}

Adam (β1=0.9, β2=0.999) steps W and b, clipped to ±4. Gradients
are accumulated over 16 samples before a step.

Each layer keeps two EMAs from this pass:

    ge_ℓ ← 0.9 ge_ℓ + 0.1 mean|g|
    ae_ℓ ← 0.9 ae_ℓ + 0.1 mean|y|

The net keeps ema, an EMA of mean|dy| at the output.

B's stall test. After warmup, once per settle window of SETTLE=48 steps,

    rel = |ema − ema_prev| / ema_prev
    tau = 0.05 * n / (n + SETTLE)

A depth insert is allowed only if rel < tau and ema is still
above a floor, and depth < 1 + floor(log2 n). A four-row file
(XOR) has tau ≈ 0.0038, not 0.05: a settle window revisits every
row a dozen times, so a fixed 5% looks "flat" by construction.

While stalled, in order:

1. drop a hidden k=1 map with W ≈ I and tiny ge, ae;
2. drop a dead extra Or (k>2, ||W||^2 tiny);
3. widen the k=1 layer with the largest ge (never past out = in);
4. otherwise insert at the interface with the largest site score:
   2·ge in front of a product, 1.5·ge after one,
   ½(ge_up+ge_down)·ge_up/(ae_down+ε) in the middle.

A mid-stack insert is identity, so f does not jump. A front insert
is a fresh affine of width max(2, o).

### 2. c-mlp

Same widths torch uses: hidden H = 8 if n ≤ 4 else H = 16.

    h = ReLU(W1 x + b1)
    y = W2 h + b2

Same MSE, same Adam, same split, compiled in C. This is the fair
clock and the fair accuracy baseline.

### 3. torch-mlp

The same two-layer ReLU net in PyTorch. Weights and split match
c-mlp. Infer microseconds include the interpreter; do not compare
them to C as if they were the same machine.

## Fair protocol

- Split: xorshift32 Fisher-Yates, seed 34972, 70% train, computed
  before init, shared by C and Python (dataset_perm).
- Inputs z-scored with Bessel n−1, matching torch.std.
- Wine labels are class-first 1..3 → one-hot 0..2. Same Y for
  every impl.
- `./bench_alts <task> type-nn-B` and `./bench_alts <task> c-mlp`.
- `python3 bench_torch.py real --kind mlp`.

## Build

    make test-alts
    make bench_alts
    TYPE_NN_DATA=./data ./bench_alts iris type-nn-B


## Controllers in the binary

All of these share the same And/Or layer and Adam trainer. Only the
insert / grow / drop rule differs.

| impl | rule |
|------|------|
| type-nn-win | F + H + rank-before-depth + B-stall after a hidden map |
| type-nn-A | refuse only the site that just failed |
| type-nn-B | insert only on a flat residual slope; tau = 0.05 n/(n+48) |
| type-nn-C | width / k / depth are separate gates |
| type-nn-D | stall energy is class margin when out>1 |
| type-nn-E | tail insert is identity |
| type-nn-F | snapshot; revert if ema does not drop |
| type-nn-G | softmax + CE (Ands are not logits — listed as a negative) |
| type-nn-H | grow width toward rank(in) |
| type-nn-I | out>1 ⇒ product emits features + readout |
| c-mlp | Linear-ReLU-Linear, fixed width |

XOR: the one-layer product already fits (win/F, 6 params, mse 0.0015).
B adds a basis because the slope test still fires once, and that extra
map is why B's XOR mse is 0.003 rather than 0.0015.


## Three scalings (original type-nn)

A layer is a list of **Ands**. Each And is a product of **Ors**.
An Or is an affine \(b + W\cdot x\). Identity of a product is the Or
\((b,W)=(1,0)\). Identity of a *sum of Ands* is the zero term
\((b,W)=(0,0)\).

1. **Or probe.** Every And keeps one dummy Or equal to 1. Back-prop
   is allowed to move it. If it leaves a 0.2-ball around \((1,0)\),
   it is now a live factor and a fresh dummy is appended. A factor
   that has collapsed to \((1,0)\) or to \(0\) is dropped, as long as
   one dummy remains. Cap: 8 Ors.

2. **And probe.** Every output index is a *sum* of And terms
   \(y_i = \sum_r \prod_t \mathrm{Or}_{i,r,t}\). One term is a dummy
   0. If back-prop moves it off 0, it is live and a fresh zero term
   is appended. Extra zero terms are dropped. Cap: 4 terms.
   This is what had gone missing; it is back in `type_nn.c` and in
   every type-nn-*.

3. **Layer probe.** Identity layer insert / drop / width, with the
   controllers type-nn-win and A–I (slope-stall, revert, rank, …).

Original type-nn still trains with per-sample SGD and snap-to-{0,1}.
type-nn-* still trains with Adam. Same probes, two trainers.

Every `type-nn-*` name (win, A–I) is the same And/Or net in `type_nn_win.c`: dummy Or and dummy And in the product, then a layer controller. c-mlp is the dense ReLU baseline, not a type-nn.
