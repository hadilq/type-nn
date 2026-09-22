# type-nn

A neural network whose layers are partition functions. The algebra is the
one in [Type Mechanics](https://hadilq.com/posts/type-mechanics/): a type
is a generating function, an **Or** is a sum type, an **And** is a product
type, and the assembly index comes from
[assembly theory](https://en.wikipedia.org/wiki/Assembly_theory). The
activation is the natural log. Width, degree and depth are not chosen by
hand: they are the three scaling problems the network solves while it
trains.

**Read the post:** [Train the knowledge](https://hadilq.com/posts/train-the-knowledge/).
It explains why a hierarchy of partition functions is our knowledge of the
data, and it carries the typeset math: the forward and backward pass, the
threshold, the evidence rule, and an appendix with every equation this
README writes in plain text.

The goal is to beat the `c-mlp` baseline on the board by finding a more
general and better-optimised scaling strategy. It must do that without
caps or constants tuned to the current data sets.

The board has two type-nn models. They share the architecture and the
three scaling problems and differ only in the scaling strategy:

| model | files | strategy |
|---|---|---|
| **type-nn** | `type_nn.c/h`, `type_nn_scale.c/h` | Knowledge changes only on significant evidence. A structure is kept, or promoted, only while it pays the Bayesian-information price of its parameters, measured on the training data. |
| **type-nn-overfit** | `type_nn_overfit*.c/h` | Frozen reference. Probes are promoted on displacement alone and dropped only when they fall back to identity. It fits the training set about 10× more tightly than `c-mlp` at similar hold-out error. |

![type-nn forward, backward and scaling](doc/type-nn-forward-backward.svg)

## Architecture

Layer ℓ maps an input vector x (width n_ℓ) to an output vector z (width
m_ℓ). Output unit k is one And over its Ors r:

```
Or_kr = w_kr · x + b_kr                  sum type, dense over x
A_k   = Π_r  Or_kr ^ a_kr                product type
z_k   = sign(A_k) · ln(1 + |A_k|)        partition function
```

- **Dense.** The output of one layer is the input of the next
  (n_{ℓ+1} = m_ℓ), and every Or reads every incoming coordinate, so a
  type-nn stack is as dense as an MLP.
- **Assembly index.** `a_kr ≥ 1` is how many times the sub-object `Or_kr`
  is used to assemble `A_k`. It is learned as a real number, with
  `o^a := sign(o)·|o|^a`, so `sign(A_k)` is the product of the Ors' signs.
  Its floor is 1 (a factor present at least once); removing a factor is a
  structural drop, not `a → 0`.
- **Partition function** on every layer, the output layer included. There
  is no temperature and no special case for hidden layers.
- **Birth.** `round(ln(1 + n·m))` layers, `n → m → … → m`, where n is the
  input width and m the output width. Every unit starts as a single random
  Or (a degree-1 And), with the same init law as `c-mlp`.
  - xor 1, iris 3, wine 4, wdbc 3, diabetes 2, ionosphere 4.

The forward pass runs in log space so a deep product never overflows:

```
ℓ_k = Σ_r a_kr · ln|Or_kr|        s_k = Π_r sign(Or_kr)        z_k = s_k · ln(1 + e^ℓ_k)
```

## Backward

```
∂z/∂A     = 1 / (1 + |A|)
∂z/∂Or_r  = s · σ(ℓ) · a_r / Or_r      (dz/dA times the cofactor; finite as A → 0)
∂z/∂a_r   = s · σ(ℓ) · ln|Or_r|
∂Or/∂w_j  = x_j      ∂Or/∂b = 1      ∂Or/∂x_j = w_j
```

σ is the logistic function. At `Or_r = 0` exactly, the slope is the
cofactor if `a_r = 1` and 0 if `a_r > 1`.

Every gradient is taken with the pre-update weights. Then each Or takes
one Adam step, and a is projected back to `a ≥ 1`. `make test` checks
every gradient against central finite differences (the error falls as h²).

## The three scaling problems

All three follow one dummy rule. A **probe** is an identity, and back-prop
is free to move it. A probe that back-prop moved out of the identity is
**promoted**, and a fresh probe takes its place. Late in training,
structure that is not worth keeping is **dropped**. How "worth keeping" is
decided is the one thing the two models do differently.

| axis | probe (scale up, early) | drop (scale down, late) |
|---|---|---|
| **Or**: width | The layer before grows a new output coordinate (a trained unit). The current layer meets it with weights born at exactly 0, so the function does not move. | Remove the coordinate and its column. |
| **And**: degree | Each And carries one identity Or (`w ≈ 0`, `b ≈ 1`); `1^a = 1` for every a. A promoted probe is replaced by a fresh one. | Remove the Or. Every And keeps at least one Or. |
| **Depth**: layers | One identity layer (unit k is the carrier `Or = x_k`) sits in the gap with the largest mean `\|∂L/∂x\|`. That gap can be between any two layers, not only at the ends. An unpromoted probe moves to the loudest gap each epoch. | Remove the layer. The output layer is never removed. |

**Near-exact depth edits.** A type-identity layer emits
`F(x) = sign(x)·ln(1 + |x|)`, which equals x only to first order. When one
is inserted, the next layer's columns absorb the best affine fit
`x ≈ α·F(x) + β`, measured on the epoch that just ran. When one is
removed, they absorb the reverse fit. Coq `fold_exact` proves this is
exact whenever the fit is.

All edits happen at epoch boundaries and never read the hold-out set.

### Schedule and growth gate

- **Phases.** With `u = step / total`: **grow** on `u < 1/3`, **fit** on
  the middle third, **prune** on `u ≥ 2/3`.
- **Growth gate.** Growth also needs an *unexplained residual*: the
  epoch's training MSE must exceed `Var(t) / N`, the constant predictor's
  loss divided by the number of training samples.
  - The model reads the targets from its own gradient,
    `t = y − m·∂L/∂y`.
  - Without the gate, Adam's scale-invariant steps on vanishing gradients
    look like "back-prop wants this probe"; one XOR seed grew to 1,464
    parameters that way.

### Dynamic threshold

Under per-sample Adam with step η, T steps of zero-mean gradient noise
move a parameter about `η·√T`. A one-signed gradient moves it up to `η·T`.
A probe of age T counts as *moved* when its RMS distance from the
identity exceeds the geometric mean of the two:

```
θ(T) = η · T^(3/4) = sqrt( η·√T · η·T )          (Coq: threshold_geometric_mean)
```

Noise falls behind θ as the probe ages. It is computed from the run
(η and age) in `tnn_threshold_up`, and both models use it for promotion.
**type-nn-overfit** also drops anything back inside the band θ(N), with
N the number of training samples (`tnno_threshold_band`).

### The prior of knowledge (type-nn)

type-nn-overfit keeps any structure back-prop moved. type-nn adds a
prior: a structure must be worth its parameters. For an item with k
parameters (an Or, a coordinate, or a layer), on `n = N·m` training
observations, it must satisfy the Bayesian information criterion:

```
n · ln( MSE_without / MSE_with )  >  k · ln n
```

- **Measured, not estimated.** `MSE_without` comes from resetting the item
  to its identity (an Or to `w=0, b=1, a=1`; a coordinate to a zero
  outgoing column; a layer to its carrier) and re-evaluating the training
  pairs back-prop saw in the epoch. Because the ablation *is* the
  identity, removing an item that failed is exact.
- **Grow.** A probe is promoted when it moved past θ(T) **and** passes the
  criterion. Otherwise it stays a probe and keeps training.
- **Prune.** Every Or and coordinate is a candidate. They are ablated in
  order of the damage each does alone. The ablated set keeps growing while
  the model criterion `C = n·ln(MSE) + K·ln n` (K = all parameters) stays
  at or below its best value seen while pruning.
- **Layers.** At most one per boundary. A layer is removed on trial with
  the fold, measured, and restored if C would rise.

Each part came from a measured failure:

- **Estimates were too far off.** Reading the damage from Adam's second
  moment (the empirical Fisher) was off by 10–20× per Or. Greedy pruning
  on those estimates collapsed iris, wine and XOR.
- **A single epoch's MSE is noisy.** Judging a boundary against that
  epoch's own MSE let one Adam loss spike license a drop from 162 to 74
  parameters.
- **The criterion alone starves growth.** Requiring it without the
  displacement test blocked promotion, because a one-epoch-old probe is
  near identity and cannot show evidence yet.
- **Stricter noise estimates broke XOR.** An unbiased variance
  `RSS/(n−K)`, or a noise floor `Var(t)/N`, in the criterion made XOR
  unlearnable: 4 observations cannot justify any XOR model.

The evidence rule runs forward passes over the training samples. It never
reads the hold-out.

### Constants that remain, and why

| constant | where | status |
|---|---|---|
| Adam β₁, β₂, ε and `TNN_LR_SCALE` = 0.1 | `common.h` | Shared by all models; cannot favour one. |
| task epochs and lr | `bench.c` | Fixed per task and shared by all models. |
| schedule thirds, 1/3 and 2/3 | both `*_scale.c` | The least-informative split. A candidate for a dynamic schedule. |
| exponent 3/4 | threshold | Derived (geometric mean), not fitted. |
| BIC price `k·ln n` | `type_nn_scale.c` | The standard criterion; n, k and the MSE come from the run. |
| probe noise amplitude η | probes | One Adam step, the smallest meaningful move. |
| `a ≥ 1` | assembly index | The definition of a factor that is present. |
| `c-mlp` hidden width 8 / 16 | `c_mlp.c` | The baseline's fixed definition. type-nn never reads it. |

There is no width cap, degree cap, depth cap, top-k, or per-dataset table.

## Board

`./bench.sh` writes [BOARD.txt](BOARD.txt). All three models go through
one harness (`bench.c`) with the same protocol:

- The same fixed 70/30 split, with standardisation fitted on the training
  rows only.
- The same per-epoch shuffle, per-sample Adam (`common.h`), mean-MSE
  gradient `(y − t)/m`, epochs, lr, and readout F.
- 5 initialisation seeds per cell, reported as mean ± sd.
- `params` counts every learnable scalar present after training: dense,
  with nothing skipped for being small.

| task | type-nn | type-nn-overfit | c-mlp | type-nn vs c-mlp | params type-nn / overfit / c-mlp |
|---|---|---|---|---|---|
| iris | 0.0215 ± 0.0069 | 0.0249 ± 0.0044 | 0.0280 ± 0.0061 | lower, within noise | 125 / 175 / **67** |
| wine | 0.0277 ± 0.0048 | 0.0209 ± 0.0105 | 0.0234 ± 0.0056 | higher, within noise | **170** / 393 / 275 |
| wdbc | 0.0448 ± 0.0048 | 0.0459 ± 0.0030 | 0.0438 ± 0.0017 | higher, within noise | **163** / 163 / 513 |
| diabetes | **0.0329 ± 0.0010** | 0.0371 ± 0.0015 | 0.0488 ± 0.0042 | **clear win** | **20** / 33 / 193 |
| ionosphere | 0.0886 ± 0.0111 | **0.0813 ± 0.0119** | 0.1461 ± 0.0155 | **clear win** | **154** / 298 / 577 |
| xor (fit only) | train 0.0000 | train 0.0000 | train 0.0000 | — | **26** / 108 / 33 |

Hold-out MSE, mean ± sd over 5 seeds. A gap is "clear" when it exceeds two
standard errors of the difference.

- **Params.** type-nn needs fewer parameters than `c-mlp` on 5 of 6
  tasks: 0.11× on diabetes, 0.27× on ionosphere, 0.32× on wdbc, 0.62× on
  wine, 0.80× on xor. Iris is the exception, at 1.87×.
- **Hold-out.** It clearly beats `c-mlp` on diabetes and ionosphere. The
  other three are within noise, in both directions.
- **Against type-nn-overfit.** Fewer parameters on every task but wdbc (a
  tie), and lower hold-out error on iris, wdbc and diabetes.

## type-nn and the MLP: pros and cons

Measured means it is on this board. Structural means it follows from the
architecture and is proved or tested in this repository, but no
experiment here has shown its benefit on real data yet. Hypothesis means
neither.

**Pros**

- **Not a black box: editable after training (structural).** Every unit
  is an explicit And of Ors with assembly indices. The same ablation that
  pruning uses can reset any Or, input coordinate or layer to its identity:
  - The reset is exact (Coq `and_probe_exact`, `drop_zero_column`), and
    its effect on the loss is measured, not guessed.
  - So after training you can find the factors that lean on a feature
    that encodes a bias in the training data (or that produce an unwanted
    behaviour), remove or reset them, and retrain the rest.
  - In an MLP a hidden unit is a single weighted sum behind a ReLU. It
    has no factor structure to cut at.

  Caveat: the weights inside an Or are still dense real numbers, and a
  debiasing edit has not been demonstrated on a dataset here.
- **No architecture hyper-parameters (measured).** Width, degree (factors
  per And) and depth are found during training; `c-mlp` needs its hidden
  width and depth chosen by hand. Training hyper-parameters remain: lr
  and epochs are set per task, exactly as for the MLP, and the schedule
  thirds are fixed.
- **Fewer parameters (measured).** 5 of 6 tasks, down to 0.11× of `c-mlp`.
  On those tasks the hold-out error is clearly better (diabetes,
  ionosphere) or statistically indistinguishable (wine, wdbc).
- **Better accuracy where it wins (measured).** Clearly lower hold-out MSE
  on diabetes and ionosphere; nowhere clearly worse.
- **A checked structure (structural).** The forward pass, the gradients
  and the function-preserving edits are proved in Coq, and the C
  gradients are checked against finite differences.
- **Towards small local models (hypothesis).** A model that grows only
  the structure its data pays for is a natural fit for local and on-device
  models, local LLMs included. Nothing here tests that: all results are on
  small tabular data sets, and there is no sequence model, batching or GPU
  path yet.

**Cons**

- **More parameters on iris** (1.87× `c-mlp`).
- **Slower training.** Measuring evidence makes training 1.8–5.2× slower
  than type-nn-overfit, and up to about 25× slower than `c-mlp` on these
  tasks.
- **Slower inference on small networks.** It ranges from 0.3× (faster)
  on ionosphere to 8.8× on iris; the logs dominate small networks.
- **Still overfits.** Training MSE sits far below `c-mlp`'s (1e-4 on
  wine). The criterion's noise estimate is the training MSE itself, so the
  prior is lenient exactly where the network overfits most.
- **Engineering.** Per-sample training in plain C, with no batching,
  vectorisation or GPU.
- **Approximate depth edits.** They are near-exact (fold), not exact: the
  log makes an exact identity layer impossible.
- **Evidence uses forward passes.** Decisions evaluate the training
  samples, not only back-prop quantities.

## Files

| file | what |
|---|---|
| `type_nn.c/h` | type-nn: structure, forward, backward, accounting, training-pair cache |
| `type_nn_scale.c/h` | type-nn scaling: probes, schedule, residual gate, measured BIC evidence |
| `type_nn_overfit*.c/h` | type-nn-overfit, frozen (own `tnno_` prefix; shares `common.h`) |
| `c_mlp.c/h` | the baseline: Linear-ReLU-Linear with the same readout F |
| `common.h` | code all models share: F, Adam, RNG |
| `bench.c`, `bench.sh` | the single harness and the board writer |
| `dataset.c/h`, `bench_time.h` | loaders, split, train-only standardisation, timer |
| `test_type_nn.c`, `test_type_nn_overfit.c`, `test_cmlp.c` | formulas, gradient checks, surgery exactness, evidence, invariants |
| `coqLang/` | Coq (Rocq 9) proofs of forward, backward, surgery, readout, threshold |
| `doc/` | the architecture diagram |
| `AUDIT.md` | why the benchmark is honest, fair and consistent, and its limits |

## Running

```
make test        # 60 type-nn + 53 type-nn-overfit + 5 c-mlp checks
make test-asan   # the same under ASan/UBSan
./bench.sh       # all models, all tasks, 5 seeds -> BOARD.txt
TNN_SEEDS=20 ./bench.sh iris
TNN_VERBOSE=1 ./bench iris type-nn      # per-seed lines on stderr
./bench wine type-nn-overfit            # one model, one task
make coq         # Rocq >= 9 with its stdlib; `nix build .#proofs` does it hermetically
make data        # fetch the UCI files if data/ is empty
```

## Proofs

`coqLang/` contains machine-checked statements of the structure the C
code implements; see [coqLang/README.md](coqLang/README.md).

- **Forward.** Composition is dense stacking, and every probe is an exact
  identity.
- **Backward.** Each gradient is the exact first-order coefficient in the
  ring.
- **Surgery.** The depth fold and the width drop are exact.
- **Readout.** F is odd, compressive, and has real derivative
  `1/(1+|A|)`.
- **Threshold.** θ sits between the noise and drift scales.

The ring-level theorems need no axioms.

## License

MIT.
