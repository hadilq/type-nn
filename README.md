# type-nn

A typed product network. The algebra is the one in
[Type Mechanics](https://hadilq.com/posts/type-mechanics/): a type is a
generating function, an **Or** is a sum-type (affine factor / incoming
coordinate), an **And** is a product-type, and a **layer** is a partition
function over those clauses. The readout is a natural log. Structure is
not a hyper-parameter — Or width, And width, and Depth are the three
scaling problems the net has to solve while it trains.

The assembly index \(a_{i,r}\) is the one from assembly theory: how many
copies of clause \(r\) the generating function already carries. Raising
\(a\) is another copy of the same type. A new product factor is a new
generator.

Initial depth is the type's own size

\[
\mathrm{depth}_0 = \lfloor 1 + \ln(n m)\rfloor
\]

with \(n\) incoming and \(m\) outgoing. Those layers share the tail's
constructor (typed product + ln) and birth at the task width \(m\).
Every layer in a type-nn model is a typed product + ln — never tanh,
ReLU, or a square \(n\to n\) map. Or-scale may add a dummy coordinate
up to the type-size cap \(1+\ln(1+n)\). A train-time depth dummy is
an identity typed product between the loudest pair. Training then
scales Or / And / Depth **up** while the residual is unexplained, and
**drops** only in the later stages. Dummy Or weights are born at 0 so
`params` only counts what back-prop moved.

## Algebra

Each Or is an affine sum over the incoming type:

\[
\mathrm{Or}_{i,r,t} = b_{i,r,t} + \sum_j W_{i,r,t,j}\, x_j
\]

An And is the product of its Ors raised to the assembly index. Dummy
Or / dummy And are identity factors (\(\times 1\)).

The typed product on a head is the layer's partition function

\[
A_k = \prod_r (\mathrm{Or}_{k,r})^{a_{k,r}},
\qquad
z_k = \mathrm{sign}(A_k)\,\ln\bigl(1 + |A_k|/\tau_k\bigr)
\]

**Every typed layer** emits \(z\). An identity hidden (depth dummy)
skips the log so the map stays \(\times 1\) until back-prop moves the
diagonal. The output of one layer is the input of the next, so a type-nn
stack is as dense as an MLP of the same depth. \(\tau_k\) is a learned
positive scale (born at \(1\)). \(a_{k,r}=1\) recovers the untyped
product.

## The three scaling problems

All three edits happen inside back-prop. Grow early
(\(u < \mathrm{sched\_grow}\)). Drop only late
(\(u \ge \mathrm{sched\_cut}\)).

| axis | scale up | scale down |
|------|----------|------------|
| **Or** | the previous layer adds a dummy output coordinate and trains it; the current layer pairs that new incoming slot with a dummy weight (born at 0) | drop a dummy coordinate of the previous layer's output |
| **And** | keep one dummy Or (\(\times 1\), \(b=1, W=0\)) in the product. If back-prop gives those dummy weights a value, append a new dummy identity Or | if two dummy identity Ors sit on the same And, drop one |
| **Depth** | birth \(\lfloor 1+\ln(nm)\rfloor\) layers. Insert a typed layer **between any two layers** (the loudest residual junction), not only at the ends | drop a layer that has become an identity |

Dummy Ors are born with a full zero weight list on the incoming type.
Without that list, back-prop can only move the bias and And-scaling
never fires.

There is no dataset-name gate and no `max_or` / `max_and` on the board
recipe. Live factors are unbounded except for a type-size fence
\(1+\ln(1+n)\) that does not read a file name. The occupancy rule is
the dummy rule.

## Board models

Each remaining model is its own `type_nn_*.c` / `type_nn_*.h`.
Ablations that were not a full type-nn recipe (or not c-mlp) are gone.

| name | file | what it does |
|------|------|----------------|
| `type-nn` | `type_nn_model.c` | ln on every typed layer; Or / And / Depth dummy rule; early grow, late drop |
| `c-mlp` | `type_nn_cmlp.c` | Linear-ReLU-Linear + ln tail (`./bench_alts TASK c-mlp`) |

The goal of type-nn is to beat `c-mlp` **by scaling**, not by staying a
tiny product.

Every model back-props mean-MSE `(y−t)/n_out`, uses per-sample Adam
with the task lr, and emits the same ln tail. `nbytes` is
`params * sizeof(double)`. XOR `hold_*` is leave-one-out. Details in
[AUDIT.md](AUDIT.md). Strategies: [SCALE.md](SCALE.md). They do not
read a dataset name.

## Fair split

xorshift32 Fisher-Yates, seed 34972, 70/30. Same cut for every impl.
Honesty limits (optimizer, loss scale, readout, `nbytes`) are in
[AUDIT.md](AUDIT.md). Lean statements of the product identities live in
`lean/`.

    ./bench_type_nn iris type-nn
    ./bench_alts iris c-mlp
    make test
    make bench

`us/infer` is mean microseconds per `predict` / `forward` over at least
200 ms of wall time, printed to 6 decimals, floored at 1 ns.
diabetes has no `hold_acc` (regression). XOR `hold_acc` is threshold
accuracy on all 4 points.
