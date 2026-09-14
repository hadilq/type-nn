# Audit: is this benchmark honest, fair, and consistent?

Split, seed, and task list are the honest part. Almost everything else
needs a label.

## What is fair

- Same 70/30 cut for every impl: xorshift32 Fisher-Yates, seed 34972.
- Same epoch count and reported lr per task.
- Hold-out is never used for a structural decision. Spawn / insert
  look only at the sample just backwarded.
- `c-mlp` is the same architecture torch-mlp trained (`Linear-ReLU-Linear`,
  H=8 if in≤4 else 16). Param counts match that formula.

## What is not apples-to-apples

1. **Optimizer.** type-nn is per-sample SGD (optionally Adam on ln-*).
   c-mlp is per-sample Adam with `lr *= 0.15` to stand in for torch's
   full-batch Adam. The 0.15 fudge is documented in `type_nn_cmlp.c`
   and is a fairness patch, not a derivation.
2. **Loss gradient scale.** type-nn back-props `y − t`. c-mlp / the old
   alt trainer back-prop `(y − t) / out` and Huber-clip the residual.
   On iris/wine (`out=3`) that is a 3× difference in step size.
3. **Readout.** type-nn tail is `tanh(z/√d)` or `sign(z) ln(1+|z|/τ)`.
   c-mlp tail is a raw linear. MSE on those two heads is not the same
   coordinate.
4. **nbytes.** type-nn counts linked-list node headers. c-mlp counts
   `params * sizeof(double)`. Do not rank memory from this column.
5. **XOR has no hold-out.** All four points are the train set.
   `hold_mse == mse`. Diabetes is regression; acc prints `n/a`.
6. **BOARD.txt was a museum.** The previous board mixed live recipes
   with type-nn-A..I, win, and torch rows from older binaries. A–I
   and win used clip+tanh, Adam-every-16, and a different init scale
   (`XOR.md`). They were not the same algebra as `type_nn.c`.

## Why small type-nn nets can beat c-mlp on MSE and lose on acc

MSE is mean squared error on the *readout*, not 0-1 class error.
A shallow product can sit close to the one-hot targets in L2 and
still flip an argmax on a hold-out row. c-mlp's ReLU hidden is a
better decision boundary on ionosphere/wine even when its MSE is
worse. That is a real model gap, not a scoring bug.

## Why the old depth probe destroyed hold-out

`scale-layer` / `Ljac` inserted an identity hidden *after* the tail
had specialized to raw x, then often dropped it the same epoch
(`max_depth=2`, drop on identity ∧ small ‖dW‖). On iris that took
hold_acc from 0.978 to 0.356. The dummy-layer story is fine; the
timing was not.

New probes:

- `depth-early` — identity hidden at the first train step, square,
  kept until 65% of the schedule and never dropped back to depth 1.
- `depth-hold` — insert on a large residual without waiting for
  Or/And caps; same late drop rule.
- `depth-born` — hidden exists *before* `init_weights` (random,
  c-mlp width) and is kept. Closest type-nn cousin of c-mlp.

## Consistency of the remaining implementations

- `scale-energy` / `scale-jac` / `scale-mix` share `type_nn_grow.c`
  and the dummy rule. mix = energy on Or, jac on And. That matches
  `PROBES.md`.
- Layer insert is the same identity constructor the tests use.
- c-mlp no longer lives behind the A–I registry.

Removed from the tree (not just the board): `type_nn_win.c`,
`type_nn_stack.c`, `type_nn_layerkit.h`, `bench_torch.py`, and the
A–I openers.


## Did we beat c-mlp?

Hold-out acc (classification) / hold-out MSE (diabetes):

| task        | best type-nn                      | c-mlp        | winner  |
|-------------|-----------------------------------|--------------|---------|
| iris        | scale-mix **0.978**               | 0.956        | type-nn |
| wine        | scale-* 0.981 (one hold-out miss) | **1.000**    | c-mlp   |
| wdbc        | scale-mix **0.953**               | 0.936        | type-nn |
| diabetes    | scale-* **0.033** MSE             | 0.048        | type-nn |
| ionosphere  | depth-born 0.906 / MSE **0.072**  | **0.943** / 0.168 | split |

Low-MSE shallow products already beat c-mlp on iris, wdbc, and
diabetes. They lose wine by one hold-out row. Ionosphere is the
remaining acc gap: energy/mix over-spawn (train MSE > 1); jac stays
sane (0.821) but thin; depth-born cuts hold MSE in half vs c-mlp
and reaches 0.906 acc without catching 0.943.

depth-early / depth-hold still print the 0.356 iris floor. Inserting
an identity product in front of a specialized tail is not the same
object as c-mlp's random ReLU hidden. depth-born is.


## Why is hold_acc sometimes n/a?

Two different reasons. They are not a scoring bug.

1. **diabetes** is a real-valued target (disease progression). There
   is no class label, so argmax accuracy is undefined. Compare
   `hold_mse`. The bench sets `Dataset.classification = 0` and prints
   `hold_acc = -1` → `n/a`.
2. **XOR used to print n/a** because the bench trained on all four
   points and skipped the accuracy path (`acc = -1`). There is no
   70/30 cut on four rows. The bench now reports threshold-0.5
   accuracy on those four points, and `hold_acc == acc`.

iris / wine / wdbc / ionosphere are classification and always have
both numbers.
