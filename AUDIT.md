# Audit: is this benchmark honest, fair, and consistent?

Split, seed, optimizer, readout, loss scale, and `nbytes` are the
same object on every board row.

## What is fair

- Same 70/30 cut for every impl: xorshift32 Fisher-Yates, seed 34972.
- Same epoch count and reported lr per task.
- Hold-out is never used for a structural decision. Spawn / insert /
  drop look only at the sample just backwarded, plus the schedule
  fraction \(u = \mathrm{step}/\mathrm{span}\).
- `c-mlp` is Linear-ReLU-Linear then the type-nn ln tail, H = 8 if
  in≤4 else 16. Param counts match that formula plus one `τ` per head.
- The board type-nn is `type_nn_model.c` on top of
  `type_nn.c` + `type_nn_ln.c` + `type_nn_grow.c` + `type_nn_layer.c`.
- Birth depth is `floor(1 + ln(n m))`. Birth width is the task type
  `m`. Neither formula reads a dataset name.
- Dummy Or weights are born at 0. `params` / `nbytes` skip
  `|w| < 1e-12`, same accounting as c-mlp (`params * sizeof(double)`).

## What was not — now fixed

1. **Optimizer.** type-nn and `c-mlp` both take per-sample Adam with
   \(β_1=0.9\), \(β_2=0.999\), \(\varepsilon=10^{-8}\). Both multiply
   the printed task lr by the same `TNN_ADAM_LR_SCALE` (0.1). There
   is no one-sided fudge.
2. **Loss gradient scale.** Both back-prop mean-MSE
   `(y − t) / out`. No Huber clip.
3. **Readout.** Both tails are
   \(y = \mathrm{sign}(z)\,\ln(1+|z|/τ)\) with learned \(τ>0\) born
   at 1. MSE is on that coordinate. type-nn also applies this ln on
   every *typed* hidden layer so the stack is as dense as an MLP.
   Identity hiddens skip the log so a depth dummy stays \(\times 1\).
4. **nbytes.** Both columns are `params * sizeof(double)`. Learned
   \(τ\) on every ln layer is counted.
5. **XOR hold-out.** Train metrics still use all 4 points (the
   Boolean). `hold_mse` / `hold_acc` are leave-one-out: train on 3,
   score the held point, average the four folds. Same protocol on
   type-nn and c-mlp. Diabetes is still regression (`hold_acc` n/a).
6. **No \(n\to n\) maps.** Birth and train-time insert are
   `network_insert_typed(at.in → m)`. Ionosphere is
   \(34\to 1\to 1\to 1\), not three dense \(34\to 34\) products.
   There is no tanh / ReLU / Linear-ReLU hidden in a type-nn model.
7. **And-scale is a dummy Or.** \(A_k=\prod_r \mathrm{Or}_{k,r}^{a_{k,r}}\)
   is one product per head. `and+` on the board is 0 for every task.
   Extra clauses are not And-scale.

## Consistency of what is left

- Dummy rule is the same object on Or, And, and Depth: inject `×1`,
  train, drop extras that fell back to identity.
- Dummy Or is born with \(b=1\) and a zero weight on every incoming
  coordinate. A NULL weight list cannot receive \(\partial L/\partial W\)
  and And-scaling is dead — that was the board bug.
- Assembly index `a_{i,r}` is strictly positive on every ln recipe.
  Each layer's typed product is the generating function
  \(A = \prod \mathrm{Or}^a\), \(z = \mathrm{sign}(A)\ln(1+|A|/τ)\).
- Cut thresholds vary with the forward contribution of each factor.
- Hidden layers run the same back-prop scaling as the tail. Insert
  may land between any two layers.

## What was dropped

`winner`, `pulse-wide`, `forge`, `phase`, `signal`, `norm`,
`assemble`, `compose`, and the old `scale` wrapper were either
partial ablations or compact controls that refused to scale up then
down. They were not a faithful type-nn. Their `type_nn_*.c` / `.h`
are gone. The board is `type-nn` vs `c-mlp` only.

## Why type-nn and c-mlp can disagree on MSE vs acc

MSE is mean squared error on the *readout*, not 0-1 class error.
A product can sit close to the one-hot targets in L2 and still flip
an argmax on a hold-out row. That is a real model gap, not a scoring
bug.

## Did we beat c-mlp?

See `BOARD.txt` from this pass (grow u<0.30, drop from 0.70, dummy Or
born with W=0 so And-scaling can fire without inflating params,
birth `floor(1+ln(n m))` typed layers of width m, type-size top-k
after the cut). `and+` is 0 on every task.

params vs c-mlp:
- xor          8 < 34
- wine       156 < 278
- diabetes    95 < 194
- wdbc       211 < 514
- ionosphere 212 < 578
- iris        85 > 70  (only miss; birth is type-width 3, four live Or promotions)

hold_mse vs c-mlp:
- iris        type-nn 0.0223 < 0.0335
- wine        type-nn 0.0116 < 0.0224
- diabetes    type-nn 0.0351 < 0.0436
- ionosphere  type-nn 0.0819 < 0.1328
- xor train MSE is 0 on both (2000 epochs, same protocol)
- wdbc still behind on hold-out (211 params vs 514; gap is model,
  not scoring). Hold acc 0.924 vs 0.942.

Fairness of the comparison is unchanged: same split, seed, epochs,
lr, Adam, mean-MSE, ln tail, nbytes. Remaining losses are model
gaps, not scoring.
