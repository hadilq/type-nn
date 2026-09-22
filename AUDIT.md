# Audit: is the benchmark honest, fair and consistent?

This file records the audit of the version this repository replaced. It
covers what was wrong, the evidence, what changed, and what is still
limited.

## 1. Honesty of the reported board

**The board was one favourable seed.** Every cell was trained from a
single initialisation seed (34972). Re-running the *original* code with
seeds 1–5 gives this (mean ± sd over 5 seeds, hold-out MSE):

| task | old board type-nn | old type-nn, seeds 1–5 | old c-mlp, seeds 1–5 | claimed | measured |
|---|---|---|---|---|---|
| wine | 0.0116 | 0.0391 ± 0.0152 | 0.0234 ± 0.0048 | type-nn 2× better | type-nn worse |
| diabetes | 0.0351 | 0.0425 ± 0.0017 | 0.0456 ± 0.0050 | clear win | within noise |
| iris | 0.0223 | 0.0273 ± 0.0036 | 0.0307 ± 0.0057 | win | within noise |
| ionosphere | 0.0819 | 0.1048 ± 0.0282 | 0.1307 ± 0.0131 | win | lower, noisy |
| wdbc | 0.0659 | 0.0602 ± 0.0052 | 0.0450 ± 0.0021 | loss | loss |

The board seed sits well inside the favourable tail on wine and diabetes.
Whatever the intent, a board tuned while looking at one seed reports that
seed.

**Fix.** `bench.c` trains every cell from `TNN_SEEDS` seeds (default 5)
and reports mean ± sd. `BOARD.txt` prints a verdict per task: clear only
if the gap exceeds two standard errors of the difference.

## 2. Fairness of the protocol

| issue in the old harness | effect | fix |
|---|---|---|
| Inputs standardised, and regression targets min-maxed, on train **and** hold-out rows | hold-out statistics leak into both models | `dataset_standardize_train` / `dataset_minmax_train` fit on training rows only |
| c-mlp reshuffled every epoch; type-nn trained in fixed order | different optimisation problem per model | one training loop, one shuffle, in `bench.c` |
| Two harnesses (`bench_type_nn.c`, `bench_cmlp.c`) with copied logic | drift risk: different XOR init seeds, different timing paths | one harness with a model vtable |
| type-nn `params` skipped weights with \|w\| < 1e-12; a post-training top-k cut each Or to 1+ln(1+fan) weights with no retraining | the "fewer parameters" claim came from a sparse model the spec does not describe | dense model; `params` counts every scalar present |
| Adam's bias correction shared one global step; parameters born late got a ~3× first step | new structure trained differently from old | one step counter per Or; c-mlp is one group from step 1 |
| XOR "hold-out" was leave-one-out on 4 points | any model that fits 3 points must mispredict the 4th: hold_acc was 0.000 for both | XOR is fit only; hold-out n/a |
| type-nn inference allocated linked lists on every forward | timing measured `malloc`, not the model | dense arrays, no allocation in forward |

What was already fair, and is kept: the same split algorithm and seed,
the same epochs and lr per task, the same Adam constants and learning-rate
scale, and the same mean-MSE gradient.

## 3. Faithfulness of the old type-nn to its design

| design | old implementation | now |
|---|---|---|
| $z_k=\operatorname{sign}(A_k)\ln(1+\lvert A_k\rvert)$ | $\ln(1+\lvert A\rvert/\tau_k)$ with a learned $\tau$; "identity" hidden layers skipped the log | exactly the design, on every layer |
| $A_k=\prod_r \mathrm{Or}_{k,r}^{a_{k,r}}$, index per Or | several Ands (clauses) per head, one index per clause | one And per unit, one index per Or |
| birth depth $\ln(1+nm)$ | $\lfloor 1+\ln(nm)\rfloor$ | $\operatorname{round}(\ln(1+nm))$ |
| And scaling | `and+` = 0 on every task; it never fired | fires (2–28 promotions per run on average, by task) |
| depth scaling between any two layers | `L+` = `L−` = 0 on every task | probe in the loudest gap; fires |
| no caps or special numbers | type-size cap 1+ln(1+n), max depth 16, fixed prune thresholds 0.004 / 0.03 / 0.05, η/10 on $a$, dataset-tuned 0.30 / 0.70 | none of these; see README, "Constants that remain" |
| dynamic threshold | fixed constants | $\theta(T)=\eta T^{3/4}$ plus a residual gate |

About 6,000 lines of ablation code (dozens of policy bits, global recipe
state, removed models referenced from comments) were deleted. The board
models are `type_nn.c/h` (+ `type_nn_scale.c/h`) and `c_mlp.c/h`.

## 4. Consistency

- **Shared code.** Both models include `common.h` for the readout $F$,
  the Adam step and the RNG, so these cannot differ between them.
- **Gradients.** `test_type_nn` checks $\partial L/\partial x$,
  $\partial L/\partial b$, $\partial L/\partial a$ and
  $\partial L/\partial w$ against central differences on random
  three-layer nets with $a>1$ and mixed signs. The error falls as $h^2$
  down to about 1e-11. The exact-zero-factor branch is checked
  separately.
- **Surgery.** With probe noise 0, a net with probes and the same net
  without them produce bit-identical outputs. The depth fold is checked
  to perturb less than a naive insert.
- **Proofs.** `coqLang/` now builds. The old build failed at
  `Basic.v:87`: `simpl` rewrote `1 * prod v` into a `match` that `ring`
  rejects. The development was extended to the forward and backward
  pass; see `coqLang/README.md`.
- **Build.** `flake.nix` no longer points at a deleted `lean/` directory
  or a nonexistent `make type-nn` target. `nix build .#proofs` checks
  the Coq development.

## 5. Current result and its limits

On the fixed split, over 5 seeds:

- type-nn has clearly lower hold-out MSE on **diabetes** and
  **ionosphere**, with 0.17× and 0.52× the parameters of `c-mlp`.
- iris, wine and wdbc are within noise.
- type-nn uses more parameters on iris, wine and xor.

Limits a reader should know:

- **One split.** Seeds vary initialisation only. A different 70/30 cut
  could move the ranking on the small sets (iris hold-out has 45 rows).
- **Shared task lr and epochs** were inherited from the old board. They
  were not tuned for either model here, nor tuned fairly for both.
- **type-nn overfits more.** Its train MSE is roughly 10× below `c-mlp`
  at similar hold-out error. The prune phase rarely removes promoted
  structure.
- **The remaining constants** (schedule thirds, the shared lr scale)
  are listed in the README. The schedule is the next candidate to make
  dynamic, after the threshold.

## 6. This iteration: two type-nn models

- **Frozen reference.** The type-nn from sections 3–5 is frozen as
  `type-nn-overfit` (`type_nn_overfit*.c/h`, symbol prefix `tnno_`). It
  reproduces its board numbers bit for bit.
- **New type-nn.** It keeps the architecture and changes only the
  scaling strategy, to measured BIC evidence; see the README.
- **Same protocol.** All three models run through the same harness with
  the same protocol.
- **Result.** The new type-nn has fewer parameters than `c-mlp` on 5 of
  6 tasks, with iris the exception. Its hold-out MSE is a clear win on
  diabetes and ionosphere and within noise on iris, wine and wdbc.
- **Cost of the evidence rule.** It evaluates the training samples of
  the epoch (never the hold-out). That is a departure from "decisions
  read only back-prop quantities", and it is stated in the README.
- **Coq.** The development moved to Rocq 9 (`From Stdlib`, local
  replacements for deprecated lemmas). It was verified here on Rocq
  9.0.0 built from source with its stdlib: every file builds with no
  warnings, and the axiom check matches the earlier 8.18 run. The
  reported `make coq` error was the missing Rocq stdlib package, which
  the flake now provides.
