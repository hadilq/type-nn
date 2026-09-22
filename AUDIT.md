# Audit: is the benchmark honest, fair and consistent?

Every claim below points at the code that enforces it or the test that
checks it.

## Honest

- **Results are distributions, not a single seed.** Every cell is trained
  from `TNN_SEEDS` initialisation seeds (default 5) and reported as
  mean ± sd. On data sets this small, one seed moves hold-out MSE by as
  much as the gaps between models. (`bench.c`, `run_cell`)
- **Wins are claimed only outside the noise.** `BOARD.txt` calls a gap
  "clear" only if it exceeds two standard errors of the difference, and
  prints "within noise" otherwise, in both directions. (`bench.sh`)
- **Parameters are counted densely.** Every learnable scalar present after
  training counts: w, b and the assembly index of every Or, whatever its
  value. Nothing is skipped for being small. (`tnn_params`, `cmlp_params`)
- **Models never see the hold-out.** The harness hands a model only
  training pairs (`train`). The hold-out rows are used only by
  `eval_mse` / `eval_acc` after training. type-nn's evidence rule
  evaluates its own cache of the epoch's *training* pairs.
- **No test on 4 points.** XOR is reported as fit only. Leaving one of
  four points out would ask every model to predict the opposite of what
  the other three imply.

## Fair

| protocol | enforced by |
|---|---|
| One harness for all models; only a vtable differs | `bench.c` |
| One fixed 70/30 split (xorshift32 Fisher–Yates, seed 34972) | `dataset_perm` |
| Standardisation, and regression min-max, fitted on training rows only | `dataset_standardize_train`, `dataset_minmax_train` |
| Same per-epoch shuffle, same mean-MSE gradient `(y − t)/m` | `train` in `bench.c` |
| Same epochs and lr per task, same `TNN_LR_SCALE` | `bench.c`, `common.h` |
| Same optimiser code, with one Adam step counter per parameter group | `tnn_adam`, `tnn_adam_bias` in `common.h` |
| Same readout F(u) = sign(u)·ln(1+\|u\|) | `tnn_F` in `common.h` |
| Same init law, U(±1/sqrt(fan_in)) | `c_mlp.c`, `tnn_or_init_random` |
| Same timer and inference loop | `time_infer` |

`c-mlp`'s hidden width (8 or 16) is the baseline's fixed definition. It
is not tuned here, and type-nn never reads it.

## Consistent

- **Gradients.** `test_type_nn` checks every gradient against central
  finite differences on random three-layer nets with mixed signs and
  assembly index `a > 1`: ∂L/∂x, ∂L/∂b, ∂L/∂a and ∂L/∂w. The error falls
  as h². The exact-zero-factor branch is checked separately.
- **Function-preserving edits.** With probe noise 0, a network with probes
  and the same network without them produce bit-identical outputs. The
  depth fold perturbs less than a naive insert.
- **Evidence rule.** The measured MSE equals a direct computation, the
  BIC ratio matches its formula, and prune edits never raise the
  criterion above max(before, best seen).
- **Invariants.** After training no probe is left, every And keeps an Or,
  widths chain between layers, and all parameters are finite with
  `a ≥ 1`.
- **Proofs.** `coqLang/` builds on Rocq 9 with no warnings. It proves:
  - dense composition and exact probes;
  - the exact first-order gradients;
  - the fold, width-drop and gauge identities;
  - the readout's real derivative;
  - the threshold bounds.

  The ring-level theorems need no axioms.
- **Memory safety.** `make test-asan` runs all three suites under
  AddressSanitizer and UndefinedBehaviorSanitizer.
- **Reproducibility.** The benchmark is deterministic: the same build
  prints the same board, digit for digit.

## Limits

- **One split.** Seeds vary initialisation only. A different 70/30 cut
  could move rankings on the small sets (the iris hold-out has 45 rows).
- **Task lr and epochs** are fixed per task and shared by all models.
  They were not tuned for any model, nor tuned fairly for all.
- **Small tabular data only.** Five UCI-style sets plus XOR. Nothing here
  measures sequence models, large inputs, or scale.
- **type-nn still overfits.** Its training MSE is far below `c-mlp`'s at
  similar hold-out error.
- **Evidence reads training samples.** type-nn's evidence rule runs
  forward passes over training samples, not only back-prop quantities.
- **Remaining constants.** The schedule thirds and the shared lr scale
  remain; see the README.
