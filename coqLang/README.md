# Coq proofs of the type-nn structure

```
make          # from this directory; Rocq >= 9 with its stdlib
make coq      # from the repository root
nix build .#proofs
```

**Requires Rocq 9 and its standard library.** Rocq 9 ships the stdlib
as a separate package (`rocq-stdlib`; in nixpkgs `coqPackages.stdlib`).
Without it, every `Require` fails with:

```
Error: Cannot find a physical path bound to logical path List with prefix Stdlib.
```

The flake provides it. The sources use the Rocq 9 spelling
`From Stdlib Require Import`. The deprecated `map_length` / `seq_length` /
`app_length` are replaced by local lemmas. Coq 8.x has no `Stdlib`
prefix; to use it, replace `From Stdlib` with `From Coq`
(`sed -i 's/From Stdlib/From Coq/' TypeNN/*.v`).

The whole development, `SignedPower.v` and `Graph.v` included, checks
with no error and no message on Rocq 9 (`nix build .#proofs`), and on
Coq 8.18 through that substitution.

The Or/And algebra is stated over `Z`. Every statement there holds in a
commutative ring, and those theorems use **no axioms**
(`Print Assumptions` reports "Closed under the global context"). The
readout, the signed power and the threshold are stated over `R`, and use
only the standard axioms of the real-number library.

Tactic note: `simpl` unfolds `1 * p` into a `match` that `ring` rejects,
so the proofs unfold only list functions (`cbn [prod app ...]`).

The theorems cover the architecture and the edits both variants use
(chain and graph). The ring files (`Basic`, `Forward`, `Backward`,
`Surgery`, `Graph`) model an assembly index as a natural number. The C
code uses a real `a ≥ 1` and the signed power `o^a = sign(o)·|o|^a`.
`SignedPower.v` states that unit over R and proves the C forward and the
C gradients. It also shows exactly where the ring model agrees with the
C unit: `o ≥ 0` or odd `a`. For `o < 0` and even `a` the two differ by
the sign (`spow_neg_even`).

The decisions — which edit to attempt, the slime-mould feedback, the
BIC evidence rule, the thresholds — are procedures over floating-point
measurements. They are tested in C (`test_type_nn_graph.c`), not proved.

| theorem | file | statement |
|---|---|---|
| `net_app` | Forward | stacking layers is composition: the output of one layer is the input of the next |
| `net_length` | Forward | a layer emits one coordinate per And |
| `and_probe_exact`, `and_probe_anywhere`, `and_probe_drop` | Forward | an identity Or (`w = 0, b = 1`) with any assembly index leaves the And unchanged, so adding or dropping it is exact |
| `width_probe_exact`, `width_probe_net` | Forward | a new producer coordinate met by zero weights leaves the consumer, and the network, unchanged |
| `identity_layerA`, `identity_layer_with_probes` | Forward | the carrier layer returns its input as A, with or without And probes |
| `andVal_exponents_one` | Basic | assembly index 1 everywhere recovers the plain product |
| `or_grad_b`, `or_grad_w`, `or_grad_x` | Backward | `∂Or/∂b = 1`, `∂Or/∂w_j = x_j`, `∂Or/∂x_j = w_j` (exact) |
| `din_chain` | Backward | `∂/∂x_j Σ_r δ_r Or_r = Σ_r δ_r w_rj` |
| `prod_multilinear` | Backward | `∂(Π o)/∂o_s` is exactly the cofactor `Π_{r≠s} o_r` |
| `prefix_suffix` | Backward | cofactor = prefix × suffix, no division |
| `powZ_first_order` | Backward | `(x+h)^a = x^a + h·a·x^(a−1) + O(h²)` |
| `andPow_first_order` | Backward | `∂A/∂o_s = a_s · o_s^(a_s−1) · Π_{r≠s} o_r^(a_r)` |
| `dead_factor`, `dead_and` | Backward | one zero factor kills every sibling cofactor; two kill all |
| `gauss_seidel` | Backward | updating W before the input gradient is rank-1 off textbook back-prop |
| `fold_exact` | Surgery | depth insert/remove: `w' = α·w`, `b' = b + w·β` is exact when `x = α·u + β` |
| `drop_column_mean`, `drop_zero_column` | Surgery | width drop is exact at the coordinate's mean, and exact for zero weight |
| `gauge_invariant` | Surgery | rescaling factors by c with `Π c = 1` leaves the And unchanged |
| `identity_test_bound` | Surgery | a layer within t of the identity moves each coordinate by at most `t·‖z‖₁` |
| `F_zero`, `F_odd`, `F_pos`, `F_neg`, `F_abs_le` | Readout | `F(A) = sign(A)·ln(1+\|A\|)`: odd, sign-preserving, `\|F(A)\| ≤ \|A\|` |
| `F_deriv_pos`, `F_deriv_neg` | Readout | `derivable_pt_lim F A (1/(1+\|A\|))` for `A ≠ 0` (a real derivative) |
| `dF_even`, `dF_pos`, `dF_le_1` | Readout | the Jacobian is even and lies in (0, 1] |
| `F_log_space`, `log_space_slope` | Readout | `F(s·e^ℓ) = s·ln(1+e^ℓ)` and `F'(A)·A = s·σ(ℓ)`: the C log-space formulas are the design |
| `threshold_between` | Threshold | `η√T ≤ η·T^(3/4) ≤ η·T` for `T ≥ 1` |
| `threshold_geometric_mean` | Threshold | `θ² = (η√T)·(η·T)` |
| `threshold_separates` | Threshold | noise / θ = θ / drift = `T^(−1/4)` |
| `spow_log`, `and_log_space`, `unit_log_space` | SignedPower | the C unit on R: `o^a = sign(o)·e^{a ln\|o\|}`, `A = s·e^ℓ`, `z = F(A) = s·ln(1+e^ℓ)` |
| `dz_da`, `dz_do_pos`, `dz_do_neg` | SignedPower | the C gradients `∂z/∂a_r = s σ(ℓ) ln\|o_r\|` and `∂z/∂o_r = s σ(ℓ) a_r / o_r` (real derivatives) |
| `spow_pos_nat`, `spow_neg_odd`, `spow_neg_even` | SignedPower | where the ring model (ℕ exponents) is the C unit: `o > 0`, or odd `a`; for `o < 0`, even `a` they differ by the sign |
| `path_graph_is_chain` | Graph | a path graph x → v₀ → v₁ → … with weight-1 edges reading every coordinate is `netForward`: the chain is the linked-list graph |
| `gain_is_weight`, `gain_absorb_exact` | Graph | an edge weight on the values equals the same weight on the destination's columns; re-estimating it with `w'c' = wc` is exact (the C renormalisation) |
| `merge_zero_block_exact`, `drop_zero_block_exact`, `dummy_edge_exact` | Graph | a merge dummy (a block of new input anywhere, met by zero weights) changes nothing, and dropping it is exact |
| `branch_exact` | Graph | a branch dummy (one new source coordinate read through one zero column) changes nothing |
| `vertex_fold_exact` | Graph | a depth edit on an edge is exact when the consumer is folded with the exact affine fit |

**Not covered:**

- floating point;
- the Adam update itself;
- the derivative of F at A = 0: only A ≠ 0 is proved, and the C code uses
  the limit value 1 there;
- the decision procedures: the slime-mould feedback, the BIC evidence
  rule and the thresholds choose *which* edit to make. The theorems
  above say what each edit does to the function.

## What the graph theorems do and do not say

- The graph is a superset of the chain: `path_graph_is_chain` shows the
  chain is the path graph, and skip edges add functions a chain of the
  same widths cannot compute. The two classes are not claimed equal.
- The edge weights are a reparameterisation (`gain_is_weight`,
  `gain_absorb_exact`). They change neither the function nor the
  optimiser's path in function space: the C code also scales Adam's step
  on a wrapped column by `1/c`. Their job is to rank edges for the late
  drops.
- Merge, branch and Or dummies are exact (`merge_zero_block_exact`,
  `branch_exact`, `Forward.width_probe_exact`).
- A depth edit is exact only when the affine fit is exact
  (`vertex_fold_exact`). The C code fits it on the epoch's training
  inputs, so in practice it is near-exact: the identity node emits F(x),
  not x.
