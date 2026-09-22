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
`From Stdlib Require Import`, and build with **no warnings on Rocq
9.0.0**. The deprecated `map_length` / `seq_length` are replaced by local
lemmas. Coq 8.x has no `Stdlib` prefix; to use it, replace
`From Stdlib` with `From Coq`.

The Or/And algebra is stated over `Z`. Every statement there holds in a
commutative ring, and those theorems use **no axioms**
(`Print Assumptions` reports "Closed under the global context"). The
readout and the threshold are stated over `R`, and use only the standard
axioms of the real-number library.

Tactic note: `simpl` unfolds `1 * p` into a `match` that `ring` rejects,
so the proofs unfold only list functions (`cbn [prod app ...]`).

The theorems cover the architecture and the scaling edits that both
type-nn models share. The evidence rule of the current type-nn (BIC on a
measured MSE) is a decision procedure over floating-point measurements.
It is tested in C (`test_type_nn.c`), not proved.

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

**Not covered:**

- floating point;
- the Adam update itself;
- real (non-integer) assembly indices in the ring theorems. The
  log-space forms cover them on `R`.
- the derivative of F at A = 0. Only A ≠ 0 is proved; the C code uses
  the limit value 1 there.
