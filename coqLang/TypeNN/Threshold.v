(* TypeNN.Threshold — the dynamic identity threshold.

   Under per-sample Adam with step size lr, T steps of zero-mean
   gradient noise move a parameter about lr·√T, and T steps of a
   one-signed gradient move it up to lr·T. type-nn decides "back-prop
   moved this probe" with

       θ(T) = lr · T^{3/4}.

     threshold_between         lr √T ≤ θ(T) ≤ lr T   for T ≥ 1
     threshold_geometric_mean  θ(T)² = (lr √T)(lr T): the log-midpoint
     threshold_separates       noise / θ = θ / drift = T^{−1/4}, which
                               shrinks as the probe ages                *)

From Stdlib Require Import Reals Lra.
Local Open Scope R_scope.

Definition theta (lr T : R) : R := lr * Rpower T (3 / 4).

Theorem threshold_between :
  forall lr T, 0 <= lr -> 1 <= T -> lr * sqrt T <= theta lr T <= lr * T.
Proof.
  intros lr T Hlr HT. unfold theta.
  assert (H1 : Rpower T (/ 2) <= Rpower T (3 / 4)) by (apply Rle_Rpower; lra).
  assert (H2 : Rpower T (3 / 4) <= Rpower T 1) by (apply Rle_Rpower; lra).
  rewrite Rpower_sqrt in H1 by lra. rewrite Rpower_1 in H2 by lra.
  split; apply Rmult_le_compat_l; assumption.
Qed.

Theorem threshold_geometric_mean :
  forall lr T, 0 < T -> theta lr T * theta lr T = (lr * sqrt T) * (lr * T).
Proof.
  intros lr T HT. unfold theta.
  replace (lr * Rpower T (3 / 4) * (lr * Rpower T (3 / 4)))
    with (lr * lr * (Rpower T (3 / 4) * Rpower T (3 / 4))) by ring.
  rewrite <- Rpower_plus.
  replace (3 / 4 + 3 / 4) with (/ 2 + 1) by field.
  rewrite Rpower_plus, Rpower_sqrt, Rpower_1 by lra. ring.
Qed.

Theorem threshold_separates :
  forall lr T, 0 < lr -> 0 < T ->
    (lr * sqrt T) / theta lr T = Rpower T (- (1 / 4)) /\
    theta lr T / (lr * T) = Rpower T (- (1 / 4)).
Proof.
  intros lr T Hlr HT. unfold theta.
  assert (Hp : forall e, 0 < Rpower T e) by (intros; unfold Rpower; apply exp_pos).
  replace (sqrt T) with (Rpower T (/ 2)) by (apply Rpower_sqrt; lra).
  replace (lr * T) with (lr * Rpower T 1) by (rewrite Rpower_1; lra).
  split.
  - replace (- (1 / 4)) with (/ 2 - 3 / 4) by field.
    unfold Rminus. rewrite Rpower_plus, Rpower_Ropp.
    pose proof (Hp (3 / 4)). field. split; lra.
  - replace (- (1 / 4)) with (3 / 4 - 1) by field.
    unfold Rminus. rewrite Rpower_plus, Rpower_Ropp.
    pose proof (Hp 1). field. split; lra.
Qed.
