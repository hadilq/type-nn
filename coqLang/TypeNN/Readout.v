(* TypeNN.Readout — the partition function z = F(A) = sign(A) ln(1+|A|).

     F_zero, F_odd, F_pos, F_neg    F(0) = 0, F is odd and keeps sign
     F_abs_le                       |F(A)| ≤ |A|  (it only compresses)
     F_deriv_pos / F_deriv_neg      F'(A) = 1/(1+|A|) on A ≠ 0 (a real
                                    derivative, not just an identity)
     dF_even, dF_pos, dF_le_1       the Jacobian is even, in (0, 1]
     F_log_space                    with A = s·e^ℓ, F(A) = s·ln(1+e^ℓ):
                                    the C code evaluates softplus(ℓ) and
                                    never forms the product A
     log_space_slope                F'(A)·A = s·σ(ℓ): the factor the C
                                    code multiplies into a_r / o_r
     mean_mse_grad_one_out          ∂L/∂y = (y − t)/m is y − t for m = 1 *)

From Stdlib Require Import Reals Lra.
Local Open Scope R_scope.

Definition sgn (A : R) : R := if Rlt_dec A 0 then -1 else 1.
Definition F (A : R) : R := sgn A * ln (1 + Rabs A).
Definition dF (A : R) : R := / (1 + Rabs A).

(* ── values ──────────────────────────────────────────────────────── *)

Lemma ln1p_nonneg : forall u, 0 <= u -> 0 <= ln (1 + u).
Proof.
  intros u Hu. destruct (Req_dec u 0) as [->|Hne].
  - rewrite Rplus_0_r, ln_1. lra.
  - rewrite <- ln_1. left. apply ln_increasing; lra.
Qed.

Lemma ln1p_lt : forall u, 0 < u -> ln (1 + u) < u.
Proof.
  intros u Hu.
  rewrite <- (ln_exp u) at 2.
  apply ln_increasing; [lra|]. apply exp_ineq1. lra.
Qed.

Theorem F_zero : F 0 = 0.
Proof. unfold F. rewrite Rabs_R0, Rplus_0_r, ln_1. ring. Qed.

Theorem F_odd : forall A, F (- A) = - F A.
Proof.
  intros A. unfold F, sgn. rewrite Rabs_Ropp.
  destruct (Rlt_dec (- A) 0), (Rlt_dec A 0); try lra.
  assert (A = 0) by lra. subst. rewrite Rabs_R0, Rplus_0_r, ln_1. ring.
Qed.

Theorem F_pos : forall A, 0 < A -> 0 < F A.
Proof.
  intros A HA. unfold F, sgn. destruct (Rlt_dec A 0); [lra|].
  rewrite Rabs_right by lra. rewrite <- ln_1.
  replace (1 * ln (1 + A)) with (ln (1 + A)) by ring.
  apply ln_increasing; lra.
Qed.

Theorem F_neg : forall A, A < 0 -> F A < 0.
Proof.
  intros A HA. replace A with (- - A) by ring. rewrite F_odd.
  pose proof (F_pos (- A)). lra.
Qed.

Theorem F_abs_le : forall A, Rabs (F A) <= Rabs A.
Proof.
  intros A. unfold F.
  assert (Hs : Rabs (sgn A) = 1)
    by (unfold sgn; destruct (Rlt_dec A 0);
        [rewrite Rabs_left by lra|rewrite Rabs_right by lra]; lra).
  rewrite Rabs_mult, Hs, Rmult_1_l.
  pose proof (Rabs_pos A) as Hp.
  rewrite Rabs_right by (apply Rle_ge, ln1p_nonneg; exact Hp).
  destruct (Req_dec (Rabs A) 0) as [H0|H0].
  - rewrite H0, Rplus_0_r, ln_1. lra.
  - left. apply ln1p_lt. lra.
Qed.

(* ── the Jacobian ────────────────────────────────────────────────── *)

Theorem dF_even : forall A, dF (- A) = dF A.
Proof. intros. unfold dF. rewrite Rabs_Ropp. reflexivity. Qed.

Theorem dF_pos : forall A, 0 < dF A.
Proof. intros. unfold dF. apply Rinv_0_lt_compat. pose proof (Rabs_pos A). lra. Qed.

Theorem dF_le_1 : forall A, dF A <= 1.
Proof.
  intros. unfold dF. pose proof (Rabs_pos A).
  rewrite <- Rinv_1. apply Rinv_le_contravar; lra.
Qed.

(* F agrees with a smooth branch on each side of 0; derivatives are
   local, so it inherits the branch's derivative. *)
Lemma deriv_local :
  forall f g A l r, 0 < r ->
    (forall y, Rabs (y - A) < r -> f y = g y) ->
    derivable_pt_lim g A l -> derivable_pt_lim f A l.
Proof.
  intros f g A l r Hr Heq Hg eps He.
  destruct (Hg eps He) as [d Hd].
  assert (Hm : 0 < Rmin d r) by (apply Rmin_pos; [apply cond_pos|exact Hr]).
  exists (mkposreal _ Hm). intros h Hh Hlt. cbn in Hlt.
  rewrite (Heq (A + h)) by (replace (A + h - A) with h by ring;
                            eapply Rlt_le_trans; [exact Hlt|apply Rmin_r]).
  rewrite (Heq A) by (replace (A - A) with 0 by ring; rewrite Rabs_R0; exact Hr).
  apply Hd; [exact Hh|]. eapply Rlt_le_trans; [exact Hlt|apply Rmin_l].
Qed.

Theorem F_deriv_pos : forall A, 0 < A -> derivable_pt_lim F A (dF A).
Proof.
  intros A HA.
  set (g := comp ln (fct_cte 1 + id)%F).
  assert (Hg : derivable_pt_lim g A (/ (1 + A) * (0 + 1))).
  { apply derivable_pt_lim_comp.
    - apply derivable_pt_lim_plus; [apply derivable_pt_lim_const|apply derivable_pt_lim_id].
    - unfold plus_fct, fct_cte, id. apply derivable_pt_lim_ln. lra. }
  replace (/ (1 + A) * (0 + 1)) with (dF A) in Hg
    by (unfold dF; rewrite Rabs_right by lra; field; lra).
  apply (deriv_local F g A (dF A) A HA); [|exact Hg].
  intros y Hy. unfold F, sgn, g, comp, plus_fct, fct_cte, id.
  apply Rabs_def2 in Hy. destruct (Rlt_dec y 0); [lra|].
  rewrite Rabs_right by lra. ring.
Qed.

Theorem F_deriv_neg : forall A, A < 0 -> derivable_pt_lim F A (dF A).
Proof.
  intros A HA.
  set (g := (- comp ln (fct_cte 1 - id))%F).
  assert (Hg : derivable_pt_lim g A (- (/ (1 - A) * (0 - 1)))).
  { apply derivable_pt_lim_opp. apply derivable_pt_lim_comp.
    - apply derivable_pt_lim_minus; [apply derivable_pt_lim_const|apply derivable_pt_lim_id].
    - unfold minus_fct, fct_cte, id. apply derivable_pt_lim_ln. lra. }
  replace (- (/ (1 - A) * (0 - 1))) with (dF A) in Hg
    by (unfold dF; rewrite Rabs_left by lra; field; lra).
  apply (deriv_local F g A (dF A) (- A)); [lra| |exact Hg].
  intros y Hy. unfold F, sgn, g, comp, opp_fct, minus_fct, fct_cte, id.
  apply Rabs_def2 in Hy. destruct (Rlt_dec y 0); [|lra].
  rewrite Rabs_left by lra. replace (1 + - y) with (1 - y) by ring. ring.
Qed.

(* ── the log-space evaluation used by the C code ─────────────────── *)

Lemma sign_unit : forall s, s = 1 \/ s = -1 -> forall l, Rabs (s * exp l) = exp l.
Proof.
  intros s Hs l. rewrite Rabs_mult. pose proof (exp_pos l).
  rewrite (Rabs_right (exp l)) by lra.
  destruct Hs as [->| ->]; [rewrite (Rabs_right 1) by lra|rewrite (Rabs_left (-1)) by lra]; ring.
Qed.

Theorem F_log_space :
  forall s l, s = 1 \/ s = -1 -> F (s * exp l) = s * ln (1 + exp l).
Proof.
  intros s l Hs. unfold F. rewrite (sign_unit s Hs l). f_equal.
  unfold sgn. pose proof (exp_pos l).
  destruct Hs as [->| ->]; destruct (Rlt_dec _ 0); lra.
Qed.

Theorem log_space_slope :
  forall s l, s = 1 \/ s = -1 ->
    dF (s * exp l) * (s * exp l) = s * (exp l / (1 + exp l)).
Proof.
  intros s l Hs. unfold dF. rewrite (sign_unit s Hs l).
  pose proof (exp_pos l). field. lra.
Qed.

(* ── loss scale shared by both board models ─────────────────────── *)

Definition mean_mse_grad (y t : R) (m : nat) : R := (y - t) / INR (Nat.max 1 m).

Lemma mean_mse_grad_one_out : forall y t, mean_mse_grad y t 1%nat = y - t.
Proof. intros. unfold mean_mse_grad. cbn. field. Qed.
