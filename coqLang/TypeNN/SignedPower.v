(* TypeNN.SignedPower — the unit exactly as the C code computes it.

   The ring model of Basic.v uses natural exponents, o^a = o·…·o, so
   (−2)^2 = 4. The C code uses a real assembly index a ≥ 1 and the
   signed power  o^a := sign(o)·|o|^a, so sign(A) = Π sign(o_r) for every
   a. This file states that unit over R and proves what the C code relies
   on:

     spow_log            o^a = sign(o)·exp(a ln|o|)                 (o ≠ 0)
     and_log_space       A = (Π sign o_r)·exp(Σ a_r ln|o_r|)        (all o_r ≠ 0)
     unit_log_space      z = F(A) = s·ln(1 + e^ℓ)  — the C forward
     d_softplus          d/dℓ ln(1+e^ℓ) = σ(ℓ)
     dz_da               ∂z/∂a_r = s·σ(ℓ)·ln|o_r|                   — the C ∂a
     dz_do_pos/neg       ∂z/∂o_r = s·σ(ℓ)·a_r / o_r                 — the C ∂o
     spow_pos_nat        for o > 0 and a ∈ ℕ the two models agree
     spow_neg_odd        for o < 0 and odd a they agree
     spow_neg_even       for o < 0 and even a they differ by the sign:
                         the ring theorems describe the C unit only where
                         o ≥ 0 or a is odd.

   In dz_da / dz_do the other factors enter as the constant c = Σ_{q≠r}
   a_q ln|o_q|, and s is the (locally constant) sign product.            *)

From Stdlib Require Import Reals Lra List.
Import ListNotations.
Local Open Scope R_scope.

From TypeNN Require Import Readout.

Definition spow (o a : R) : R :=
  if Req_EM_T o 0 then 0 else sgn o * Rpower (Rabs o) a.

Lemma spow_log : forall o a, o <> 0 -> spow o a = sgn o * exp (a * ln (Rabs o)).
Proof.
  intros o a H. unfold spow, Rpower.
  destruct (Req_EM_T o 0) as [E|_]; [contradiction|reflexivity].
Qed.

(* An And as a list of (Or value, assembly index). *)
Fixpoint andR (u : list (R * R)) : R :=
  match u with
  | [] => 1
  | (o, a) :: us => spow o a * andR us
  end.

Fixpoint sgnProd (u : list (R * R)) : R :=
  match u with
  | [] => 1
  | (o, _) :: us => sgn o * sgnProd us
  end.

Fixpoint ell (u : list (R * R)) : R :=
  match u with
  | [] => 0
  | (o, a) :: us => a * ln (Rabs o) + ell us
  end.

Definition nonzero (u : list (R * R)) : Prop := Forall (fun oa => fst oa <> 0) u.

Theorem and_log_space : forall u, nonzero u -> andR u = sgnProd u * exp (ell u).
Proof.
  induction u as [|[o a] us ih]; intros H; cbn [andR sgnProd ell].
  - rewrite exp_0. ring.
  - inversion H as [|? ? Ho Hus]; subst. cbn [fst] in Ho.
    rewrite (spow_log o a Ho), (ih Hus), exp_plus. ring.
Qed.

Lemma sgn_unit : forall o, sgn o = 1 \/ sgn o = -1.
Proof. intros o. unfold sgn. destruct (Rlt_dec o 0); [right|left]; reflexivity. Qed.

Lemma sgnProd_unit : forall u, sgnProd u = 1 \/ sgnProd u = -1.
Proof.
  induction u as [|[o a] us ih]; cbn [sgnProd]; [left; reflexivity|].
  destruct (sgn_unit o) as [-> | ->]; destruct ih as [-> | ->];
    [left|right|right|left]; ring.
Qed.

(* The C forward: z = s · ln(1 + e^ℓ) with s = Π sign(o_r), ℓ = Σ a_r ln|o_r|. *)
Theorem unit_log_space :
  forall u, nonzero u -> F (andR u) = sgnProd u * ln (1 + exp (ell u)).
Proof.
  intros u H. rewrite (and_log_space u H). apply F_log_space. apply sgnProd_unit.
Qed.

(* ── gradients ─────────────────────────────────────────────────── *)

Definition sigma (l : R) : R := exp l / (1 + exp l).

Lemma one_plus_exp_pos : forall l, 0 < 1 + exp l.
Proof. intros l. pose proof (exp_pos l). lra. Qed.

Lemma d_softplus : forall l, derivable_pt_lim (fun l => ln (1 + exp l)) l (sigma l).
Proof.
  intros l.
  assert (H1 : derivable_pt_lim (fun l => 1 + exp l) l (exp l)).
  { replace (exp l) with (0 + exp l) by ring.
    apply (derivable_pt_lim_plus (fun _ => 1) exp l 0 (exp l)).
    - apply derivable_pt_lim_const.
    - apply derivable_pt_lim_exp. }
  assert (H2 : derivable_pt_lim ln (1 + exp l) (/ (1 + exp l)))
    by (apply derivable_pt_lim_ln; apply one_plus_exp_pos).
  pose proof (derivable_pt_lim_comp (fun l => 1 + exp l) ln l (exp l) (/ (1 + exp l)) H1 H2) as H.
  unfold sigma. replace (exp l / (1 + exp l)) with (/ (1 + exp l) * exp l)
    by (field; apply Rgt_not_eq, one_plus_exp_pos).
  exact H.
Qed.

Lemma d_affine : forall c L a, derivable_pt_lim (fun a => c + a * L) a L.
Proof.
  intros c L a.
  assert (H : derivable_pt_lim (fun a => c + a * L) a (0 + (1 * L + a * 0))).
  { apply (derivable_pt_lim_plus (fun _ => c) (fun a => a * L) a 0).
    - apply derivable_pt_lim_const.
    - apply (derivable_pt_lim_mult (fun a => a) (fun _ => L) a 1 0).
      + apply derivable_pt_lim_id.
      + apply derivable_pt_lim_const. }
  replace (0 + (1 * L + a * 0)) with L in H by ring. exact H.
Qed.

Lemma d_scale : forall (s : R) f x l,
  derivable_pt_lim f x l -> derivable_pt_lim (fun x => s * f x) x (s * l).
Proof.
  intros s f x l H.
  replace (s * l) with (0 * f x + s * l) by ring.
  apply (derivable_pt_lim_mult (fun _ => s) f x 0 l); [apply derivable_pt_lim_const|exact H].
Qed.

(* ∂z/∂a_r = s σ(ℓ) ln|o_r|, with ℓ = c + a_r L and L = ln|o_r|. *)
Theorem dz_da :
  forall s c L a,
    derivable_pt_lim (fun a => s * ln (1 + exp (c + a * L))) a (s * sigma (c + a * L) * L).
Proof.
  intros s c L a.
  pose proof (derivable_pt_lim_comp (fun a => c + a * L) (fun l => ln (1 + exp l)) a L
                (sigma (c + a * L)) (d_affine c L a) (d_softplus (c + a * L))) as H.
  rewrite Rmult_assoc. apply d_scale. exact H.
Qed.

(* ∂z/∂o_r = s σ(ℓ) a_r / o_r, for o_r > 0 (ℓ = c + a_r ln o_r) … *)
Theorem dz_do_pos :
  forall s c a o, 0 < o ->
    derivable_pt_lim (fun o => s * ln (1 + exp (c + a * ln o))) o
                     (s * sigma (c + a * ln o) * (a / o)).
Proof.
  intros s c a o Ho.
  assert (Hl : derivable_pt_lim (fun o => c + a * ln o) o (a / o)).
  { replace (a / o) with (0 + (0 * ln o + a * / o)) by (field; lra).
    apply (derivable_pt_lim_plus (fun _ => c) (fun o => a * ln o) o 0).
    - apply derivable_pt_lim_const.
    - apply (derivable_pt_lim_mult (fun _ => a) ln o 0 (/ o)).
      + apply derivable_pt_lim_const.
      + apply derivable_pt_lim_ln. exact Ho. }
  pose proof (derivable_pt_lim_comp (fun o => c + a * ln o) (fun l => ln (1 + exp l)) o
                (a / o) (sigma (c + a * ln o)) Hl (d_softplus (c + a * ln o))) as H.
  rewrite Rmult_assoc. apply d_scale. exact H.
Qed.

(* … and for o_r < 0 (ℓ = c + a_r ln(−o_r)): the same formula. *)
Theorem dz_do_neg :
  forall s c a o, o < 0 ->
    derivable_pt_lim (fun o => s * ln (1 + exp (c + a * ln (- o)))) o
                     (s * sigma (c + a * ln (- o)) * (a / o)).
Proof.
  intros s c a o Ho.
  assert (Hn : derivable_pt_lim (fun o => ln (- o)) o (/ o)).
  { assert (Hopp : derivable_pt_lim (fun o => - o) o (Ropp 1)).
    { apply (derivable_pt_lim_opp (fun o => o) o 1). apply derivable_pt_lim_id. }
    pose proof (derivable_pt_lim_comp (fun o => - o) ln o (Ropp 1) (/ (- o)) Hopp
                  (derivable_pt_lim_ln (- o) ltac:(lra))) as H.
    replace (/ o) with (/ (- o) * Ropp 1) by (field; lra). exact H. }
  assert (Hl : derivable_pt_lim (fun o => c + a * ln (- o)) o (a / o)).
  { replace (a / o) with (0 + (0 * ln (- o) + a * / o)) by (field; lra).
    apply (derivable_pt_lim_plus (fun _ => c) (fun o => a * ln (- o)) o 0).
    - apply derivable_pt_lim_const.
    - apply (derivable_pt_lim_mult (fun _ => a) (fun o => ln (- o)) o 0 (/ o)).
      + apply derivable_pt_lim_const.
      + exact Hn. }
  pose proof (derivable_pt_lim_comp (fun o => c + a * ln (- o)) (fun l => ln (1 + exp l)) o
                (a / o) (sigma (c + a * ln (- o))) Hl (d_softplus (c + a * ln (- o)))) as H.
  rewrite Rmult_assoc. apply d_scale. exact H.
Qed.

(* ── where the ring model (natural exponents) describes the C unit ── *)

Lemma sgn_pos : forall o, 0 < o -> sgn o = 1.
Proof. intros o H. unfold sgn. destruct (Rlt_dec o 0); [lra|reflexivity]. Qed.

Lemma sgn_neg : forall o, o < 0 -> sgn o = -1.
Proof. intros o H. unfold sgn. destruct (Rlt_dec o 0); [reflexivity|lra]. Qed.

Theorem spow_pos_nat : forall o n, 0 < o -> spow o (INR n) = o ^ n.
Proof.
  intros o n H. unfold spow.
  destruct (Req_EM_T o 0) as [E|_]; [lra|].
  rewrite sgn_pos by exact H. rewrite Rabs_right by lra.
  rewrite Rpower_pow by exact H. ring.
Qed.

Lemma pow_neg_split : forall o n, o ^ n = (-1) ^ n * (- o) ^ n.
Proof.
  intros o n. rewrite <- Rpow_mult_distr. f_equal. ring.
Qed.

Theorem spow_neg_odd : forall o k, o < 0 -> spow o (INR (S (2 * k))) = o ^ S (2 * k).
Proof.
  intros o k H. unfold spow.
  destruct (Req_EM_T o 0) as [E|_]; [lra|].
  rewrite sgn_neg by exact H. rewrite Rabs_left by exact H.
  rewrite Rpower_pow by lra.
  rewrite (pow_neg_split o (S (2 * k))), pow_1_odd. ring.
Qed.

Theorem spow_neg_even : forall o k, o < 0 -> spow o (INR (2 * k)) = - (o ^ (2 * k)).
Proof.
  intros o k H. unfold spow.
  destruct (Req_EM_T o 0) as [E|_]; [lra|].
  rewrite sgn_neg by exact H. rewrite Rabs_left by exact H.
  rewrite Rpower_pow by lra.
  rewrite (pow_neg_split o (2 * k)), pow_1_even. ring.
Qed.
