(* TypeNN.Backward — the algebra behind every line of back-prop.

   Or and And are polynomial, so their derivatives can be stated
   exactly in a ring: perturb one input by h and read off the
   coefficient of h.

     or_grad_b / or_grad_w / or_grad_x   ∂Or/∂b = 1, ∂Or/∂w_j = x_j, ∂Or/∂x_j = w_j
     din_chain                           ∂/∂x_j Σ_r δ_r Or_r = Σ_r δ_r w_{r,j}
     prod_multilinear                    ∂Π/∂o_s = Π_{r≠s} o_r exactly (cofactor)
     prefix_suffix                       cofactor = prefix × suffix, no divide
     powZ_first_order                    ∂o^a/∂o = a o^{a−1}
     andPow_first_order                  ∂A/∂o_s = a_s o_s^{a_s−1} Π_{r≠s} o_r^{a_r}
     dead_factor / dead_and              zero factors freeze the cofactors
     gauss_seidel                        updating W before din is rank-1 off
                                         textbook back-prop (why the C code
                                         accumulates din with the old W)   *)

From Stdlib Require Import List ZArith Lia.
Import ListNotations.
Local Open Scope Z_scope.

From TypeNN Require Import Basic.

(* ── Or ──────────────────────────────────────────────────────────── *)

Theorem or_grad_b :
  forall b W x h, orVal (mkOr (b + h) W) x = orVal (mkOr b W) x + h.
Proof. intros. unfold orVal. cbn [bias wts]. ring. Qed.

Lemma dot_setIdx_left :
  forall j W x h,
    (j < length W)%nat -> length W = length x ->
    dot (setIdx j (nth j W 0 + h) W) x = dot W x + h * nth j x 0.
Proof.
  induction j as [|j ih]; intros W x h Hj Hl.
  - destruct W as [|a ws]; [cbn in Hj; lia|].
    destruct x as [|c cs]; [discriminate Hl|].
    cbn [setIdx nth dot]. ring.
  - destruct W as [|a ws]; [cbn in Hj; lia|].
    destruct x as [|c cs]; [discriminate Hl|].
    cbn [length] in Hj, Hl. injection Hl as Hl.
    cbn [setIdx nth dot]. rewrite (ih ws cs h) by lia. ring.
Qed.

Lemma dot_comm : forall u v, dot u v = dot v u.
Proof.
  induction u as [|a us ih]; intros v; destruct v as [|b vs]; cbn [dot]; try reflexivity.
  rewrite ih. ring.
Qed.

Lemma setIdx_length : forall {A} j (v : A) l, length (setIdx j v l) = length l.
Proof.
  intros A j; induction j as [|j ih]; intros v l; destruct l; cbn [setIdx length]; auto.
Qed.

Theorem or_grad_w :
  forall b W x j h,
    (j < length W)%nat -> length W = length x ->
    orVal (mkOr b (setIdx j (nth j W 0 + h) W)) x
    = orVal (mkOr b W) x + h * nth j x 0.
Proof.
  intros. unfold orVal. cbn [bias wts]. rewrite dot_setIdx_left by assumption. ring.
Qed.

Theorem or_grad_x :
  forall b W x j h,
    (j < length x)%nat -> length W = length x ->
    orVal (mkOr b W) (setIdx j (nth j x 0 + h) x)
    = orVal (mkOr b W) x + h * nth j W 0.
Proof.
  intros b W x j h Hj Hl. unfold orVal. cbn [bias wts].
  rewrite (dot_comm W), (dot_comm W x).
  rewrite dot_setIdx_left by (auto; lia). ring.
Qed.

(* The input gradient of a layer: every Or r contributes δ_r w_{r,j}. *)
Fixpoint weighted (ds : list (Z * Or)) (x : list Z) : Z :=
  match ds with
  | [] => 0
  | (d, o) :: rest => d * orVal o x + weighted rest x
  end.

Fixpoint dinOf (ds : list (Z * Or)) (j : nat) : Z :=
  match ds with
  | [] => 0
  | (d, o) :: rest => d * nth j (wts o) 0 + dinOf rest j
  end.

Theorem din_chain :
  forall ds x j h,
    (j < length x)%nat ->
    Forall (fun p => length (wts (snd p)) = length x) ds ->
    weighted ds (setIdx j (nth j x 0 + h) x) = weighted ds x + h * dinOf ds j.
Proof.
  induction ds as [|[d o] rest ih]; intros x j h Hj Hf; cbn [weighted dinOf].
  - ring.
  - inversion Hf as [|? ? Ho Hr]; subst. cbn [snd] in Ho.
    destruct o as [b W]. cbn [wts] in Ho |- *.
    rewrite or_grad_x by assumption. rewrite (ih x j h Hj Hr). ring.
Qed.

(* ── And ─────────────────────────────────────────────────────────── *)

Theorem prod_multilinear :
  forall s os h,
    (s < length os)%nat ->
    prod (setIdx s (nth s os 0 + h) os) = prod os + h * prod (dropIdx s os).
Proof.
  induction s as [|s ih]; intros os h Hs.
  - destruct os as [|a xs]; [cbn in Hs; lia|]. cbn [setIdx nth prod dropIdx]. ring.
  - destruct os as [|a xs]; [cbn in Hs; lia|].
    cbn [length] in Hs. cbn [setIdx nth prod dropIdx].
    rewrite ih by lia. ring.
Qed.

Lemma dropIdx_firstn_skipn :
  forall {A} s (l : list A), dropIdx s l = firstn s l ++ skipn (S s) l.
Proof.
  intros A s; induction s as [|s ih]; intros l.
  - destruct l; reflexivity.
  - destruct l as [|a xs]; [reflexivity|cbn [dropIdx firstn skipn]; rewrite ih; reflexivity].
Qed.

(* The C code computes the cofactor as prefix × suffix, never dividing. *)
Theorem prefix_suffix :
  forall s os, prod (firstn s os) * prod (skipn (S s) os) = prod (dropIdx s os).
Proof. intros. rewrite dropIdx_firstn_skipn, prod_app. reflexivity. Qed.

Theorem powZ_first_order :
  forall a x h, exists R,
    powZ (x + h) a = powZ x a + h * Z.of_nat a * powZ x (Nat.pred a) + h * h * R.
Proof.
  induction a as [|a ih]; intros x h.
  - exists 0. cbn [powZ Z.of_nat]. ring.
  - destruct (ih x h) as [R HR].
    destruct a as [|a'].
    + exists 0. cbn [powZ Nat.pred Z.of_nat]. ring.
    + exists (powZ x a' * Z.of_nat (S a') + x * R + h * R).
      cbn [Nat.pred] in HR |- *. cbn [powZ] in HR |- *.
      rewrite HR. rewrite !Nat2Z.inj_succ. ring.
Qed.

Fixpoint andPow (vals : list Z) (exps : list nat) : Z :=
  match vals, exps with
  | v :: vs, a :: es => powZ v a * andPow vs es
  | _, _ => 1
  end.

Theorem andPow_first_order :
  forall s vals exps h,
    (s < length vals)%nat -> length vals = length exps ->
    exists R,
      andPow (setIdx s (nth s vals 0 + h) vals) exps
      = andPow vals exps
        + h * (Z.of_nat (nth s exps 0%nat) * powZ (nth s vals 0) (Nat.pred (nth s exps 0%nat))
               * andPow (dropIdx s vals) (dropIdx s exps))
        + h * h * R.
Proof.
  induction s as [|s ih]; intros vals exps h Hs Hl.
  - destruct vals as [|v vs]; [cbn in Hs; lia|].
    destruct exps as [|a es]; [discriminate Hl|].
    destruct (powZ_first_order a v h) as [R HR].
    exists (R * andPow vs es).
    cbn [setIdx nth andPow dropIdx]. rewrite HR. ring.
  - destruct vals as [|v vs]; [cbn in Hs; lia|].
    destruct exps as [|a es]; [discriminate Hl|].
    cbn [length] in Hs, Hl. injection Hl as Hl.
    destruct (ih vs es h ltac:(lia) Hl) as [R HR].
    exists (powZ v a * R).
    cbn [setIdx nth andPow dropIdx]. rewrite HR. ring.
Qed.

(* ── dead factors ────────────────────────────────────────────────── *)

Lemma prod_eq_zero_of_in : forall l, In 0 l -> prod l = 0.
Proof.
  induction l as [|a xs ih]; intros H; cbn [In prod] in *.
  - contradiction.
  - destruct H as [H|H]; [subst; ring|rewrite (ih H); ring].
Qed.

Lemma mem_dropIdx_of_ne :
  forall s t l, t <> s -> nth_error l t = Some 0 -> In 0 (dropIdx s l).
Proof.
  induction s as [|s ih]; intros t l Ht Hl.
  - destruct l as [|a xs]; [destruct t; discriminate Hl|].
    destruct t as [|t']; [contradiction Ht; reflexivity|].
    cbn in Hl. cbn [dropIdx]. apply nth_error_In in Hl. exact Hl.
  - destruct l as [|a xs]; [destruct t; discriminate Hl|].
    destruct t as [|t'].
    + cbn in Hl. injection Hl as ->. cbn [dropIdx In]. left. reflexivity.
    + cbn in Hl. cbn [dropIdx In]. right. eapply ih; [|exact Hl]. congruence.
Qed.

Theorem dead_factor :
  forall s t os, t <> s -> nth_error os t = Some 0 -> prod (dropIdx s os) = 0.
Proof. intros. apply prod_eq_zero_of_in. eapply mem_dropIdx_of_ne; eauto. Qed.

Theorem dead_and :
  forall t u os, t <> u -> nth_error os t = Some 0 -> nth_error os u = Some 0 ->
  forall s, prod (dropIdx s os) = 0.
Proof.
  intros t u os Htu Ht Hu s.
  destruct (Nat.eq_dec t s) as [H|H].
  - subst. apply dead_factor with (t := u); auto.
  - apply dead_factor with (t := t); auto.
Qed.

(* ── update order ────────────────────────────────────────────────── *)

Fixpoint gsBackward (eta x : Z) (g w : list Z) : Z :=
  match g, w with
  | a :: gs, b :: ws => a * (b - eta * a * x) + gsBackward eta x gs ws
  | _, _ => 0
  end.

Theorem gauss_seidel :
  forall g w, length g = length w ->
  forall eta x, gsBackward eta x g w = dot g w - eta * x * sqSum g.
Proof.
  induction g as [|a gs ih]; intros w Hlen eta x.
  - destruct w; cbn [gsBackward dot sqSum]; ring.
  - destruct w as [|b ws]; [discriminate Hlen|].
    cbn [length] in Hlen. injection Hlen as Hl.
    cbn [gsBackward dot sqSum]. rewrite (ih ws Hl eta x). ring.
Qed.
