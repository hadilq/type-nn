(* TypeNN.Forward — the forward pass of the architecture.

   z = act(A),  A_k = Π_r (w_{k,r}·x + b_{k,r})^{a_{k,r}},
   and the output of one layer is the input of the next.

   Theorems
     net_app              stacking is composition (dense like an MLP)
     net_length           a layer emits one coordinate per And
     and_probe_exact      an identity Or (w=0, b=1) with ANY assembly
                          index leaves the And unchanged (And scale up)
     and_probe_drop       … so dropping it is exact too (And scale down)
     width_probe_exact    the producer grows a coordinate, the consumer
                          meets it with 0 weights: same function (Or up)
     width_probe_net      … and the whole network output is unchanged
     identity_layerA      a carrier layer returns its input as A
                          (the type-level identity used by depth scaling) *)

From Stdlib Require Import List ZArith Lia.
Import ListNotations.
Local Open Scope Z_scope.

From TypeNN Require Import Basic.

(* ── composition ─────────────────────────────────────────────────── *)

Theorem net_app :
  forall act L1 L2 x,
    netForward act (L1 ++ L2) x = netForward act L2 (netForward act L1 x).
Proof.
  intros act L1; induction L1 as [|L rest ih]; intros L2 x; cbn [app netForward].
  - reflexivity.
  - apply ih.
Qed.

Theorem net_length :
  forall act L Ls x, length (netForward act (Ls ++ [L]) x) = length L.
Proof.
  intros act L Ls x. rewrite net_app. cbn [netForward].
  rewrite len_map. apply layerA_length.
Qed.

(* ── And scaling: the identity Or probe ─────────────────────────── *)

Theorem and_probe_exact :
  forall u n a x, andVal (u ++ [(oneOr n, a)]) x = andVal u x.
Proof.
  intros u n a x. rewrite andVal_app. cbn [andVal].
  rewrite oneOr_val, powZ_one_base. ring.
Qed.

(* The probe may sit anywhere in the product. *)
Theorem and_probe_anywhere :
  forall u v n a x, andVal (u ++ (oneOr n, a) :: v) x = andVal (u ++ v) x.
Proof.
  intros. rewrite !andVal_app. cbn [andVal].
  rewrite oneOr_val, powZ_one_base. ring.
Qed.

Corollary and_probe_drop :
  forall u v n a x, andVal (u ++ v) x = andVal (u ++ (oneOr n, a) :: v) x.
Proof. intros. symmetry. apply and_probe_anywhere. Qed.

Theorem and_probe_layer :
  forall L n a x,
    layerA (map (fun u => u ++ [(oneOr n, a)]) L) x = layerA L x.
Proof.
  intros. unfold layerA. rewrite map_map.
  apply map_ext. intros u. apply and_probe_exact.
Qed.

(* ── Or scaling: a new coordinate met by weights born at 0 ──────── *)

Definition widenOr (o : Or) : Or := mkOr (bias o) (wts o ++ [0]).
Definition widenUnit (u : Unit) : Unit := map (fun oa => (widenOr (fst oa), snd oa)) u.
Definition widenLayer (L : Layer) : Layer := map widenUnit L.

(* Every Or of the layer reads exactly n coordinates. *)
Definition wfLayer (n : nat) (L : Layer) : Prop :=
  Forall (fun u => Forall (fun oa => length (wts (fst oa)) = n) u) L.

Lemma dot_app_zero :
  forall W x, length W = length x -> forall v, dot (W ++ [0]) (x ++ [v]) = dot W x.
Proof.
  induction W as [|a ws ih]; intros x Hl v.
  - destruct x; [|discriminate Hl]. cbn [app dot]. ring.
  - destruct x as [|c cs]; [discriminate Hl|].
    cbn [length] in Hl. injection Hl as Hl.
    cbn [app dot]. rewrite (ih cs Hl v). reflexivity.
Qed.

Lemma widenOr_val :
  forall o x, length (wts o) = length x ->
  forall v, orVal (widenOr o) (x ++ [v]) = orVal o x.
Proof.
  intros o x Hl v. unfold orVal, widenOr. cbn [bias wts].
  rewrite dot_app_zero by exact Hl. reflexivity.
Qed.

Lemma widenUnit_val :
  forall u x,
    Forall (fun oa => length (wts (fst oa)) = length x) u ->
    forall v, andVal (widenUnit u) (x ++ [v]) = andVal u x.
Proof.
  induction u as [|[o a] us ih]; intros x Hf v.
  - reflexivity.
  - inversion Hf as [|? ? Ho Hus]; subst.
    unfold widenUnit in *. cbn [map fst snd andVal] in *.
    rewrite widenOr_val by exact Ho.
    rewrite (ih x Hus v). reflexivity.
Qed.

Theorem width_probe_exact :
  forall L x, wfLayer (length x) L ->
  forall v, layerA (widenLayer L) (x ++ [v]) = layerA L x.
Proof.
  induction L as [|u us ih]; intros x Hwf v.
  - reflexivity.
  - inversion Hwf as [|? ? Hu Hus]; subst.
    unfold layerA, widenLayer in *. cbn [map].
    rewrite widenUnit_val by exact Hu.
    f_equal. apply ih. exact Hus.
Qed.

(* Network level: the producer P grows unit q (trained freely), the
   consumer C is widened. What C computes does not move, whatever q is. *)
Theorem width_probe_net :
  forall act P q C x,
    wfLayer (length P) C ->
    layerA (widenLayer C) (map act (layerA (P ++ [q]) x))
    = layerA C (map act (layerA P x)).
Proof.
  intros act P q C x Hwf.
  unfold layerA at 2. rewrite map_app, map_app. cbn [map].
  apply width_probe_exact.
  unfold layerA. rewrite !len_map. exact Hwf.
Qed.

(* ── Depth scaling: the type-level identity ─────────────────────── *)

Definition idUnit (n i : nat) : Unit := [(carrier n i, 1%nat)].
Definition idLayer (n : nat) : Layer := map (idUnit n) (seq 0 n).

Theorem identity_layerA :
  forall x, layerA (idLayer (length x)) x = x.
Proof.
  intros x. unfold layerA, idLayer. rewrite map_map.
  apply (nth_ext _ _ 0 0).
  - rewrite len_map, len_seq. reflexivity.
  - intros i Hi. rewrite len_map, len_seq in Hi.
    set (f := fun i0 : nat => andVal (idUnit (length x) i0) x).
    rewrite (nth_indep (map f (seq 0 (length x))) 0 (f 0%nat))
      by (rewrite len_map, len_seq; exact Hi).
    rewrite map_nth, seq_nth by exact Hi. cbn [plus].
    unfold f, idUnit. cbn [andVal]. rewrite powZ_1.
    destruct (nth_error x i) as [v|] eqn:E.
    + rewrite (carrier_val (length x) i x eq_refl v E).
      apply nth_error_nth with (d := 0) in E. rewrite E. ring.
    + apply nth_error_None in E. lia.
Qed.

(* With the And probe on every unit it is still the identity. *)
Theorem identity_layer_with_probes :
  forall x a,
    layerA (map (fun u => u ++ [(oneOr (length x), a)]) (idLayer (length x))) x = x.
Proof.
  intros. rewrite and_probe_layer. apply identity_layerA.
Qed.
