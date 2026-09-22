(* TypeNN.Basic — the Or / And / layer algebra.

   Scalars are [Z]: every statement here holds in any commutative ring,
   and [Z] keeps the proofs decidable and free of real analysis. The
   partition function ln(1+|A|) and everything that needs R lives in
   Readout.v.

   An Or is an affine form        b + Σ_j W_j x_j          (sum type)
   An And is a product of powers  Π_r Or_r^{a_r}           (product type)
   a_r is the assembly index of Or_r in the And.

   Tactic note. [simpl] also unfolds [Z.mul] and turns [1 * p] into a
   [match] on [p] that [ring] cannot read; that was the error in the
   previous version (prod_app, line 87). Here only the list functions
   are unfolded, with [cbn [f ...]].                                     *)

From Stdlib Require Import List ZArith Lia.
Import ListNotations.
Local Open Scope Z_scope.

Fixpoint prod (l : list Z) : Z :=
  match l with
  | [] => 1
  | a :: xs => a * prod xs
  end.

Fixpoint dot (u v : list Z) : Z :=
  match u, v with
  | a :: us, b :: vs => a * b + dot us vs
  | _, _ => 0
  end.

Fixpoint powZ (x : Z) (a : nat) : Z :=
  match a with
  | O => 1
  | S k => x * powZ x k
  end.

Fixpoint sqSum (l : list Z) : Z :=
  match l with
  | [] => 0
  | a :: xs => a * a + sqSum xs
  end.

Fixpoint dropIdx {A} (s : nat) (l : list A) : list A :=
  match s, l with
  | _, [] => []
  | O, _ :: xs => xs
  | S n, a :: xs => a :: dropIdx n xs
  end.

(* Replace the s-th element. *)
Fixpoint setIdx {A} (s : nat) (v : A) (l : list A) : list A :=
  match s, l with
  | _, [] => []
  | O, _ :: xs => v :: xs
  | S n, a :: xs => a :: setIdx n v xs
  end.

Fixpoint basis (n i : nat) : list Z :=
  match n, i with
  | O, _ => []
  | S k, O => 1 :: repeat 0 k
  | S k, S j => 0 :: basis k j
  end.

Fixpoint zipWith (f : Z -> Z -> Z) (xs ys : list Z) : list Z :=
  match xs, ys with
  | x :: xs', y :: ys' => f x y :: zipWith f xs' ys'
  | _, _ => []
  end.

Definition iabs (a : Z) : Z := Z.abs a.

Fixpoint l1 (l : list Z) : Z :=
  match l with
  | [] => 0
  | a :: xs => iabs a + l1 xs
  end.

(* ── the architecture ──────────────────────────────────────────── *)

Record Or := mkOr { bias : Z; wts : list Z }.

Definition orVal (o : Or) (x : list Z) : Z := bias o + dot (wts o) x.

(* An And is a list of (Or, assembly index). *)
Definition Unit := list (Or * nat).

Fixpoint andVal (u : Unit) (x : list Z) : Z :=
  match u with
  | [] => 1
  | (o, a) :: us => powZ (orVal o x) a * andVal us x
  end.

(* A layer is one And per output coordinate: it emits the vector of
   partition-function arguments A_k. *)
Definition Layer := list Unit.

Definition layerA (L : Layer) (x : list Z) : list Z :=
  map (fun u => andVal u x) L.

(* Network: the output of one layer is the input of the next. [act] is
   the partition function, applied coordinatewise (on R it is
   sign(A) ln(1+|A|); on Z it is any map). *)
Fixpoint netForward (act : Z -> Z) (Ls : list Layer) (x : list Z) : list Z :=
  match Ls with
  | [] => x
  | L :: rest => netForward act rest (map act (layerA L x))
  end.

(* The dummies. *)
Definition oneOr (n : nat) : Or := mkOr 1 (repeat 0 n).       (* value 1 *)
Definition carrier (n i : nat) : Or := mkOr 0 (basis n i).    (* value x_i *)

(* ── basic lemmas ──────────────────────────────────────────────── *)

(* Stated here so the development is warning-free on Coq 8.18 and on
   Rocq 9.x, where len_map / len_seq became length_map / length_seq. *)
Lemma len_map : forall {A B} (f : A -> B) l, length (map f l) = length l.
Proof. intros A B f l; induction l as [|a xs ih]; cbn [map length]; congruence. Qed.

Lemma len_seq : forall n s, length (seq s n) = n.
Proof. induction n as [|n ih]; intros s; cbn [seq length]; [reflexivity|rewrite ih; reflexivity]. Qed.

Lemma prod_app : forall u v, prod (u ++ v) = prod u * prod v.
Proof.
  induction u as [|a xs ih]; intros v; cbn [prod app].
  - ring.
  - rewrite ih. ring.
Qed.

Lemma prod_repeat_one : forall m, prod (repeat 1 m) = 1.
Proof.
  induction m as [|n ih]; cbn [prod repeat]; [reflexivity|rewrite ih; ring].
Qed.

Lemma powZ_one_base : forall a, powZ 1 a = 1.
Proof.
  induction a as [|k ih]; cbn [powZ]; [reflexivity|rewrite ih; ring].
Qed.

Lemma powZ_1 : forall x, powZ x 1%nat = x.
Proof. intros. cbn [powZ]. ring. Qed.

Lemma dot_repeat_zero : forall n x, dot (repeat 0 n) x = 0.
Proof.
  induction n as [|n ih]; intros x; cbn [repeat dot].
  - reflexivity.
  - destruct x; cbn [dot]; [reflexivity|rewrite ih; ring].
Qed.

Lemma dot_nil_right : forall w, dot w [] = 0.
Proof. intros []; reflexivity. Qed.

Lemma iabs_nonneg : forall a, 0 <= iabs a.
Proof. intros. unfold iabs. apply Z.abs_nonneg. Qed.

Lemma iabs_add_le : forall a b, iabs (a + b) <= iabs a + iabs b.
Proof. intros. unfold iabs. apply Z.abs_triangle. Qed.

Lemma l1_nonneg : forall x, 0 <= l1 x.
Proof.
  induction x as [|a xs ih]; cbn [l1]; [lia|].
  pose proof (iabs_nonneg a). lia.
Qed.

Lemma basis_length : forall n i, length (basis n i) = n.
Proof.
  induction n as [|k ih]; intros i; cbn [basis].
  - reflexivity.
  - destruct i; cbn [length]; [rewrite repeat_length; reflexivity|rewrite ih; reflexivity].
Qed.

Lemma dot_basis :
  forall n i z, length z = n ->
  forall v, nth_error z i = Some v -> dot (basis n i) z = v.
Proof.
  induction n as [|k ih]; intros i z Hz v Hv.
  - destruct z; [|discriminate Hz]. destruct i; discriminate Hv.
  - destruct z as [|a xs]; [discriminate Hz|].
    cbn [length] in Hz. injection Hz as Hl.
    destruct i as [|j].
    + cbn in Hv. inversion Hv; subst. cbn [basis dot].
      rewrite dot_repeat_zero. ring.
    + cbn in Hv. cbn [basis dot]. rewrite (ih j xs Hl v Hv). ring.
Qed.

Lemma oneOr_val : forall n x, orVal (oneOr n) x = 1.
Proof. intros. unfold orVal, oneOr. cbn [bias wts]. rewrite dot_repeat_zero. ring. Qed.

Lemma carrier_val :
  forall n i x, length x = n ->
  forall v, nth_error x i = Some v -> orVal (carrier n i) x = v.
Proof.
  intros n i x Hx v Hv. unfold orVal, carrier. cbn [bias wts].
  rewrite (dot_basis n i x Hx v Hv). ring.
Qed.

Lemma andVal_app : forall u v x, andVal (u ++ v) x = andVal u x * andVal v x.
Proof.
  induction u as [|[o a] us ih]; intros v x; cbn [andVal app].
  - ring.
  - rewrite ih. ring.
Qed.

Lemma layerA_length : forall L x, length (layerA L x) = length L.
Proof. intros. unfold layerA. apply len_map. Qed.

(* Assembly index 1 everywhere recovers the plain product of Ors. *)
Theorem andVal_exponents_one :
  forall os x, andVal (map (fun o => (o, 1%nat)) os) x = prod (map (fun o => orVal o x) os).
Proof.
  induction os as [|o os ih]; intros x; cbn [map andVal prod].
  - reflexivity.
  - rewrite powZ_1, ih. reflexivity.
Qed.
