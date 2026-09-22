(* TypeNN.Surgery — structural edits that preserve the function.

     fold_exact           depth insert/remove: when the junction is the
                          affine image x = α u + β of what the identity
                          layer emits, the consumer with w' = α w and
                          b' = b + w·β computes exactly the same Or
     drop_column_mean     width drop: removing coordinate j and adding
                          w_j·m to the bias is exact whenever x_j = m
     drop_zero_column     … and exact for every x when w_j = 0
     gauge_invariant      rescaling Or factors by c with Π c = 1 leaves
                          the And unchanged (the product has a gauge)
     identity_test_bound  a layer within t of the identity moves each
                          coordinate by at most t ‖z‖₁                  *)

From Stdlib Require Import List ZArith Lia.
Import ListNotations.
Local Open Scope Z_scope.

From TypeNN Require Import Basic.

(* ── depth: the fold ─────────────────────────────────────────────── *)

Definition affineMap (al be u : list Z) : list Z :=
  zipWith Z.add (zipWith Z.mul al u) be.

Definition foldOr (al be : list Z) (o : Or) : Or :=
  mkOr (bias o + dot (wts o) be) (zipWith Z.mul al (wts o)).

Lemma dot_fold :
  forall W al be u,
    length al = length W -> length be = length W -> length u = length W ->
    dot W (affineMap al be u) = dot (zipWith Z.mul al W) u + dot W be.
Proof.
  induction W as [|w ws ih]; intros al be u Ha Hb Hu.
  - destruct al, be, u; unfold affineMap; cbn [zipWith dot]; ring.
  - destruct al as [|a als]; [discriminate Ha|].
    destruct be as [|b bes]; [discriminate Hb|].
    destruct u as [|x us]; [discriminate Hu|].
    cbn [length] in Ha, Hb, Hu.
    injection Ha as Ha. injection Hb as Hb. injection Hu as Hu.
    unfold affineMap in *. cbn [zipWith dot].
    rewrite (ih als bes us Ha Hb Hu). ring.
Qed.

Theorem fold_exact :
  forall o al be u,
    length al = length (wts o) -> length be = length (wts o) ->
    length u = length (wts o) ->
    orVal (foldOr al be o) u = orVal o (affineMap al be u).
Proof.
  intros o al be u Ha Hb Hu. unfold orVal, foldOr. cbn [bias wts].
  rewrite dot_fold by assumption. ring.
Qed.

(* ── width: dropping a coordinate ───────────────────────────────── *)

Lemma dot_dropIdx :
  forall j W x,
    (j < length W)%nat -> length W = length x ->
    dot W x = dot (dropIdx j W) (dropIdx j x) + nth j W 0 * nth j x 0.
Proof.
  induction j as [|j ih]; intros W x Hj Hl.
  - destruct W as [|a ws]; [cbn in Hj; lia|].
    destruct x as [|c cs]; [discriminate Hl|].
    cbn [dropIdx nth dot]. ring.
  - destruct W as [|a ws]; [cbn in Hj; lia|].
    destruct x as [|c cs]; [discriminate Hl|].
    cbn [length] in Hj, Hl. injection Hl as Hl.
    cbn [dropIdx nth dot]. rewrite (ih ws cs) by lia. ring.
Qed.

Theorem drop_column_mean :
  forall j o x m,
    (j < length (wts o))%nat -> length (wts o) = length x -> nth j x 0 = m ->
    orVal (mkOr (bias o + nth j (wts o) 0 * m) (dropIdx j (wts o))) (dropIdx j x)
    = orVal o x.
Proof.
  intros j o x m Hj Hl Hm. unfold orVal. cbn [bias wts].
  rewrite (dot_dropIdx j (wts o) x Hj Hl), Hm. ring.
Qed.

Theorem drop_zero_column :
  forall j o x,
    (j < length (wts o))%nat -> length (wts o) = length x -> nth j (wts o) 0 = 0 ->
    orVal (mkOr (bias o) (dropIdx j (wts o))) (dropIdx j x) = orVal o x.
Proof.
  intros j o x Hj Hl H0. unfold orVal. cbn [bias wts].
  rewrite (dot_dropIdx j (wts o) x Hj Hl), H0. ring.
Qed.

(* ── the gauge of a product ─────────────────────────────────────── *)

Lemma prod_zipWith_mul :
  forall c o, length c = length o -> prod (zipWith Z.mul c o) = prod c * prod o.
Proof.
  induction c as [|a xs ih]; intros o Hlen.
  - destruct o; [cbn [zipWith prod]; ring|discriminate Hlen].
  - destruct o as [|b bs]; [discriminate Hlen|].
    cbn [length] in Hlen. injection Hlen as Hl.
    cbn [zipWith prod]. rewrite (ih bs Hl). ring.
Qed.

Theorem gauge_invariant :
  forall c o, length c = length o -> prod c = 1 -> prod (zipWith Z.mul c o) = prod o.
Proof.
  intros c o Hlen Hc. rewrite prod_zipWith_mul by exact Hlen. rewrite Hc. ring.
Qed.

(* ── the identity band ──────────────────────────────────────────── *)

Lemma dot_sub :
  forall w v z, length w = length v -> dot (zipWith Z.sub w v) z = dot w z - dot v z.
Proof.
  induction w as [|a xs ih]; intros v z Hlen.
  - destruct v; [|discriminate Hlen]. cbn [zipWith dot]. ring.
  - destruct v as [|b bs]; [discriminate Hlen|].
    cbn [length] in Hlen. injection Hlen as Hl.
    destruct z as [|c cs].
    + cbn [zipWith]. rewrite !dot_nil_right. ring.
    + cbn [zipWith dot]. rewrite (ih bs cs Hl). ring.
Qed.

Lemma dot_bound :
  forall t, 0 <= t -> forall d z,
    (forall a, In a d -> iabs a <= t) -> iabs (dot d z) <= t * l1 z.
Proof.
  intros t Ht. induction d as [|a xs ih]; intros z Hd.
  - cbn [dot]. unfold iabs. rewrite Z.abs_0.
    apply Z.mul_nonneg_nonneg; [exact Ht|apply l1_nonneg].
  - destruct z as [|c cs].
    + rewrite dot_nil_right. unfold iabs. rewrite Z.abs_0.
      apply Z.mul_nonneg_nonneg; [exact Ht|apply l1_nonneg].
    + cbn [dot l1].
      assert (Hrest : forall b, In b xs -> iabs b <= t) by (intros; apply Hd; right; assumption).
      assert (H1 : iabs a <= t) by (apply Hd; left; reflexivity).
      pose proof (iabs_add_le (a * c) (dot xs cs)) as Hadd.
      pose proof (ih cs Hrest) as H3.
      unfold iabs in *. rewrite Z.abs_mul in Hadd.
      assert (H2 : Z.abs a * Z.abs c <= t * Z.abs c)
        by (apply Z.mul_le_mono_nonneg_r; [apply Z.abs_nonneg|exact H1]).
      nia.
Qed.

Theorem identity_test_bound :
  forall t, 0 <= t ->
  forall n i w z, length w = n -> length z = n ->
  forall v, nth_error z i = Some v ->
  (forall a, In a (zipWith Z.sub w (basis n i)) -> iabs a <= t) ->
  iabs (dot w z - v) <= t * l1 z.
Proof.
  intros t Ht n i w z Hw Hz v Hv Hclose.
  assert (Hlen : length w = length (basis n i)) by (rewrite Hw, basis_length; reflexivity).
  pose proof (dot_bound t Ht (zipWith Z.sub w (basis n i)) z Hclose) as HB.
  rewrite dot_sub in HB by exact Hlen.
  rewrite (dot_basis n i z Hz v Hv) in HB.
  exact HB.
Qed.
