(* TypeNN.Graph — the graph variant: semantics and its function-preserving
   edits.

   A graph-type-nn is a list of nodes in topological order. Node i reads
   the concatenation of its in-edges; an edge reads coordinates `sel` of a
   source (the model input x, or an earlier node) and wraps every value it
   passes with its weight `gain` (the slime-mould edge weight c = in_w ·
   out_w of the C code). Every node is the chain's unit (layerA, act).
   The C code stores an explicit pair map k -> j; listing the edges' blocks
   in column order gives exactly this concatenation.

   Theorems
     path_graph_is_chain     a path graph (x -> v0 -> v1 -> …, every edge
                             reading all coordinates with weight 1) is
                             netForward: the chain is the graph whose
                             topology is a linked list
     gain_is_weight          an edge weight c on the values is the same as
                             c on the destination's weights
     gain_absorb_exact       re-estimating an edge weight c -> c' while the
                             destination absorbs w' c' = w c leaves the Or
                             unchanged (the C renormalisation)
     merge_zero_block_exact  a merge dummy: a block of new input anywhere in
                             the input, met by zero weights, leaves the Or
                             unchanged (and dropping such a block is exact)
     dummy_edge_exact        the same with the block at the end
     branch_exact            a branch dummy: one new source coordinate read
                             through one zero column
     vertex_fold_exact       a depth edit on an edge is exact when the
                             consumer is folded with the exact affine fit
                             (Surgery.fold_exact on the edge's block)

   Not stated here: the choice of which edit to attempt (feedback, BIC,
   thresholds) — those are decisions over floating-point measurements,
   tested in C. Skip edges make the graph's function class a superset of
   the chain's; nothing here claims they are equal.                    *)

From Stdlib Require Import List ZArith Lia.
Import ListNotations.
Local Open Scope Z_scope.

From TypeNN Require Import Basic.
From TypeNN Require Import Surgery.

(* ── semantics ──────────────────────────────────────────────────── *)

Record Edge := mkEdge { src : option nat; sel : list nat; gain : Z }.
Record Node := mkNode { ins : list Edge; layer : Layer }.

Definition srcVec (x : list Z) (outs : list (list Z)) (s : option nat) : list Z :=
  match s with None => x | Some i => nth i outs [] end.

Definition edgeVal (x : list Z) (outs : list (list Z)) (e : Edge) : list Z :=
  map (fun k => gain e * nth k (srcVec x outs (src e)) 0) (sel e).

Definition nodeIn (x : list Z) (outs : list (list Z)) (n : Node) : list Z :=
  concat (map (edgeVal x outs) (ins n)).

Fixpoint graphRun (act : Z -> Z) (x : list Z) (outs : list (list Z)) (ns : list Node)
  : list (list Z) :=
  match ns with
  | [] => outs
  | n :: rest => graphRun act x (outs ++ [map act (layerA (layer n) (nodeIn x outs n))]) rest
  end.

(* The readout is the last node. *)
Definition graphForward (act : Z -> Z) (x : list Z) (ns : list Node) : list Z :=
  last (graphRun act x [] ns) x.

(* ── the chain is the path graph ───────────────────────────────── *)

Lemma len_app : forall {A} (l1 l2 : list A), length (l1 ++ l2) = (length l1 + length l2)%nat.
Proof. intros A l1 l2. induction l1 as [|a l ih]; cbn [app length]; [reflexivity|rewrite ih; reflexivity]. Qed.

Fixpoint chainFrom (prev : option nat) (w i : nat) (Ls : list Layer) : list Node :=
  match Ls with
  | [] => []
  | L :: rest => mkNode [mkEdge prev (seq 0 w) 1] L :: chainFrom (Some i) (length L) (S i) rest
  end.

Definition chainNodes (n : nat) (Ls : list Layer) : list Node := chainFrom None n 0 Ls.

Lemma select_all : forall v, map (fun k => 1 * nth k v 0) (seq 0 (length v)) = v.
Proof.
  intros v. apply (nth_ext _ _ 0 0).
  - rewrite len_map, len_seq. reflexivity.
  - intros i Hi. rewrite len_map, len_seq in Hi.
    set (f := fun k : nat => 1 * nth k v 0).
    rewrite (nth_indep (map f (seq 0 (length v))) 0 (f 0%nat))
      by (rewrite len_map, len_seq; exact Hi).
    rewrite map_nth, seq_nth by exact Hi. cbn [plus]. unfold f. ring.
Qed.

Lemma run_chain :
  forall act x Ls prev w i outs v,
    srcVec x outs prev = v -> length v = w -> last outs x = v -> i = length outs ->
    last (graphRun act x outs (chainFrom prev w i Ls)) x = netForward act Ls v.
Proof.
  intros act x Ls. induction Ls as [|L rest ih]; intros prev w i outs v Hs Hw Hl Hi.
  - cbn [chainFrom graphRun netForward]. exact Hl.
  - cbn [chainFrom graphRun netForward].
    assert (Hin : nodeIn x outs (mkNode [mkEdge prev (seq 0 w) 1] L) = v).
    { unfold nodeIn, edgeVal. cbn [ins map concat src sel gain]. rewrite app_nil_r.
      rewrite Hs, <- Hw. apply select_all. }
    rewrite Hin.
    set (z := map act (layerA L v)).
    apply (ih (Some i) (length L) (S i) (outs ++ [z]) z).
    + cbn [srcVec]. rewrite Hi. apply nth_middle.
    + unfold z. rewrite len_map. apply layerA_length.
    + apply last_last.
    + rewrite len_app. cbn [length]. lia.
Qed.

Theorem path_graph_is_chain :
  forall act Ls x, graphForward act x (chainNodes (length x) Ls) = netForward act Ls x.
Proof.
  intros act Ls x. unfold graphForward, chainNodes.
  apply (run_chain act x Ls None (length x) 0 [] x); reflexivity.
Qed.

(* ── edge weights ──────────────────────────────────────────────── *)

Lemma dot_app :
  forall a b c d, length a = length c -> dot (a ++ b) (c ++ d) = dot a c + dot b d.
Proof.
  induction a as [|p ps ih]; intros b c d H.
  - destruct c; [|discriminate H]. cbn [app dot]. ring.
  - destruct c as [|q qs]; [discriminate H|]. cbn [length] in H. injection H as H.
    cbn [app dot]. rewrite (ih b qs d H). ring.
Qed.

Theorem gain_is_weight :
  forall c w z, dot w (map (Z.mul c) z) = dot (map (fun a => a * c) w) z.
Proof.
  intros c. induction w as [|a ws ih]; intros z; cbn [map dot]; [reflexivity|].
  destruct z as [|b bs]; cbn [map dot]; [reflexivity|]. rewrite ih. ring.
Qed.

Lemma dot_absorb :
  forall c c' wb' wb zb,
    Forall2 (fun p q => p * c' = q * c) wb' wb ->
    dot wb' (map (Z.mul c') zb) = dot wb (map (Z.mul c) zb).
Proof.
  intros c c' wb' wb zb H. revert zb.
  induction H as [|p q ps qs Hpq Hrest ih]; intros zb; cbn [dot]; [reflexivity|].
  destruct zb as [|b bs]; cbn [map dot]; [reflexivity|].
  rewrite (ih bs). replace (p * (c' * b)) with ((p * c') * b) by ring.
  rewrite Hpq. ring.
Qed.

Theorem gain_absorb_exact :
  forall b w1 wb' wb w2 x1 zb x2 c c',
    length w1 = length x1 -> length wb = length zb ->
    Forall2 (fun p q => p * c' = q * c) wb' wb ->
    orVal (mkOr b (w1 ++ wb' ++ w2)) (x1 ++ map (Z.mul c') zb ++ x2)
    = orVal (mkOr b (w1 ++ wb ++ w2)) (x1 ++ map (Z.mul c) zb ++ x2).
Proof.
  intros b w1 wb' wb w2 x1 zb x2 c c' H1 Hb Hf.
  assert (Hb' : length wb' = length (map (Z.mul c') zb))
    by (rewrite len_map, <- Hb; eapply Forall2_length; exact Hf).
  assert (Hb2 : length wb = length (map (Z.mul c) zb)) by (rewrite len_map; exact Hb).
  unfold orVal. cbn [bias wts].
  rewrite (dot_app w1 (wb' ++ w2) x1 _ H1), (dot_app w1 (wb ++ w2) x1 _ H1).
  rewrite (dot_app wb' w2 _ x2 Hb'), (dot_app wb w2 _ x2 Hb2).
  rewrite (dot_absorb c c' wb' wb zb Hf). ring.
Qed.

(* ── merge and branch dummies ──────────────────────────────────── *)

Theorem merge_zero_block_exact :
  forall b w1 w2 x1 blk x2,
    length w1 = length x1 ->
    orVal (mkOr b (w1 ++ repeat 0 (length blk) ++ w2)) (x1 ++ blk ++ x2)
    = orVal (mkOr b (w1 ++ w2)) (x1 ++ x2).
Proof.
  intros b w1 w2 x1 blk x2 H. unfold orVal. cbn [bias wts].
  rewrite (dot_app w1 _ x1 _ H), (dot_app w1 w2 x1 x2 H).
  rewrite (dot_app (repeat 0 (length blk)) w2 blk x2) by (apply repeat_length).
  rewrite dot_repeat_zero. ring.
Qed.

Corollary drop_zero_block_exact :
  forall b w1 w2 x1 blk x2,
    length w1 = length x1 ->
    orVal (mkOr b (w1 ++ w2)) (x1 ++ x2)
    = orVal (mkOr b (w1 ++ repeat 0 (length blk) ++ w2)) (x1 ++ blk ++ x2).
Proof. intros. symmetry. apply merge_zero_block_exact. assumption. Qed.

Lemma dot_app_repeat_zero :
  forall w extra x,
    length w = length x ->
    dot (w ++ repeat 0 (length extra)) (x ++ extra) = dot w x.
Proof.
  intros w extra x H. rewrite (dot_app w _ x extra H), dot_repeat_zero. ring.
Qed.

Theorem dummy_edge_exact :
  forall o x extra,
    length (wts o) = length x ->
    orVal (mkOr (bias o) (wts o ++ repeat 0 (length extra))) (x ++ extra) = orVal o x.
Proof.
  intros o x extra H. unfold orVal. cbn [bias wts].
  rewrite (dot_app_repeat_zero (wts o) extra x H). reflexivity.
Qed.

Corollary branch_exact :
  forall b w1 w2 x1 v x2,
    length w1 = length x1 ->
    orVal (mkOr b (w1 ++ [0] ++ w2)) (x1 ++ [v] ++ x2) = orVal (mkOr b (w1 ++ w2)) (x1 ++ x2).
Proof.
  intros b w1 w2 x1 v x2 H.
  exact (merge_zero_block_exact b w1 w2 x1 [v] x2 H).
Qed.

(* ── depth on an edge ──────────────────────────────────────────── *)

(* The block an edge delivers changes from xb to the new node's output ub.
   If xb = α ub + β coordinatewise, folding the consumer's block weights
   (w' = α w, b' = b + w·β) gives exactly the old Or. *)
Corollary vertex_fold_exact :
  forall o al be ub,
    length al = length (wts o) -> length be = length (wts o) ->
    length ub = length (wts o) ->
    orVal (foldOr al be o) ub = orVal o (affineMap al be ub).
Proof. exact fold_exact. Qed.
