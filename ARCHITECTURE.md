# type-nn — architecture

This file is the high-level design of type-nn and its variants. It is the
reference that the code, the proofs and the board are audited against.
The current gaps between it and the code, and the fixes, are in
[PLAN.md](PLAN.md).

Background: [Type Mechanics](https://hadilq.com/posts/type-mechanics/)
(types as generating functions: an Or is a sum type, an And is a product
type) and [Train the knowledge](https://hadilq.com/posts/train-the-knowledge/)
(the hierarchy of partition functions, the dummy rule, the threshold and
the evidence rule).

## 1. The unit: a layer is a partition function

For input x, unit k of a layer is one And over its Ors r:

```
Or_{k,r} = w_{k,r} · x + b_{k,r}                   sum type, dense over x
A_k      = Π_r  Or_{k,r} ^ a_{k,r}                 product type
z_k      = sign(A_k) · ln(1 + |A_k|)               partition function
```

- **Assembly index** `a_{k,r} ≥ 1`, from assembly theory, counts how many
  times the sub-object `Or_{k,r}` is reused to assemble `A_k`. It is
  learned as a real number, with the signed power
  `o^a := sign(o)·|o|^a`, so `sign(A_k) = Π_r sign(Or_{k,r})`.
- **Natural log is the activation**, applied on every layer, the output
  layer included.
- **Log-space evaluation** (no overflow for deep products):
  `ℓ_k = Σ_r a_{k,r} ln|Or_{k,r}|`, `s_k = Π_r sign(Or_{k,r})`,
  `z_k = s_k · ln(1 + e^{ℓ_k})`.
- **Dense.** The output of one layer is the input of the next and every
  Or reads every incoming coordinate, so a type-nn is as dense as an MLP
  of the same widths.
- **The forward pass only evaluates.** All structural change happens on
  the back-propagation side, at epoch boundaries, driven by what
  back-prop did to the probes.

## 2. Structure is learned: three scaling problems, one dummy rule

**The dummy rule.** Every new piece of structure is born as a *dummy*: an
identity, up to a small noise, so the network computes the same function
the moment it appears. Back-propagation is free to move it.
- If back-prop moves a dummy out of its identity band, it is promoted to
  live structure and a fresh dummy takes its place.
- Late in training, live structure that has fallen back to identity (up
  to the threshold) is dropped.

| axis | what it scales | dummy (scale up, early) | drop (scale down, late) |
|---|---|---|---|
| **Or** | the width of a junction: the output size of the previous layer, which is the input size of the current one | The previous layer gains a new output element, a trained unit. In the current layer it is paired with a **dummy weight** (born at 0) in every Or. | Drop the element when its outgoing weights are under the threshold. |
| **And** | the degree of a unit: the number of Ors in its product | Each And carries one **dummy noisy identity Or** (`w ≈ 0, b ≈ 1`; `1^a = 1` for every a). If back-prop gives it weight, it is promoted and a new dummy identity Or joins the list. | If, after the thresholds, an And holds two identity Ors, one is dropped. An And always keeps at least one Or. |
| **Depth** | the number of layers | An **identity layer** (unit k carries `Or = x_k`) is inserted **between any two layers**, not only at the ends. | A layer that has become an identity, up to the threshold, is removed. |

**Birth.** The network is born with `round(ln(1 + n·m))` layers, where n
is the input size and m the output size of the model, shaped
`n → m → … → m`.

**Schedule.** Scale up early and drop only late. The current code uses
fixed thirds of the run: grow, then fit, then prune. Growth also needs an
unexplained residual: training MSE > `Var(t)/N`.

**Threshold.** A dummy of age T has *moved* when its distance from
identity exceeds `θ(T) = η·T^{3/4}`. This is the geometric mean of
Adam's noise walk `η√T` and a consistent drift `ηT`, so it is derived,
not fitted.

**Near-exact depth edits.** The depth dummy is an identity at the type
level: its carrier Or is `x_k`, so `A_k = x_k` exactly. Because every
layer applies the partition function, the layer emits `F(x) = sign(x)
ln(1+|x|)`, which is x only to first order. No finite And of affine Ors
can undo F, because F⁻¹ is not a polynomial. The small probe noise is not
the cause: it is about 0.3 % of the residual. On insertion, the next
layer's columns absorb the affine fit `x ≈ αF(x) + β`; on removal, they
absorb the reverse fit. The edit is exact when the fit is (Coq
`fold_exact`).

## 3. Variants

All variants share §1, the dummy rule of §2, the harness and the
optimizer (per-sample Adam, `common.h`).

| model | topology | keep / drop rule | files |
|---|---|---|---|
| **type-nn** | chain | Evidence. Promote when moved past θ **and** BIC pays: `n ln(MSE_without/MSE_with) > k ln n`, with MSE_without *measured* by resetting the item to its identity. Prune by joint ablation while `C = n ln MSE + K ln n` stays ≤ the best value seen. | `type_nn.c/h`, `type_nn_scale.c/h` |
| **type-nn-overfit** | chain | Displacement only. Promote when moved past θ; drop when back inside the band `θ(N)`. Frozen reference. | `type_nn_overfit*.c/h` |
| **graph-type-nn** | graph (§5) | type-nn's evidence rule on every axis (And, Or, merge, depth), with slime-mould edge weights that order the late edge drops. | `type_nn_graph.c/h` |
| c-mlp | Linear–ReLU–Linear | none (baseline, fixed width 8/16, same readout F) | `c_mlp.c/h` |

## 4. Chain variant: layers as a linked list

The **Layer scale has the structure of a linked list**: an ordered
sequence of layers, where a layer can be inserted or removed at any
position. The implementation stores it as an array of layer pointers
with insert and remove at any index, which is the same abstract list.

- The **depth probe** is one identity layer. It sits in the gap with the
  largest mean `|∂L/∂x|`, and each epoch an unpromoted probe moves to
  the loudest gap.
- The **Or probe** lives on every junction `i → i+1`.
- The **And probe** lives in every unit.

## 5. Graph variant: layers become nodes

When the model is a graph, the layers are called **nodes**. Every node is
exactly the unit of §1, computed by the same C function as a chain layer.
The linked list of layers becomes a directed acyclic graph in
topological order, and the dummy rule of §2 runs on the graph instead of
the list.

- **Reading x.** The first layer of nodes reads the model input x and
  nothing else. Every other node reads the outputs of other nodes.
- **Edges.** An edge carries some output coordinates of its source to
  some input columns of its destination, through an explicit coordinate
  map.
- **The chain is a special case.** It is the graph whose topology is a
  linked list: x → v₀ → v₁ → … (Coq `path_graph_is_chain`).

### 5.1 Birth
- `round(ln(1 + n·m))` nodes, each **as wide as the chain's layer**
  (`n → m → … → m`), with the chain's init law and random-number order.
- The topology starts as a **small-world graph**: the backbone
  x → v₀ → … → v_{D−1} plus forward Watts–Strogatz shortcuts. The
  shortcuts are the lattice `i → i+h` for `h = 2…⌈ln D⌉`, each rewired
  with probability `1/D`.
- Every shortcut is born as a **dummy edge** (zero columns), so at birth
  the graph computes exactly the chain's function. A test checks this
  bit for bit.

### 5.2 Edge weights (the slime mould)
Each edge wraps a node's input with a weight and multiplies the value it
passes by it:

```
x_v[j] = c_e · z_u[k],      c_e = in_w_e · out_w_e
in_w_e  = I_e / Σ_{e' into v} I_e'      I = accumulated inference feedback
out_w_e = B_e / Σ_{e' out of u} B_e'    B = accumulated back-prop feedback
```

- **Inference feedback** `I_e`, for each input x: the mean |value
  entering the edge.
- **Back-prop feedback** `B_e`, for each output y: the mean |∂L/∂z|
  demanded at the edge's destination.
- **Accumulation.** Both are averaged over the edge's lifetime.
- **Normalisation.** The in-weights of a node sum to 1, and so do the
  out-weights of a node (or of x).
- **No veto.** The negative feedback acts through the out-weights.

The weights are re-estimated at every epoch boundary. The destination
absorbs each change, `w' = w·c/c'`, so the function does not move (Coq
`gain_absorb_exact`). Adam's step on a wrapped column is scaled by
`1/c`, so the optimiser does not move either: the effective weight `c·w`
learns at the same rate whatever the tube's thickness. Without that
scaling, a thin tube learns c× slower in function space. That measurably
hurt ionosphere: hold MSE 0.109 with the plain weight against 0.086 with
the scaled step (one split, 5 seeds). The weights therefore do one job,
the one they were added for: they are the slime mould's measure for
dropping edges.

### 5.3 Two kinds of node (in the inference direction)
- **Merging node.** Two or more inferences flow into one node. Merging
  only extends the Or list: the merge dummy for a new source u gives every
  Or of v **dummy weights** for u's outputs (zero columns). This is the
  Or dummy rule extended to one set of dummy weights per incoming source.
- **Branching node.** One node's inference splits into branches that
  carry *different outputs* to *different* successors. An Or dummy (a new
  trained output) is read by every out-edge. Late pruning removes each
  (edge, output) column its destination does not pay for, so different
  outputs end up routed to different nodes.
- A node can merge and branch at once.

### 5.4 The dummy rule on the graph
One rule, with type-nn's evidence and thresholds on every axis.
- **Promotion.** A dummy is promoted when it moved past `θ(T)` **and**
  pays its BIC price on the epoch's training data.
- **Growth.** Only in the grow third, and only while the residual is
  unexplained.
- **Pruning.** In the last third, edges go first, thinnest first: an
  edge is removed when its block does not pay under the criterion. Then
  the joint ablation pool removes Ors, coordinates and single columns.

| axis | dummy (early) | how it is chosen | identity used to judge it / drop it (late) |
|---|---|---|---|
| And | one noisy identity Or per unit | every unit | Or → 1 |
| Or | a trained new output of a node, read by every out-edge through zero columns | every non-readout node | zero outgoing columns |
| Merge | zero block on v for all of u's outputs | one per destination. The source is the node with the largest inference feedback; an unpromoted dummy moves when a stronger source appears | zero block |
| Depth | an identity node on an edge; the consumer is folded on the epoch's training inputs | the edge with the largest back-prop gradient at its junction; an unpromoted dummy moves each grow epoch | carrier node, removed with the reverse fold |

### 5.5 Status
Implemented in `type_nn_graph.c` and tested in `test_type_nn_graph.c`:
- finite-difference gradients;
- exact dummies and drops;
- birth equal to the chain;
- feedback accumulation;
- invariants under randomised edits, run under ASan.

The open items are in PLAN.md §7.

## 6. Faithfulness criteria

An implementation is faithful when:

1. every node or layer computes §1, with real assembly indices and the
   signed power;
2. every structural edit is a dummy: the forward changes by at most the
   declared probe noise, or for depth, by at most the measured fold
   residual;
3. promotion uses back-prop movement past θ, growth happens only early,
   drops only late, and the birth depth and widths are as in §2;
4. in the graph, the feedbacks are epoch aggregates over all x and y,
   and any state that selects structure is either part of the forward
   function or derived from it;
5. no constant is tuned to the board's data sets; every remaining
   constant is listed with its justification;
6. the benchmark gives every model the same split, standardisation,
   epochs, lr, loss and optimizer, reports structure next to accuracy,
   and never reads the hold-out before training ends.

The chain variants and graph-type-nn meet these criteria, with their
deviations documented: the near-exact depth fold, and the derived
constants listed in the README ("Constants that remain"). The audit, the fixes and what
is still open are in [PLAN.md](PLAN.md).

## 7. Proofs

`coqLang/` holds machine-checked statements of §1, §2 and the surgery
edits: composition, exact identity probes, exact gradients in the ring,
the fold, and the column drop. It also has the readout F and the
threshold. `SignedPower.v` states the unit as the C code computes it
(real `a`, signed power, log space, ∂z/∂o, ∂z/∂a), and shows where the
natural-exponent ring model agrees with it. `Graph.v` gives the DAG
semantics: the chain as the path graph, edge weights as a
reparameterisation, and exact merge, branch and drop edits.
