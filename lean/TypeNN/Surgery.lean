/-
  TypeNN.Surgery — growing and shrinking the type expression.

  Prop 2   the gauge symmetry of a product
  Prop 6   zero-padding the input is exact
  Prop 7   widening a layer leaves the composite unchanged
  Prop 8   a new rank factor initialised at the product unit is exact
  Prop 11  the identity insert is exact
  Prop 12  the identity test bounds how far a splice moves the composite
-/
import TypeNN.Basic

namespace TypeNN

/-! ### Proposition 2 — gauge

Rescaling the factors of an And by `c₁,…,c_k` with `∏ c_r = 1` leaves the And
invariant. The orbit is `(k-1)`-dimensional per output unit, so the map from
parameters to functions is not injective and the loss is flat along the orbit.
That degeneracy is why a plain SGD step on a product is badly scaled and why a
per-coordinate second-moment normaliser is not a luxury. -/

theorem prod_zipWith_mul :
    ∀ (c o : List Int), c.length = o.length →
      prod (List.zipWith (· * ·) c o) = prod c * prod o := by
  intro c
  induction c with
  | nil => intro o h; cases o <;> simp_all [prod]
  | cons a as ih =>
    intro o h
    cases o with
    | nil => simp at h
    | cons b bs =>
      have hl : as.length = bs.length := by simpa using h
      simp [prod, ih bs hl]
      ac_rfl

/-- **Proposition 2.** A gauge transformation with `∏ c_r = 1` fixes the And. -/
theorem gauge_invariant (c o : List Int) (hlen : c.length = o.length)
    (hc : prod c = 1) :
    andVal (List.zipWith (· * ·) c o) = andVal o := by
  simp [andVal, prod_zipWith_mul c o hlen, hc]

/-! ### Propositions 6 and 7 — widening is exact

New input columns are zero-filled, so an Or is unchanged as a function and is
independent of the new coordinates until training writes into them. Applied to
the *successor* of a widened layer, this says the whole network map is
unchanged even though the new And units carry random weights: they meet zero
columns. Growth is free. Shrinking is not, and the code does not pretend it is. -/

/-- **Proposition 6.** Zero-padding the weight row leaves the Or value alone,
whatever arrives on the new coordinates. -/
theorem dot_zero_pad :
    ∀ (W x : List Int), W.length = x.length → ∀ (m : Nat) (y : List Int),
      dot (W ++ List.replicate m 0) (x ++ y) = dot W x := by
  intro W
  induction W with
  | nil =>
    intro x h m y
    cases x with
    | nil => simp [dot, dot_replicate_zero]
    | cons _ _ => simp at h
  | cons a as ih =>
    intro x h m y
    cases x with
    | nil => simp at h
    | cons b bs =>
      have hl : as.length = bs.length := by simpa using h
      simp [dot, ih bs hl m y]

theorem orVal_zero_pad (b : Int) (W x : List Int) (h : W.length = x.length)
    (m : Nat) (y : List Int) :
    orVal b (W ++ List.replicate m 0) (x ++ y) = orVal b W x := by
  simp [orVal, dot_zero_pad W x h m y]

/-- **Proposition 7.** Widening layer `ℓ` and zero-padding layer `ℓ+1` leaves
every Or of layer `ℓ+1` unchanged, so the composite is unchanged — even though
the new And units of layer `ℓ` were initialised at random. -/
theorem widen_preserves (b : Int) (W A : List Int) (h : W.length = A.length)
    (m : Nat) (Anew : List Int) :
    orVal b (W ++ List.replicate m 0) (A ++ Anew) = orVal b W A :=
  orVal_zero_pad b W A h m Anew

/-! ### Proposition 8 — growing the rank is exact

A new Or factor initialised at the unit of the product (`W = 0`, `b = 1`) is
identically `1`, so the And does not move at the instant of surgery and the
degree rises only once the optimiser walks the new row off the unit. Rank
growth is therefore always safe to attempt: it never costs the polynomial that
has already been paid for. -/

/-- **Proposition 8.** -/
theorem grow_rank_exact (os : List Int) (m : Nat) :
    andVal (os ++ List.replicate m 1) = andVal os := by
  simp [andVal, prod_append, prod_replicate_one]

/-- The unit factor really is the constant `1`, whatever the input. -/
theorem unit_factor_is_one (n : Nat) (x : List Int) :
    orVal 1 (List.replicate n 0) x = 1 := by
  simp [orVal, dot_replicate_zero]

/-! ### Proposition 11 — the identity insert is exact

An inserted layer whose first factor is the identity row and whose remaining
factors are product units computes `z ↦ z`, so `F_new = F_old` immediately
after the surgery. In the implementation the hypothesis `‖z‖_∞ ≤ β` is needed
because Ors clip at `β = 4` while Ands clip at `γ = 32`; here the algebra is
unclipped, so the identity is exact. -/

theorem dot_basis :
    ∀ (n i : Nat) (z : List Int), z.length = n → ∀ v, z[i]? = some v →
      dot (basis n i) z = v := by
  intro n
  induction n with
  | zero =>
    intro i z hz v hv
    cases z with
    | nil => simp at hv
    | cons _ _ => simp at hz
  | succ k ih =>
    intro i z hz v hv
    cases z with
    | nil => simp at hz
    | cons a as =>
      have hl : as.length = k := by simpa using hz
      cases i with
      | zero => simp at hv; simp [basis, dot, dot_replicate_zero, hv]
      | succ j => simp at hv; simp [basis, dot, ih j as hl v hv]

/-- **Proposition 11.** The identity layer's `i`-th And returns `z i`. -/
theorem identity_insert (n i k : Nat) (z : List Int) (hz : z.length = n)
    (v : Int) (hv : z[i]? = some v) :
    andVal (orVal 0 (basis n i) z :: List.replicate k 1) = v := by
  simp [andVal, prod, prod_replicate_one, orVal, dot_basis n i z hz v hv]

/-! ### Proposition 12 — the identity test is a bound, not a theorem

`type_nn_over.c` deletes a hidden layer when it is square, rank one, and every
weight is within `ϑ = 0.08` of the identity. That test does not make the splice
free: it makes the error at most `ϑ ‖z‖₁`. Since `0.08` is not a small number,
calling it "the identity test" is an abbreviation. -/

theorem dot_sub :
    ∀ (w v z : List Int), w.length = v.length →
      dot (List.zipWith (· - ·) w v) z = dot w z - dot v z := by
  intro w
  induction w with
  | nil => intro v z h; cases v <;> simp_all [dot]
  | cons a as ih =>
    intro v z h
    cases v with
    | nil => simp at h
    | cons b bs =>
      have hl : as.length = bs.length := by simpa using h
      cases z with
      | nil => simp [dot]
      | cons c cs =>
        simp only [List.zipWith_cons_cons, dot, ih bs cs hl, Int.sub_mul]
        omega

/-- If every coefficient of a row is bounded by `t`, the row's action on `z` is
bounded by `t ‖z‖₁`. -/
theorem dot_bound (t : Int) (ht : 0 ≤ t) :
    ∀ (d z : List Int), (∀ a ∈ d, iabs a ≤ t) → iabs (dot d z) ≤ t * l1 z := by
  intro d
  induction d with
  | nil =>
    intro z _
    have := l1_nonneg z
    simp only [dot, iabs]
    have : 0 ≤ t * l1 z := Int.mul_nonneg ht this
    omega
  | cons a as ih =>
    intro z hd
    cases z with
    | nil =>
      have : 0 ≤ t * l1 ([] : List Int) := by simp [l1]
      simp only [dot_nil_right, iabs]
      omega
    | cons c cs =>
      have hrest : ∀ b ∈ as, iabs b ≤ t := fun b hb => hd b (List.mem_cons_of_mem a hb)
      have h1 : iabs a ≤ t := hd a (List.mem_cons_self ..)
      have hstep : iabs (a * c + dot as cs) ≤ iabs (a * c) + iabs (dot as cs) :=
        iabs_add_le _ _
      have h2 : iabs (a * c) ≤ t * iabs c := by
        rw [iabs_mul]
        exact Int.mul_le_mul_of_nonneg_right h1 (iabs_nonneg c)
      have h3 : iabs (dot as cs) ≤ t * l1 cs := ih cs hrest
      have : t * l1 (c :: cs) = t * iabs c + t * l1 cs := by
        simp [l1, Int.mul_add]
      simp only [dot]
      omega

/-- **Proposition 12.** A layer that passes the identity test within `t` moves
each output coordinate by at most `t ‖z‖₁`. -/
theorem identity_test_bound (t : Int) (ht : 0 ≤ t) (n i : Nat) (w z : List Int)
    (hw : w.length = n) (hz : z.length = n) (v : Int) (hv : z[i]? = some v)
    (hclose : ∀ a ∈ List.zipWith (· - ·) w (basis n i), iabs a ≤ t) :
    iabs (dot w z - v) ≤ t * l1 z := by
  have hlen : w.length = (basis n i).length := by
    rw [hw, basis_length]
  have := dot_bound t ht (List.zipWith (· - ·) w (basis n i)) z hclose
  rwa [dot_sub w (basis n i) z hlen, dot_basis n i z hz v hv] at this

end TypeNN
