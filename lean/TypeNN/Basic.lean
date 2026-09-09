/-
  TypeNN.Basic — the And/Or layer as an algebraic object.

  Scalars are `Int`. Every proof in this development uses only the axioms of a
  linearly ordered commutative ring, which `Int` satisfies and so does `ℝ`; the
  carrier is fixed to `Int` purely so that the library builds against bare Lean
  core with no Mathlib dependency. Nothing below divides.

  An Or is an affine form  b + Σ_j W_j x_j.
  An And is the product of its Or factors.
  Clipping is deliberately absent: as argued in the post, the implementation's
  backward pass is the gradient of the *unclipped* network (a straight-through
  estimator), so the unclipped algebra is what the theorems are about.
-/

namespace TypeNN



/-- Product of a list of factors. This is the And. -/
def prod : List Int → Int
  | []      => 1
  | a :: as => a * prod as

/-- Truncating inner product. Used for the linear part of an Or. -/
def dot : List Int → List Int → Int
  | [],      _       => 0
  | _,       []      => 0
  | a :: as, b :: bs => a * b + dot as bs

/-- One Or unit: `b + Σ_j W_j x_j`. -/
def orVal (b : Int) (W x : List Int) : Int := b + dot W x

/-- One And unit: the product of its Or factors. -/
def andVal (os : List Int) : Int := prod os

/-- Sum of squares, used by the Gauss–Seidel correction term. -/
def sqSum : List Int → Int
  | []      => 0
  | a :: as => a * a + sqSum as

/-- `dropIdx s l` deletes the element at index `s`. `∏ (dropIdx s os)` is `∂A/∂O_s`. -/
def dropIdx : Nat → List Int → List Int
  | _,   []      => []
  | 0,   _ :: as => as
  | n+1, a :: as => a :: dropIdx n as

/-- `basis n i` is the `i`-th standard basis row of width `n`. -/
def basis : Nat → Nat → List Int
  | 0,   _   => []
  | n+1, 0   => 1 :: List.replicate n 0
  | n+1, i+1 => 0 :: basis n i

/-- Absolute value, kept division-free so the bound proofs stay in the ring. -/
def iabs (a : Int) : Int := (a.natAbs : Int)

/-- `linf` is the ℓ^∞ norm; `l1` is the ℓ^1 norm. -/
def linf : List Int → Int
  | []      => 0
  | a :: as => max (iabs a) (linf as)

def l1 : List Int → Int
  | []      => 0
  | a :: as => iabs a + l1 as

/-! ### Elementary facts -/

theorem prod_append (u v : List Int) : prod (u ++ v) = prod u * prod v := by
  induction u with
  | nil => simp [prod]
  | cons a as ih => simp [prod, ih, Int.mul_assoc]

theorem prod_cons (a : Int) (l : List Int) : prod (a :: l) = a * prod l := rfl

theorem prod_replicate_one (m : Nat) : prod (List.replicate m (1 : Int)) = 1 := by
  induction m with
  | zero => simp [prod]
  | succ n ih => simp [List.replicate, prod, ih]

theorem dot_replicate_zero (n : Nat) (x : List Int) :
    dot (List.replicate n (0 : Int)) x = 0 := by
  induction n generalizing x with
  | zero => cases x <;> simp [dot, List.replicate]
  | succ m ih => cases x <;> simp [dot, List.replicate, ih]

theorem dot_nil_right (w : List Int) : dot w [] = 0 := by
  cases w <;> rfl

theorem iabs_mul (a b : Int) : iabs (a * b) = iabs a * iabs b := by
  simp [iabs, Int.natAbs_mul]

theorem iabs_add_le (a b : Int) : iabs (a + b) ≤ iabs a + iabs b := by
  simp only [iabs]; omega

theorem iabs_nonneg (a : Int) : 0 ≤ iabs a := by
  simp only [iabs]; omega

theorem basis_length : ∀ (n i : Nat), (basis n i).length = n := by
  intro n
  induction n with
  | zero => intro i; simp [basis]
  | succ k ih =>
    intro i
    cases i with
    | zero => simp [basis]
    | succ j => simp [basis, ih j]

theorem l1_nonneg (x : List Int) : 0 ≤ l1 x := by
  induction x with
  | nil => simp [l1]
  | cons a as ih => have := iabs_nonneg a; simp only [l1]; omega

end TypeNN
