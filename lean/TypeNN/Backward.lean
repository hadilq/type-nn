/-
  TypeNN.Backward — the three backward-pass results.

  Lemma 3   prefix/suffix computes ∂A/∂O_s without dividing
  Theorem 5 the dead-factor theorem
  Prop 4    the Gauss–Seidel correction of the frozen core
-/
import TypeNN.Basic

namespace TypeNN

/-! ### Lemma 3 — prefix/suffix, division-free

`∂A_i/∂O_{i,s} = ∏_{r ≠ s} O_{i,r}`, and the prefix/suffix pair computes it
with `2(k-1)` multiplications and no division. Division by `O_{i,s}` is not
merely slower, it is undefined exactly where the interesting case lives, so
the statement below is what justifies `compute_accums` and `lk_dor`. -/

theorem dropIdx_eq_take_append_drop :
    ∀ (s : Nat) (l : List Int), dropIdx s l = l.take s ++ l.drop (s+1) := by
  intro s
  induction s with
  | zero => intro l; cases l <;> simp [dropIdx]
  | succ n ih =>
    intro l
    cases l with
    | nil => simp [dropIdx]
    | cons a as => simp [dropIdx, ih as]

/-- **Lemma 3.** The prefix product times the suffix product is the product of
every factor except the `s`-th. -/
theorem prefix_suffix (s : Nat) (os : List Int) :
    prod (os.take s) * prod (os.drop (s+1)) = prod (dropIdx s os) := by
  rw [dropIdx_eq_take_append_drop, prod_append]

/-! ### Theorem 5 — the dead-factor theorem

If any factor of an And is zero then every *other* factor has zero gradient.
If two factors are zero then the whole And has zero gradient and gradient
descent can never leave it. This is the only structural failure mode of a pure
product layer, and it is what `or_is_dead` and `drop_dead_or` escape. -/

theorem prod_eq_zero_of_mem : ∀ (l : List Int), (0 : Int) ∈ l → prod l = 0 := by
  intro l
  induction l with
  | nil => intro h; cases h
  | cons a as ih =>
    intro h
    rcases List.mem_cons.mp h with h1 | h2
    · simp [prod, ← h1]
    · simp [prod, ih h2]

theorem mem_dropIdx_of_ne :
    ∀ (s t : Nat) (l : List Int), t ≠ s → l[t]? = some 0 → (0 : Int) ∈ dropIdx s l := by
  intro s
  induction s with
  | zero =>
    intro t l hts hl
    cases l with
    | nil => simp at hl
    | cons a as =>
      cases t with
      | zero => exact absurd rfl hts
      | succ m => simp [dropIdx]; exact List.mem_of_getElem? hl
  | succ n ih =>
    intro t l hts hl
    cases l with
    | nil => simp at hl
    | cons a as =>
      cases t with
      | zero => simp at hl; simp [dropIdx, hl]
      | succ m =>
        have : m ≠ n := fun h => hts (by omega)
        simp at hl
        exact List.mem_cons_of_mem a (ih m as this hl)

/-- **Theorem 5 (part 1).** A zero factor at index `t` kills the gradient of
every sibling `s ≠ t`. -/
theorem dead_factor (s t : Nat) (os : List Int) (hst : t ≠ s)
    (ht : os[t]? = some 0) : prod (dropIdx s os) = 0 :=
  prod_eq_zero_of_mem _ (mem_dropIdx_of_ne s t os hst ht)

/-- **Theorem 5 (part 2).** Two distinct zero factors freeze the whole And:
*every* factor gradient vanishes, so the point is an absorbing state of the
gradient flow and no step can leave it. -/
theorem dead_and (t u : Nat) (os : List Int) (htu : t ≠ u)
    (ht : os[t]? = some 0) (hu : os[u]? = some 0) :
    ∀ s, prod (dropIdx s os) = 0 := by
  intro s
  by_cases h : t = s
  · exact dead_factor s u os (by omega) hu
  · exact dead_factor s t os h ht

/-! ### Proposition 4 — the Gauss–Seidel correction

`type_nn.c` applies the SGD step inside `or_backward`, i.e. *before*
`layer_backward` forms the input-side gradient, so the vector handed to the
previous layer is built from weights that have already moved. The discrepancy
is not vague: it is exactly rank one along `x`, with coefficient
`η ‖∇_O L‖₂²`. Textbook backprop is the `η → 0` limit. -/

/-- The backward sweep as the frozen core actually performs it: each weight is
updated with its own gradient, and the input-side sum reads the updated value. -/
def gsBackward (eta x : Int) : List Int → List Int → Int
  | [],      _       => 0
  | _,       []      => 0
  | g :: gs, w :: ws => g * (w - eta * g * x) + gsBackward eta x gs ws

private theorem gs_step (g w eta x A B : Int) :
    g * (w - eta * g * x) + (A - eta * x * B)
      = (g * w + A) - eta * x * (g * g + B) := by
  simp [Int.mul_add, Int.sub_eq_add_neg, Int.neg_add, Int.mul_neg]
  ac_rfl

/-- **Proposition 4.** The propagated gradient equals the exact one minus
`η · ‖∇_O L‖₂² · x`. -/
theorem gauss_seidel :
    ∀ (g w : List Int), g.length = w.length → ∀ (eta x : Int),
      gsBackward eta x g w = dot g w - eta * x * sqSum g := by
  intro g
  induction g with
  | nil => intro w _ eta x; simp [gsBackward, dot, sqSum]
  | cons a as ih =>
    intro w hlen eta x
    cases w with
    | nil => simp at hlen
    | cons b bs =>
      have hl : as.length = bs.length := by simpa using hlen
      simp only [gsBackward, dot, sqSum, ih bs hl eta x]
      exact gs_step a b eta x (dot as bs) (sqSum as)

end TypeNN
