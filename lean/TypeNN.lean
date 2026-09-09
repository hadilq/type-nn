/-
  Type-NN — machine-checked statements of the results in the post
  https://hadilq.com/posts/type-neural-network/

  Build with `lake build`, or `make lean` from the repository root.

  What is proved here:

    Proposition 2   TypeNN.gauge_invariant       gauge symmetry of a product
    Lemma 3         TypeNN.prefix_suffix         ∂A/∂O_s without dividing
    Proposition 4   TypeNN.gauss_seidel          the in-place backward correction
    Theorem 5       TypeNN.dead_factor           one zero factor freezes the siblings
                    TypeNN.dead_and              two zero factors freeze the And
    Proposition 6   TypeNN.orVal_zero_pad        zero-padding the input is exact
    Proposition 7   TypeNN.widen_preserves       widening preserves the composite
    Proposition 8   TypeNN.grow_rank_exact       a unit factor is exact
    Proposition 11  TypeNN.identity_insert       the identity insert is exact
    Proposition 12  TypeNN.identity_test_bound   the identity test is only a bound

  Not formalised here, and said so in the post: Proposition 1 (degree of the
  composite) needs a theory of polynomial degree, and Propositions 9 and 10
  (the ones-drop error bound and the dead-drop rescue) are stated over the
  clipped algebra.
-/
import TypeNN.Basic
import TypeNN.Backward
import TypeNN.Surgery

-- Audit: none of these depend on `sorryAx`.
#print axioms TypeNN.gauge_invariant
#print axioms TypeNN.prefix_suffix
#print axioms TypeNN.gauss_seidel
#print axioms TypeNN.dead_factor
#print axioms TypeNN.dead_and
#print axioms TypeNN.orVal_zero_pad
#print axioms TypeNN.widen_preserves
#print axioms TypeNN.grow_rank_exact
#print axioms TypeNN.identity_insert
#print axioms TypeNN.identity_test_bound
