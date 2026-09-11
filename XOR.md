# Why original type-nn hits 0 XOR MSE and type-nn-* does not

Same four points, seed 34972, 400 epochs, lr 0.08, one k=2 product
at birth (6 parameters). Measured:

    orig  dynamic=0   depth 1   mse 0
    orig  dynamic=1   depth 1   mse 0          (never inserted)
    win   dynamic=0   depth 1   mse 3.06e-4
    win   dynamic=1   depth 1   mse 1.48e-3    (F/win: insert refused)
    B     dynamic=1   depth 2   mse 3.07e-3    (slope test added a basis)

The And/Or identities are the same object:

    Or_t = clip(b_t + W_t · x, ±4)
    y    = clip(Or_1 * Or_2, ±32)

Original type-nn implements that as linked Or/And nodes.
type-nn-* implements it as dense W[r, j] in layerkit.
Forward and ∂y/∂Or_t = Π_{u≠t} Or_u are identical, including the
clips. We did not lose the product.

What changed is everything around the product.

## 1. The update (this is most of the gap)

Original, every sample:

    W ← clip(W − lr · ∂L/∂W, ±4)
    then snap:  |W| < 1e-7 → 0,  |W−1| < 1e-7 → 1

type-nn-*, every 16 samples (4 full XOR epochs):

    ḡ = mean of 16 raw gradients
    Adam(β1=0.9, β2=0.999) step on ḡ
    clip to ±4, no snap

On four points, original takes 400 × 4 = 1600 SGD steps.
type-nn-* takes 400 × 4 / 16 = 100 Adam steps, each on a smoothed
gradient. Adam’s second-moment estimate on a 4-row file is almost
a constant; it never finishes the last 1e-3.

The snap is why original prints exact 0.000000. XOR’s closed form
uses weights in {0, ±1, 2}. Once SGD lands inside 1e-7 of 0 or 1,
the parameter is frozen on that value and the product becomes an
exact Boolean. type-nn-* has no such lock, so it asymptotes at
~10^{-3}.

## 2. The residual that is sent backward

Original:

    ∂L/∂y = y − y*              (comment in type_nn.c: "we use e")

type-nn-* trainer (type_nn_alt_train):

    ∂L/∂y = (y − y*) / o        (matches torch mean-MSE)

For XOR, o = 1, so this is the same number. It matters on iris/wine
(o = 3) and is *not* why XOR differs.

## 3. Init

Original:          W, b ~ U(−0.15, +0.15)
type-nn-* win:     W, b ~ U(−s, s), s = 0.15 / √in
                   XOR in=2 ⇒ s ≈ 0.106

Same seed 34972, different scale, different first product. Both
are fine for XOR; this is a small effect next to SGD vs Adam.

## 4. Epoch order

Original walks the four rows in fixed order 00, 01, 10, 11.
type-nn-* shuffles each epoch with xorshift32. On four points that
changes the path; it does not change the representable set.

## 5. Dynamic insert on top of (1)

win/F see a stall, try an insert, revert it (add=0). The wasted
settle windows are why dyn=1 is 1.5e-3 and dyn=0 is 3e-4 — same
6 weights, less time spent descending.

B’s slope test still fires once (τ(n=4)=0.0038), so B is no longer
the 6-parameter product. The extra affine is why B is 3e-3.

Original dynamic=1 never inserts on XOR: its residual is already
falling under SGD, so the stall predicate is false.

## What was essential

Not a different And. Not a different XOR encoding.

    per-sample SGD + snap-to-{0,1}

That pair is what turns “XOR is representable by a product of two
affines” into “XOR mse is 0.” The optimizations (Adam, minibatch
16, init 1/√in, slope-stall inserts) are the right tools on UCI
tables and the wrong tools on four Boolean rows.

A fair “did we keep the algebra?” check is win with dynamic off:
mse 3e-4, same 6 weights, same four outputs to two decimals.
The remaining 3e-4 is Adam + no snap, not a broken product.
