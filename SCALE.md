# Scaling Or, And, and Depth

The typed product has three axes. None of the knobs below read a
dataset name, an input arity table, or a hold-out score. They only
see the current type (the live product) and the residual of the
sample that just went backward.

## Why a raw product loses

A dense Or on \(d\) features is one affine. A product of \(k\) of
them is degree \(k\) in \(d\) variables. Stacking those products
without an activation explodes. c-mlp dodges this by composing two
linear maps of bounded width plus ReLU. type-nn does the same with
types: every *typed* layer emits

\[
z_k = \mathrm{sign}(A_k)\,\ln(1+|A_k|/\tau_k)
\]

and the next layer consumes \(z\). An identity hidden (the depth
dummy) skips that log so the map stays \(\times 1\) until back-prop
moves it. That is why a type-nn stack is as dense as an MLP.

## The dummy rule (all three axes)

1. Before back-prop the list contains exactly one identity.
2. Back-prop is allowed to move that identity.
3. After back-prop, extra identities drop. Grow early; drop late.

### Or — incoming width

When Or scales **up**, the *previous* layer adds a new output
coordinate and trains it. The *current* layer pairs that slot with a
dummy weight (born at 0). When Or scales **down**, that coordinate
is dropped from the previous layer's output and the current layer
realigns.

Birth width is the task type \(m\), so the stack is
\(n\to m \to \cdots \to m\) of typed products + ln. A square
\(n\to n\) map is not a type-nn layer. Or-scale may add dummy
coordinates up to \(\max(m,1+\ln(1+n))+1\). The tail's task width
never moves.

### And — product factors

Keep one dummy Or (\(\times 1\), \(b\approx 1, W=0\) on the live
incoming type) on every And. The zero weight list is required: a
dummy with no weights can only move its bias, and And-scaling
never fires. \(W=0\) also means a dummy does not inflate `params`
until back-prop moves it.

If back-prop gives those dummy weights a value, the dummy is no
longer identity: append a new dummy identity Or. If two dummy
identity Ors sit on the same And, drop one.

Raising the assembly index \(a\) copies an existing clause
(\(A\mapsto A^{a}\)). The board product is one And per head —
\(A_k=\prod_r \mathrm{Or}_{k,r}^{a_{k,r}}\) — so And-scale does
not open a second clause. \(a\) is only forced \(>0\). Dummy Or
bias is born with a \(10^{-3}\) kick off \(1\) so
\(\partial Z/\partial a = Z\log|A|\) is live.

A live-Or fence \(1+\ln(1+n)\) follows the incoming type size. It
does not read a file name. XOR (\(n\le 2\)) stays degree 2.

### Depth

Birth: stack to \(\lfloor 1+\ln(n m)\rfloor\) copies of the tail's
typed product, each of width \(m\). Grow may insert one more layer
**between any two layers** — the junction whose \(\|\partial L/\partial x\|_1\)
is largest. Shrink: drop a near-`×1` hidden that fell back to identity.

## Schedule

    u = step / (n_train · epochs)

    grow   u < 0.30           replace missing dummies; insert depth
    cut    0.30 ≤ u < 0.70    keep one dummy; snap |w| below signal
    shrink u ≥ 0.70           no new dummy; drop extra ×1 Or / width / layer

    thresh(signal) = hi + (lo − hi) · signal / (signal + 1)

A dummy Or must leave ×1 by a real margin. Identity hidden layers
are the depth dummy: they still carry one dummy Or and may grow a
dummy outgoing coordinate, but they skip the log until the diagonal
moves.

Live Or support after the cut is \(1+\ln(1+\mathrm{fan})\): a
type-size top-k, not a dataset table. Dummy weights stay 0 until
back-prop moves them.

What we are *not* doing: `max_or = f(dataset)`, hidden width = 8/16
from c-mlp, or any gate that looks at the file name.

## Initial depth

    depth_0 = floor(1 + ln(n m))

xor 1, iris 3, wine 4, wdbc 4 (out=1), ionosphere 4 (out=1), diabetes 3.
