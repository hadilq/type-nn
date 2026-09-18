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
types: every layer emits

\[
z_k = \mathrm{sign}(A_k)\,\ln(1+|A_k|/\tau_k)
\]

and the next layer consumes \(z\). That is why a type-nn stack is as
dense as an MLP.

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

The tail's task width never moves.

### And — product factors

Keep one dummy Or (\(\times 1\)) on every And. If back-prop gives it
weight, the dummy is no longer identity: append a new dummy identity
Or. If two dummy identity Ors sit on the same And, drop one.

Raising the assembly index \(a\) is preferred before opening a new
And clause (another generator). \(a_{\max}=3\) is the clip already
in the log-space product, not a dataset constant.

### Depth

Birth: stack to \(\lfloor 1+\ln(n m)\rfloor\) copies of the tail's
typed product. Grow may insert one more layer **between any two
layers** — the junction whose \(\|\partial L/\partial x\|_1\) is
largest. Shrink: drop a near-`×1` hidden that fell back to identity.

## Schedule

    u = step / (n_train · epochs)

    grow   u < 0.30           replace missing dummies; insert depth
    cut    0.30 ≤ u < 0.70    keep one dummy; snap |w| below signal
    shrink u ≥ 0.70           no new dummy; drop extra ×1 Or / width / layer

    thresh(signal) = hi + (lo − hi) · signal / (signal + 1)

A dummy Or must leave ×1 by a real margin and the residual must
still be loud. Identity hidden layers are the depth dummy: they
do not also open dummy Ors or extra width until they move.

What we are *not* doing: `max_or = f(dataset)`, hidden width = 8/16
from c-mlp, or any gate that looks at the file name.

## Initial depth

    depth_0 = floor(1 + ln(n m))

xor 1, iris 3, wine 4, wdbc 4 (out=1), ionosphere 4 (out=1), diabetes 3.
