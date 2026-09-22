# type-nn: forward, backward, and structural edits

![type-nn](type-nn-forward-backward.svg)

The code is `type_nn.c` (forward/backward) and `type_nn_scale.c`
(edits). The theorems named below are in `../coqLang/TypeNN/`.

## Forward

For unit $k$ of a layer with input $x\in\mathbb R^{n}$:

$$
o_r = w_r\cdot x + b_r,\qquad
\ell=\sum_r a_r\ln|o_r|,\qquad
s=\prod_r\operatorname{sign}(o_r),\qquad
z = s\,\operatorname{softplus}(\ell).
$$

This is the design written in log space. With $|A|=e^{\ell}$ and
$A=s\,e^{\ell}$, $z=\operatorname{sign}(A)\ln(1+|A|)$ (Coq
`F_log_space`). If any $o_r=0$, then $s=0$, $\ell=-\infty$ and $z=0$.
The output vector $z$ is the next layer's $x$ (Coq `net_app`).

## Backward

Let $g=\partial L/\partial z$ for the unit, and $\sigma$ the logistic
function. Differentiating $z=s\,\operatorname{softplus}(\ell)$:

$$
\frac{\partial z}{\partial o_r}=s\,\sigma(\ell)\,\frac{a_r}{o_r},\qquad
\frac{\partial z}{\partial a_r}=s\,\sigma(\ell)\ln|o_r| .
$$

Why this is the same as the product-form chain rule:

- $s\,\sigma(\ell)=F'(A)\cdot A$ (Coq `log_space_slope`).
- $A\,a_r/o_r=a_r|o_r|^{a_r-1}\prod_{q\neq r}o_q^{a_q}$, the cofactor
  (Coq `andPow_first_order`, `prod_multilinear`).

So $\partial z/\partial o_r=F'(A)\,\partial A/\partial o_r$. As $A\to0$,
$\sigma(\ell)\approx|A|$, and the expression stays finite for $a_r\ge1$.

At $o_r=0$ exactly, the formula divides by zero, so the code uses the
limit instead:

- the cofactor $\prod_{q\ne r}o_q^{a_q}$ times $F'(0)=1$ when $a_r=1$;
- $0$ when $a_r>1$;
- $\partial z/\partial a_r=0$ in both cases.

Then for $\delta_r=g\,\partial z/\partial o_r$:

$$
\frac{\partial L}{\partial w_{r,j}}=\delta_r x_j,\quad
\frac{\partial L}{\partial b_r}=\delta_r,\quad
\frac{\partial L}{\partial a_r}=g\,\frac{\partial z}{\partial a_r},\quad
\frac{\partial L}{\partial x_j}\mathrel{+}=\delta_r w_{r,j}
$$

(Coq `or_grad_w`, `or_grad_b`, `or_grad_x`, `din_chain`).

$\partial L/\partial x$ accumulates with the weights *before* the Or's
Adam step. Updating first would be rank-1 off textbook back-prop (Coq
`gauss_seidel`). Each Or keeps its own Adam step counter, so structure
born late is bias-corrected like structure born at step 1. $a$ is
projected to $a\ge1$.

The output gradient is mean-MSE, $\partial L/\partial y=(y-t)/m$, the
same for both models. Coq `mean_mse_grad` covers only the $m=1$ case.

## Edits

| edit | what changes | why the function does not move |
|---|---|---|
| And probe in | append Or $(w\approx0,b\approx1)$ | $1^{a}=1$ (`and_probe_exact`, `and_probe_anywhere`) |
| Or probe in | producer gets a unit; consumer gets a 0 column | `width_probe_exact`, `width_probe_net` |
| depth probe in | carrier layer $x_k\mapsto x_k$ at the loudest gap; the consumer folds $x\approx\alpha F(x)+\beta$ | A-level identity (`identity_layerA`); fold exact when the fit is (`fold_exact`) |
| And drop | remove an Or inside the identity band | it is $\times(1+O(\theta))$ |
| Or drop | remove a coordinate whose outgoing weights are in the band; bias $+{=}\,w_j\,\mathbb E[x_j]$ | exact at the mean (`drop_column_mean`), exact for zero weight (`drop_zero_column`) |
| depth drop | remove a layer within the band of the identity; the consumer folds $F(x)\approx\alpha' x+\beta'$ | `fold_exact`, `identity_test_bound` |

**Band and promotion.**

- The distance from the identity is the RMS over the item's parameters,
  measured against the identity values:
  - $w=0$, $b=1$, $a=1$ for an Or;
  - weight 0 for a coordinate's outgoing column;
  - carrier plus identity Ors for a layer.
- A probe of age $T$ steps is promoted when that distance exceeds
  $\theta(T)=\eta T^{3/4}$.
- In the prune phase an item is dropped when its distance is at most
  $\theta(N)$, with $N$ the samples per epoch.
- Why $3/4$: see the README and Coq `threshold_between`,
  `threshold_geometric_mean`, `threshold_separates`.

**Evidence (type-nn only).** type-nn-overfit uses the rules above
unchanged. type-nn keeps the edits and changes the decision:

- **Promote.** A probe is promoted when it passes the displacement test
  **and**
  $n\ln(\mathrm{MSE}_{\text{without}}/\mathrm{MSE}_{\text{with}}) > k\ln n$.
- **Prune.** Items are ablated to identity in order of the damage each
  does alone. An ablation is kept while
  $C=n\ln\mathrm{MSE}+K\ln n$ stays at or below its best value seen
  while pruning; otherwise it is undone. Ablated items are then removed,
  which is exact because they are identities.
- **Layers.** A layer is removed on trial with the fold and restored if
  $C$ would rise.

**Order within an epoch boundary.** Depth edits come first, because they
read the junction statistics that the width edits reset. Width edits
come next, then And edits. At most one layer edit happens per boundary.
