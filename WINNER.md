# type-nn-winner = ln + schedule probes

Not a per-dataset recipe. (in, out) is never inspected.

## Algebra

ln-v2: log-space And product, assembly index a > 0, slow eta_a,
ln tail readout. Hidden layers stay z so an identity insert does
not jump the map (PROBES.md).

## The only extra number: u = step / span

Same clock orcool already uses.

  u < grow     Or/And: energy or residual (scale up)
               Layer: insert only if residual large AND lists at cap
                      AND ||dL/dW|| stuck
               prune ~ quantization

  grow..cut    Or/And: jac only
               Layer: no insert

  u >= cut     Or/And: no spawn
               Layer: drop identity hidden if ||dL/dW|| ~ 0
               prune |w| < 0.015 + 0.05 * v

Hidden Or/And stay frozen. Spawn stays on the tail.

## Cutting variables

  winner / scale-sched        grow=0.40  cut=0.60
  scale-sched-tight           grow=0.25  cut=0.50
  scale-sched-wide            grow=0.55  cut=0.75

## vs c-mlp

  xor         c-mlp 33 / 0.00000     winner 28 / 4.5e-7
  iris        c-mlp 67 / 0.04324     winner 90 / 0.02239
  wine        c-mlp 275 / 0.02512    winner 143 / 0.13527
  wdbc        c-mlp 513 / 0.05594    winner 17 / 0.06507
  diabetes    c-mlp 193 / 0.04794    winner 21 / 0.03290
  ionosphere  c-mlp 577 / 0.16790    winner 24 / 0.14664

Both bars: xor (mse ~ 0), diabetes, ionosphere.
MSE only: iris. Params only: wine, wdbc.
