# Slimming models that already beat c-mlp hold_mse

c-mlp param counts: xor 33, iris 67, wine 275, wdbc 513,
diabetes 193, ionosphere 577.

Ideas tried (plus-form tokens, last token wins the cap):

1. **slim-cap** — `max_or = 2`. Spawn still uses energy/jac, but a
   clause cannot grow a third Or. Biggest lever on iris/wine mix.
2. **slim-budget** — at most one live And + one dummy And per head
   (`AP_BUDGET`). Stops And-list blow-up.
3. **slim-prune** — drop `|w| < 0.05` after each sample and do not
   refill those coordinates. Param count is live weights only.
4. **slim-k** — keep the 3 largest `|w|` on every live Or. Dummy
   Ors are left alone so they can still leave 1.
5. **slim-narrow** — depth insert uses `max(out, ceil(in/2))`
   hidden units instead of 8/16 or `2·in`.

Compare `scale-jac` / `scale-mix` with and without each knob.
Target: `hold_mse < c-mlp` **and** `params < c-mlp`.
