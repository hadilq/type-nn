# type-nn

C And-Or net. An And is a product of Ors. Dummy Or / dummy And are
identity factors (`×1`). After back-prop, extra identities drop.

Scale recipes have **no max_or and no max_and**. Live factors are
unbounded. The only occupancy rule is the dummy rule (at most one
identity Or per And, one identity And per output). A missing dummy
is replaced only when a back-prop gate says the current Jacobian
cannot explain the incoming residual.

Current recipes (`./bench_type_nn TASK name`):

- `scale-keep` — always replace a missing dummy (raw product)
- `scale-energy` — spawn if `|parent| > κ · Σ|live child grad|`
- `scale-jac` — spawn if residual large and `Σ|∂L/∂W|_live` small
- `scale-slack` / `scale-sign` — specialization / sign-conflict
- `scale-mix` — energy on Or, jac on And
- `scale-ej` — energy and jac together
- `ln-v2w+scale-mix` — log tail + typed `a` + energy-on-Or

Layer insert (`Ljac`, `scale-layer`) is the same dummy rule one
level up. It is **not** shipped: one identity hidden on iris drops
hold from 0.978 to 0.356. Details: [PROBES.md](PROBES.md), board
in [BOARD.txt](BOARD.txt).

**type-nn-ln** lives in `type_nn_ln.c` / `type_nn_ln.h`. Same
linked net, type exponent \(a_{i,r}\) on every And, log tail:

    Or_{i,r,t} = b_{i,r,t} + Σ_j W_{i,r,t,j} x_j
    A_{i,r}    = sign-exp clip(Σ_t log|Or|)     # LN_LOGZ
    z_i        = sign-exp clip(Σ_r a_{i,r} log|A_{i,r}|)
    y_i        = sign(z_i) ln(1 + |z_i|/√d)     # tail only

## Fair split

xorshift32 Fisher-Yates, seed 34972, 70/30. Same cut for every impl.

    ./bench_type_nn iris scale-keep
    ./bench_type_nn ionosphere ln-v2w+scale-mix
    make test
    make bench
