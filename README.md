# type-nn

C And-Or net. An And is a product of Ors. Dummy Or / dummy And are
identity factors (`×1`). After back-prop, extra identities drop.

Scale recipes have **no max_or and no max_and**. Live factors are
unbounded. The only occupancy rule is the dummy rule (at most one
identity Or per And, one identity And per output).

Depth recipes insert an identity hidden the same way. `depth-early`
does it at step 0 and keeps it. `depth-born` allocates the hidden
before init (random, c-mlp width). Details: [AUDIT.md](AUDIT.md),
[PROBES.md](PROBES.md), board in [BOARD.txt](BOARD.txt).

Current board (`./bench_type_nn TASK name`):

- `scale-energy` / `scale-jac` / `scale-mix` — the three width gates
- `depth-early` / `depth-hold` / `depth-born` — the depth gates
- `scale-mix+depth-early`, `scale-energy+depth-hold`,
  `scale-ej+depth-born`, … — both axes
- `c-mlp` — Linear-ReLU-Linear baseline (`./bench_alts TASK c-mlp`)

Each board model has its own `type_nn_*.c` / `type_nn_*.h` wrapper.

## Fair split

xorshift32 Fisher-Yates, seed 34972, 70/30. Same cut for every impl.

    ./bench_type_nn iris scale-mix+depth-early
    ./bench_alts iris c-mlp
    make test
    make bench

`us/infer` is mean microseconds per `predict` / `forward`. The bench
keeps calling until 50 ms of wall time so the column cannot print 0.
