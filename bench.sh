#!/usr/bin/env bash
# Run type-nn, alternate layouts, and (if available) PyTorch benches.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

CC="${CC:-gcc}"
CFLAGS="${CFLAGS:--std=c11 -O2 -Wall -Wextra -I.}"
TASK="${1:-all}"

if [ -z "${TYPE_NN_DATA:-}" ]; then
  if [ -d "$ROOT/data" ] && [ -f "$ROOT/data/iris.data" ]; then
    export TYPE_NN_DATA="$ROOT/data"
  elif [ -d /tmp/type-nn-data ] && [ -f /tmp/type-nn-data/iris.data ]; then
    export TYPE_NN_DATA=/tmp/type-nn-data
  fi
fi

ALTS="type_nn_stack.c type_nn_idins.c type_nn_bpsite.c type_nn_proj.c type_nn_over.c"

echo "== building bench_type_nn + bench_alts =="
$CC $CFLAGS -o bench_type_nn type_nn.c bench_type_nn.c dataset.c -lm
$CC $CFLAGS -o bench_alts bench_alts.c dataset.c $ALTS -lm

echo "== type-nn (linked lists, frozen)  TYPE_NN_DATA=${TYPE_NN_DATA:-unset} =="
./bench_type_nn "$TASK" | tee /tmp/type_nn_bench.jsonl

echo "== type-nn-* layouts (arena soa gemm csr hotcold q8 tape opt) =="
./bench_alts "$TASK" | tee /tmp/alt_bench.jsonl

: > /tmp/torch_bench.jsonl
if python3 -c "import torch" >/dev/null 2>&1; then
    echo "== pytorch (CPU, 1 thread) =="
    python3 bench_torch.py "$TASK" | tee /tmp/torch_bench.jsonl
else
    echo "== pytorch skipped (python3 + torch not installed) =="
fi

python3 - << 'PY'
import json, collections
rows = []
for path in ("/tmp/type_nn_bench.jsonl", "/tmp/alt_bench.jsonl", "/tmp/torch_bench.jsonl"):
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("{"):
                    rows.append(json.loads(line))
    except FileNotFoundError:
        pass

print()
print("task         impl             hold_acc     acc  params   nbytes  us/infer   train_s  hold_mse      mse")
print("-" * 108)
by = collections.defaultdict(list)
for r in rows:
    by[r["task"]].append(r)
order = ("xor", "iris", "wine", "wdbc", "diabetes", "ionosphere")
for task in order:
    block = by.get(task, [])
    def acc_key(v):
        if v is None or v < 0:
            return -1.0
        return float(v)
    block.sort(key=lambda r: (
        -acc_key(r.get("hold_acc", r.get("acc", -1))),
        -acc_key(r.get("acc", -1)),
        r.get("params", 1 << 30),
        r.get("nbytes", 1 << 30),
        r.get("us_per_infer", 1e300),
        r.get("train_s", 1e300),
    ))
    for r in block:
        acc = r.get("acc", -1)
        acc_s = "   n/a" if acc is None or acc < 0 else f"{acc:6.3f}"
        hm = r.get("hold_mse", r.get("mse", 0))
        ha = r.get("hold_acc", acc)
        ha_s = "   n/a" if ha is None or ha < 0 else f"{ha:6.3f}"
        print(f"{r['task']:<12} {r['impl']:<16} {ha_s} {acc_s} {r['params']:7d} {r['nbytes']:7d} "
              f"{r['us_per_infer']:8.3f} {r['train_s']:8.4f} "
              f"{hm:8.5f} {r['mse']:8.5f}")
    if block:
        print()
print("Notes:")
print("  • infer_s  = wall seconds for infer_n forward passes.")
print("  • us/infer = infer_s / infer_n × 1e6  (microseconds per forward).")
print("  • mse / acc      = 70% train split (XOR uses all 4 rows).")
print("  • hold_mse / hold_acc = held-out 30% never trained on.")
print("  • type-nn-static = fixed k=1 / k=2 / k=1 stack (was type-nn-proj2).")
print("  • type-nn-opt    = unified dynamic policy.")
print("  • type-nn-over   = over-add I-maps, drop only if W≈I.")
print("  • type-nn-bpsite = early proj+readout from BP sites.")
print("  • Sorted by (hold_acc desc, acc desc, params, nbytes, us/infer, train_s).")
PY
