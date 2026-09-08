#!/usr/bin/env bash
# Run type-nn and (if available) PyTorch benches; print a comparison table.
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

echo "== building bench_type_nn =="
$CC $CFLAGS -o bench_type_nn type_nn.c bench_type_nn.c dataset.c -lm

echo "== type-nn (C, linked-list AND-OR)  TYPE_NN_DATA=${TYPE_NN_DATA:-unset} =="
./bench_type_nn "$TASK" | tee /tmp/type_nn_bench.jsonl

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
for path in ("/tmp/type_nn_bench.jsonl", "/tmp/torch_bench.jsonl"):
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("{"):
                    rows.append(json.loads(line))
    except FileNotFoundError:
        pass

print()
print("task         impl         train_s   us/infer    rss_kb   params       mse      acc")
print("-" * 88)
by = collections.defaultdict(list)
for r in rows:
    by[r["task"]].append(r)
order = ("xor", "quadratic", "mlp32x16x8", "iris", "wine", "wdbc", "diabetes")
for task in order:
    for r in by.get(task, []):
        acc = r.get("acc", -1)
        acc_s = "   n/a" if acc is None or acc < 0 else f"{acc:6.3f}"
        print(f"{r['task']:<12} {r['impl']:<12} {r['train_s']:8.4f}  "
              f"{r['us_per_infer']:8.3f}  {r['rss_kb']:8d}  {r['params']:7d}  "
              f"{r['mse']:.6f}  {acc_s}")
    if task in by:
        print()
print("Notes:")
print("  • type-nn     = product of affine units (Type Mechanics AND/OR).")
print("  • torch-mlp   = Linear+ReLU + Adam — the production default.")
print("  • torch-poly  = Linear on explicit degree-2 features.")
print("  • iris/wine/wdbc/diabetes are fetched by flake.nix (or `make data`).")
print("  • acc is argmax / 0.5-threshold train accuracy; n/a for regression.")
print("  • PyTorch rss includes the interpreter + MKL; type-nn rss is the C process.")
PY
