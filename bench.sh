#!/usr/bin/env bash
# Run Vortex and (if available) PyTorch benches and print a side-by-side table.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

CC="${CC:-gcc}"
CFLAGS="${CFLAGS:--std=c11 -O2 -Wall -Wextra -I.}"
TASK="${1:-all}"

echo "== building bench_vortex =="
$CC $CFLAGS -o bench_vortex bench_vortex.c vortex.c -lm

echo "== vortex (C, linked-list AND-OR) =="
./bench_vortex "$TASK" | tee /tmp/vortex_bench.jsonl

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
for path in ("/tmp/vortex_bench.jsonl", "/tmp/torch_bench.jsonl"):
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("{"):
                    rows.append(json.loads(line))
    except FileNotFoundError:
        pass

print()
print("task         impl         train_s   us/infer    rss_kb   params       mse")
print("-" * 78)
by = collections.defaultdict(list)
for r in rows:
    by[r["task"]].append(r)
for task in ("xor", "quadratic", "mlp32x16x8"):
    for r in by.get(task, []):
        print(f"{r['task']:<12} {r['impl']:<12} {r['train_s']:8.4f}  "
              f"{r['us_per_infer']:8.3f}  {r['rss_kb']:8d}  {r['params']:7d}  "
              f"{r['mse']:.6f}")
    print()
print("Notes:")
print("  • torch-mlp  = Linear+ReLU + Adam — the production default.")
print("  • torch-poly = Linear on explicit degree-2 features (same class as a 2-OR Vortex layer).")
print("  • Vortex rss is the C process; PyTorch rss includes the interpreter + MKL/OpenMP.")
print("  • Inference is one sample at a time on CPU, 1 thread — matching Vortex's API.")
print("  • On wide dense layers PyTorch's GEMM wins at train time; Vortex wins on tiny polynomial tasks.")
PY
