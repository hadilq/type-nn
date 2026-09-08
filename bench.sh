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

ALTS="type_nn_stack.c type_nn_soa.c type_nn_gemm.c type_nn_arena.c type_nn_csr.c type_nn_hotcold.c type_nn_q8.c type_nn_tape.c type_nn_opt.c"

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
print("task         impl         train_s   us/infer    nbytes   params       mse      acc")
print("-" * 90)
by = collections.defaultdict(list)
for r in rows:
    by[r["task"]].append(r)
order = ("xor", "quadratic", "mlp32x16x8", "iris", "wine", "wdbc", "diabetes")
impl_order = ("type-nn","type-nn-arena","type-nn-soa","type-nn-gemm",
              "type-nn-csr","type-nn-hotcold","type-nn-q8","type-nn-tape",
              "type-nn-opt","torch-mlp","torch-poly")
for task in order:
    block = by.get(task, [])
    block.sort(key=lambda r: impl_order.index(r["impl"]) if r["impl"] in impl_order else 99)
    for r in block:
        acc = r.get("acc", -1)
        acc_s = "   n/a" if acc is None or acc < 0 else f"{acc:6.3f}"
        print(f"{r['task']:<12} {r['impl']:<12} {r['train_s']:8.4f}  "
              f"{r['us_per_infer']:8.3f}  {r['nbytes']:8d}  {r['params']:7d}  "
              f"{r['mse']:.6f}  {acc_s}")
    if block:
        print()
print("Notes:")
print("  • type-nn         = original linked lists (type_nn.c frozen).")
print("  • type-nn-arena   = And/Or nodes in a slab, integer next.")
print("  • type-nn-soa     = SoA W[or][in].")
print("  • type-nn-gemm    = blocked GEMV affine map.")
print("  • type-nn-csr     = CSR affine map.")
print("  • type-nn-hotcold = first 16 features dense.")
print("  • type-nn-q8      = int8 weights + per-Or scale.")
print("  • type-nn-tape    = Wengert-list reverse mode.")
print("  • type-nn-opt     = int8 + unrolled dot, smallest+fastest combo.")
print("  • All type-nn-* variants have And/Or layers, grow/shrink, insert/remove.")
PY
