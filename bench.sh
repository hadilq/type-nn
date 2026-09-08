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

ALTS="type_nn_stack.c type_nn_opt.c type_nn_proj.c type_nn_bpest.c type_nn_bpsite.c type_nn_bpdeep.c type_nn_idins.c type_nn_typefact.c type_nn_initd.c"

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
print("task         impl               params   nbytes  us/infer   train_s       mse      acc  dscale ddepth dparams  ladd ldrop")
print("-" * 108)
by = collections.defaultdict(list)
for r in rows:
    by[r["task"]].append(r)
order = ("xor", "quadratic", "mlp32x16x8", "iris", "wine", "wdbc", "diabetes")
for task in order:
    block = by.get(task, [])
    block.sort(key=lambda r: (
        r.get("mse", 1e300),
        r.get("params", 1 << 30),
        r.get("nbytes", 1 << 30),
        r.get("us_per_infer", 1e300),
        r.get("train_s", 1e300),
    ))
    for r in block:
        acc = r.get("acc", -1)
        acc_s = "   n/a" if acc is None or acc < 0 else f"{acc:6.3f}"
        print(f"{r['task']:<12} {r['impl']:<18} {r['params']:7d}  {r['nbytes']:7d}  "
              f"{r['us_per_infer']:8.3f}  {r['train_s']:8.4f}  "
              f"{r['mse']:.6f}  {acc_s}  "
              f"{r.get('dyn_scale', 0):6.1f} {r.get('dyn_depth', 0):6d} {r.get('dyn_params', 0):7d} {r.get('layer_add', 0):5d} {r.get('layer_drop', 0):5d}")
    if block:
        print()
print("Notes:")
print("  • dscale = Σ_layers (|ΔOr|+|ΔAnd|+|ΔOr|·|ΔAnd|) after train.")
print("    Or = sum-type count (out·k), And = product-type count (out).")
print("  • ddepth = final depth − start depth.\n  • ladd / ldrop = raw layer inserts / deletes during train (churn if both > 0).")
print("  • dparams = param_count after train − param_count at init.")
print("  • type-nn         = original linked lists (type_nn.c frozen).")
print("  • type-nn-arena   = And/Or nodes in a slab, integer next.")
print("  • type-nn-soa     = SoA W[or][in].")
print("  • type-nn-gemm    = blocked GEMV affine map.")
print("  • type-nn-csr     = CSR affine map.")
print("  • type-nn-hotcold = first 16 features dense.")
print("  • type-nn-q8      = int8 weights + per-Or scale.")
print("  • type-nn-tape    = Wengert-list reverse mode.")
print("  • type-nn-opt-q8  = previous packed opt (int8); XOR MSE is not exactly 0.")
print("  • type-nn-opt     = double W + adaptive SoA/GEMV, one-pass backward.")
print("  • type-nn-bp      = prefix/suffix dOr, fused dx, still SGD.")
print("  • type-nn-mom     = SGD + momentum (μ=0.9) on And/Or weights.")
print("  • type-nn-adam    = Adam (β1=0.9, β2=0.999) on And/Or weights.")
print("  • type-nn-bpgemm  = Wᵀ dOr + dOr xᵀ blocked backward.")
print("  • Rows sorted by (mse, params, nbytes, us/infer, train_s), all ascending.")
print("  • type-nn-dyn     = residual-driven grow k / insert / widen hidden.")
print("  • All type-nn-* variants have And/Or layers, grow/shrink, insert/remove.")
PY
