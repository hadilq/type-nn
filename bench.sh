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

ALTS="type_nn_stack.c type_nn_win.c type_nn_cmlp.c"

echo "== building bench_type_nn + bench_alts =="
$CC $CFLAGS -o /tmp/bench_type_nn type_nn.c type_nn_ln.c type_nn_layer.c type_nn_grow.c bench_type_nn.c dataset.c -lm
$CC $CFLAGS -o /tmp/bench_alts bench_alts.c dataset.c $ALTS -lm

echo "== type-nn (Or/And probes, no layer probe)  TYPE_NN_DATA=${TYPE_NN_DATA:-unset} =="
/tmp/bench_type_nn "$TASK" | tee /tmp/type_nn_bench.jsonl
# Iteration 5: no max_or / max_and. Width and depth from BP statistics.
for mode in ln-v2w ln-adam ln-v2w+adam \
            ln-v2w+scale-mix ln-v2w+scale-mix+adam \
            ln-v2w+scale-energy ln-v2w+scale-energy+adam \
            scale-keep scale-mix; do
  echo "== type-nn-$mode =="
  /tmp/bench_type_nn "$TASK" "$mode" | tee -a /tmp/type_nn_bench.jsonl
done

echo "== type-nn-win + c-mlp  TYPE_NN_DATA=${TYPE_NN_DATA:-unset} =="
/tmp/bench_alts "$TASK" | tee /tmp/alt_bench.jsonl

: > /tmp/torch_bench.jsonl
if python3 -c "import torch" >/dev/null 2>&1; then
    echo "== pytorch (CPU, 1 thread) =="
    python3 bench_torch.py "$TASK" | tee /tmp/torch_bench.jsonl
else
    echo "== pytorch skipped (python3 + torch not installed) =="
fi

python3 - << 'PY'
import json, collections, pathlib
rows = []
for path in ("/tmp/type_nn_bench.jsonl", "/tmp/alt_bench.jsonl", "/tmp/torch_bench.jsonl"):
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("{"):
                    line = (line.replace("-nan", "null").replace("nan", "null")
                                .replace("-inf", "null").replace("inf", "null"))
                    rows.append(json.loads(line))
    except FileNotFoundError:
        pass
# last row for (impl, task) wins — reruns replace earlier attempts
uniq = {}
for r in rows:
    uniq[(r.get("impl"), r.get("task"))] = r
rows = list(uniq.values())

hdr = ("task         impl             hold_acc     acc  params   nbytes  us/infer   train_s"
       "  or+ or- and+ and-  L+  L-  hold_mse      mse")
bar = "-" * len(hdr)
lines = []
lines.append("type-nn fair hold-out board")
lines.append("===========================")
lines.append("")
lines.append("Split: xorshift32 Fisher-Yates, seed 34972, 70/30. Same cut for every impl.")
lines.append("type-nn = original linked net: Or + And probes, NO layer probe.")
lines.append("type-nn-* = same And/Or algebra + layer controllers A–I / win.")
lines.append("or+/or- = dummy Or (×1) promoted / collapsed.")
lines.append("and+/and- = dummy And (product ≡ 1) promoted / collapsed.")
lines.append("L+/L- = identity layer insert / drop (always 0 for original type-nn).")
lines.append("Sorted by (hold_acc desc, acc desc, params, nbytes, us/infer, train_s).")
lines.append("")
lines.append(hdr)
lines.append(bar)
by = collections.defaultdict(list)
for r in rows:
    by[r["task"]].append(r)
order = ("xor", "iris", "wine", "wdbc", "diabetes", "ionosphere")
def acc_key(v):
    if v is None or v < 0:
        return -1.0
    return float(v)
for task in order:
    block = by.get(task, [])
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
        def fnum(v):
            if v is None:
                return "     nan"
            try:
                import math
                if not math.isfinite(float(v)):
                    return "     nan"
            except (TypeError, ValueError):
                return "     nan"
            return f"{float(v):8.5f}"
        hm = r.get("hold_mse", r.get("mse", 0))
        ha = r.get("hold_acc", acc)
        ha_s = "   n/a" if ha is None or ha < 0 else f"{ha:6.3f}"
        lines.append(
            f"{r['task']:<12} {r['impl']:<16} {ha_s} {acc_s} {r['params']:7d} {r['nbytes']:7d} "
            f"{r.get('us_per_infer', 0):8.4f} {r.get('train_s', 0):8.4f} "
            f"{int(r.get('or_add', 0)):4d} {int(r.get('or_drop', 0)):3d} "
            f"{int(r.get('and_add', 0)):4d} {int(r.get('and_drop', 0)):4d} "
            f"{int(r.get('layer_add', 0)):3d} {int(r.get('layer_drop', 0)):3d} "
            f"{fnum(hm)} {fnum(r.get('mse'))}")
    if block:
        lines.append("")
text = "\n".join(lines) + "\n"
print()
print(text, end="")
pathlib.Path("BOARD.txt").write_text(text)
print("wrote BOARD.txt")
PY
