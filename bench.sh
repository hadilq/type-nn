#!/usr/bin/env bash
# Run the type-nn scale/depth board and the c-mlp baseline.
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

MODELS="type_nn_scale_energy.c type_nn_scale_jac.c type_nn_scale_mix.c \
        type_nn_depth_early.c type_nn_depth_hold.c type_nn_depth_born.c \
        type_nn_scale_mix_early.c type_nn_scale_energy_hold.c type_nn_scale_ej_born.c \
        type_nn_scale_jac_early.c type_nn_scale_jac_hold.c type_nn_scale_mix_hold.c \
        type_nn_slim_cap.c type_nn_slim_prune.c type_nn_slim_k.c type_nn_winner.c type_nn_scale_sched.c"

echo "== building bench_type_nn + bench_alts =="
$CC $CFLAGS -o /tmp/bench_type_nn type_nn.c type_nn_ln.c type_nn_layer.c type_nn_grow.c \
    bench_type_nn.c dataset.c $MODELS -lm
$CC $CFLAGS -o /tmp/bench_alts bench_alts.c dataset.c type_nn_alt.c type_nn_cmlp.c -lm

echo "== type-nn scale + depth  TYPE_NN_DATA=${TYPE_NN_DATA:-unset} =="
: > /tmp/type_nn_bench.jsonl
# Three scaling probes (energy, jac, mix) and the Or/And/Depth combos.
for mode in winner scale-sched scale-sched-tight scale-sched-wide scale-energy scale-jac scale-mix \
            depth-early depth-hold depth-born \
            scale-mix+depth-hold scale-jac+depth-early \
            scale-jac+slim-cap scale-mix+slim-cap \
            scale-jac+slim-prune scale-mix+slim-prune \
            scale-jac+slim-k scale-mix+slim-k \
            scale-jac+slim-budget scale-mix+slim-budget \
            scale-jac+slim-cap+slim-budget \
            scale-mix+slim-cap+slim-k+slim-prune \
            scale-mix+depth-hold+slim-narrow; do
  echo "== type-nn-$mode =="
  /tmp/bench_type_nn "$TASK" "$mode" | tee -a /tmp/type_nn_bench.jsonl
done

echo "== c-mlp  TYPE_NN_DATA=${TYPE_NN_DATA:-unset} =="
/tmp/bench_alts "$TASK" c-mlp | tee /tmp/alt_bench.jsonl

python3 - << 'PY'
import json, collections, pathlib
rows = []
for path in ("/tmp/type_nn_bench.jsonl", "/tmp/alt_bench.jsonl"):
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
uniq = {}
for r in rows:
    uniq[(r.get("impl"), r.get("task"))] = r
rows = list(uniq.values())

hdr = ("task         impl                         hold_mse  params   train_s      mse  hold_acc     acc   nbytes    us/infer"
       "  or+ or- and+ and-  L+  L-")
bar = "-" * len(hdr)
lines = []
lines.append("type-nn fair hold-out board")
lines.append("===========================")
lines.append("")
lines.append("Split: xorshift32 Fisher-Yates, seed 34972, 70/30. Same cut for every impl.")
lines.append("Board models: the three scale probes (energy, jac, mix), the Or/And")
lines.append("spawn they drive, and the depth probes (early / hold / born).")
lines.append("c-mlp is the only non-type-nn row (Linear-ReLU-Linear baseline).")
lines.append("or+/or- = dummy Or (×1) promoted / collapsed.")
lines.append("and+/and- = dummy And (product ≡ 1) promoted / collapsed.")
lines.append("L+/L- = identity layer insert / drop.")
lines.append("us/infer = mean microseconds per forward over >=200 ms wall (never 0).")
lines.append("hold_acc is n/a on diabetes (regression) only. XOR prints threshold acc")
lines.append("on all 4 points (no 70/30 cut exists).")
lines.append("Sorted by (hold_mse, params, train_s, mse, hold_acc asc, acc asc).")
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
    def mse_key(v):
        if v is None:
            return 1e300
        try:
            x = float(v)
        except (TypeError, ValueError):
            return 1e300
        import math
        return x if math.isfinite(x) else 1e300
    def acc_asc(v):
        if v is None or v < 0:
            return 1e300
        return float(v)
    block.sort(key=lambda r: (
        mse_key(r.get("hold_mse", r.get("mse"))),
        r.get("params", 1 << 30),
        r.get("train_s", 1e300),
        mse_key(r.get("mse")),
        acc_asc(r.get("hold_acc", r.get("acc"))),
        acc_asc(r.get("acc")),
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
        impl = r['impl']
        if len(impl) > 28:
            impl = impl[:28]
        us = r.get('us_per_infer')
        if us is None:
            inf_s = r.get('infer_s') or 0.0
            inf_n = r.get('infer_n') or 0
            us = (inf_s * 1e6 / inf_n) if inf_n else 0.0
        try:
            us = float(us)
        except (TypeError, ValueError):
            us = 0.0
        if us <= 0.0:
            us = 1e-6
        lines.append(
            f"{r['task']:<12} {impl:<28} {fnum(hm)} {r['params']:7d} "
            f"{r.get('train_s', 0):8.4f} {fnum(r.get('mse'))} {ha_s} {acc_s} "
            f"{r['nbytes']:7d} {us:10.6f} "
            f"{int(r.get('or_add', 0)):4d} {int(r.get('or_drop', 0)):3d} "
            f"{int(r.get('and_add', 0)):4d} {int(r.get('and_drop', 0)):4d} "
            f"{int(r.get('layer_add', 0)):3d} {int(r.get('layer_drop', 0)):3d}")
    if block:
        lines.append("")
text = "\n".join(lines) + "\n"
print()
print(text, end="")
pathlib.Path("BOARD.txt").write_text(text)
print("wrote BOARD.txt")
PY
