#!/usr/bin/env bash
# Build the harness, run every model on every task, write BOARD.txt.
#   ./bench.sh [task|all]        TNN_SEEDS=5 by default
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
TASK="${1:-all}"
if [ -z "${TYPE_NN_DATA:-}" ] && [ -f "$ROOT/data/iris.data" ]; then
  export TYPE_NN_DATA="$ROOT/data"
fi
make -s bench
./bench "$TASK" all | tee bench.jsonl

python3 - <<'PY'
import json, math, collections, pathlib
rows = [json.loads(l) for l in open("bench.jsonl") if l.startswith("{")]
by = collections.defaultdict(dict)
for r in rows:
    by[r["task"]][r["impl"]] = r
S = rows[0]["seeds"] if rows else 0

def f(v, w=7, p=4):
    return f"{'n/a':>{w}}" if v is None else f"{v:{w}.{p}f}"

out = []
out += ["type-nn, type-nn-overfit vs c-mlp — hold-out board", "=" * 50, "",
        "Split      one fixed 70/30 cut (xorshift32 Fisher-Yates, seed 34972);",
        "           standardisation fitted on the training rows only.",
        f"Seeds      {S} initialisation seeds per cell; every number is the",
        "           mean over seeds, ± is the standard deviation.",
        "Protocol   one harness (bench.c) for both models: same shuffle, same",
        "           per-sample Adam (common.h), same mean-MSE, same epochs/lr,",
        "           same readout F(u) = sign(u) ln(1+|u|).",
        "params     every learnable scalar present after training (dense;",
        "           no weight is skipped for being small).",
        "L0 -> L    layers at birth -> after training. type-nn births",
        "           round(ln(1 + n m)) layers.",
        "or/and/L   type-nn probes promoted (+) and live items pruned (-) on the",
        "           Or-width, And-degree and depth axes.",
        "models     type-nn          evidence-driven scaling (BIC on measured MSE)",
        "           type-nn-overfit  the previous type-nn, frozen",
        "           c-mlp            Linear-ReLU-Linear baseline",
        "verdict    each type-nn against c-mlp: hold_mse difference, clear if it",
        "           exceeds two standard errors, otherwise within noise; and the",
        "           params ratio.",
        "xor        fit only: 4 points have no hold-out.", ""]
hdr = (f"{'task':<11}{'impl':<16}{'hold_mse':>9}{'±':>8}{'hold_acc':>9}{'±':>7}"
       f"{'train_mse':>10}{'params':>8}{'±':>6}{'train_s':>8}{'us/inf':>8}"
       f"{'L0->L':>9}{'or+':>6}{'or-':>5}{'and+':>6}{'and-':>5}{'L+':>5}{'L-':>5}")
out += [hdr, "-" * len(hdr)]
IMPLS = ("type-nn", "type-nn-overfit", "c-mlp")
for task in ("xor", "iris", "wine", "wdbc", "diabetes", "ionosphere"):
    cell = by.get(task, {})
    for impl in IMPLS:
        r = cell.get(impl)
        if not r: continue
        out.append(
            f"{task:<11}{impl:<16}{f(r['hold_mse'],9)}{f(r['hold_mse_sd'],8)}"
            f"{f(r['hold_acc'],9,3)}{f(r['hold_acc_sd'],7,3)}{r['mse']:10.5f}"
            f"{r['params']:8.1f}{r['params_sd']:6.1f}{r['train_s']:8.3f}"
            f"{r['us_per_infer']:8.3f}{r['init_layers']:5.0f}->{r['layers']:<3.1f}"
            f"{r['or_add']:6.1f}{r['or_drop']:5.1f}{r['and_add']:6.1f}"
            f"{r['and_drop']:5.1f}{r['layer_add']:5.1f}{r['layer_drop']:5.1f}")
    c = cell.get("c-mlp")
    for name in ("type-nn", "type-nn-overfit"):
        t = cell.get(name)
        if not (t and c):
            continue
        pr = t["params"] / c["params"]
        pw = "fewer" if pr < 1 else "more"
        if t["hold_mse"] is None:
            out.append(f"{'':<11}{name}: fit only; params {pr:.2f}x c-mlp ({pw})")
            continue
        d = t["hold_mse"] - c["hold_mse"]
        se = math.sqrt((t["hold_mse_sd"]**2 + c["hold_mse_sd"]**2) / max(S, 1))
        word = "clear" if abs(d) > 2 * se else "within noise"
        who = name if d < 0 else "c-mlp"
        out.append(f"{'':<11}{name}: {who} lower hold_mse by {abs(d):.4f} (2se {2*se:.4f}) -> {word};"
                   f" params {pr:.2f}x c-mlp ({pw})")
    out.append("")
text = "\n".join(out) + "\n"
pathlib.Path("BOARD.txt").write_text(text)
print(); print(text, end="")
PY
