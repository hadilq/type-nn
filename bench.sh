#!/usr/bin/env bash
# Build the harness, run every model on every task, write BOARD.md.
#   ./bench.sh [task|all]          TNN_SEEDS=5, TNN_SPLITS=1 by default
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
P = rows[0].get("splits", 1) if rows else 0
TASKS = ("xor", "iris", "wine", "wdbc", "diabetes", "ionosphere", "spambase",
         "synth", "mnist", "mnist_large", "video", "video_large")
IMPLS = ("type-nn", "type-nn-overfit", "graph-type-nn", "c-mlp", "c-mlp-matched")

def f(v, w=7, p=4):
    return f"{'n/a':>{w}}" if v is None else f"{v:{w}.{p}f}"

def compare(a, b):
    """(difference a-b, 2 standard errors, label). Paired over splits when
    there are several; otherwise unpaired over the initialisation seeds."""
    if a["hold_mse"] is None or b["hold_mse"] is None:
        return None
    d = a["hold_mse"] - b["hold_mse"]
    sa, sb = a.get("hold_mse_splits", []), b.get("hold_mse_splits", [])
    if len(sa) > 1 and len(sa) == len(sb):
        ds = [x - y for x, y in zip(sa, sb)]
        m = sum(ds) / len(ds)
        v = sum((x - m) ** 2 for x in ds) / (len(ds) - 1)
        se, how = math.sqrt(v / len(ds)), f"paired over {len(ds)} splits"
    else:
        n = max(a["seeds"], 1)
        se, how = math.sqrt((a["hold_mse_sd"] ** 2 + b["hold_mse_sd"] ** 2) / n), f"over {n} seeds"
    return d, 2 * se, how

def verdict(name_a, a, name_b, b):
    pr = a["params"] / b["params"] if b["params"] else float("nan")
    pw = "fewer" if pr < 1 else "more"
    c = compare(a, b)
    if c is None:
        return f"{name_a} vs {name_b}: fit only; params {pr:.2f}x ({pw})", None, pr
    d, two_se, how = c
    word = "clear" if abs(d) > two_se else "within noise"
    who = name_a if d < 0 else name_b
    return (f"{name_a} vs {name_b}: {who} lower hold_mse by {abs(d):.4f} "
            f"(2se {two_se:.4f}, {how}) -> {word}; params {pr:.2f}x ({pw})"), (d, two_se), pr

out = []
out += ["type-nn board: graph-type-nn, type-nn, type-nn-overfit, c-mlp", "=" * 62, "",
        f"Splits     {P} x 70/30 split(s) (xorshift32 Fisher-Yates; split 0 is seed 34972);",
        "           standardisation fitted on the training rows of each split only.",
        f"Seeds      {S} initialisation seeds per split; every number is the mean over",
        "           all (split, seed) runs, ± is their standard deviation.",
        "Protocol   one harness (bench.c) for every model: same shuffle, same",
        "           per-sample Adam (common.h), same mean-MSE, same epochs/lr,",
        "           same readout F(u) = sign(u) ln(1+|u|). The hold-out is read",
        "           only after training ends.",
        "params     every learnable scalar present after training (dense; no weight",
        "           is skipped for being small). Edge weights of the graph are set",
        "           by feedback, not by gradient, and are not counted.",
        "L0->L      layers (chain) or nodes (graph) at birth -> after training.",
        "E0->E      edges at birth -> after training, x -> first layer included.",
        "or/and     Or-width and And-degree dummies promoted (+) and live items",
        "           pruned (-); L+/L- layers or nodes, E+/E- graph edges.",
        "models     graph-type-nn    type-nn on a DAG, slime-mould edge weights",
        "           type-nn          evidence-driven scaling (BIC on measured MSE)",
        "           type-nn-overfit  displacement-only scaling (frozen reference)",
        "           c-mlp            Linear-ReLU-Linear, fixed hidden width 8 / 16",
        "           c-mlp-matched    the widest c-mlp with params <= graph-type-nn's",
        "verdict    hold_mse difference, 'clear' if it exceeds two standard errors",
        "           (paired over splits when there are several), and the params ratio.",
        "goal       graph-type-nn: hold_mse not clearly worse than type-nn, with",
        "           fewer parameters.",
        "xor        fit only: 4 points have no hold-out.", ""]
hdr = (f"{'task':<12}{'impl':<16}{'hold_mse':>9}{'±':>8}{'hold_acc':>9}{'±':>7}"
       f"{'train_mse':>10}{'params':>8}{'±':>7}{'train_s':>8}{'us/inf':>8}"
       f"{'L0->L':>10}{'E0->E':>10}{'or+':>6}{'or-':>5}{'and+':>6}{'and-':>5}"
       f"{'L+':>5}{'L-':>5}{'E+':>5}{'E-':>5}")
out += [hdr, "-" * len(hdr)]
goal_met, goal_n, missing = [], 0, []
for task in TASKS:
    cell = by.get(task, {})
    if not cell:
        missing.append(task)
        continue
    for impl in IMPLS:
        r = cell.get(impl)
        if not r: continue
        g = impl == "graph-type-nn"
        e = (lambda k: f"{r.get(k, 0):5.1f}") if g else (lambda k: f"{'':>5}")
        struct = (f"{r['init_layers']:5.0f}->{r['layers']:<3.1f}"
                  f"{r.get('init_edges', 0):5.0f}->{r.get('edges', 0):<3.1f}"
                  f"{r['or_add']:6.1f}{r['or_drop']:5.1f}{r['and_add']:6.1f}"
                  f"{r['and_drop']:5.1f}{r['layer_add']:5.1f}{r['layer_drop']:5.1f}"
                  f"{e('edge_add')}{e('edge_drop')}")
        out.append(
            f"{task:<12}{impl:<16}{f(r['hold_mse'],9)}{f(r['hold_mse_sd'],8)}"
            f"{f(r['hold_acc'],9,3)}{f(r['hold_acc_sd'],7,3)}{f(r['mse'],10,5)}"
            f"{r['params']:8.1f}{r['params_sd']:7.1f}{r['train_s']:8.3f}"
            f"{f(r['us_per_infer'],8,3)}{struct}")
    pairs = (("graph-type-nn", "type-nn"), ("graph-type-nn", "c-mlp"),
             ("graph-type-nn", "c-mlp-matched"), ("type-nn", "c-mlp"))
    for a, b in pairs:
        if a in cell and b in cell:
            line, c, pr = verdict(a, cell[a], b, cell[b])
            out.append(f"{'':<12}{line}")
            if (a, b) == ("graph-type-nn", "type-nn"):
                goal_n += 1
                not_worse = c is None or not (c[0] > c[1])
                if not_worse and pr < 1:
                    goal_met.append(task)
    out.append("")
if goal_n:
    out.append(f"goal met on {len(goal_met)} / {goal_n} tasks: {', '.join(goal_met) or '-'}")
if missing:
    out.append(f"not run (data not found: use nix develop, or not selected): {', '.join(missing)}")
text = "\n".join(out) + "\n"
pathlib.Path("BOARD.md").write_text(text)
print(); print(text, end="")
PY
