#!/usr/bin/env python3
"""CPU PyTorch baselines for the type-nn bench suite.

Two implementations per task:

  torch-mlp   – the usual production style: Linear + ReLU + Adam
  torch-poly  – Linear on polynomial features (degree 2).  This is the
                closest *functional* cousin of type-nn (rank-2 polynomial).

Everything runs on CPU with one thread so the comparison is against a
strong library, not against a GPU kernel launcher.
"""
from __future__ import annotations

import argparse
import json
import resource
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def emit(impl: str, task: str, train_s: float, infer_s: float, infer_n: int,
         params: int, mse: float, n: int = 0, acc: float = -1.0,
         hold_mse: float = None, hold_acc: float = None) -> None:
    us = (infer_s * 1e6 / infer_n) if infer_n else 0.0
    print(json.dumps({
        "impl": impl,
        "task": task,
        "train_s": round(train_s, 6),
        "infer_s": round(infer_s, 6),
        "infer_n": infer_n,
        "us_per_infer": round(us, 3),
        "rss_kb": rss_kb(),
        "hwm_kb": rss_kb(),
        "params": params,
        "nbytes": params * 4,
        "mse": float(mse),
        "depth": None,
        "n": n,
        "acc": acc,
        "dyn_scale": 0.0,
        "dyn_depth": 0,
        "dyn_params": 0,
        "hold_mse": float(hold_mse) if hold_mse is not None else float(mse),
        "hold_acc": float(hold_acc) if hold_acc is not None else float(acc),
    }))


def nparams(mod: nn.Module) -> int:
    return sum(p.numel() for p in mod.parameters())


class MLP(nn.Module):
    def __init__(self, widths: list[int]):
        super().__init__()
        layers = []
        for a, b in zip(widths, widths[1:-1]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers.append(nn.Linear(widths[-2], widths[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def poly_features(x: torch.Tensor) -> torch.Tensor:
    """[1, x_i, x_i x_j for i<=j] — affine + rank-2 monomials."""
    n, d = x.shape
    cols = [torch.ones(n, 1, dtype=x.dtype), x]
    feats = [x[:, i:i+1] * x[:, j:j+1] for i in range(d) for j in range(i, d)]
    if feats:
        cols.append(torch.cat(feats, dim=1))
    return torch.cat(cols, dim=1)


class PolyLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # 1 + d + d(d+1)/2
        feat = 1 + in_dim + in_dim * (in_dim + 1) // 2
        self.lin = nn.Linear(feat, out_dim)

    def forward(self, x):
        return self.lin(poly_features(x))


def fit_sgd(mod: nn.Module, X: torch.Tensor, Y: torch.Tensor,
            epochs: int, lr: float, batch: int | None = None) -> float:
    opt = torch.optim.Adam(mod.parameters(), lr=lr)
    n = X.shape[0]
    t0 = time.perf_counter()
    mod.train()
    for _ in range(epochs):
        if batch is None or batch >= n:
            pred = mod(X)
            loss = F.mse_loss(pred, Y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        else:
            perm = torch.randperm(n)
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                pred = mod(X[idx])
                loss = F.mse_loss(pred, Y[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
    train_s = time.perf_counter() - t0
    return train_s


@torch.no_grad()
def infer_loop(mod: nn.Module, X: torch.Tensor, reps: int) -> float:
    mod.eval()
    n = X.shape[0]
    t0 = time.perf_counter()
    for i in range(reps):
        _ = mod(X[i % n: i % n + 1])
    return time.perf_counter() - t0


@torch.no_grad()
def final_mse(mod: nn.Module, X: torch.Tensor, Y: torch.Tensor) -> float:
    mod.eval()
    return float(F.mse_loss(mod(X), Y).item())


def bench_xor(kind: str) -> None:
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    if kind == "mlp":
        mod = MLP([2, 8, 1])
        lr, epochs = 0.05, 400
        impl = "torch-mlp"
    else:
        mod = PolyLinear(2, 1)
        lr, epochs = 0.1, 400
        impl = "torch-poly"
    train_s = fit_sgd(mod, X, Y, epochs, lr)
    reps = 20000
    infer_s = infer_loop(mod, X, reps)
    emit(impl, "xor", train_s, infer_s, reps, nparams(mod), final_mse(mod, X, Y))


def bench_quadratic(kind: str) -> None:
    g = torch.Generator().manual_seed(7)
    N = 64
    X = torch.rand(N, 2, generator=g) * 2 - 1
    Y = (X[:, 0:1] * X[:, 1:2] + 0.25 * X[:, 0:1])
    if kind == "mlp":
        mod = MLP([2, 16, 1])
        impl = "torch-mlp"
        lr = 0.02
    else:
        mod = PolyLinear(2, 1)
        impl = "torch-poly"
        lr = 0.05
    train_s = fit_sgd(mod, X, Y, 200, lr)
    reps = 5000
    infer_s = infer_loop(mod, X, reps)
    emit(impl, "quadratic", train_s, infer_s, reps, nparams(mod), final_mse(mod, X, Y))


def bench_mlp_scale(kind: str) -> None:
    g = torch.Generator().manual_seed(11)
    N, IN, HID, OUT = 256, 32, 16, 8
    X = torch.rand(N, IN, generator=g) * 2 - 1
    Y = (torch.rand(N, OUT, generator=g) * 2 - 1) * 0.3
    if kind == "mlp":
        mod = MLP([IN, HID, OUT])
        impl = "torch-mlp"
        lr = 0.01
    else:
        mod = PolyLinear(IN, OUT)
        impl = "torch-poly"
        lr = 0.01
    train_s = fit_sgd(mod, X, Y, 30, lr, batch=64)
    reps = 1000
    infer_s = infer_loop(mod, X, reps)
    emit(impl, "mlp32x16x8", train_s, infer_s, reps, nparams(mod), final_mse(mod, X, Y))



import os

def find_data(name: str) -> str | None:
    dirs = []
    env = os.environ.get("TYPE_NN_DATA")
    if env:
        dirs.append(env)
    dirs += ["data", "./data", "/tmp/type-nn-data", "/usr/share/type-nn"]
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def standardize(X: torch.Tensor) -> torch.Tensor:
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True).clamp_min(1e-8)
    return (X - mean) / std


def minmax(Y: torch.Tensor) -> torch.Tensor:
    lo = Y.min(dim=0, keepdim=True).values
    hi = Y.max(dim=0, keepdim=True).values
    return (Y - lo) / (hi - lo).clamp_min(1e-8)


def accuracy(mod: nn.Module, X: torch.Tensor, Y: torch.Tensor) -> float:
    mod.eval()
    with torch.no_grad():
        pred = mod(X)
        if Y.shape[1] == 1:
            return float(((pred >= 0.5) == (Y >= 0.5)).float().mean())
        return float((pred.argmax(1) == Y.argmax(1)).float().mean())


def load_iris(path: str):
    xs, ys = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            xs.append([float(v) for v in parts[:4]])
            lab = parts[4]
            k = 1 if "versicolor" in lab else 2 if "virginica" in lab else 0
            ys.append([1.0 if i == k else 0.0 for i in range(3)])
    X = standardize(torch.tensor(xs, dtype=torch.float32))
    Y = torch.tensor(ys, dtype=torch.float32)
    return X, Y


def load_wine(path: str):
    xs, ys = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            cls = int(float(parts[0])) - 1
            xs.append([float(v) for v in parts[1:14]])
            ys.append([1.0 if i == cls else 0.0 for i in range(3)])
    return standardize(torch.tensor(xs)), torch.tensor(ys)


def load_wdbc(path: str):
    xs, ys = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            lab = 1.0 if parts[1] in ("M", "m") else 0.0
            xs.append([float(v) for v in parts[2:32]])
            ys.append([lab])
    return standardize(torch.tensor(xs)), torch.tensor(ys)


def load_diabetes(path: str):
    xs, ys = [], []
    with open(path) as f:
        header = f.readline()
        _ = header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 11:
                continue
            vals = [float(v) for v in parts[:11]]
            xs.append(vals[:10])
            ys.append([vals[10]])
    X = standardize(torch.tensor(xs))
    Y = minmax(torch.tensor(ys))
    return X, Y


def load_ionosphere(path: str):
    xs, ys = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            xs.append([float(v) for v in parts[:34]])
            ys.append([1.0 if parts[-1].lower().startswith("g") else 0.0])
    return standardize(torch.tensor(xs)), torch.tensor(ys)


def xorshift32(state: int) -> int:
    x = state & 0xFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= (x >> 17) & 0xFFFFFFFF
    x ^= (x << 5) & 0xFFFFFFFF
    return x & 0xFFFFFFFF


def split_perm(n: int, seed: int = 34972) -> list[int]:
    """Same Fisher–Yates + xorshift32 as dataset_perm() in dataset.c."""
    perm = list(range(n))
    s = seed & 0xFFFFFFFF
    if s == 0:
        s = 1
    for i in range(n, 1, -1):
        s = xorshift32(s)
        j = s % i
        perm[i - 1], perm[j] = perm[j], perm[i - 1]
    return perm


def bench_real(kind: str, task: str, filename: str, loader, widths, epochs, lr, reps):
    path = find_data(filename)
    if path is None:
        print(f"skip {task}: {filename} not found", flush=True)
        return
    X, Y = loader(path)
    in_dim, out_dim = X.shape[1], Y.shape[1]
    n = X.shape[0]
    perm = split_perm(n, 34972)
    ntr = max(1, (n * 7) // 10)
    tr = torch.tensor(perm[:ntr], dtype=torch.long)
    te = torch.tensor(perm[ntr:], dtype=torch.long)
    Xtr, Ytr = X[tr], Y[tr]
    Xte, Yte = X[te], Y[te]
    torch.manual_seed(34972)
    if kind == "mlp":
        hid = widths[0]
        mod = MLP([in_dim, hid, out_dim])
        impl = "torch-mlp"
    else:
        mod = PolyLinear(in_dim, out_dim)
        impl = "torch-poly"
    train_s = fit_sgd(mod, Xtr, Ytr, epochs, lr)
    infer_s = infer_loop(mod, X, reps)
    acc = accuracy(mod, Xtr, Ytr) if task != "diabetes" else -1.0
    hold_acc = accuracy(mod, Xte, Yte) if task != "diabetes" and len(te) else acc
    emit(impl, task, train_s, infer_s, reps, nparams(mod), final_mse(mod, Xtr, Ytr),
         n=n, acc=acc, hold_mse=final_mse(mod, Xte, Yte) if len(te) else None,
         hold_acc=hold_acc)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("task", nargs="?", default="all",
                   choices=["all", "xor", "quadratic", "mlp32x16x8",
                            "iris", "wine", "wdbc", "diabetes", "ionosphere", "real"])
    p.add_argument("--kind", default="both", choices=["mlp", "poly", "both"])
    args = p.parse_args()
    kinds = ["mlp", "poly"] if args.kind == "both" else [args.kind]
    synth = {
        "xor": bench_xor,
        "quadratic": bench_quadratic,
        "mlp32x16x8": bench_mlp_scale,
    }
    real = [
        ("iris", "iris.data", load_iris, [8], 250, 0.05, 2000),
        ("wine", "wine.data", load_wine, [16], 200, 0.03, 2000),
        ("wdbc", "wdbc.data", load_wdbc, [16], 80, 0.02, 1000),
        ("diabetes", "diabetes.tab.txt", load_diabetes, [16], 150, 0.02, 2000),
        ("ionosphere", "ionosphere.data", load_ionosphere, [16], 120, 0.02, 1000),
    ]
    want_synth = args.task in ("all", "xor", "quadratic", "mlp32x16x8")
    want_real = args.task in ("all", "real", "iris", "wine", "wdbc", "diabetes", "ionosphere")
    for kind in kinds:
        if want_synth:
            run = synth if args.task == "all" else {args.task: synth[args.task]} if args.task in synth else {}
            for fn in run.values():
                fn(kind)
        if want_real:
            for task, fnm, loader, widths, epochs, lr, reps in real:
                if args.task in ("all", "real", task):
                    bench_real(kind, task, fnm, loader, widths, epochs, lr, reps)


if __name__ == "__main__":
    main()
