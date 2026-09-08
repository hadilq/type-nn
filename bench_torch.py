#!/usr/bin/env python3
"""CPU PyTorch baselines for the Vortex bench suite.

Two implementations per task:

  torch-mlp   – the usual production style: Linear + ReLU + Adam
  torch-poly  – Linear on polynomial features (degree 2).  This is the
                closest *functional* cousin of Vortex (rank-2 polynomial).

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
         params: int, mse: float) -> None:
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
        "nbytes": params * 4,  # fp32 weights only; excludes allocator slop
        "mse": float(mse),
        "depth": None,
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("task", nargs="?", default="all",
                   choices=["all", "xor", "quadratic", "mlp32x16x8"])
    p.add_argument("--kind", default="both", choices=["mlp", "poly", "both"])
    args = p.parse_args()
    kinds = ["mlp", "poly"] if args.kind == "both" else [args.kind]
    tasks = {
        "xor": bench_xor,
        "quadratic": bench_quadratic,
        "mlp32x16x8": bench_mlp_scale,
    }
    run = tasks if args.task == "all" else {args.task: tasks[args.task]}
    for kind in kinds:
        for fn in run.values():
            fn(kind)


if __name__ == "__main__":
    main()
