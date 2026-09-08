#!/usr/bin/env python3
"""Label entropy, a crude H(Y|X), and train-vs-holdout MSE for torch-mlp."""
import math, os, random, sys
from collections import Counter

DATA = os.environ.get("TYPE_NN_DATA") or os.path.join(
    os.path.dirname(__file__), "..", "data")

def entropy(labels):
    n = len(labels)
    h = 0.0
    for c in Counter(labels).values():
        p = c / n
        if p > 0:
            h -= p * math.log2(p)
    return h

def load_iris(path):
    X, y = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            X.append([float(v) for v in p[:4]])
            y.append(p[4])
    return X, y, "cls"

def load_wine(path):
    X, y = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            y.append(p[0])
            X.append([float(v) for v in p[1:]])
    return X, y, "cls"

def load_wdbc(path):
    X, y = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            y.append(p[1])
            X.append([float(v) for v in p[2:]])
    return X, y, "cls"

def load_diabetes(path):
    X, y = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == "#" or "AGE" in line.upper():
                if line and line[0] != "#" and any(ch.isalpha() for ch in line):
                    continue
                if not line or line[0] == "#":
                    continue
            p = line.replace(",", " ").split()
            if len(p) < 11:
                continue
            try:
                nums = [float(v) for v in p]
            except ValueError:
                continue
            X.append(nums[:-1])
            y.append(nums[-1])
    return X, y, "reg"

def zscore(X):
    d = len(X[0])
    mu = [sum(r[j] for r in X) / len(X) for j in range(d)]
    sd = []
    for j in range(d):
        v = sum((r[j] - mu[j]) ** 2 for r in X) / len(X)
        sd.append(math.sqrt(v) if v > 1e-12 else 1.0)
    return [[(r[j] - mu[j]) / sd[j] for j in range(d)] for r in X]

def dist(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))

def knn1_loocv(X, y):
    n = len(X)
    err = 0
    for i in range(n):
        best, bj = None, None
        for j in range(n):
            if i == j:
                continue
            d = dist(X[i], X[j])
            if best is None or d < best:
                best, bj = d, j
        if y[bj] != y[i]:
            err += 1
    return err / n

def knn1_loocv_mse(X, y):
    n = len(X)
    acc = 0.0
    for i in range(n):
        best, bj = None, None
        for j in range(n):
            if i == j:
                continue
            d = dist(X[i], X[j])
            if best is None or d < best:
                best, bj = d, j
        acc += (y[bj] - y[i]) ** 2
    return acc / n

def hist_entropy(vals, bins=16):
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return 0.0
    w = (hi - lo) / bins
    cnt = [0] * bins
    for v in vals:
        k = int((v - lo) / w)
        if k >= bins:
            k = bins - 1
        cnt[k] += 1
    n = len(vals)
    h = 0.0
    for c in cnt:
        if c:
            p = c / n
            h -= p * math.log2(p)
    return h

def holdout_torch(X, Y, epochs, lr, seed=0):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        return None
    torch.manual_seed(seed)
    n, d = X.shape[0], X.shape[1]
    o = Y.shape[1]
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    cut = max(1, int(0.7 * n))
    tr, te = idx[:cut], idx[cut:]
    Xt = torch.tensor(X[tr], dtype=torch.float32)
    Yt = torch.tensor(Y[tr], dtype=torch.float32)
    Xe = torch.tensor(X[te], dtype=torch.float32)
    Ye = torch.tensor(Y[te], dtype=torch.float32)
    hid = 8 if d < 10 else 16
    mod = nn.Sequential(nn.Linear(d, hid), nn.ReLU(), nn.Linear(hid, o))
    opt = torch.optim.Adam(mod.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(mod(Xt), Yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        tr_mse = float(F.mse_loss(mod(Xt), Yt))
        te_mse = float(F.mse_loss(mod(Xe), Ye))
        if o > 1:
            tr_acc = float((mod(Xt).argmax(1) == Yt.argmax(1)).float().mean())
            te_acc = float((mod(Xe).argmax(1) == Ye.argmax(1)).float().mean())
        else:
            tr_acc = te_acc = -1.0
    return tr_mse, te_mse, tr_acc, te_acc, hid

def onehot(y):
    labs = sorted(set(y))
    m = {l: i for i, l in enumerate(labs)}
    Y = [[0.0] * len(labs) for _ in y]
    for i, yi in enumerate(y):
        Y[i][m[yi]] = 1.0
    return Y, labs

def main():
    loaders = {
        "iris": ("iris.data", load_iris, 400, 0.02),
        "wine": ("wine.data", load_wine, 400, 0.02),
        "wdbc": ("wdbc.data", load_wdbc, 200, 0.01),
        "diabetes": ("diabetes.tab.txt", load_diabetes, 300, 0.01),
    }
    print(f"{'task':<10} {'n':>4} {'in':>3} {'H(Y)':>6} {'1NN-err':>8} {'note':<28} "
          f"{'trMSE':>8} {'teMSE':>8} {'trAcc':>6} {'teAcc':>6}")
    print("-" * 108)
    for name, (fn, loader, ep, lr) in loaders.items():
        path = os.path.join(DATA, fn)
        if not os.path.isfile(path):
            print(name, "missing", path)
            continue
        X, y, kind = loader(path)
        Xz = zscore(X)
        if kind == "cls":
            hy = entropy(y)
            err = knn1_loocv(Xz, y)
            note = "low H(Y|X)" if err < 0.08 else ("mixed" if err < 0.2 else "harder")
            Y, labs = onehot(y)
            import array
            try:
                import numpy as np
                Xa = np.array(Xz, dtype=float)
                Ya = np.array(Y, dtype=float)
            except ImportError:
                print(name, "numpy missing")
                continue
            ht = holdout_torch(Xa, Ya, ep, lr)
        else:
            # scale y to ~[0,1] like the C bench
            lo, hi = min(y), max(y)
            ys = [(v - lo) / (hi - lo) if hi > lo else 0.0 for v in y]
            hy = hist_entropy(ys)
            err = knn1_loocv_mse(Xz, ys)
            note = f"reg 1NN-MSE={err:.3f}"
            try:
                import numpy as np
                Xa = np.array(Xz, dtype=float)
                Ya = np.array([[v] for v in ys], dtype=float)
            except ImportError:
                continue
            ht = holdout_torch(Xa, Ya, ep, lr)
        tr = te = ta = ea = float("nan")
        if ht:
            tr, te, ta, ea, hid = ht
        print(f"{name:<10} {len(X):4d} {len(X[0]):3d} {hy:6.3f} {err:8.3f} {note:<28} "
              f"{tr:8.4f} {te:8.4f} {ta:6.3f} {ea:6.3f}")
    print()
    print("H(Y) = label entropy in bits (classes) or histogram entropy (diabetes).")
    print("1NN-err = leave-one-out 1-NN error on z-scored X (proxy for H(Y|X)/Bayes).")
    print("tr/te = torch-mlp 70/30 split, same family as bench_torch (train-set scores")
    print("        in ./bench.sh are NOT held-out).")

if __name__ == "__main__":
    main()
