"""How much of a person's pressure level is recoverable from many unlabelled windows of that person?

The error splits into a level and a movement. The movement is being tracked already, at a within
subject correlation of 0.63. The level is the four and a half millimetres that separate the
population error from the error a model makes when handed each subject's own mean, and nothing
tried so far recovers it: fifteen hand-built band descriptors gave a correlation of 0.08 against
subject mean pressure on unseen people.

That was fifteen numbers built by hand, though, and it is not evidence about what the trained
encoder holds. The encoder is a hundred and ninety-two dimensions wide and was optimised against
pressure, so the honest version of the question is this one:

    average a trained model's own representation over M windows of a person it has never seen, and
    fit one shared rule from that average to that person's mean pressure.  How close does it get?

The answer decides the architecture. If averaging many windows recovers the level, then what
belongs beside the raw branch is a memory of the person accumulated from their own unlabelled
stream, and the level error is an engineering problem. If it does not, the level is not in the
signal, no second branch can put it there, and only a cuff reading will do.

M is swept so the shape of the curve is visible: a level that emerges only with many windows is a
different finding from one that is present in a single window.

Both the subjects used to fit the rule and the subjects used to score it are test subjects, never
seen during the encoder's training, so the encoder cannot have memorised either group.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CONVERTED = ROOT / "converted"
CACHE = ROOT / "cache"
RUNS = ROOT / "runs" / "s4_pat_control"
REPORTS = ROOT / "reports"
from network_loader import load_q1_module  # noqa: E402

TEST_TAG = "calfree_test"
MODELS = ("RAW_w", "CWC3_w", "MAGONLY")
M_VALUES = (1, 5, 20, 100, 400)
N_FIT_SUBJECTS = 96
N_REPEATS = 20
MIN_WINDOWS = 400
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
SEED = 42


@torch.no_grad()
def features(tag, device, q1, batch=512):
    """The representation feeding each model's final linear layer, for every test window."""
    ck = torch.load(RUNS / f"{tag}_weights.pt", map_location="cpu", weights_only=False)
    arm = ck["arm"]
    model = (q1.GatedRawCWCFusionNet(dropout=0.5) if arm != "RAW"
             else q1.RawResNet1D(dropout=0.5))
    model.load_state_dict(ck["state_dict"])
    model = model.to(device).eval()
    mean, std = ck["signal_mean"], ck["signal_std"]
    X = np.load(CONVERTED / TEST_TAG / "X.npy", mmap_mode="r")
    IM = np.load(CACHE / TEST_TAG / "images.npy", mmap_mode="r") if arm != "RAW" else None
    out = []
    for s in range(0, X.shape[0], batch):
        idx = np.arange(s, min(s + batch, X.shape[0]))
        x = (np.asarray(X[idx], dtype=np.float32) - mean[None, :, None]) / std[None, :, None]
        xb = torch.from_numpy(x).to(device)
        if arm == "RAW":
            e = model.encoder(xb)
            h = torch.cat([e.mean(-1), e.amax(-1)], dim=1)
        else:
            img = np.asarray(IM[idx, :3], dtype=np.float32)
            img[:, 0] /= 255.0
            img[:, 1:] = img[:, 1:] / 127.5 - 1.0
            if arm == "MAGONLY":
                img[:, 1:] = 0.0
            ib = torch.from_numpy(img).to(device)
            e = model.raw_encoder(xb)
            h_raw = model.raw_proj(torch.cat([e.mean(-1), e.amax(-1)], dim=1))
            h_cwc = model.cwc_proj(model.cwc_encoder(ib).mean(dim=(2, 3)))
            g = model.gate(torch.cat([h_raw, h_cwc], dim=1))
            h = g * h_raw + (1.0 - g) * h_cwc
        for layer in model.head[:-1]:
            h = layer(h)
        out.append(h.float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float64)


def ridge(X, y, a):
    mu, ym = X.mean(0), y.mean()
    Xc = X - mu
    return mu, np.linalg.solve(Xc.T @ Xc + a * np.eye(X.shape[1]), Xc.T @ (y - ym)), ym


def pick_alpha(X, y, alphas, folds=4, seed=0):
    fold = np.random.default_rng(seed).permutation(len(X)) % folds
    best, best_a = np.inf, alphas[0]
    for a in alphas:
        e = []
        for f in range(folds):
            tr, te = fold != f, fold == f
            mu, w, b = ridge(X[tr], y[tr], a)
            e.append(np.abs((X[te] - mu) @ w + b - y[te]).mean())
        if np.mean(e) < best:
            best, best_a = float(np.mean(e)), a
    return best_a


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q1 = load_q1_module()
    Y = np.asarray(np.load(CONVERTED / TEST_TAG / "Y.npy"), dtype=np.float64)
    S = np.load(CONVERTED / TEST_TAG / "S.npy")
    subs = np.array([s for s in np.unique(S) if (S == s).sum() >= MIN_WINDOWS])
    rows_by = {int(s): np.flatnonzero(S == s) for s in subs}
    print(f"{len(subs)} test subjects with at least {MIN_WINDOWS} windows; "
          f"the encoder saw none of them\n", flush=True)

    F = {}
    for tag in MODELS:
        F[tag] = features(tag, device, q1)
        print(f"  {tag}: {F[tag].shape}", flush=True)

    rng = np.random.default_rng(SEED)
    results = {}
    for j, target in enumerate(("SBP", "DBP")):
        ys = np.array([Y[rows_by[int(s)], j].mean() for s in subs])
        print("\n" + "=" * 88)
        print(f"--- {target} ---  predicting each unseen subject's MEAN pressure")
        print("%-12s %6s %10s %10s %10s %10s   %s"
              % ("model", "M", "MAE", "baseline", "gain", "pearson r", "95% CI on gain"))
        for tag in MODELS:
            for M in M_VALUES:
                maes, bases, rs = [], [], []
                for rep in range(N_REPEATS):
                    r2 = np.random.default_rng(1000 + rep)
                    # average M windows of each subject: the memory a device would accumulate
                    A = np.stack([F[tag][r2.choice(rows_by[int(s)], size=M, replace=False)].mean(0)
                                  for s in subs])
                    fit = r2.permutation(len(subs))[:N_FIT_SUBJECTS]
                    m = np.zeros(len(subs), dtype=bool)
                    m[fit] = True
                    mu, sd = A[m].mean(0), A[m].std(0)
                    sd[sd < 1e-9] = 1.0
                    Z = (A - mu) / sd
                    a = pick_alpha(Z[m], ys[m], ALPHAS, seed=rep)
                    cmu, w, b = ridge(Z[m], ys[m], a)
                    pred = (Z[~m] - cmu) @ w + b
                    maes.append(np.abs(pred - ys[~m]).mean())
                    bases.append(np.abs(ys[~m] - ys[m].mean()).mean())
                    rs.append(float(np.corrcoef(pred, ys[~m])[0, 1]))
                maes, bases = np.asarray(maes), np.asarray(bases)
                g = bases - maes
                lo, hi = float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))
                print("%-12s %6d %10.4f %10.4f %10.4f %10.4f   [%+.4f,%+.4f]%s"
                      % (tag, M, maes.mean(), bases.mean(), g.mean(), float(np.mean(rs)),
                         lo, hi, " *" if lo * hi > 0 else ""))
                results.setdefault(target, {}).setdefault(tag, {})[M] = {
                    "mae": float(maes.mean()), "baseline": float(bases.mean()),
                    "gain": float(g.mean()), "pearson_r": float(np.mean(rs)),
                    "ci": [lo, hi], "significant": bool(lo * hi > 0)}

    print("\n* interval over repeats excludes zero")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "s29_level_information.json").write_text(json.dumps(
        {"analysis_id": "s29_level_information", "n_subjects": int(len(subs)),
         "fit_subjects": N_FIT_SUBJECTS, "repeats": N_REPEATS, "m_values": list(M_VALUES),
         "results": results}, indent=2), encoding="utf-8")
    print("saved reports/s29_level_information.json")


if __name__ == "__main__":
    main()
