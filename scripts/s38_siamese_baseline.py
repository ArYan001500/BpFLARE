"""The two baselines a hostile reviewer will demand before believing the paired objective.

An independent audit named the prior art most likely to sink this: Schlesinger, Vigderhouse, Eytan
and Moshe, ICASSP 2020. They take one calibration window from a patient together with its known
pressure, pass it and the target window through a shared encoder, subtract the two feature vectors
with the sign preserved, and regress the pressure difference from the calibration value; adding the
calibration pressure back gives an absolute estimate. Patients are disjoint between train and test.

That is operationally the same idea as ours and it is five years old. The one thing it does not do
is factorise the comparison. Their network sees both windows at once and is free to use any joint
function of the pair. Ours never sees a pair at inference: it learns a single-window scalar whose
differences are supervised, so the pair only exists in the loss. Whether that factorisation is
worth anything is an empirical question and this script asks it, on our split, with our calibration
procedure, with the same encoder and the same number of gradient steps.

Two baselines are produced:

    CALONLY   no model at all. After k calibration readings, predict their mean for every
              remaining window of that subject. This is the floor that any calibrated method must
              clear, and the audit lists it as mandatory.
    SIAMESE   the Schlesinger construction: shared encoder, signed feature subtraction, a small
              comparator regressing the pressure difference, absolute value recovered by adding the
              calibration pressure and averaging over the k calibration windows.

Both are scored exactly as the paired model was: the same subjects, the same k, the same draws,
and calibration windows never appear among the scored windows.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
CONVERTED = ROOT / "converted"
REPORTS = ROOT / "reports"
RUNS = ROOT / "runs" / "s4_pat_control"
sys.path.insert(0, str(ROOT / "scripts"))
from network_loader import load_q1_module  # noqa: E402
from s35_within_subject_objective import (  # noqa: E402
    SUBJ_PER_BATCH, WIN_PER_SUBJ, TARGET_WEIGHTS, weighted_huber)

TEST_TAG = "calfree_test"
LR, WEIGHT_DECAY, DROPOUT, SEED = 1e-4, 1e-4, 0.5, 42
MAX_EPOCHS, CHECKS_PER_EPOCH, PATIENCE_CHECKS = 6, 4, 8
K_VALUES = (1, 3, 5, 10, 20)
N_DRAWS = 20                 # fewer than the 200 used elsewhere: the siamese needs k times more
MIN_WINDOWS = 100            # forward passes per draw, and 20 already stabilises the mean
CAL_SEED = 42


class Siamese(nn.Module):
    """Shared encoder, signed feature subtraction, comparator regressing the pressure difference."""

    def __init__(self, q1, dropout=DROPOUT):
        super().__init__()
        base = q1.RawResNet1D(dropout=dropout)
        self.encoder = base.encoder
        with torch.no_grad():
            dim = self.pool(self.encoder(torch.zeros(1, 2, 1250))).shape[1]
        self.comparator = nn.Sequential(nn.Linear(dim, 128), nn.SiLU(),
                                        nn.Dropout(dropout), nn.Linear(128, 3))

    @staticmethod
    def pool(e):
        return torch.cat([e.mean(-1), e.amax(-1)], dim=1)

    def features(self, x):
        return self.pool(self.encoder(x))

    def compare(self, f_target, f_cal):
        # signed, not absolute, so the direction of the pressure change survives
        return self.comparator(f_target - f_cal)

    def forward(self, x_target, x_cal):
        return self.compare(self.features(x_target), self.features(x_cal))


def calibrated_mae(pred_fn, Y, rows_by, subs, draws, k, j):
    """MAE per subject after k calibration windows fix the estimate, those windows excluded."""
    out = []
    for s in subs:
        rows = rows_by[int(s)]
        v = []
        for cal in draws[k][int(s)]:
            m = np.ones(len(rows), dtype=bool)
            m[np.searchsorted(rows, np.sort(cal))] = False
            ev = rows[m]
            v.append(np.abs(pred_fn(ev, cal, j) - Y[ev, j]).mean())
        out.append(float(np.mean(v)))
    return np.asarray(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q1 = load_q1_module()
    d, dt = CONVERTED / "train", CONVERTED / TEST_TAG
    X = np.load(d / "X.npy", mmap_mode="r")
    Y = np.asarray(np.load(d / "Y.npy"), dtype=np.float64)
    S = np.load(d / "S.npy")
    tr, va = np.load(d / "rows_train.npy"), np.load(d / "rows_validation.npy")
    Xt = np.load(dt / "X.npy", mmap_mode="r")
    Yt = np.asarray(np.load(dt / "Y.npy"), dtype=np.float64)
    St = np.load(dt / "S.npy")

    rows_of = {int(s): tr[S[tr] == s] for s in np.unique(S[tr])
               if (S[tr] == s).sum() >= WIN_PER_SUBJ}
    train_subjects = np.array(sorted(rows_of))
    sub = tr[:: max(1, len(tr) // 40000)]
    xs = np.asarray(X[sub], dtype=np.float64)
    mean = xs.mean(axis=(0, 2)).astype(np.float32)
    std = np.maximum(xs.std(axis=(0, 2)), 1e-6).astype(np.float32)

    def signals(Xa, idx):
        x = (np.asarray(Xa[idx], dtype=np.float32) - mean[None, :, None]) / std[None, :, None]
        return torch.from_numpy(x).to(device)

    subs = np.array([s for s in np.unique(St) if (St == s).sum() >= MIN_WINDOWS])
    rows_by = {int(s): np.flatnonzero(St == s) for s in subs}
    rng = np.random.default_rng(CAL_SEED)
    draws = {k: {int(s): [rng.choice(rows_by[int(s)], size=k, replace=False)
                          for _ in range(N_DRAWS)] for s in subs} for k in K_VALUES}
    boot = np.random.default_rng(7).integers(0, len(subs), size=(2000, len(subs)))
    print(f"{len(subs)} test subjects, {N_DRAWS} draws per subject per k\n", flush=True)

    # ---------------- the siamese ----------------
    ck_path = RUNS / "SIAMESE_weights.pt"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = Siamese(q1, DROPOUT).to(device)
    if ck_path.is_file() or args.skip_train:
        model.load_state_dict(torch.load(ck_path, map_location=device, weights_only=False)["state_dict"])
        print("loaded existing siamese weights\n", flush=True)
    else:
        steps = len(tr) // (SUBJ_PER_BATCH * WIN_PER_SUBJ)
        check_at = {int(round(steps * f / CHECKS_PER_EPOCH)) - 1
                    for f in range(1, CHECKS_PER_EPOCH + 1)}
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=MAX_EPOCHS * steps, eta_min=1e-6)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=4096.0)
        srng = np.random.default_rng(args.seed)
        best = {"score": float("inf"), "at": -1.0, "state": None}
        since, started, stop = 0, time.perf_counter(), False
        iu = torch.triu_indices(WIN_PER_SUBJ, WIN_PER_SUBJ, offset=1)
        print("===== SIAMESE (Schlesinger-style direct delta) =====", flush=True)

        @torch.no_grad()
        def val_delta():
            """Mean absolute error on the pressure difference between same-subject window pairs."""
            model.eval()
            errs = []
            vs = np.unique(S[va])[:120]
            for s in vs:
                r = va[S[va] == s]
                if len(r) < 8:
                    continue
                r = r[:32]
                f = model.features(signals(X, r))
                dt_ = torch.from_numpy((Y[r][:, None, :] - Y[r][None, :, :]).astype(np.float32)
                                       ).to(device)
                n = len(r)
                a, b = torch.triu_indices(n, n, offset=1)
                dp = model.compare(f[a], f[b])
                errs.append(float((dp - dt_[a, b]).abs()[:, 0].mean()))
            model.train()
            return float(np.mean(errs))

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            for s0 in range(steps):
                picked = srng.choice(train_subjects, size=SUBJ_PER_BATCH, replace=False)
                idx = np.concatenate([srng.choice(rows_of[int(p)], size=WIN_PER_SUBJ,
                                                  replace=False) for p in picked])
                xb = signals(X, idx)
                yb = torch.from_numpy(Y[idx].astype(np.float32)).to(device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    f = model.features(xb).view(SUBJ_PER_BATCH, WIN_PER_SUBJ, -1)
                    yv = yb.view(SUBJ_PER_BATCH, WIN_PER_SUBJ, -1)
                    fa = f[:, iu[0]].reshape(-1, f.shape[-1])
                    fb = f[:, iu[1]].reshape(-1, f.shape[-1])
                    ta = yv[:, iu[0]].reshape(-1, 3)
                    tb = yv[:, iu[1]].reshape(-1, 3)
                    loss = weighted_huber(model.compare(fa, fb), ta - tb, TARGET_WEIGHTS)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()

                if s0 in check_at:
                    at = epoch - 1 + (s0 + 1) / steps
                    v = val_delta()
                    flag = ""
                    if v < best["score"]:
                        best = {"score": v, "at": at,
                                "state": {k: t.detach().cpu().clone()
                                          for k, t in model.state_dict().items()}}
                        since, flag = 0, "  *"
                    else:
                        since += 1
                    print(f"  {at:5.2f}ep  delta-SBP MAE={v:.4f}{flag}", flush=True)
                    if since >= PATIENCE_CHECKS:
                        print(f"  stop at {at:.2f}, best {best['at']:.2f}", flush=True)
                        stop = True
                        break
            if stop:
                break
        model.load_state_dict(best["state"])
        torch.save({"state_dict": best["state"], "best_at": best["at"],
                    "signal_mean": mean, "signal_std": std}, ck_path)
        print(f"  siamese trained in {(time.perf_counter() - started) / 60:.1f} min\n", flush=True)

    model.eval()
    with torch.no_grad():
        F = torch.cat([model.features(signals(Xt, np.arange(s0, min(s0 + 512, len(Yt)))))
                       for s0 in range(0, len(Yt), 512)])
    print(f"siamese test features {tuple(F.shape)}\n", flush=True)

    @torch.no_grad()
    def siamese_pred(ev, cal, j):
        """Average over the k calibration windows of (predicted difference + calibration value)."""
        fe = F[ev]
        out = torch.zeros(len(ev), device=device)
        yc = torch.from_numpy(Yt[cal, j].astype(np.float32)).to(device)
        for i in range(len(cal)):
            d = model.compare(fe, F[cal[i]].unsqueeze(0).expand_as(fe))[:, j]
            out += d + yc[i]
        return (out / len(cal)).double().cpu().numpy()

    def calonly_pred(ev, cal, j):
        return np.full(len(ev), Yt[cal, j].mean())

    # our own arms, scored with exactly the same draws
    P = {t: np.asarray(np.load(RUNS / f"WS_{t}_test_pred.npy"), dtype=np.float64)
         for t in ("ABS", "PAIRED") if (RUNS / f"WS_{t}_test_pred.npy").is_file()}

    def offset_pred_factory(pred):
        def f(ev, cal, j):
            return pred[ev, j] + (Yt[cal, j] - pred[cal, j]).mean()
        return f

    methods = {"CALONLY": calonly_pred, "SIAMESE": siamese_pred}
    for t, p in P.items():
        methods[t] = offset_pred_factory(p)

    results, store = {}, {}
    for j, target in enumerate(("SBP", "DBP")):
        print("=" * 74)
        print(f"--- {target} ---  absolute MAE after k calibration windows")
        print("%-10s %9s %9s %9s %9s %9s" % ("method", *[f"k={k}" for k in K_VALUES]))
        for name, fn in methods.items():
            row = []
            for k in K_VALUES:
                e = calibrated_mae(fn, Y=Yt, rows_by=rows_by, subs=subs, draws=draws, k=k, j=j)
                store[(name, target, k)] = e
                row.append(e.mean())
                results.setdefault(target, {}).setdefault(name, {})[k] = float(e.mean())
            print("%-10s %9.4f %9.4f %9.4f %9.4f %9.4f" % (name, *row))
        print("\nPAIRED minus each other method, 95% subject bootstrap")
        for name in methods:
            if name == "PAIRED":
                continue
            cells = []
            for k in K_VALUES:
                dd = store[("PAIRED", target, k)][boot].mean(1) - store[(name, target, k)][boot].mean(1)
                lo, hi = float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))
                cells.append("%+.3f%s" % (store[("PAIRED", target, k)].mean()
                                          - store[(name, target, k)].mean(),
                                          "*" if lo * hi > 0 else " "))
                results[target].setdefault(f"PAIRED_vs_{name}", {})[k] = {
                    "delta": float(store[("PAIRED", target, k)].mean()
                                   - store[(name, target, k)].mean()),
                    "ci": [lo, hi], "significant": bool(lo * hi > 0)}
            print("%-10s %9s %9s %9s %9s %9s" % (name, *cells))
        print("* interval excludes zero; negative favours PAIRED\n")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "s38_siamese_baseline.json").write_text(json.dumps(
        {"analysis_id": "s38_siamese_baseline",
         "created_utc": datetime.now(timezone.utc).isoformat(),
         "n_subjects": int(len(subs)), "draws": N_DRAWS, "results": results}, indent=2),
        encoding="utf-8")
    print("saved reports/s38_siamese_baseline.json")


if __name__ == "__main__":
    main()
