"""Combine the level estimators we already have instead of looking for a better one.

Six attempts to extract more level information from the waveform have now failed, and the ceiling
measurements say the information is not there to extract. That closes one route and leaves another
untouched: several weak estimators already exist, they reach the level by different paths, and
nobody has put them side by side. Averaging estimators does not need new information. It needs their
errors to be less than perfectly correlated, which is a different and much weaker requirement.

Six components are available without training anything:

  ridge on the RAW_w encoder features          the estimator the paper reports
  ridge on the WS_PAIRED encoder features      a different encoder, so different errors
  the mean output of four absolute-loss models over the warm-up windows, which are themselves
  level estimates and have never been scored as such

The error correlation matrix is printed before anything else, because if those errors move together
the rest of the run cannot help and should be read as a negative result rather than tuned.

Combination weights are fitted on the validation subjects, never on the test subjects, using warm-up
sets drawn the same way they are drawn at scoring time. The tracking stage is also ensembled over
the four stored pairwise checkpoints, which is free for the same reason.

Scoring is the strict protocol of s64: the warm-up windows set the level and the offset, and they
are excluded from the windows that are scored.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CONVERTED = ROOT / "converted"
RUNS = ROOT / "runs" / "s4_pat_control"
REPORTS = ROOT / "reports"
sys.path.insert(0, str(ROOT / "scripts"))
from network_loader import load_q1_module  # noqa: E402
from s29_level_information import pick_alpha, ridge  # noqa: E402
from s37_backbones import BACKBONES  # noqa: E402

ABS_MODELS = ("RAW_w", "WS_ABS", "BB_INCEPTION_ABS", "BB_TCN_ABS")
PAIRED_MODELS = ("WS_PAIRED", "WS_PAIRED_s43", "BB_INCEPTION_PAIRED", "BB_TCN_PAIRED")
RIDGE_FEATURES = ("RAW_w", "WS_PAIRED")
M_EVAL, N_DRAWS, N_FIT_DRAWS = (1, 5, 10, 20, 60, 120), 50, 10
MIN_TEST_WINDOWS, MIN_FIT_WINDOWS = 100, 20
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 1e4, 1e5)
STACK_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
SEED = 42


@torch.no_grad()
def predict(tag, X, rows, device, q1, batch=512):
    ck = torch.load(RUNS / f"{tag}_weights.pt", map_location="cpu", weights_only=False)
    if "INCEPTION" in tag:
        model = BACKBONES["INCEPTION"](0.5)
    elif "TCN" in tag:
        model = BACKBONES["TCN"](0.5)
    else:
        model = q1.RawResNet1D(dropout=0.5)
    model.load_state_dict(ck["state_dict"])
    model = model.to(device).eval()
    mean, std = ck["signal_mean"], ck["signal_std"]
    out = np.empty((len(rows), 3), dtype=np.float32)
    for s in range(0, len(rows), batch):
        sel = rows[s: s + batch]
        x = (np.asarray(X[sel], dtype=np.float32) - mean[None, :, None]) / std[None, :, None]
        out[s: s + batch] = model(torch.from_numpy(x).to(device)).float().cpu().numpy()
    return out


def cached_predictions(device, q1, Xv, val_rows, Xt, n_test):
    path = RUNS / "s68_abs_predictions.npz"
    if path.is_file():
        z = np.load(path)
        return ({t: z[f"val_{t}"] for t in ABS_MODELS}, {t: z[f"test_{t}"] for t in ABS_MODELS})
    val, test = {}, {}
    for t in ABS_MODELS:
        print(f"  predicting with {t} ...", flush=True)
        val[t] = predict(t, Xv, val_rows, device, q1)
        test[t] = predict(t, Xt, np.arange(n_test), device, q1)
    np.savez(path, **{f"val_{t}": val[t] for t in ABS_MODELS},
             **{f"test_{t}": test[t] for t in ABS_MODELS})
    return val, test


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q1 = load_q1_module()

    Yt = np.asarray(np.load(CONVERTED / "calfree_test" / "Y.npy"), dtype=np.float64)
    St = np.load(CONVERTED / "calfree_test" / "S.npy")
    Xt = np.load(CONVERTED / "calfree_test" / "X.npy", mmap_mode="r")
    Yv = np.asarray(np.load(CONVERTED / "train" / "Y.npy"), dtype=np.float64)
    Sv = np.load(CONVERTED / "train" / "S.npy")
    Xv = np.load(CONVERTED / "train" / "X.npy", mmap_mode="r")
    val_rows = np.load(CONVERTED / "train" / "rows_validation.npy")

    subs = np.array([s for s in np.unique(St) if (St == s).sum() >= MIN_TEST_WINDOWS])
    rows_by = {int(s): np.flatnonzero(St == s) for s in subs}
    counts = {int(g): int(c) for g, c in zip(*np.unique(Sv[val_rows], return_counts=True))}
    fit_subs = np.array([g for g, c in counts.items() if c >= MIN_FIT_WINDOWS])
    vrows_by = {int(g): np.flatnonzero(Sv[val_rows] == g) for g in fit_subs}
    truth = np.stack([Yt[rows_by[int(s)], :2].mean(0) for s in subs])
    vtruth = np.stack([Yv[val_rows][vrows_by[int(g)], :2].mean(0) for g in fit_subs])

    abs_val, abs_test = cached_predictions(device, q1, Xv, val_rows, Xt, len(Yt))
    track = {t: np.asarray(np.load(RUNS / f"{t}_test_pred.npy"), dtype=np.float64)
             for t in PAIRED_MODELS}
    track["ENSEMBLE"] = np.mean([track[t] for t in PAIRED_MODELS], axis=0)

    # per-window scalars for the two ridge components, so any window set reduces to a mean
    scal_v, scal_t = {}, {}
    for tag in RIDGE_FEATURES:
        z = np.load(RUNS / f"{tag}_feat_cache.npz")
        fv, ft = z["val"].astype(np.float64), z["test"].astype(np.float64)
        G = np.stack([fv[vrows_by[int(g)]].mean(0) for g in fit_subs])
        mu, sd = G.mean(0), G.std(0)
        sd[sd < 1e-9] = 1.0
        for j in (0, 1):
            a = pick_alpha((G - mu) / sd, vtruth[:, j], ALPHAS, seed=SEED)
            cmu, w, b = ridge((G - mu) / sd, vtruth[:, j], a)
            scal_v[(tag, j)] = ((fv - mu) / sd - cmu) @ w + b
            scal_t[(tag, j)] = ((ft - mu) / sd - cmu) @ w + b

    names = [f"ridge({t})" for t in RIDGE_FEATURES] + [f"absmean({t})" for t in ABS_MODELS]

    def components(j, which, rows_map, keys, draws):
        """(draws, groups, components) level estimates from each warm-up set."""
        out = np.zeros((len(draws[int(keys[0])]), len(keys), len(names)))
        for i, g in enumerate(keys):
            for d, warm in enumerate(draws[int(g)]):
                col = [ (scal_v if which == "val" else scal_t)[(t, j)][warm].mean()
                        for t in RIDGE_FEATURES]
                col += [(abs_val if which == "val" else abs_test)[t][warm, j].mean()
                        for t in ABS_MODELS]
                out[d, i] = col
        return out

    results = {}
    for M in M_EVAL:
        tdraws = {int(s): [np.random.default_rng(SEED + 7 * d).choice(
            rows_by[int(s)], size=M, replace=False) for d in range(N_DRAWS)] for s in subs}
        vdraws = {int(g): [np.random.default_rng(SEED + 13 * d).choice(
            vrows_by[int(g)], size=min(M, len(vrows_by[int(g)])), replace=False)
            for d in range(N_FIT_DRAWS)] for g in fit_subs}

        print("\n" + "=" * 90)
        print(f"===== warm-up of {M} windows, {M * 10 / 60:.1f} minutes =====")
        for j, target in ((0, "SBP"), (1, "DBP")):
            CV = components(j, "val", vrows_by, list(fit_subs), vdraws)
            CT = components(j, "test", rows_by, list(subs), tdraws)

            print(f"\n--- {target} --- each estimator on its own, on the test subjects")
            print("  %-22s %8s %8s" % ("component", "r", "level MAE"))
            errs = []
            for c, n in enumerate(names):
                r = float(np.mean([np.corrcoef(CT[d, :, c], truth[:, j])[0, 1]
                                   for d in range(N_DRAWS)]))
                print("  %-22s %8.3f %8.3f"
                      % (n, r, float(np.abs(CT[:, :, c] - truth[None, :, j]).mean())))
                errs.append((CT[:, :, c] - truth[None, :, j]).mean(0))
            E = np.stack(errs)
            C = np.corrcoef(E)
            print("  error correlation between components: mean off-diagonal %.3f, "
                  "range %.3f to %.3f"
                  % (float((C.sum() - len(C)) / (len(C) ** 2 - len(C))),
                     float(C[~np.eye(len(C), dtype=bool)].min()),
                     float(C[~np.eye(len(C), dtype=bool)].max())))

            # combinations
            flat_x = CV.reshape(-1, len(names))
            flat_y = np.repeat(vtruth[None, :, j], CV.shape[0], axis=0).reshape(-1)
            mu2, sd2 = flat_x.mean(0), flat_x.std(0)
            sd2[sd2 < 1e-9] = 1.0
            a = pick_alpha((flat_x - mu2) / sd2, flat_y, STACK_ALPHAS, seed=SEED)
            cmu, w, b = ridge((flat_x - mu2) / sd2, flat_y, a)
            combos = {"best single (ridge RAW_w)": CT[:, :, 0],
                      "simple average of all six": CT.mean(2),
                      "stacked, weights fitted on validation": ((CT - mu2) / sd2 - cmu) @ w + b}

            print("  %-40s %9s | %-24s %9s" % ("level rule", "level MAE",
                                               "tracking model", "two-stage"))
            for cname, hat in combos.items():
                lmae = float(np.abs(hat - truth[None, :, j]).mean())
                for tname in ("WS_PAIRED", "ENSEMBLE"):
                    e = np.zeros(len(subs))
                    for i, s in enumerate(subs):
                        rows = rows_by[int(s)]
                        v = []
                        for d, warm in enumerate(tdraws[int(s)]):
                            m_ = np.ones(len(rows), dtype=bool)
                            m_[np.searchsorted(rows, np.sort(warm))] = False
                            ev = rows[m_]
                            off = hat[d, i] - track[tname][warm, j].mean()
                            v.append(np.abs(track[tname][ev, j] + off - Yt[ev, j]).mean())
                        e[i] = float(np.mean(v))
                    print("  %-40s %9.4f | %-24s %9.4f"
                          % (cname if tname == "WS_PAIRED" else "", lmae, tname, e.mean()),
                          flush=True)
                    results.setdefault(str(M), {}).setdefault(target, {})[
                        f"{cname} + {tname}"] = {"level_mae": lmae,
                                                 "two_stage_mae": float(e.mean())}

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "s68_level_ensemble.json").write_text(json.dumps(
        {"analysis_id": "s68_level_ensemble",
         "created_utc": datetime.now(timezone.utc).isoformat(),
         "components": names, "paired_models": list(PAIRED_MODELS),
         "draws": N_DRAWS, "results": results}, indent=2), encoding="utf-8")
    print("\nsaved reports/s68_level_ensemble.json")


if __name__ == "__main__":
    main()
