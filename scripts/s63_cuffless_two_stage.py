"""The two-stage cuffless system: estimate the level without a cuff, then track with the pairwise model.

The paper measures the two halves separately and never joins them. Section 4.1 shows that a
subject's mean pressure is barely recoverable from the waveform, and Section 4.4 shows that the
pairwise objective tracks movement well once a cuff supplies the level. What it does not report is
the number you get when the cuff is removed altogether and the level comes from the waveform too,
which is exactly the system a reader, or a supervisor, will ask about.

Joining them is not the same as reading the two halves off the page. The level estimator is scored
here on how much absolute error it leaves behind, not on a correlation, and the tracking model is
asked to carry an offset that is wrong by several millimetres rather than one that is right.

The level estimator is fitted on the held-out validation subjects rather than on part of the test
set, which is stricter than the within-test split used for Section 4.1 and lets every one of the
144 CalFree subjects be scored. Nothing here is trained: both encoders and the pairwise model are
the frozen checkpoints from the main experiments.

Five arms are reported against the same subjects:

  absolute loss, no calibration   what a calibration-free model does today, the 12.77 in the paper
  pairwise + population level     the pairwise model with no subject information at all, a floor
  pairwise + cuffless level       the two-stage system being asked about
  pairwise + cuff, k=10           the paper's calibrated result, for scale
  pairwise + oracle level         the true subject mean handed over, the ceiling
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

TEST_TAG = "calfree_test"
LEVEL_SOURCES = ("WS_PAIRED", "RAW_w")   # whose encoder supplies the per-subject feature
BEST = "RAW_w"                            # the one carried into the arm table
M_VALUES = (1, 5, 20, 100, 400)           # unlabelled windows behind the cuffless level
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 1e4, 1e5)
MIN_TEST_WINDOWS = 100
MIN_FIT_WINDOWS = 20
K_CUFF = 10
N_DRAWS = 200
SEED = 42


@torch.no_grad()
def encode(tag, X, rows, device, q1, batch=512):
    """Mean-and-max pooled encoder features for the given rows of X, 512 dimensions."""
    ck = torch.load(RUNS / f"{tag}_weights.pt", map_location="cpu", weights_only=False)
    model = q1.RawResNet1D(dropout=0.5)
    model.load_state_dict(ck["state_dict"])
    model = model.to(device).eval()
    mean, std = ck["signal_mean"], ck["signal_std"]
    out = np.empty((len(rows), 512), dtype=np.float32)
    for s in range(0, len(rows), batch):
        sel = rows[s: s + batch]
        x = (np.asarray(X[sel], dtype=np.float32) - mean[None, :, None]) / std[None, :, None]
        e = model.encoder(torch.from_numpy(x).to(device))
        out[s: s + batch] = torch.cat([e.mean(-1), e.amax(-1)], dim=1).float().cpu().numpy()
    return out


def group_mean(A, S, rows, groups):
    """One row per group: that group's mean over the given rows."""
    idx = {int(g): np.flatnonzero(S[rows] == g) for g in groups}
    return np.stack([A[idx[int(g)]].mean(0) for g in groups])


def bootstrap(per_subject, boot):
    d = per_subject[boot].mean(1)
    return float(per_subject.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q1 = load_q1_module()

    Yt = np.asarray(np.load(CONVERTED / TEST_TAG / "Y.npy"), dtype=np.float64)
    St = np.load(CONVERTED / TEST_TAG / "S.npy")
    Xt = np.load(CONVERTED / TEST_TAG / "X.npy", mmap_mode="r")
    subs = np.array([s for s in np.unique(St) if (St == s).sum() >= MIN_TEST_WINDOWS])
    rows_by = {int(s): np.flatnonzero(St == s) for s in subs}
    test_rows = np.arange(len(Yt))

    Xv = np.load(CONVERTED / "train" / "X.npy", mmap_mode="r")
    Yv = np.asarray(np.load(CONVERTED / "train" / "Y.npy"), dtype=np.float64)
    Sv = np.load(CONVERTED / "train" / "S.npy")
    val_rows = np.load(CONVERTED / "train" / "rows_validation.npy")
    counts = {int(g): int(c) for g, c in zip(*np.unique(Sv[val_rows], return_counts=True))}
    fit_subs = np.array([g for g, c in counts.items() if c >= MIN_FIT_WINDOWS])
    print(f"{len(subs)} test subjects, {len(fit_subs)} validation subjects for the level rule; "
          f"the encoders saw neither group\n", flush=True)

    pred = {a: np.asarray(np.load(RUNS / f"{a}_test_pred.npy"), dtype=np.float64)
            for a in ("WS_PAIRED", "WS_ABS")}

    # per-subject truth and per-subject mean of the pairwise model's own output
    truth = {j: np.array([Yt[rows_by[int(s)], j].mean() for s in subs]) for j in (0, 1)}
    model_mean = {j: np.array([pred["WS_PAIRED"][rows_by[int(s)], j].mean() for s in subs])
                  for j in (0, 1)}

    level, level_quality, feat_windows, ridge_fit = {}, {}, {}, {}
    for tag in LEVEL_SOURCES:
        cache = RUNS / f"{tag}_feat_cache.npz"
        if cache.is_file():
            z = np.load(cache)
            fv_w, ft_w = z["val"], z["test"]
            print(f"  reusing cached {tag} features", flush=True)
        else:
            print(f"  encoding with {tag} ...", flush=True)
            fv_w = encode(tag, Xv, val_rows, device, q1)
            ft_w = encode(tag, Xt, test_rows, device, q1)
            np.savez(cache, val=fv_w, test=ft_w)
        feat_windows[tag] = ft_w
        Fv = group_mean(fv_w, Sv, val_rows, fit_subs)
        Ft = group_mean(ft_w, St, test_rows, subs)
        mu, sd = Fv.mean(0), Fv.std(0)
        sd[sd < 1e-9] = 1.0
        Zv, Zt = (Fv - mu) / sd, (Ft - mu) / sd
        for j, target in ((0, "SBP"), (1, "DBP")):
            yv = np.array([Yv[val_rows][Sv[val_rows] == g, j].mean() for g in fit_subs])
            a = pick_alpha(Zv, yv, ALPHAS, seed=SEED)
            cmu, w, b = ridge(Zv, yv, a)
            hat = (Zt - cmu) @ w + b
            level[(tag, j)] = hat
            ridge_fit[(tag, j)] = (mu, sd, cmu, w, b)
            level_quality[f"{tag}_{target}"] = {
                "alpha": float(a),
                "pearson_r": float(np.corrcoef(hat, truth[j])[0, 1]),
                "mae_of_subject_mean": float(np.abs(hat - truth[j]).mean()),
                "mae_of_population_mean": float(np.abs(yv.mean() - truth[j]).mean())}
            print("    %-9s %s  r=%.3f   level MAE %.3f vs %.3f for the population mean"
                  % (tag, target, level_quality[f"{tag}_{target}"]["pearson_r"],
                     level_quality[f"{tag}_{target}"]["mae_of_subject_mean"],
                     level_quality[f"{tag}_{target}"]["mae_of_population_mean"]), flush=True)
        pop = {j: np.full(len(subs), np.array(
            [Yv[val_rows][Sv[val_rows] == g, j].mean() for g in fit_subs]).mean()) for j in (0, 1)}

    # cuff calibration, identical protocol to the paper: k windows, 200 draws, excluded from scoring
    rng = np.random.default_rng(SEED)
    draws = {int(s): [rng.choice(rows_by[int(s)], size=K_CUFF, replace=False)
                      for _ in range(N_DRAWS)] for s in subs}
    boot = np.random.default_rng(7).integers(0, len(subs), size=(2000, len(subs)))

    results, per_sub = {}, {}
    for j, target in ((0, "SBP"), (1, "DBP")):
        arms = {"absolute loss, no calibration": ("WS_ABS", None),
                "absolute + population level": ("WS_ABS", pop[j]),
                "absolute + cuffless level": ("WS_ABS", level[(BEST, j)]),
                "absolute + oracle level": ("WS_ABS", truth[j]),
                "pairwise + population level": ("WS_PAIRED", pop[j]),
                "pairwise + cuffless level": ("WS_PAIRED", level[(BEST, j)]),
                "pairwise + cuff, k=10": ("WS_PAIRED", "cuff"),
                "pairwise + oracle level": ("WS_PAIRED", truth[j])}

        print("\n" + "=" * 78)
        print(f"--- {target} ---  absolute error on the {len(subs)} unseen CalFree subjects")
        print("%-38s %9s   %s" % ("arm", "MAE", "95% CI"))
        for name, (arm, lv) in arms.items():
            e = np.empty(len(subs))
            for i, s in enumerate(subs):
                rows = rows_by[int(s)]
                if lv is None:
                    e[i] = np.abs(pred[arm][rows, j] - Yt[rows, j]).mean()
                elif isinstance(lv, str):
                    v = []
                    for cal in draws[int(s)]:
                        m = np.ones(len(rows), dtype=bool)
                        m[np.searchsorted(rows, np.sort(cal))] = False
                        ev = rows[m]
                        off = (Yt[cal, j] - pred[arm][cal, j]).mean()
                        v.append(np.abs(pred[arm][ev, j] + off - Yt[ev, j]).mean())
                    e[i] = float(np.mean(v))
                else:
                    off = lv[i] - pred[arm][rows, j].mean()
                    e[i] = np.abs(pred[arm][rows, j] + off - Yt[rows, j]).mean()
            per_sub[(name, target)] = e
            m, lo, hi = bootstrap(e, boot)
            results.setdefault(target, {})[name] = {"mae": m, "ci": [lo, hi]}
            print("%-38s %9.4f   [%.2f, %.2f]" % (name, m, lo, hi))

        base = "absolute loss, no calibration"
        print("\n%-38s %s" % ("", f"paired difference against '{base}'"))
        for name in arms:
            if name == base:
                continue
            d = per_sub[(name, target)] - per_sub[(base, target)]
            dd = per_sub[(name, target)][boot].mean(1) - per_sub[(base, target)][boot].mean(1)
            lo, hi = float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))
            print("%-38s %+9.4f   [%+.3f, %+.3f]%s"
                  % (name, d.mean(), lo, hi, " *" if lo * hi > 0 else ""))
            results[target][name]["vs_calibration_free"] = {
                "delta": float(d.mean()), "ci": [lo, hi], "significant": bool(lo * hi > 0)}
        print("* interval excludes zero; negative favours the arm")

        print("\n%-38s %s" % ("", "same level treatment, pairwise minus absolute"))
        for lvl in ("population level", "cuffless level", "oracle level"):
            a, b = f"pairwise + {lvl}", f"absolute + {lvl}"
            d = per_sub[(a, target)] - per_sub[(b, target)]
            dd = per_sub[(a, target)][boot].mean(1) - per_sub[(b, target)][boot].mean(1)
            lo, hi = float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))
            print("%-38s %+9.4f   [%+.3f, %+.3f]%s"
                  % (lvl, d.mean(), lo, hi, " *" if lo * hi > 0 else ""))
            results[target].setdefault("objective_at_matched_level", {})[lvl] = {
                "delta": float(d.mean()), "ci": [lo, hi], "significant": bool(lo * hi > 0)}


    # How long a cuffless warm-up has to be: the level rule is given M unlabelled windows only.
    print()
    print("=" * 78)
    print("--- how many unlabelled windows the cuffless level needs ---")
    print("%6s %8s   %9s %9s   %9s %9s" % ("M", "minutes", "SBP r", "SBP MAE", "DBP r", "DBP MAE"))
    sweep = {}
    for M in M_VALUES:
        row = {}
        for j, target in ((0, "SBP"), (1, "DBP")):
            mu, sd, cmu, w, b = ridge_fit[(BEST, j)]
            rs, es = [], []
            for rep in range(20):
                r2 = np.random.default_rng(3000 + rep)
                F = np.stack([feat_windows[BEST][r2.choice(rows_by[int(x)], size=M,
                                                           replace=False)].mean(0) for x in subs])
                hat = (((F - mu) / sd) - cmu) @ w + b
                rs.append(float(np.corrcoef(hat, truth[j])[0, 1]))
                e = np.array([np.abs(pred["WS_PAIRED"][rows_by[int(x)], j]
                                     + (hat[i] - pred["WS_PAIRED"][rows_by[int(x)], j].mean())
                                     - Yt[rows_by[int(x)], j]).mean()
                              for i, x in enumerate(subs)])
                es.append(e.mean())
            row[target] = {"pearson_r": float(np.mean(rs)), "mae": float(np.mean(es))}
        sweep[M] = row
        print("%6d %8.1f   %9.3f %9.4f   %9.3f %9.4f"
              % (M, M * 10 / 60, row["SBP"]["pearson_r"], row["SBP"]["mae"],
                 row["DBP"]["pearson_r"], row["DBP"]["mae"]))

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "s63_cuffless_two_stage.json").write_text(json.dumps(
        {"analysis_id": "s63_cuffless_two_stage",
         "created_utc": datetime.now(timezone.utc).isoformat(),
         "n_test_subjects": int(len(subs)), "n_fit_subjects": int(len(fit_subs)),
         "k_cuff": K_CUFF, "draws": N_DRAWS,
         "level_estimator": level_quality, "warmup_sweep": sweep, "results": results}, indent=2), encoding="utf-8")
    print("\nsaved reports/s63_cuffless_two_stage.json")


if __name__ == "__main__":
    main()
