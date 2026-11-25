"""Does the two-stage cuffless system survive the dataset where the objective loses?

On PulseDB VitalDB, estimating a subject's level from their own unlabelled windows and tracking
with the pairwise model beats the calibration-free baseline by about 1.5 mmHg. VitalDB subjects are
anaesthetised and their pressure moves a great deal. UCI records barely move, and the paper already
shows the pairwise objective losing there, with the mean of a few reference readings beating every
model. If the threshold is a general property rather than a fact about one comparison, the
two-stage system should lose here too, for the same reason and in the same direction.

Predicting that a method fails is a weaker claim than predicting that it works, but it is the one
the paper's own threshold makes, and a referee will ask for it. Everything is frozen: the encoders,
the tracking models and the ridge that maps encoder features to a level are all fitted on VitalDB
and applied to UCI without retraining or refitting any normalisation.

UCI ships no subject identifiers, so a record is the finest grouping and every interval is
bootstrapped over records. Records are short, with a median of about forty windows, so the warm-up
has to stay small and a record is scored only when it has enough windows left after the warm-up is
removed.
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

# The seed-43 twin kept only its test predictions, not its weights, so it cannot be run on a new
# dataset. The ensemble here is the three checkpoints that can be, which is the same set the paper
# reports as its three-model arm.
PAIRED_MODELS = ("WS_PAIRED", "BB_INCEPTION_PAIRED", "BB_TCN_PAIRED")
M_EVAL, N_DRAWS = (5, 10, 20), 50
MIN_SCORED, MIN_FIT_WINDOWS = 10, 20
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 1e4, 1e5)
SEED = 42
UCI_TAG = os.environ.get("UCI_TAG", "uci_external")
SFX = "" if UCI_TAG == "uci_external" else "_" + UCI_TAG.split("_")[-1]
FIXED = os.environ.get("UCI_FIXED") == "1"
SFX += "_fixed" if FIXED else ""


def build(tag, q1):
    if "INCEPTION" in tag:
        return BACKBONES["INCEPTION"](0.5)
    if "TCN" in tag:
        return BACKBONES["TCN"](0.5)
    return q1.RawResNet1D(dropout=0.5)


@torch.no_grad()
def run_model(tag, X, device, q1, want_features, batch=512):
    ck = torch.load(RUNS / f"{tag}_weights.pt", map_location="cpu", weights_only=False)
    model = build(tag, q1)
    model.load_state_dict(ck["state_dict"])
    model = model.to(device).eval()
    mean, std = ck["signal_mean"], ck["signal_std"]
    n = X.shape[0]
    out = np.empty((n, 512 if want_features else 3), dtype=np.float32)
    for s in range(0, n, batch):
        x = (np.asarray(X[s: s + batch], dtype=np.float32)
             - mean[None, :, None]) / std[None, :, None]
        xb = torch.from_numpy(x).to(device)
        if want_features:
            e = model.encoder(xb)
            out[s: s + batch] = torch.cat([e.mean(-1), e.amax(-1)], dim=1).float().cpu().numpy()
        else:
            out[s: s + batch] = model(xb).float().cpu().numpy()
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q1 = load_q1_module()

    Y = np.asarray(np.load(CONVERTED / UCI_TAG / "Y.npy"), dtype=np.float64)
    S = np.load(CONVERTED / UCI_TAG / "S.npy")
    X = np.load(CONVERTED / UCI_TAG / "X.npy", mmap_mode="r")

    cache = RUNS / f"s69_uci_cache{SFX.replace('_fixed', '')}.npz"
    if cache.is_file():
        z = np.load(cache)
        feats = z["feats"].astype(np.float64)
        pred = {t: z[t].astype(np.float64) for t in ("WS_ABS",) + PAIRED_MODELS}
        print("reusing cached UCI passes", flush=True)
    else:
        print(f"running frozen models over {len(Y):,} UCI windows ...", flush=True)
        feats = run_model("RAW_w", X, device, q1, True).astype(np.float64)
        pred = {}
        for t in ("WS_ABS",) + PAIRED_MODELS:
            print(f"  {t} ...", flush=True)
            pred[t] = run_model(t, X, device, q1, False).astype(np.float64)
        np.savez(cache, feats=feats.astype(np.float32),
                 **{t: v.astype(np.float32) for t, v in pred.items()})
    pred["ENSEMBLE"] = np.mean([pred[t] for t in PAIRED_MODELS], axis=0)

    # the level rule, fitted on the VitalDB validation subjects and never refitted here
    Yv = np.asarray(np.load(CONVERTED / "train" / "Y.npy"), dtype=np.float64)
    Sv = np.load(CONVERTED / "train" / "S.npy")
    val_rows = np.load(CONVERTED / "train" / "rows_validation.npy")
    counts = {int(g): int(c) for g, c in zip(*np.unique(Sv[val_rows], return_counts=True))}
    fit_subs = np.array([g for g, c in counts.items() if c >= MIN_FIT_WINDOWS])
    fv = np.load(RUNS / "RAW_w_feat_cache.npz")["val"].astype(np.float64)
    G = np.stack([fv[Sv[val_rows] == g].mean(0) for g in fit_subs])
    mu, sd = G.mean(0), G.std(0)
    sd[sd < 1e-9] = 1.0
    scalar = {}
    for j in (0, 1):
        yv = np.array([Yv[val_rows][Sv[val_rows] == g, j].mean() for g in fit_subs])
        a = pick_alpha((G - mu) / sd, yv, ALPHAS, seed=SEED)
        cmu, w, b = ridge((G - mu) / sd, yv, a)
        scalar[j] = ((feats - mu) / sd - cmu) @ w + b     # per-window level scalar on UCI

    results = {}
    for M in M_EVAL:
        # UCI_FIXED=1 scores every warm-up length on the records eligible at the longest one
        need = (max(M_EVAL) if FIXED else M) + MIN_SCORED
        recs = np.array([r for r in np.unique(S) if (S == r).sum() >= need])
        rows_by = {int(r): np.flatnonzero(S == r) for r in recs}
        truth = np.stack([Y[rows_by[int(r)], :2].mean(0) for r in recs])
        draws = {int(r): [np.random.default_rng(SEED + 7 * d).choice(
            rows_by[int(r)], size=M, replace=False) for d in range(N_DRAWS)] for r in recs}
        boot = np.random.default_rng(7).integers(0, len(recs), size=(2000, len(recs)))

        print("\n" + "=" * 84)
        print(f"--- warm-up of {M} windows ({M * 10 / 60:.1f} min), "
              f"{len(recs):,} UCI records ---")
        for j, target in ((0, "SBP"), (1, "DBP")):
            print("  %-42s %9s   %s" % ("arm", "MAE", "95% CI"))
            per = {}
            arms = ["absolute loss, no calibration",
                    "warm-up mean, no model",
                    "two-stage cuffless, single tracker",
                    "two-stage cuffless, tracker ensemble",
                    "pairwise + cuff on the warm-up windows"]
            for name in arms:
                e = np.zeros(len(recs))
                for i, r in enumerate(recs):
                    rows = rows_by[int(r)]
                    v = []
                    for d, warm in enumerate(draws[int(r)]):
                        m_ = np.ones(len(rows), dtype=bool)
                        m_[np.searchsorted(rows, np.sort(warm))] = False
                        ev = rows[m_]
                        if name.startswith("absolute"):
                            v.append(np.abs(pred["WS_ABS"][ev, j] - Y[ev, j]).mean())
                        elif name.startswith("warm-up mean"):
                            v.append(np.abs(Y[warm, j].mean() - Y[ev, j]).mean())
                        elif name.startswith("pairwise + cuff"):
                            off = (Y[warm, j] - pred["WS_PAIRED"][warm, j]).mean()
                            v.append(np.abs(pred["WS_PAIRED"][ev, j] + off - Y[ev, j]).mean())
                        else:
                            t = "WS_PAIRED" if "single" in name else "ENSEMBLE"
                            off = scalar[j][warm].mean() - pred[t][warm, j].mean()
                            v.append(np.abs(pred[t][ev, j] + off - Y[ev, j]).mean())
                    e[i] = float(np.mean(v))
                per[name] = e
                dd = e[boot].mean(1)
                print("  %-42s %9.4f   [%.2f, %.2f]"
                      % (name, e.mean(), np.percentile(dd, 2.5), np.percentile(dd, 97.5)),
                      flush=True)
                results.setdefault(str(M), {}).setdefault(target, {})[name] = {
                    "mae": float(e.mean()),
                    "ci": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]}

            base = "absolute loss, no calibration"
            for name in arms[1:]:
                d = per[name] - per[base]
                dd = per[name][boot].mean(1) - per[base][boot].mean(1)
                lo, hi = float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))
                results[str(M)][target][name]["vs_calibration_free"] = {
                    "delta": float(d.mean()), "ci": [lo, hi], "significant": bool(lo * hi > 0)}
            n = "two-stage cuffless, tracker ensemble"
            print("  two-stage minus calibration-free: %+.3f [%+.3f, %+.3f]%s\n"
                  % (results[str(M)][target][n]["vs_calibration_free"]["delta"],
                     *results[str(M)][target][n]["vs_calibration_free"]["ci"],
                     " *" if results[str(M)][target][n]["vs_calibration_free"]["significant"]
                     else ""))

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"s69_two_stage_uci{SFX}.json").write_text(json.dumps(
        {"analysis_id": "s69_two_stage_uci",
         "created_utc": datetime.now(timezone.utc).isoformat(),
         "draws": N_DRAWS, "group_unit": "record", "results": results}, indent=2),
        encoding="utf-8")
    print("saved reports/s69_two_stage_uci.json")


if __name__ == "__main__":
    main()
