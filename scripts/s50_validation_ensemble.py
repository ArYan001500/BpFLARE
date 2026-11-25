"""Build the calibration-free ensemble by choosing its members on validation subjects only.

Averaging the three models that happen to score best on the test set is not a result, it is a
measurement of the test set. This selects members by greedy forward selection on the held-out
validation subjects, who are disjoint from both the training and the test subjects, and only then
reports the test error. The rule is fixed before the test set is touched: start empty, repeatedly
add whichever remaining model most improves the validation score, stop when nothing improves it,
and allow a model to be added more than once so that a strong model can carry more weight.

The candidate pool is every model trained during this project that produces an absolute pressure,
which excludes the pairwise and Siamese models because they produce a level-free quantity.

Which weights the selection score uses is itself a choice a referee can question, so passing
--criterion all runs the whole selection three times over one shared set of predictions, once
weighting systolic alone, once diastolic alone and once both. The three answers are then directly
comparable and cost a single pass over the pool.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CONVERTED = ROOT / "converted"
CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
RUNS = ROOT / "runs" / "s4_pat_control"
sys.path.insert(0, str(ROOT / "scripts"))
from network_loader import load_q1_module  # noqa: E402
from s24_balanced_fusion import BalancedGatedNet  # noqa: E402
from s37_backbones import InceptionTime1D, TCN1D  # noqa: E402

TEST_TAG = "calfree_test"
MAX_MEMBERS = 8
WEIGHTS = (2.0, 1.2, 0.8)
CRITERIA = {"combined": WEIGHTS, "sbp": (1.0, 0.0, 0.0), "dbp": (0.0, 1.0, 0.0)}

# tag -> (constructor key, whether the phase channels are zeroed)
POOL = {
    "WS_ABS": ("raw", False), "RAW_w": ("raw", False), "RAW_s43w": ("raw", False),
    "RAW_seed44": ("raw", False), "RAWBASE": ("raw", False),
    "REG_CTRL": ("raw", False), "REG_WD": ("raw", False),
    "REG_AUG": ("raw", False), "REG_AUGWD": ("raw", False),
    "BB_INCEPTION_ABS": ("inception", False), "BB_TCN_ABS": ("tcn", False),
    "BW_RESNET": ("raw", False), "BW_INCEPTION": ("inception", False),
    "BW_TCN": ("tcn", False),
    "CWC3_w": ("gated", False), "CWC3_s43w": ("gated", False),
    "MAGONLY": ("gated", True), "MAGONLY_s43w": ("gated", True),
    "PHASESHUF": ("gated", False), "HARM": ("gated", False),
    "BALANCED": ("balanced", False), "BALANCED_MAG": ("balanced", True),
}


def build(kind, q1):
    if kind == "raw":
        return q1.RawResNet1D(dropout=0.5)
    if kind == "gated":
        return q1.GatedRawCWCFusionNet(dropout=0.5)
    if kind == "balanced":
        return BalancedGatedNet(q1, 0.5)
    return InceptionTime1D(0.5) if kind == "inception" else TCN1D(0.5)


def select_and_report(criterion, preds_va, preds_te, Yva, Yt):
    """Run the greedy selection under one weighting and write out its report."""
    global WEIGHTS
    WEIGHTS = CRITERIA[criterion]
    bar = "=" * 70
    print(f"\n{bar}\nselection criterion: {criterion}, weights {WEIGHTS}", flush=True)

    chosen, current = [], None
    print("greedy forward selection on the validation subjects")
    for step in range(MAX_MEMBERS):
        best, best_tag = None, None
        for tag, p in preds_va.items():
            trial = p if current is None else (current * len(chosen) + p) / (len(chosen) + 1)
            s = score(trial, Yva)
            if best is None or s < best:
                best, best_tag = s, tag
        before = score(current, Yva) if current is not None else float("inf")
        if best >= before - 1e-6:
            print(f"  no further improvement at step {step + 1}")
            break
        current = (preds_va[best_tag] if current is None
                   else (current * len(chosen) + preds_va[best_tag]) / (len(chosen) + 1))
        chosen.append(best_tag)
        print(f"  step {step + 1}: add {best_tag:<20s} validation score {best:.4f}")

    ens_te = np.mean([preds_te[t] for t in chosen], axis=0)
    single = min(preds_va, key=lambda t: score(preds_va[t], Yva))
    out = {"members": chosen, "single_best_on_validation": single}
    print("\n%-34s %9s %9s" % ("model", "SBP MAE", "DBP MAE"))
    for name, p in (("best single, chosen on validation", preds_te[single]),
                    ("validation-selected ensemble", ens_te)):
        sbp = float(np.abs(p[:, 0] - Yt[:, 0]).mean())
        dbp = float(np.abs(p[:, 1] - Yt[:, 1]).mean())
        print("%-34s %9.4f %9.4f" % (name, sbp, dbp))
        out[name] = {"SBP": sbp, "DBP": dbp}
    print("\npublished protocol-matched results: 12.27 / 7.94 and 12.37 / 7.89")
    spread = [float(np.abs(preds_te[t][:, 0] - Yt[:, 0]).mean()) for t in preds_te]
    out["pool_spread"] = {"min": min(spread), "max": max(spread),
                          "range": max(spread) - min(spread), "n": len(spread)}
    print(f"spread across the {len(spread)} candidate models: "
          f"{min(spread):.2f} to {max(spread):.2f}, a range of {max(spread) - min(spread):.2f}")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"s50_validation_ensemble_{criterion}.json").write_text(
        json.dumps({"analysis_id": "s50_validation_ensemble", "criterion": criterion,
                    "results": out}, indent=2), encoding="utf-8")
    print(f"saved reports/s50_validation_ensemble_{criterion}.json", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--criterion", default="combined", choices=list(CRITERIA) + ["all"])
    args = ap.parse_args()
    criteria = list(CRITERIA) if args.criterion == "all" else [args.criterion]
    global WEIGHTS
    WEIGHTS = CRITERIA["combined"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q1 = load_q1_module()
    d, dt = CONVERTED / "train", CONVERTED / TEST_TAG
    X = np.load(d / "X.npy", mmap_mode="r")
    Y = np.asarray(np.load(d / "Y.npy"), dtype=np.float64)
    S = np.load(d / "S.npy")
    va = np.load(d / "rows_validation.npy")
    Xt = np.load(dt / "X.npy", mmap_mode="r")
    Yt = np.asarray(np.load(dt / "Y.npy"), dtype=np.float64)
    IM = np.load(CACHE / "train" / "images.npy", mmap_mode="r")
    IMt = np.load(CACHE / TEST_TAG / "images.npy", mmap_mode="r")
    print(f"{len(va):,} validation windows from {len(np.unique(S[va])):,} subjects, "
          f"{len(Yt):,} test windows\n", flush=True)
    Yva = Y[np.sort(va)]

    preds_va, preds_te = {}, {}
    for tag, (kind, zero_phase) in POOL.items():
        w = RUNS / f"{tag}_weights.pt"
        if not w.is_file():
            print(f"  {tag}: no weights, skipped")
            continue
        ck = torch.load(w, map_location=device, weights_only=False)
        model = build(kind, q1).to(device)
        try:
            model.load_state_dict(ck["state_dict"])
        except RuntimeError:
            print(f"  {tag}: state dict does not fit a {kind} model, skipped")
            continue
        model.eval()
        mean, std = ck["signal_mean"], ck["signal_std"]
        needs_image = kind in ("gated", "balanced")

        def run(Xa, IMa, idx):
            out = []
            with torch.no_grad():
                for s0 in range(0, len(idx), 512):
                    sel = np.sort(idx[s0: s0 + 512])
                    x = (np.asarray(Xa[sel], dtype=np.float32)
                         - mean[None, :, None]) / std[None, :, None]
                    xb = torch.from_numpy(x).to(device)
                    if needs_image:
                        img = np.asarray(IMa[sel, :3], dtype=np.float32)
                        img[:, 0] /= 255.0
                        img[:, 1:] = img[:, 1:] / 127.5 - 1.0
                        if zero_phase:
                            img[:, 1:] = 0.0
                        out.append(model(xb, torch.from_numpy(img).to(device))
                                   .float().cpu().numpy())
                    else:
                        out.append(model(xb).float().cpu().numpy())
            return np.concatenate(out).astype(np.float64)

        preds_va[tag] = run(X, IM, va)
        preds_te[tag] = run(Xt, IMt, np.arange(len(Yt)))
        print(f"  {tag:<20s} validation score {score(preds_va[tag], Yva):.4f}", flush=True)
        del model
        torch.cuda.empty_cache()

    for criterion in criteria:
        select_and_report(criterion, preds_va, preds_te, Yva, Yt)


def score(pred, Y):
    """The same weighted selection score used during training."""
    return float(sum(w * np.abs(pred[:, j] - Y[:, j]).mean() for j, w in enumerate(WEIGHTS)))


if __name__ == "__main__":
    main()
