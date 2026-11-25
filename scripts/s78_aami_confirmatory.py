"""Confirmatory two-stage evaluation on the 116 VitalDB AAMI subjects.

Implements PAPER/Cuffless Two Stage/notes/protocol_aami_confirmatory.md (commit 08fe6c7) exactly.
Frozen system: stage-1 ridge on RAW_w features fitted on the 194 validation subjects (as s70);
primary tracker WS_PAIRED; secondary mean of WS_PAIRED, BB_INCEPTION_PAIRED, BB_TCN_PAIRED;
comparator WS_ABS. M = 20, 50 draws (SEED + 7 d), warm-up windows excluded from scoring.
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
sys.path.insert(0, str(ROOT / "scripts"))
from s70_two_stage_headline import (CONVERTED, RUNS, REPORTS, N_DRAWS, N_BOOT, MIN_FIT_WINDOWS, TARGETS,  # noqa: E402
                                    encode, level_scalar, interval, agreement, load_q1_module)
from s37_backbones import BACKBONES  # noqa: E402

M, MIN_WINDOWS, SEED = 20, 40, 42
TRACKERS3 = ("WS_PAIRED", "BB_INCEPTION_PAIRED", "BB_TCN_PAIRED")
CACHE = RUNS / "s78_aami_predictions.npz"


@torch.no_grad()
def predict(tag, X, device, q1, batch=512):
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
    out = np.empty((len(X), 3), dtype=np.float32)
    for s in range(0, len(X), batch):
        x = (np.asarray(X[s: s + batch], dtype=np.float32) - mean[None, :, None]) / std[None, :, None]
        out[s: s + batch] = model(torch.from_numpy(x).to(device)).float().cpu().numpy()
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q1 = load_q1_module()
    parts = [p for p in ("aami_cal", "aami_test")]
    Y = np.concatenate([np.load(CONVERTED / p / "Y.npy") for p in parts]).astype(np.float64)
    S = np.concatenate([np.load(CONVERTED / p / "S.npy") for p in parts])
    ids = json.loads((CONVERTED / "aami_cal" / "subjects.json").read_text())
    assert ids == json.loads((CONVERTED / "aami_test" / "subjects.json").read_text())
    train_ids = set(json.loads((CONVERTED / "train" / "subjects.json").read_text()))
    calfree_ids = set(json.loads((CONVERTED / "calfree_test" / "subjects.json").read_text()))
    assert not (set(ids) & train_ids) and not (set(ids) & calfree_ids), "subject overlap"

    if CACHE.is_file():
        z = np.load(CACHE)
        preds, feat = {k: z[k] for k in z.files if k != "RAW_w_feat"}, z["RAW_w_feat"]
    else:
        X = np.concatenate([np.load(CONVERTED / p / "X.npy", mmap_mode="r") for p in parts])
        preds = {t: predict(t, X, device, q1) for t in TRACKERS3 + ("WS_ABS",)}
        feat = encode("RAW_w", X, np.arange(len(X)), device, q1, False)
        np.savez(CACHE, RAW_w_feat=feat, **preds)

    Yv = np.asarray(np.load(CONVERTED / "train" / "Y.npy"), dtype=np.float64)
    Sv = np.load(CONVERTED / "train" / "S.npy")
    val_rows = np.load(CONVERTED / "train" / "rows_validation.npy")
    fv = np.load(RUNS / "RAW_w_feat_cache.npz")["val"]
    counts = {int(g): int(c) for g, c in zip(*np.unique(Sv[val_rows], return_counts=True))}
    fit_subs = np.array([g for g, c in counts.items() if c >= MIN_FIT_WINDOWS])
    lvl = level_scalar(fv, feat, Sv, val_rows, Yv, fit_subs)

    u, c = np.unique(S, return_counts=True)
    subs = u[c >= MIN_WINDOWS]
    rows_by = {int(s): np.flatnonzero(S == s) for s in subs}
    draws = {int(s): [np.random.default_rng(SEED + 7 * d).choice(rows_by[int(s)], size=M, replace=False)
                      for d in range(N_DRAWS)] for s in subs}
    boot = np.random.default_rng(7).integers(0, len(subs), size=(N_BOOT, len(subs)))
    track = {"single": preds["WS_PAIRED"].astype(np.float64),
             "three": np.mean([preds[t] for t in TRACKERS3], axis=0).astype(np.float64)}
    base = preds["WS_ABS"].astype(np.float64)

    out = {"analysis_id": "s78_aami_confirmatory", "created_utc": datetime.now(timezone.utc).isoformat(),
           "protocol_commit": "08fe6c7", "n_subjects_total": int(len(u)), "n_included": int(len(subs)),
           "n_excluded": int(len(u) - len(subs)), "n_scored_windows_per_draw": int(sum(len(r) - M for r in rows_by.values())),
           "results": {}}
    for j, t in TARGETS:
        per = {k: np.zeros(len(subs)) for k in ("single", "three", "base")}
        errs = {k: [] for k in per}
        refs = []
        for i, s in enumerate(subs):
            rows = rows_by[int(s)]
            acc = {k: [] for k in per}
            for warm in draws[int(s)]:
                m = np.ones(len(rows), bool)
                m[np.searchsorted(rows, np.sort(warm))] = False
                ev = rows[m]
                for k in ("single", "three"):
                    est = track[k][ev, j] + (lvl[j][warm].mean() - track[k][warm, j].mean())
                    e = est - Y[ev, j]
                    acc[k].append(np.abs(e).mean())
                    errs[k].append(e)
                e = base[ev, j] - Y[ev, j]
                acc["base"].append(np.abs(e).mean())
                errs["base"].append(e)
                refs.append(Y[ev, j])
            for k in per:
                per[k][i] = np.mean(acc[k])
        ref = np.concatenate(refs)
        res = {}
        for k in per:
            res[k] = {"mae": float(per[k].mean()), "ci": interval(per[k], boot)}
        for k in ("single", "three"):
            d = per[k] - per["base"]
            res[f"{k}_minus_base"] = {"delta": float(d.mean()), "ci": interval(d, boot),
                                      "subjects_improved": int((d < 0).sum())}
        res["agreement"] = {k: agreement(np.concatenate(errs[k])) for k in per}
        if t == "SBP":
            strata = {"below 90": ref < 90, "90 to 140": (ref >= 90) & (ref <= 140), "above 140": ref > 140}
            res["strata_pooled_windows"] = {
                name: {"n_window_draws": int(msk.sum()),
                       **{k: float(np.abs(np.concatenate(errs[k])[msk]).mean()) for k in per}}
                for name, msk in strata.items()}
        out["results"][t] = res
        print(t, json.dumps({k: v for k, v in res.items() if k != "agreement"}, indent=1), flush=True)
    out["primary_confirmed"] = bool(out["results"]["SBP"]["single_minus_base"]["ci"][1] < 0)
    print("primary confirmed:", out["primary_confirmed"])
    (REPORTS / "s78_aami_confirmatory.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
