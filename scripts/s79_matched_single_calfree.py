"""Matched single-backbone contrast on CalFree: two-stage with the single residual pairwise tracker
WS_PAIRED against the absolute residual network WS_ABS, same windows, same draws, as in s70.
Reviewers asked for this instead of four-tracker ensemble vs one absolute network."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from s70_two_stage_headline import (CONVERTED, RUNS, REPORTS, MIN_TEST_WINDOWS, MIN_FIT_WINDOWS, N_BOOT,  # noqa: E402
                                    TARGETS, level_scalar, draws_for, score, interval)

Y = np.asarray(np.load(CONVERTED / "calfree_test" / "Y.npy"), dtype=np.float64)
S = np.load(CONVERTED / "calfree_test" / "S.npy")
Yv = np.asarray(np.load(CONVERTED / "train" / "Y.npy"), dtype=np.float64)
Sv = np.load(CONVERTED / "train" / "S.npy")
val_rows = np.load(CONVERTED / "train" / "rows_validation.npy")
subs = np.array([s for s in np.unique(S) if (S == s).sum() >= MIN_TEST_WINDOWS])
rows_by = {int(s): np.flatnonzero(S == s) for s in subs}
counts = {int(g): int(c) for g, c in zip(*np.unique(Sv[val_rows], return_counts=True))}
fit_subs = np.array([g for g, c in counts.items() if c >= MIN_FIT_WINDOWS])
boot = np.random.default_rng(7).integers(0, len(subs), size=(N_BOOT, len(subs)))
track = np.load(RUNS / "WS_PAIRED_test_pred.npy").astype(np.float64)
base = np.load(RUNS / "WS_ABS_test_pred.npy").astype(np.float64)
z = np.load(RUNS / "RAW_w_feat_cache.npz")
lvl = level_scalar(z["val"], z["test"], Sv, val_rows, Yv, fit_subs)
dr = draws_for(subs, rows_by, 20)
out = {}
for j, t in TARGETS:
    two, cf = score(subs, rows_by, dr, Y, track, base, lvl, j)
    d = two - cf
    out[t] = {"single_two_stage": {"mae": float(two.mean()), "ci": interval(two, boot)},
              "absolute": {"mae": float(cf.mean()), "ci": interval(cf, boot)},
              "difference": {"delta": float(d.mean()), "ci": interval(d, boot)},
              "subjects_improved": int((d < 0).sum())}
    print(t, json.dumps(out[t]))
(REPORTS / "s79_matched_single_calfree.json").write_text(json.dumps(
    {"analysis_id": "s79_matched_single_calfree", "created_utc": datetime.now(timezone.utc).isoformat(),
     "results": out}, indent=2), encoding="utf-8")
