"""True-level ceiling for one tracker under the strict protocol of s70/s72.

s72 reports the ceiling only for the four-tracker ensemble (6.974 SBP). The manuscript compares the
ten-reading result of a single tracker with a true-mean ceiling, and the single-tracker figure it had
(6.886, s35 'tracked') re-centres on all windows, which is a different protocol. Here: M = 20 warm-up
windows drawn with the s68/s70 seeds, level = the subject's true mean over all windows, offset =
level - mean tracker output over the warm-up, warm-up windows excluded from scoring. The ensemble row
must reproduce s72 before the single-tracker row is trusted.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "s4_pat_control"
PAIRED = ("WS_PAIRED", "WS_PAIRED_s43", "BB_INCEPTION_PAIRED", "BB_TCN_PAIRED")
M, N_DRAWS, SEED = 20, 50, 42

Y = np.load(ROOT / "converted/calfree_test/Y.npy").astype(np.float64)
S = np.load(ROOT / "converted/calfree_test/S.npy")
subs = np.array([s for s in np.unique(S) if (S == s).sum() >= 100])
rows_by = {int(s): np.flatnonzero(S == s) for s in subs}
pred = {t: np.load(RUNS / f"{t}_test_pred.npy").astype(np.float64) for t in PAIRED}
pred["ENSEMBLE"] = np.mean([pred[t] for t in PAIRED], axis=0)
draws = {int(s): [np.random.default_rng(SEED + 7 * d).choice(rows_by[int(s)], size=M, replace=False)
                  for d in range(N_DRAWS)] for s in subs}

out = {}
for name in ("ENSEMBLE", "WS_PAIRED"):
    out[name] = {}
    for j, t in ((0, "SBP"), (1, "DBP")):
        e = []
        for s in subs:
            rows = rows_by[int(s)]
            truth = Y[rows, j].mean()
            v = []
            for warm in draws[int(s)]:
                keep = np.ones(len(rows), bool)
                keep[np.searchsorted(rows, np.sort(warm))] = False
                ev = rows[keep]
                off = truth - pred[name][warm, j].mean()
                v.append(np.abs(pred[name][ev, j] + off - Y[ev, j]).mean())
            e.append(np.mean(v))
        out[name][t] = float(np.mean(e))
print(json.dumps(out, indent=1))
s72 = json.loads((ROOT / "reports/s72_two_stage_ablation.json").read_text())["results"]
for t in ("SBP", "DBP"):
    ref = s72[t]["true level (ceiling)"]["mae"]
    assert abs(out["ENSEMBLE"][t] - ref) < 1e-6, (t, out["ENSEMBLE"][t], ref)
print("ensemble reproduces s72")
(ROOT / "reports/s77_true_level_single_tracker.json").write_text(json.dumps(
    {"analysis_id": "s77_true_level_single_tracker", "created_utc": datetime.now(timezone.utc).isoformat(),
     "warmup_windows": M, "draws": N_DRAWS, "results": out}, indent=2))
