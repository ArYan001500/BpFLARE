"""Is there any time order left in PulseDB's window sequence within a subject?

The manuscript says a chronological warm-up is undefined on this data because the windows of a
subject come shuffled. The number it quoted had no saved report, so this recomputes it on the
CalFree test subjects, the subjects the warm-up is drawn from, and saves it.

Statistic: for each subject, the Pearson correlation between consecutive systolic labels in file
order (lag one); averaged over subjects, with a 2000-replicate bootstrap over subjects. The same
statistic after randomly permuting each subject's windows, averaged over 20 permutations, is the
value expected with no order at all.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def lag1(y):
    return float(np.corrcoef(y[:-1], y[1:])[0, 1])


def main() -> None:
    Y = np.load(ROOT / "converted" / "calfree_test" / "Y.npy", mmap_mode="r")
    S = np.load(ROOT / "converted" / "calfree_test" / "S.npy")
    rng = np.random.default_rng(42)
    obs, shuf = [], []
    for s in np.unique(S):
        y = np.asarray(Y[S == s, 0], dtype=np.float64)
        obs.append(lag1(y))
        shuf.append(np.mean([lag1(rng.permutation(y)) for _ in range(20)]))
    obs, shuf = np.array(obs), np.array(shuf)
    boot = lambda v: [float(x) for x in np.percentile(
        [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)], [2.5, 97.5])]
    d = obs - shuf
    res = {"n_subjects": int(len(obs)),
           "file_order": {"mean": float(obs.mean()), "ci": boot(obs)},
           "shuffled": {"mean": float(shuf.mean()), "ci": boot(shuf)},
           "difference": {"mean": float(d.mean()), "ci": boot(d)}}
    print(json.dumps(res, indent=1))
    (REPORTS / "s75_window_order.json").write_text(json.dumps(
        {"analysis_id": "s75_window_order", "created_utc": datetime.now(timezone.utc).isoformat(),
         "target": "SBP", "split": "calfree_test", "results": res}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
