"""What the paired objective is worth end to end, once a cuff reading supplies the level.

The paired arm does not predict pressure. It predicts how far a window sits from the rest of that
person's recording, and its output is only defined up to a constant per person. That is not a
defect: the level is exactly the part of the problem that no model can solve from the signal, and
in any real deployment it arrives from a single cuff measurement at enrolment. What the paired
objective bought is a tracking error of 6.886 mmHg against the control's 8.408.

The number that matters to a reader is the absolute one. Here each held-out subject is given k
windows with a reference pressure, the offset between model and truth is estimated on those k
windows alone, and the remaining windows of that subject are scored in absolute terms. The
calibration windows are never scored. Every arm sees exactly the same calibration windows for a
given subject and draw, so no arm can win by being handed easier ones, and many draws are averaged
because at k=1 a single unlucky window would decide the answer.

Two references are reported alongside: the tracking error, which is what the absolute error would
be if the cuff reading were perfect, and the previously measured 8.82 mmHg that the ordinary
absolute-loss model reached under this same protocol at k=10.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONVERTED = ROOT / "converted"
RUNS = ROOT / "runs" / "s4_pat_control"
REPORTS = ROOT / "reports"
TEST_TAG = "calfree_test"
ARMS = ("ABS", "CENTERED", "PAIRED", "ABSPAIRED")
K_VALUES = (1, 3, 5, 10, 20)
N_DRAWS = 200
MIN_WINDOWS = 100
SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="")
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    args = ap.parse_args()
    Y = np.asarray(np.load(CONVERTED / TEST_TAG / "Y.npy"), dtype=np.float64)
    S = np.load(CONVERTED / TEST_TAG / "S.npy")
    subs = np.array([s for s in np.unique(S) if (S == s).sum() >= MIN_WINDOWS])
    rows_by = {int(s): np.flatnonzero(S == s) for s in subs}
    rng = np.random.default_rng(SEED)
    draws = {k: {int(s): [rng.choice(rows_by[int(s)], size=k, replace=False)
                          for _ in range(N_DRAWS)] for s in subs} for k in K_VALUES}
    boot = np.random.default_rng(7).integers(0, len(subs), size=(2000, len(subs)))
    P = {a: np.asarray(np.load(RUNS / f"WS_{a}{args.suffix}_test_pred.npy"),
                   dtype=np.float64)
         for a in args.arms if (RUNS / f"WS_{a}{args.suffix}_test_pred.npy").is_file()}
    print(f"{len(subs)} test subjects with at least {MIN_WINDOWS} windows, "
          f"{N_DRAWS} draws per subject per k\n")

    results, per_subject = {}, {}
    for j, target in enumerate(("SBP", "DBP")):
        print("=" * 78)
        print(f"--- {target} ---  absolute error after a k-window cuff calibration")
        print("%-12s %9s %9s %9s %9s %9s" % ("arm", *[f"k={k}" for k in K_VALUES]))
        for arm, pred in P.items():
            row = []
            for k in K_VALUES:
                e = []
                for s in subs:
                    rows = rows_by[int(s)]
                    v = []
                    for cal in draws[k][int(s)]:
                        m = np.ones(len(rows), dtype=bool)
                        m[np.searchsorted(rows, np.sort(cal))] = False
                        ev = rows[m]
                        off = (Y[cal, j] - pred[cal, j]).mean()
                        v.append(np.abs(pred[ev, j] + off - Y[ev, j]).mean())
                    e.append(float(np.mean(v)))
                e = np.asarray(e)
                per_subject[(arm, target, k)] = e
                row.append(e.mean())
                results.setdefault(target, {}).setdefault(arm, {})[k] = float(e.mean())
            print("%-12s %9.4f %9.4f %9.4f %9.4f %9.4f" % (arm, *row))
        print()
        base = "ABS"
        if base in P:
            print("%-12s %s" % ("", "change versus the absolute-loss control, 95% CI"))
            for arm in P:
                if arm == base:
                    continue
                cells = []
                for k in K_VALUES:
                    d = (per_subject[(arm, target, k)][boot].mean(1)
                         - per_subject[(base, target, k)][boot].mean(1))
                    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
                    cells.append("%+.3f%s" % (per_subject[(arm, target, k)].mean()
                                              - per_subject[(base, target, k)].mean(),
                                              "*" if lo * hi > 0 else " "))
                    results[target].setdefault(f"{arm}_vs_{base}", {})[k] = {
                        "delta": float(per_subject[(arm, target, k)].mean()
                                       - per_subject[(base, target, k)].mean()),
                        "ci": [lo, hi], "significant": bool(lo * hi > 0)}
                print("%-12s %9s %9s %9s %9s %9s" % (arm, *cells))
            print("* interval excludes zero; negative favours the arm\n")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"s36_calibrated_end_to_end{args.suffix}.json").write_text(json.dumps(
        {"analysis_id": "s36_calibrated_end_to_end", "n_subjects": int(len(subs)),
         "draws": N_DRAWS, "k_values": list(K_VALUES), "results": results}, indent=2),
        encoding="utf-8")
    print(f"saved reports/s36_calibrated_end_to_end{args.suffix}.json")


if __name__ == "__main__":
    main()
