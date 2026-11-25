"""When is a tracking model worth more than the cuff reading it was calibrated with?

Two results look contradictory until they are put on the same axis. On PulseDB VitalDB the paired
objective beats the absolute one by 1.6 mmHg. On the MIMIC-II-derived UCI set the same frozen
models lose by 0.7. The datasets differ in exactly the property that should decide it: a VitalDB
subject's own pressure moves with a standard deviation of 13.3 mmHg, a UCI record's by 3.9.

So the two are not contradictory, they are two points on one curve. This measures that curve
directly. Every group -- a VitalDB subject or a UCI record -- contributes its own within-group
pressure standard deviation and its own calibrated error under three methods, and the difference
between them is plotted against that standard deviation. Where the curve crosses zero is the point
below which the calibration mean is the better estimator and above which a model that tracks
earns its place.

That threshold is a more useful thing to report than either dataset's average, because it tells a
reader whether the method applies to their population before they run anything.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONVERTED = ROOT / "converted"
REPORTS = ROOT / "reports"
RUNS = ROOT / "runs" / "s4_pat_control"

K = 10
N_DRAWS = 30
CAL_SEED = 42
UCI_TAG = __import__("os").environ.get("UCI_TAG", "uci_external")
SFX = "" if UCI_TAG == "uci_external" else "_" + UCI_TAG.split("_")[-1]
SOURCES = {"VitalDB subjects": ("calfree_test", 100), "UCI records": (UCI_TAG, 20)}
EDGES = (0, 3, 5, 8, 12, 18, 100)


def calibrated(pred, Y, rows, draws, j):
    """Mean absolute error for one group after a k-window offset, calibration windows excluded."""
    v = []
    for cal in draws:
        m = np.ones(len(rows), dtype=bool)
        m[np.searchsorted(rows, np.sort(cal))] = False
        ev = rows[m]
        est = (np.full(len(ev), Y[cal, j].mean()) if pred is None
               else pred[ev, j] + (Y[cal, j] - pred[cal, j]).mean())
        v.append(np.abs(est - Y[ev, j]).mean())
    return float(np.mean(v))


def main() -> None:
    global K
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=K,
                    help="calibration windows per group; the paper reports 10")
    args = ap.parse_args()
    K = args.k
    print(f"calibration windows per group: {K}\n", flush=True)
    rows_all = []
    for label, (tag, min_w) in SOURCES.items():
        d = CONVERTED / tag
        Y = np.asarray(np.load(d / "Y.npy"), dtype=np.float64)
        S = np.load(d / "S.npy")
        pa = np.load(RUNS / "WS_ABS_test_pred.npy") if tag == "calfree_test" else None
        pb = np.load(RUNS / "WS_PAIRED_test_pred.npy") if tag == "calfree_test" else None
        if tag != "calfree_test":
            pa = np.load(RUNS / f"UCI_WS_ABS_pred{SFX}.npy") if (RUNS / f"UCI_WS_ABS_pred{SFX}.npy").is_file() \
                else None
            pb = np.load(RUNS / f"UCI_WS_PAIRED_pred{SFX}.npy") \
                if (RUNS / f"UCI_WS_PAIRED_pred{SFX}.npy").is_file() else None
        if pa is None or pb is None:
            print(f"  {label}: predictions missing, run s43 with prediction saving first")
            continue
        groups = [g for g in np.unique(S) if (S == g).sum() >= min_w]
        rng = np.random.default_rng(CAL_SEED)
        for g in groups:
            r = np.flatnonzero(S == g)
            draws = [rng.choice(r, size=K, replace=False) for _ in range(N_DRAWS)]
            rows_all.append({
                "source": label, "sd": float(Y[r, 0].std()), "n": int(len(r)),
                "abs": calibrated(pa, Y, r, draws, 0),
                "paired": calibrated(pb, Y, r, draws, 0),
                "calonly": calibrated(None, Y, r, draws, 0)})
        print(f"  {label}: {len(groups):,} groups scored", flush=True)

    if not rows_all:
        raise SystemExit("nothing to score")
    sd = np.array([r["sd"] for r in rows_all])
    ea = np.array([r["abs"] for r in rows_all])
    eb = np.array([r["paired"] for r in rows_all])
    ec = np.array([r["calonly"] for r in rows_all])
    src = np.array([r["source"] for r in rows_all])
    rng = np.random.default_rng(7)

    print(f"\n{len(sd):,} groups in total, within-group SBP standard deviation "
          f"{sd.min():.1f} to {sd.max():.1f} mmHg\n")
    print("=" * 92)
    print("SBP error at k=10 against how much the group's own pressure moves")
    print("%-14s %7s %9s %9s %9s   %-24s %-24s"
          % ("sd band", "groups", "cal only", "absolute", "paired",
             "paired - absolute", "paired - cal only"))
    out = {}
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (sd >= lo) & (sd < hi)
        if m.sum() < 10:
            continue
        idx = rng.integers(0, int(m.sum()), size=(2000, int(m.sum())))
        cells = []
        for x, y in ((eb[m], ea[m]), (eb[m], ec[m])):
            dd = x[idx].mean(1) - y[idx].mean(1)
            l, h = np.percentile(dd, 2.5), np.percentile(dd, 97.5)
            cells.append("%+7.3f [%+6.2f,%+6.2f]%s" % (x.mean() - y.mean(), l, h,
                                                       "*" if l * h > 0 else " "))
        name = f"{lo}-{hi if hi < 99 else '+'}"
        print("%-14s %7d %9.3f %9.3f %9.3f   %-24s %-24s"
              % (name, m.sum(), ec[m].mean(), ea[m].mean(), eb[m].mean(), *cells))
        idxb = rng.integers(0, int(m.sum()), size=(2000, int(m.sum())))
        db = eb[m][idxb].mean(1) - ea[m][idxb].mean(1)
        lo_b, hi_b = float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))
        out[name] = {"n": int(m.sum()), "cal_only": float(ec[m].mean()),
                     "difference_ci": [lo_b, hi_b], "difference_significant": bool(lo_b * hi_b > 0),
                     "absolute": float(ea[m].mean()), "paired": float(eb[m].mean()),
                     "paired_minus_absolute": float(eb[m].mean() - ea[m].mean()),
                     "paired_minus_calonly": float(eb[m].mean() - ec[m].mean())}
    print("* interval excludes zero; negative favours the paired objective")

    # where the advantage changes sign, found by scanning the threshold rather than fitting a line
    print("\nthreshold scan: groups whose own SBP standard deviation exceeds t")
    print("%8s %8s %10s %10s   %s" % ("t", "groups", "absolute", "paired", "difference, 95% CI"))
    cross = None
    for t in (0, 2, 4, 6, 8, 10, 12, 14, 16):
        m = sd >= t
        if m.sum() < 20:
            continue
        idx = rng.integers(0, int(m.sum()), size=(2000, int(m.sum())))
        dd = eb[m][idx].mean(1) - ea[m][idx].mean(1)
        l, h = np.percentile(dd, 2.5), np.percentile(dd, 97.5)
        d = eb[m].mean() - ea[m].mean()
        print("%8.0f %8d %10.3f %10.3f   %+.3f [%+.2f,%+.2f]%s"
              % (t, m.sum(), ea[m].mean(), eb[m].mean(), d, l, h, "*" if l * h > 0 else " "))
        out.setdefault("threshold_scan", {})[str(t)] = {
            "n": int(m.sum()), "delta": float(d), "ci": [float(l), float(h)],
            "significant": bool(l * h > 0)}
        if cross is None and d < 0 and l * h > 0:
            cross = t
    if cross is not None:
        print(f"\nthe paired objective becomes significantly better once the group's own SBP "
              f"standard deviation exceeds about {cross} mmHg")
    for s in np.unique(src):
        m = src == s
        print(f"  {s}: median within-group sd {np.median(sd[m]):.2f}, "
              f"{100 * (sd[m] >= (cross or 0)).mean():.0f} percent above that line")

    REPORTS.mkdir(exist_ok=True)
    name = (f"s45_crossover{SFX}.json" if K == 10 else f"s45_crossover_k{K}{SFX}.json")
    (REPORTS / name).write_text(json.dumps(
        {"analysis_id": "s45_crossover", "k": K, "draws": N_DRAWS,
         "n_groups": len(sd), "crossover_sd": cross, "bands": out,
         "groups": rows_all}, indent=2), encoding="utf-8")
    print(f"\nsaved reports/{name}")


if __name__ == "__main__":
    main()
