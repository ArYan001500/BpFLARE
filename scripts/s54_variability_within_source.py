"""Separate the effect of within-group variability from the effect of which dataset a group is in.

The pooled threshold scan cannot tell these apart, because almost every VitalDB subject sits above
the crossover and almost every UCI record below it. A cumulative scan inside one dataset is also the
wrong instrument: it spends its power on a shrinking tail and reports nothing when that tail runs
out, which is not the same as reporting no effect.

Two estimators are used instead. Non-overlapping bands show the shape of the relation inside each
dataset without the tail dominating. A slope of the per-group difference on the group's own standard
deviation uses every group at once, so a real trend does not have to survive on 42 records to be
visible. Both are bootstrapped over groups.

The third quantity is the one the referee's objection is really about: the slope after each dataset
has been centred on its own mean. That removes any constant offset between the two populations,
including the part that comes from one being in-domain and the other out-of-domain, and leaves only
the variation explained by variability inside a dataset. If that slope is negative in both datasets
separately, the relation is a property of populations rather than an artefact of the gap between
these two.

Nothing is retrained. This reads the per-group errors s45 already stored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
EDGES = (0, 3, 5, 8, 12, 18, 1000)
BOOT = 2000
SEED = 7


def boot_mean_ci(x, rng):
    b = x[rng.integers(0, len(x), size=(BOOT, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def boot_slope_ci(x, y, rng):
    """Least-squares slope of y on x with a bootstrap interval over groups."""
    idx = rng.integers(0, len(x), size=(BOOT, len(x)))
    slopes = np.empty(BOOT)
    for i in range(BOOT):
        xi, yi = x[idx[i]], y[idx[i]]
        xc = xi - xi.mean()
        denom = (xc * xc).sum()
        slopes[i] = (xc * (yi - yi.mean())).sum() / denom if denom > 0 else np.nan
    xc = x - x.mean()
    slope = float((xc * (y - y.mean())).sum() / (xc * xc).sum())
    lo, hi = float(np.nanpercentile(slopes, 2.5)), float(np.nanpercentile(slopes, 97.5))
    return slope, lo, hi, bool(lo * hi > 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    SFX = "" if __import__("os").environ.get("UCI_TAG", "uci_external") == "uci_external" else "_all"
    f = REPORTS / (f"s45_crossover{SFX}.json" if args.k == 10 else f"s45_crossover_k{args.k}{SFX}.json")
    rows = json.loads(f.read_text(encoding="utf-8"))["groups"]
    src = np.array([r["source"] for r in rows])
    sd = np.array([r["sd"] for r in rows])
    delta = np.array([r["paired"] - r["abs"] for r in rows])
    print(f"{len(rows):,} groups from {f.name}, k = {args.k}\n")

    results = {}
    for s in sorted(set(src)):
        m = src == s
        rng = np.random.default_rng(SEED)
        print("=" * 74)
        print(f"{s}: {m.sum():,} groups, median own SBP sd {np.median(sd[m]):.2f}")
        print("%-12s %7s %10s   %s" % ("band", "groups", "mean diff", "95% interval"))
        bands = {}
        for lo_e, hi_e in zip(EDGES[:-1], EDGES[1:]):
            b = m & (sd >= lo_e) & (sd < hi_e)
            if b.sum() < 15:
                print("%-12s %7d   too few to report" % (f"{lo_e} to {hi_e}", b.sum()))
                continue
            mu, l, h = boot_mean_ci(delta[b], rng)
            star = " *" if l * h > 0 else ""
            label = f"{lo_e} to {hi_e}" if hi_e < 1000 else f"above {lo_e}"
            print("%-12s %7d %10.3f   [%+.2f, %+.2f]%s" % (label, b.sum(), mu, l, h, star))
            bands[label] = {"n": int(b.sum()), "delta": mu, "ci": [l, h],
                            "significant": bool(l * h > 0)}

        sl, sl_lo, sl_hi, sl_sig = boot_slope_ci(sd[m], delta[m], rng)
        print(f"  slope of the difference on the group's own sd: {sl:+.4f} mmHg per mmHg "
              f"[{sl_lo:+.4f}, {sl_hi:+.4f}]{' *' if sl_sig else ''}")
        results[s] = {"n_groups": int(m.sum()), "median_sd": float(np.median(sd[m])),
                      "bands": bands,
                      "slope": {"value": sl, "ci": [sl_lo, sl_hi], "significant": sl_sig}}
        print()

    # the same slope with each dataset centred on itself, which removes any constant offset
    xc, yc = sd.copy(), delta.copy()
    for s in set(src):
        m = src == s
        xc[m] -= xc[m].mean()
        yc[m] -= yc[m].mean()
    rng = np.random.default_rng(SEED)
    sl, lo, hi, sig = boot_slope_ci(xc, yc, rng)
    print("=" * 74)
    print(f"within-dataset slope, each source centred on its own mean: {sl:+.4f} mmHg per mmHg "
          f"[{lo:+.4f}, {hi:+.4f}]{' *' if sig else ''}")
    results["within_dataset_centred"] = {"slope": sl, "ci": [lo, hi], "significant": sig}

    name = "s54_variability_within_source" + ("" if args.k == 10 else f"_k{args.k}") + SFX
    (REPORTS / f"{name}.json").write_text(json.dumps(
        {"analysis_id": "s54_variability_within_source", "k": args.k, "source_file": f.name,
         "results": results}, indent=2), encoding="utf-8")
    print(f"\nsaved reports/{name}.json")


if __name__ == "__main__":
    main()
