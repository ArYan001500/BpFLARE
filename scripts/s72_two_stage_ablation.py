"""One ablation table for the two-stage system, every row under the same protocol and the same draws.

The component comparisons were first run one question at a time across s63 to s68, and they do not
share a setting: some used one tracker and some four, some the permissive protocol and some the
strict one, some ten warm-up windows and some twenty. A table assembled from those runs would place
numbers side by side that differ in details a reader cannot see. This run recomputes every row that
removes or replaces one component, at the headline setting of s70: strict protocol, 20 warm-up
windows, the same 50 draws per subject, the same 144 subjects.

Each row changes one thing relative to the full system and is paired against it subject by subject:

  full system                   level from RAW_w features, tracker = mean of four pairwise networks
  one tracker                   tracker = the single ResNet pairwise network
  absolute network as tracker   tracker = the absolute-loss network; asks whether the gain comes
                                from the pairwise objective or merely from re-centring
  level from pairwise encoder   stage-1 features from the pairwise network instead of RAW_w
  population level              stage 1 replaced by one constant, the validation population mean
  stage 1 alone                 every window predicted as the stage-1 level, no tracking at all
  no calibration                the absolute-loss network on its own, the calibration-free baseline
  true level (ceiling)          stage 1 replaced by the subject's true mean; not deployable, it
                                bounds what a perfect level estimate could buy
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from s70_two_stage_headline import (CONVERTED, M_HEADLINE, MIN_FIT_WINDOWS,  # noqa: E402
                                    MIN_TEST_WINDOWS, N_BOOT, REPORTS, RUNS, TARGETS, TRACKERS,
                                    FOUR_TRACKERS, SINGLE, SUFFIX,
                                    draws_for, interval, level_scalar)


def main() -> None:
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
    draws = draws_for(subs, rows_by, M_HEADLINE)

    load = lambda t: np.asarray(np.load(RUNS / f"{t}_test_pred.npy"), dtype=np.float64)  # noqa: E731
    ensemble = np.mean([load(t) for t in TRACKERS], axis=0)
    single, absolute = load("WS_PAIRED"), load("WS_ABS")

    z = np.load(RUNS / "RAW_w_feat_cache.npz")
    lvl_raw = level_scalar(z["val"], z["test"], Sv, val_rows, Yv, fit_subs)
    zp = np.load(RUNS / "WS_PAIRED_feat_cache.npz")
    lvl_pair = level_scalar(zp["val"], zp["test"], Sv, val_rows, Yv, fit_subs)
    pop = {j: float(np.mean([Yv[val_rows][Sv[val_rows] == g, j].mean() for g in fit_subs]))
           for j, _ in TARGETS}
    true_mean = {j: {int(s): float(Y[rows_by[int(s)], j].mean()) for s in subs} for j, _ in TARGETS}

    # (tracker, level source); level source is a per-window scalar dict, a constant, or a rule
    arms = {
        "full system": (ensemble, "raw"),
        ("four trackers" if SINGLE else "one tracker"): (np.mean([load(t) for t in FOUR_TRACKERS], axis=0) if SINGLE else single, "raw"),
        "absolute network as tracker": (absolute, "raw"),
        "level from pairwise encoder": (ensemble, "pair"),
        "population level": (ensemble, "pop"),
        "stage 1 alone": (None, "raw"),
        "no calibration": (absolute, "none"),
        "true level (ceiling)": (ensemble, "true"),
    }

    results = {}
    for j, t in TARGETS:
        per = {}
        for name, (track, source) in arms.items():
            e = np.zeros(len(subs))
            for i, s in enumerate(subs):
                rows = rows_by[int(s)]
                v = []
                for warm in draws[int(s)]:
                    m = np.ones(len(rows), dtype=bool)
                    m[np.searchsorted(rows, np.sort(warm))] = False
                    ev = rows[m]
                    if source == "none":
                        est = track[ev, j]
                    else:
                        if source == "raw":
                            level = lvl_raw[j][warm].mean()
                        elif source == "pair":
                            level = lvl_pair[j][warm].mean()
                        elif source == "pop":
                            level = pop[j]
                        else:
                            level = true_mean[j][int(s)]
                        est = (np.full(len(ev), level) if track is None
                               else track[ev, j] + (level - track[warm, j].mean()))
                    v.append(np.abs(est - Y[ev, j]).mean())
                e[i] = float(np.mean(v))
            per[name] = e
        base = per["full system"]
        print(f"\n--- {t} ---  strict protocol, {M_HEADLINE} warm-up windows, {len(subs)} subjects")
        print("  %-30s %8s  %-16s %10s  %s" % ("arm", "MAE", "95% CI", "vs full", "95% CI"))
        for name, e in per.items():
            d = e - base
            row = {"mae": float(e.mean()), "ci": interval(e, boot)}
            if name != "full system":
                row["minus_full"] = {"delta": float(d.mean()), "ci": interval(d, boot)}
            results.setdefault(t, {})[name] = row
            print("  %-30s %8.3f  [%5.2f, %5.2f]  %s"
                  % (name, row["mae"], *row["ci"],
                     "" if name == "full system" else
                     "%+9.3f  [%+.2f, %+.2f]" % (row["minus_full"]["delta"],
                                                 *row["minus_full"]["ci"])), flush=True)

    (REPORTS / f"s72_two_stage_ablation{SUFFIX}.json").write_text(json.dumps(
        {"analysis_id": "s72_two_stage_ablation",
         "created_utc": datetime.now(timezone.utc).isoformat(),
         "protocol": "strict, 20 warm-up windows, same 50 draws per subject as s70",
         "n_test_subjects": int(len(subs)), "results": results}, indent=2), encoding="utf-8")
    print("\nsaved reports/s72_two_stage_ablation.json")


if __name__ == "__main__":
    main()
