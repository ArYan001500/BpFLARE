"""BSPC asks that sex/gender be considered in reporting. Post hoc, descriptive: cohort demographics and the
primary single-tracker contrast stratified by the sex recorded in PulseDB (VitalDB field Gender: M/F),
on CalFree and on the 108 confirmatory subjects. Same protocol as s79/s81 (M = 20, 50 draws, warm-up
windows not scored); intervals are subject bootstraps within each sex; the interaction is the
difference between the female and male mean differences, bootstrapped within sex.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from s70_two_stage_headline import CONVERTED, RUNS, REPORTS, MIN_FIT_WINDOWS, N_BOOT, level_scalar  # noqa: E402
from s81_seeds_and_comparators import cohort, per_subject, two_stage, plain  # noqa: E402


def summarise(S, G, D, subs):
    first = {int(s): np.flatnonzero(S == s)[0] for s in subs}
    g = np.array([G[first[int(s)]] for s in subs])
    d = np.array([D[first[int(s)]] for s in subs], dtype=np.float64)
    return g, {"n": int(len(subs)), "female": int((g == 0).sum()), "male": int((g == 1).sum()),
               "unknown": int((g < 0).sum()),
               "age_mean": float(np.nanmean(d[:, 0])), "age_sd": float(np.nanstd(d[:, 0], ddof=1)),
               "bmi_mean": float(np.nanmean(d[:, 1])), "bmi_sd": float(np.nanstd(d[:, 1], ddof=1))}


def stratify(a, b, g):
    rng = np.random.default_rng(7)
    out, boots = {}, {}
    for code, name in ((0, "female"), (1, "male")):
        m = g == code
        d = a[m] - b[m]
        bt = d[rng.integers(0, m.sum(), size=(N_BOOT, m.sum()))].mean(1)
        boots[name] = bt
        out[name] = {"n": int(m.sum()), "two_stage": float(a[m].mean()), "absolute": float(b[m].mean()),
                     "delta": float(d.mean()), "ci": [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
                     "improved": int((d < 0).sum())}
    inter = boots["female"] - boots["male"]
    out["female_minus_male"] = {"delta": out["female"]["delta"] - out["male"]["delta"],
                                "ci": [float(np.percentile(inter, 2.5)), float(np.percentile(inter, 97.5))]}
    return out


def main():
    Yv = np.asarray(np.load(CONVERTED / "train" / "Y.npy"), dtype=np.float64)
    Sv = np.load(CONVERTED / "train" / "S.npy")
    val_rows = np.load(CONVERTED / "train" / "rows_validation.npy")
    counts = {int(g): int(c) for g, c in zip(*np.unique(Sv[val_rows], return_counts=True))}
    fit_subs = np.array([g for g, c in counts.items() if c >= MIN_FIT_WINDOWS])
    zf = np.load(RUNS / "RAW_w_feat_cache.npz")
    res = {}

    # training and validation demographics
    Gt, Dt = np.load(CONVERTED / "train" / "G.npy"), np.load(CONVERTED / "train" / "D.npy")
    res["train_and_validation"] = summarise(Sv, Gt, Dt, np.unique(Sv))[1]

    Y = np.asarray(np.load(CONVERTED / "calfree_test" / "Y.npy"), dtype=np.float64)
    S = np.load(CONVERTED / "calfree_test" / "S.npy")
    G, D = np.load(CONVERTED / "calfree_test" / "G.npy"), np.load(CONVERTED / "calfree_test" / "D.npy")
    subs, rows_by, draws, _ = cohort(S, 100)
    g, res["calfree_demographics"] = summarise(S, G, D, subs)
    lvl = level_scalar(zf["val"], zf["test"], Sv, val_rows, Yv, fit_subs)
    P = lambda t: np.load(RUNS / f"{t}_test_pred.npy").astype(np.float64)  # noqa: E731
    for j, t in ((0, "SBP"), (1, "DBP")):
        a = per_subject(Y, subs, rows_by, draws, j, two_stage(P("WS_PAIRED"), lvl, j))
        b = per_subject(Y, subs, rows_by, draws, j, plain(P("WS_ABS"), j))
        res.setdefault("calfree", {})[t] = stratify(a, b, g)

    parts = ("aami_cal", "aami_test")
    Y = np.concatenate([np.load(CONVERTED / p / "Y.npy") for p in parts]).astype(np.float64)
    S = np.concatenate([np.load(CONVERTED / p / "S.npy") for p in parts])
    G = np.concatenate([np.load(CONVERTED / p / "G.npy") for p in parts])
    D = np.concatenate([np.load(CONVERTED / p / "D.npy") for p in parts])
    z = np.load(RUNS / "s78_aami_predictions.npz")
    lvl = level_scalar(zf["val"], z["RAW_w_feat"], Sv, val_rows, Yv, fit_subs)
    subs, rows_by, draws, _ = cohort(S, 40)
    g, res["aami_demographics"] = summarise(S, G, D, subs)
    for j, t in ((0, "SBP"), (1, "DBP")):
        a = per_subject(Y, subs, rows_by, draws, j, two_stage(z["WS_PAIRED"].astype(np.float64), lvl, j))
        b = per_subject(Y, subs, rows_by, draws, j, plain(z["WS_ABS"].astype(np.float64), j))
        res.setdefault("aami", {})[t] = stratify(a, b, g)

    print(json.dumps(res, indent=1))
    (REPORTS / "s83_sex_and_demographics.json").write_text(json.dumps(
        {"analysis_id": "s83_sex_and_demographics", "created_utc": datetime.now(timezone.utc).isoformat(),
         "post_hoc": True, "results": res}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
