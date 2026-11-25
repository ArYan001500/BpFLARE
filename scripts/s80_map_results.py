"""Mean arterial pressure, post hoc (not in the pre-registered protocol): the networks output MAP as a third
target but the manuscript reported only SBP and DBP. Same frozen systems and protocol as s70/s78/s79,
with a stage-1 ridge for MAP fitted on the same 194 validation subjects (same alpha grid, same seed).
Also window-level detection of MAP < 65 mmHg (descriptive; windows are not in temporal order, so this
is not episode detection)."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from s70_two_stage_headline import (CONVERTED, RUNS, REPORTS, MIN_TEST_WINDOWS, MIN_FIT_WINDOWS, N_BOOT,  # noqa: E402
                                    N_DRAWS, ALPHAS, SEED, interval)
from s29_level_information import pick_alpha, ridge  # noqa: E402

J, M, HYPO = 2, 20, 65.0


def map_level(fv, ft, Sv, val_rows, Yv, fit_subs):
    G = np.stack([fv[Sv[val_rows] == g].mean(0) for g in fit_subs]).astype(np.float64)
    mu, sd = G.mean(0), G.std(0)
    sd[sd < 1e-9] = 1.0
    yv = np.array([Yv[val_rows][Sv[val_rows] == g, J].mean() for g in fit_subs])
    a = pick_alpha((G - mu) / sd, yv, ALPHAS, seed=SEED)
    cmu, w, b = ridge((G - mu) / sd, yv, a)
    return ((ft.astype(np.float64) - mu) / sd - cmu) @ w + b


def evaluate(Y, S, trackers, base, lvl, min_windows):
    u, c = np.unique(S, return_counts=True)
    subs = u[c >= min_windows]
    rows_by = {int(s): np.flatnonzero(S == s) for s in subs}
    boot = np.random.default_rng(7).integers(0, len(subs), size=(N_BOOT, len(subs)))
    per = {k: np.zeros(len(subs)) for k in list(trackers) + ["absolute"]}
    hyp = {k: [] for k in per}
    truth = []
    for i, s in enumerate(subs):
        rows = rows_by[int(s)]
        acc = {k: [] for k in per}
        for d in range(N_DRAWS):
            warm = np.random.default_rng(SEED + 7 * d).choice(rows, size=M, replace=False)
            m = np.ones(len(rows), bool)
            m[np.searchsorted(rows, np.sort(warm))] = False
            ev = rows[m]
            for k, tr in trackers.items():
                est = tr[ev, J] + (lvl[warm].mean() - tr[warm, J].mean())
                acc[k].append(np.abs(est - Y[ev, J]).mean())
                hyp[k].append(est < HYPO)
            acc["absolute"].append(np.abs(base[ev, J] - Y[ev, J]).mean())
            hyp["absolute"].append(base[ev, J] < HYPO)
            truth.append(Y[ev, J] < HYPO)
        for k in per:
            per[k][i] = np.mean(acc[k])
    truth = np.concatenate(truth)
    out = {"n_subjects": int(len(subs)), "prevalence_map_below_65": float(truth.mean())}
    for k in per:
        p = np.concatenate(hyp[k])
        out[k] = {"mae": float(per[k].mean()), "ci": interval(per[k], boot),
                  "map65": {"sensitivity": float((p & truth).sum() / truth.sum()),
                            "specificity": float((~p & ~truth).sum() / (~truth).sum()),
                            "ppv": float((p & truth).sum() / max(p.sum(), 1))}}
        if k != "absolute":
            dd = per[k] - per["absolute"]
            out[k]["minus_absolute"] = {"delta": float(dd.mean()), "ci": interval(dd, boot)}
    return out


def main():
    Yv = np.asarray(np.load(CONVERTED / "train" / "Y.npy"), dtype=np.float64)
    Sv = np.load(CONVERTED / "train" / "S.npy")
    val_rows = np.load(CONVERTED / "train" / "rows_validation.npy")
    counts = {int(g): int(c) for g, c in zip(*np.unique(Sv[val_rows], return_counts=True))}
    fit_subs = np.array([g for g, c in counts.items() if c >= MIN_FIT_WINDOWS])
    zf = np.load(RUNS / "RAW_w_feat_cache.npz")
    res = {}
    # CalFree
    Y = np.asarray(np.load(CONVERTED / "calfree_test" / "Y.npy"), dtype=np.float64)
    S = np.load(CONVERTED / "calfree_test" / "S.npy")
    load = lambda t: np.load(RUNS / f"{t}_test_pred.npy").astype(np.float64)  # noqa: E731
    four = ("WS_PAIRED", "WS_PAIRED_s43", "BB_INCEPTION_PAIRED", "BB_TCN_PAIRED")
    lvl = map_level(zf["val"], zf["test"], Sv, val_rows, Yv, fit_subs)
    res["calfree"] = evaluate(Y, S, {"single": load("WS_PAIRED"), "four": np.mean([load(t) for t in four], 0)},
                              load("WS_ABS"), lvl, MIN_TEST_WINDOWS)
    print("calfree", json.dumps(res["calfree"]), flush=True)
    # AAMI
    parts = ("aami_cal", "aami_test")
    Y = np.concatenate([np.load(CONVERTED / p / "Y.npy") for p in parts]).astype(np.float64)
    S = np.concatenate([np.load(CONVERTED / p / "S.npy") for p in parts])
    z = np.load(RUNS / "s78_aami_predictions.npz")
    lvl = map_level(zf["val"], z["RAW_w_feat"], Sv, val_rows, Yv, fit_subs)
    three = np.mean([z[t] for t in ("WS_PAIRED", "BB_INCEPTION_PAIRED", "BB_TCN_PAIRED")], 0).astype(np.float64)
    res["aami"] = evaluate(Y, S, {"single": z["WS_PAIRED"].astype(np.float64), "three": three},
                           z["WS_ABS"].astype(np.float64), lvl, 40)
    print("aami", json.dumps(res["aami"]), flush=True)
    (REPORTS / "s80_map_results.json").write_text(json.dumps(
        {"analysis_id": "s80_map_results", "created_utc": datetime.now(timezone.utc).isoformat(),
         "post_hoc": True, "results": res}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
