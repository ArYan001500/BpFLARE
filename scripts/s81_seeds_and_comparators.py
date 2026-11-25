"""Round-2 review requests, all post hoc, same strict protocol as s70/s78/s79 (M = 20, 50 draws,
warm-up windows excluded from scoring, stage-1 ridge on RAW_w features fitted on the 194 validation
subjects).

1. Training-seed replication of the matched contrast: residual pairwise tracker (two-stage) against the
   residual absolute network of the same seed, seeds 42, 43, 44, 45 on CalFree (seed 43 from saved
   predictions; its weights were not kept) and seeds 42, 44, 45 on the 108 AAMI subjects.
2. Comparator chosen on absolute error: BW_RESNET, the best single absolute network on validation in s50,
   against the seed-42 two-stage single tracker, on CalFree and AAMI.
3. Same-anchor contrast with an interval: WS_ABS used as the tracker with the identical level against
   WS_PAIRED (CalFree).
4. Absolute ensemble of s50 (greedy picks, equal weights) on the same scored windows, with paired intervals.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from s70_two_stage_headline import (CONVERTED, RUNS, REPORTS, N_DRAWS, N_BOOT, MIN_FIT_WINDOWS, SEED,  # noqa: E402
                                    TARGETS, level_scalar, interval, load_q1_module)
from s78_aami_confirmatory import predict  # noqa: E402

M = 20
ENSEMBLE = ["BW_RESNET", "BALANCED", "BW_TCN", "RAW_s43w", "RAW_w", "BW_TCN", "RAW_seed44"]


def cohort(S, min_windows):
    u, c = np.unique(S, return_counts=True)
    subs = u[c >= min_windows]
    rows_by = {int(s): np.flatnonzero(S == s) for s in subs}
    draws = {int(s): [np.random.default_rng(SEED + 7 * d).choice(rows_by[int(s)], size=M, replace=False)
                      for d in range(N_DRAWS)] for s in subs}
    return subs, rows_by, draws, np.random.default_rng(7).integers(0, len(subs), size=(N_BOOT, len(subs)))


def per_subject(Y, subs, rows_by, draws, j, est_fn):
    out = np.zeros(len(subs))
    for i, s in enumerate(subs):
        rows = rows_by[int(s)]
        v = []
        for warm in draws[int(s)]:
            m = np.ones(len(rows), bool)
            m[np.searchsorted(rows, np.sort(warm))] = False
            ev = rows[m]
            v.append(np.abs(est_fn(ev, warm) - Y[ev, j]).mean())
        out[i] = np.mean(v)
    return out


def two_stage(track, lvl, j):
    return lambda ev, warm: track[ev, j] + (lvl[j][warm].mean() - track[warm, j].mean())


def plain(pred, j):
    return lambda ev, warm: pred[ev, j]


def contrast(a, b, boot):
    d = a - b
    return {"a": float(a.mean()), "b": float(b.mean()), "delta": float(d.mean()), "ci": interval(d, boot),
            "a_ci": interval(a, boot), "b_ci": interval(b, boot), "a_better_in": int((d < 0).sum()), "n": int(len(d))}


def main():
    Yv = np.asarray(np.load(CONVERTED / "train" / "Y.npy"), dtype=np.float64)
    Sv = np.load(CONVERTED / "train" / "S.npy")
    val_rows = np.load(CONVERTED / "train" / "rows_validation.npy")
    counts = {int(g): int(c) for g, c in zip(*np.unique(Sv[val_rows], return_counts=True))}
    fit_subs = np.array([g for g, c in counts.items() if c >= MIN_FIT_WINDOWS])
    zf = np.load(RUNS / "RAW_w_feat_cache.npz")
    res = {"calfree": {}, "aami": {}}

    # ---------------- CalFree
    Y = np.asarray(np.load(CONVERTED / "calfree_test" / "Y.npy"), dtype=np.float64)
    S = np.load(CONVERTED / "calfree_test" / "S.npy")
    subs, rows_by, draws, boot = cohort(S, 100)
    lvl = level_scalar(zf["val"], zf["test"], Sv, val_rows, Yv, fit_subs)
    P = lambda t: np.load(RUNS / f"{t}_test_pred.npy").astype(np.float64)  # noqa: E731
    pairs = {"42": ("WS_PAIRED", "WS_ABS"), "43": ("WS_PAIRED_s43", "WS_ABS_s43"),
             "44": ("WS_PAIRED_s44", "WS_ABS_s44"), "45": ("WS_PAIRED_s45", "WS_ABS_s45")}
    ens = np.mean([P(t) for t in ENSEMBLE], axis=0)
    for j, t in TARGETS:
        r = res["calfree"].setdefault(t, {"seeds": {}})
        for sd, (pa, ab) in pairs.items():
            r["seeds"][sd] = contrast(per_subject(Y, subs, rows_by, draws, j, two_stage(P(pa), lvl, j)),
                                      per_subject(Y, subs, rows_by, draws, j, plain(P(ab), j)), boot)
        single = per_subject(Y, subs, rows_by, draws, j, two_stage(P("WS_PAIRED"), lvl, j))
        four = per_subject(Y, subs, rows_by, draws, j, two_stage(np.mean([P(x) for x in (
            "WS_PAIRED", "WS_PAIRED_s43", "BB_INCEPTION_PAIRED", "BB_TCN_PAIRED")], 0), lvl, j))
        r["vs_abs_error_selected_BW_RESNET"] = contrast(single, per_subject(Y, subs, rows_by, draws, j, plain(P("BW_RESNET"), j)), boot)
        r["same_anchor_pairwise_vs_absolute_tracker"] = contrast(single, per_subject(Y, subs, rows_by, draws, j, two_stage(P("WS_ABS"), lvl, j)), boot)
        e = per_subject(Y, subs, rows_by, draws, j, plain(ens, j))
        r["single_vs_absolute_ensemble"] = contrast(single, e, boot)
        r["four_vs_absolute_ensemble"] = contrast(four, e, boot)
        print("calfree", t, json.dumps({k: (round(v["delta"], 3), [round(x, 3) for x in v["ci"]]) for k, v in r.items() if k != "seeds"}),
              {k: (round(v["a"], 2), round(v["b"], 2), round(v["delta"], 2), [round(x, 2) for x in v["ci"]]) for k, v in r["seeds"].items()}, flush=True)

    # ---------------- AAMI
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q1 = load_q1_module()
    parts = ("aami_cal", "aami_test")
    Y = np.concatenate([np.load(CONVERTED / p / "Y.npy") for p in parts]).astype(np.float64)
    S = np.concatenate([np.load(CONVERTED / p / "S.npy") for p in parts])
    z = np.load(RUNS / "s78_aami_predictions.npz")
    lvl = level_scalar(zf["val"], z["RAW_w_feat"], Sv, val_rows, Yv, fit_subs)
    cache = RUNS / "s81_aami_predictions.npz"
    if cache.is_file():
        extra = dict(np.load(cache))
    else:
        X = np.concatenate([np.load(CONVERTED / p / "X.npy", mmap_mode="r") for p in parts])
        extra = {t: predict(t, X, device, q1) for t in ("WS_PAIRED_s44", "WS_ABS_s44", "WS_PAIRED_s45", "WS_ABS_s45", "BW_RESNET")}
        np.savez(cache, **extra)
    A = {**{k: z[k] for k in ("WS_PAIRED", "WS_ABS")}, **extra}
    A = {k: v.astype(np.float64) for k, v in A.items()}
    subs, rows_by, draws, boot = cohort(S, 40)
    for j, t in TARGETS:
        r = res["aami"].setdefault(t, {"seeds": {}})
        for sd, (pa, ab) in {"42": ("WS_PAIRED", "WS_ABS"), "44": ("WS_PAIRED_s44", "WS_ABS_s44"),
                             "45": ("WS_PAIRED_s45", "WS_ABS_s45")}.items():
            r["seeds"][sd] = contrast(per_subject(Y, subs, rows_by, draws, j, two_stage(A[pa], lvl, j)),
                                      per_subject(Y, subs, rows_by, draws, j, plain(A[ab], j)), boot)
        single = per_subject(Y, subs, rows_by, draws, j, two_stage(A["WS_PAIRED"], lvl, j))
        r["vs_abs_error_selected_BW_RESNET"] = contrast(single, per_subject(Y, subs, rows_by, draws, j, plain(A["BW_RESNET"], j)), boot)
        print("aami", t, {k: (round(v["a"], 2), round(v["b"], 2), round(v["delta"], 2), [round(x, 2) for x in v["ci"]]) for k, v in r["seeds"].items()},
              round(r["vs_abs_error_selected_BW_RESNET"]["delta"], 3), [round(x, 3) for x in r["vs_abs_error_selected_BW_RESNET"]["ci"]], flush=True)
    (REPORTS / "s81_seeds_and_comparators.json").write_text(json.dumps(
        {"analysis_id": "s81_seeds_and_comparators", "created_utc": datetime.now(timezone.utc).isoformat(),
         "post_hoc": True, "results": res}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
