"""s101: label-free fluctuation index in deployable form. For each subject and draw d (seed 42 + 7d, 50 draws),
20 random warm-up windows; index = SD of the model output over those windows. Pearson r across subjects between the
index (one draw at a time, then r averaged over draws; also the index averaged over draws) and the true SD of the
reference over all windows of the subject. Pairwise tracker vs absolute network, ResNet seed 42. Subjects >= 40 windows.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
C, R = ROOT / "converted", ROOT / "runs" / "s4_pat_control"
f64 = lambda a: np.asarray(a, dtype=np.float64)  # noqa: E731
Ya = np.concatenate([f64(np.load(C / p / "Y.npy")) for p in ("aami_cal", "aami_test")])
Sa = np.concatenate([np.load(C / p / "S.npy") for p in ("aami_cal", "aami_test")])
Yt, St = f64(np.load(C / "calfree_test" / "Y.npy")), np.load(C / "calfree_test" / "S.npy")
za = np.load(R / "s78_aami_predictions.npz")
M, NDRAW = 20, 50
out = {"protocol": __doc__}
for cname, Y, S, get in (("calfree", Yt, St, lambda t: f64(np.load(R / f"{t}_test_pred.npy"))),
                         ("aami", Ya, Sa, lambda t: f64(za[t]))):
    out[cname] = {}
    subs = [s for s in np.unique(S) if (S == s).sum() >= 40]
    idx = {s: np.flatnonzero(S == s) for s in subs}
    for tag in ("WS_PAIRED", "WS_ABS"):
        P = get(tag)
        for j, t in ((0, "SBP"), (1, "DBP")):
            true_sd = np.array([Y[idx[s], j].std() for s in subs])
            V = np.zeros((NDRAW, len(subs)))
            for d in range(NDRAW):
                rng = np.random.default_rng(42 + 7 * d)
                for k, s in enumerate(subs):
                    w = rng.choice(idx[s], M, replace=False)
                    V[d, k] = P[w, j].std()
            r_draw = [np.corrcoef(V[d], true_sd)[0, 1] for d in range(NDRAW)]
            out[cname].setdefault(tag, {})[t] = {
                "n_subjects": len(subs),
                "r_single_draw_mean": round(float(np.mean(r_draw)), 3),
                "r_single_draw_range": [round(float(min(r_draw)), 3), round(float(max(r_draw)), 3)],
                "r_index_averaged_over_draws": round(float(np.corrcoef(V.mean(0), true_sd)[0, 1]), 3),
                "median_index_over_true_sd": round(float(np.median(V.mean(0) / true_sd)), 2)}
print(json.dumps({k: v for k, v in out.items() if k != "protocol"}, indent=1))
(ROOT / "reports" / "s101_fluctuation_index_warmup.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
