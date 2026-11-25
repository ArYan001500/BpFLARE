"""DBP with the penalty selected under fold-wise scaling (1e4 instead of 1e3), single tracker, CalFree and
AAMI, strict protocol; post hoc sensitivity to s82."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from s29_level_information import ridge  # noqa: E402
from s70_two_stage_headline import CONVERTED, RUNS, REPORTS, MIN_FIT_WINDOWS  # noqa: E402
from s81_seeds_and_comparators import cohort, per_subject, two_stage, plain, contrast  # noqa: E402

Yv = np.asarray(np.load(CONVERTED / "train" / "Y.npy"), dtype=np.float64)
Sv = np.load(CONVERTED / "train" / "S.npy")
val_rows = np.load(CONVERTED / "train" / "rows_validation.npy")
counts = {int(g): int(c) for g, c in zip(*np.unique(Sv[val_rows], return_counts=True))}
fit_subs = np.array([g for g, c in counts.items() if c >= MIN_FIT_WINDOWS])
zf = np.load(RUNS / "RAW_w_feat_cache.npz")
G = np.stack([zf["val"][Sv[val_rows] == g].mean(0) for g in fit_subs]).astype(np.float64)
mu, sd = G.mean(0), G.std(0)
sd[sd < 1e-9] = 1.0
y = np.array([Yv[val_rows][Sv[val_rows] == g, 1].mean() for g in fit_subs])


def level(ft, a):
    cmu, w, b = ridge((G - mu) / sd, y, a)
    return {1: ((ft.astype(np.float64) - mu) / sd - cmu) @ w + b}


out = {}
Y = np.asarray(np.load(CONVERTED / "calfree_test" / "Y.npy"), dtype=np.float64)
S = np.load(CONVERTED / "calfree_test" / "S.npy")
subs, rows_by, draws, boot = cohort(S, 100)
P = lambda t: np.load(RUNS / f"{t}_test_pred.npy").astype(np.float64)  # noqa: E731
base = per_subject(Y, subs, rows_by, draws, 1, plain(P("WS_ABS"), 1))
for a in (1e3, 1e4):
    out[f"calfree_alpha_{a:g}"] = contrast(per_subject(Y, subs, rows_by, draws, 1, two_stage(P("WS_PAIRED"), level(zf["test"], a), 1)), base, boot)
parts = ("aami_cal", "aami_test")
Y = np.concatenate([np.load(CONVERTED / p / "Y.npy") for p in parts]).astype(np.float64)
S = np.concatenate([np.load(CONVERTED / p / "S.npy") for p in parts])
z = np.load(RUNS / "s78_aami_predictions.npz")
subs, rows_by, draws, boot = cohort(S, 40)
base = per_subject(Y, subs, rows_by, draws, 1, plain(z["WS_ABS"].astype(np.float64), 1))
for a in (1e3, 1e4):
    out[f"aami_alpha_{a:g}"] = contrast(per_subject(Y, subs, rows_by, draws, 1, two_stage(z["WS_PAIRED"].astype(np.float64), level(z["RAW_w_feat"], a), 1)), base, boot)
for k, v in out.items():
    print(k, round(v["a"], 3), round(v["b"], 3), round(v["delta"], 3), [round(x, 3) for x in v["ci"]], v["a_better_in"])
(REPORTS / "s82b_dbp_alpha_sensitivity.json").write_text(json.dumps(
    {"analysis_id": "s82b_dbp_alpha_sensitivity", "created_utc": datetime.now(timezone.utc).isoformat(),
     "post_hoc": True, "results": out}, indent=2), encoding="utf-8")
