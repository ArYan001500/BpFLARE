"""Data for Fig. 1 (FLARE overview): one CalFree subject chosen for illustration (subject 52: within-subject r and
amplitude ratio close to the cohort means of Table 6), draw 0 of the main protocol (seed 42, 20 warm-up windows).
Writes runs/fig1/fig1_data.npz."""
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
C, R = ROOT / "converted", ROOT / "runs" / "s4_pat_control"
OUT = ROOT / "runs" / "fig1"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules.setdefault("network_loader", types.SimpleNamespace(load_q1_module=None))
from s29_level_information import pick_alpha, ridge  # noqa: E402

SUBJ, M, SEED = 52, 20, 42
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 1e4, 1e5)
f64 = lambda a: np.asarray(a, dtype=np.float64)  # noqa: E731
Y, S = f64(np.load(C / "train" / "Y.npy")), np.load(C / "train" / "S.npy")
va, tr = np.load(C / "train" / "rows_validation.npy"), np.load(C / "train" / "rows_train.npy")
Yt, St = f64(np.load(C / "calfree_test" / "Y.npy")), np.load(C / "calfree_test" / "S.npy")
zc = np.load(R / "RAW_w_feat_cache.npz")
Pp, Pa = f64(np.load(R / "WS_PAIRED_test_pred.npy")), f64(np.load(R / "WS_ABS_test_pred.npy"))
Sv, Yv = S[va], Y[va]
vs = np.array([s for s in np.unique(Sv) if (Sv == s).sum() >= 40])
G = np.stack([zc["val"][Sv == s].mean(0) for s in vs])
yv = np.array([Yv[Sv == s, 0].mean() for s in vs])
mu, sd = G.mean(0), G.std(0)
sd[sd < 1e-9] = 1.0
a = pick_alpha((G - mu) / sd, yv, ALPHAS, seed=SEED)
cmu, w, b = ridge((G - mu) / sd, yv, a)
rows = np.flatnonzero(St == SUBJ)
warm = np.random.default_rng(SEED).choice(rows, M, replace=False)
score = ((zc["test"][rows] - mu) / sd - cmu) @ w + b
L_hat = float((((zc["test"][warm] - mu) / sd - cmu) @ w + b).mean())
scored = np.setdiff1d(rows, warm)
flare = Pp[scored, 0] + (L_hat - Pp[warm, 0].mean())
X = np.load(C / "calfree_test" / "X.npy", mmap_mode="r")
np.savez(OUT / "fig1_data.npz", ecg=np.asarray(X[warm[0], 0]), ppg=np.asarray(X[warm[0], 1]),
         ecg2=np.asarray(X[warm[1], 0]), ppg2=np.asarray(X[warm[1], 1]),
         L_hat=L_hat, true_mean=float(Yt[rows, 0].mean()), pop_mean=float(np.median(Y[tr, 0])),
         y=Yt[scored, 0], flare=flare, abs_net=Pa[scored, 0],
         dev_ref=Yt[rows, 0] - Yt[rows, 0].mean(), dev_pair=Pp[rows, 0] - Pp[rows, 0].mean(),
         dev_abs=Pa[rows, 0] - Pa[rows, 0].mean(), alpha=a)
print("subject", SUBJ, "windows", len(rows), "L_hat", round(L_hat, 1), "true", round(float(Yt[rows, 0].mean()), 1),
      "train median", round(float(np.median(Y[tr, 0])), 1), "alpha", a,
      "MAE FLARE", round(float(np.abs(flare - Yt[scored, 0]).mean()), 2),
      "MAE abs", round(float(np.abs(Pa[scored, 0] - Yt[scored, 0]).mean()), 2))
