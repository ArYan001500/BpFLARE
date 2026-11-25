"""s96: FLARE across warm-up lengths M = 1, 3, 5, 6, 10, 20, 60, 120 windows (10 s to 20 min), CalFree.
For every M the gate threshold and the ECG weight are re-chosen on the validation subjects with at least M + 20
windows (same rules as s93); the absolute network is scored on the same windows. With M = 1 the fluctuation index
is zero by construction, so every subject falls in the low branch. 50 draws, subject bootstrap 2000. Exploratory.
"""
import json
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
C, R, OUT93 = ROOT / "converted", ROOT / "runs" / "s4_pat_control", ROOT / "runs" / "s93"
sys.path.insert(0, str(ROOT / "scripts"))
import bp_networks as q1  # noqa: E402
sys.modules.setdefault("network_loader", types.SimpleNamespace(load_q1_module=lambda *a: q1))
from s29_level_information import pick_alpha, ridge  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

MS, N_DRAWS, N_BOOT, SEED = (1, 3, 5, 6, 10, 20, 60, 120), 50, 2000, 42
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 1e4, 1e5)
f64 = lambda a: np.asarray(a, dtype=np.float64)  # noqa: E731
rng = np.random.default_rng(7)
Y, S = f64(np.load(C / "train" / "Y.npy")), np.load(C / "train" / "S.npy")
tr, va = np.load(C / "train" / "rows_train.npy"), np.load(C / "train" / "rows_validation.npy")
Yt, St = f64(np.load(C / "calfree_test" / "Y.npy")), np.load(C / "calfree_test" / "S.npy")
zc = np.load(R / "RAW_w_feat_cache.npz")
cohorts = {"val": (Y[va], S[va]), "calfree": (Yt, St)}
T = {"val": f64(np.load(R / "WS_PAIRED_val_pred_s89.npy")), "calfree": f64(np.load(R / "WS_PAIRED_test_pred.npy"))}
Acf = f64(np.load(R / "WS_ABS_test_pred.npy"))
feats = {"val": zc["val"], "calfree": zc["test"]}
E_win = {"val": np.load(OUT93 / "ecg_window_features_validation.npy"), "calfree": np.load(OUT93 / "ecg_window_features_calfree.npy")}
E_tr_win = np.load(OUT93 / "ecg_window_features_train_fit.npy")
tr_subs = np.array([s for s in np.unique(S[tr]) if (S[tr] == s).sum() >= 20])
med = np.nanmedian(E_tr_win, 0)
E_tr = np.stack([np.nanmedian(E_tr_win[i * 20:(i + 1) * 20], 0) for i in range(len(tr_subs))])
E_tr = np.where(np.isnan(E_tr), med, E_tr)
vs_all = np.array([s for s in np.unique(S[va]) if (S[va] == s).sum() >= 40])
ts = np.unique(St)


def fit_rule(G, y):
    mu, sd = G.mean(0), G.std(0)
    sd[sd < 1e-9] = 1.0
    a = pick_alpha((G - mu) / sd, y, ALPHAS, seed=SEED)
    cmu, w, b = ridge((G - mu) / sd, y, a)
    return lambda F: ((np.asarray(F, np.float64) - mu) / sd - cmu) @ w + b


def ci(x):
    i = rng.integers(0, len(x), (N_BOOT, len(x)))
    return [round(float(x.mean()), 3), [round(float(v), 3) for v in np.percentile(x[i].mean(1), [2.5, 97.5])]]


report = {"protocol": __doc__}
for j, t in ((0, "SBP"), (1, "DBP")):
    Yv, Sv = cohorts["val"]
    G = np.stack([feats["val"][Sv == s].mean(0) for s in vs_all]).astype(np.float64)
    yv = np.array([Yv[Sv == s, j].mean() for s in vs_all])
    score = {"calfree": fit_rule(G, yv)(feats["calfree"]), "val": np.empty(len(Sv))}
    folds = np.random.default_rng(SEED).permutation(len(vs_all)) % 4
    for k in range(4):
        m = np.isin(Sv, vs_all[folds == k])
        score["val"][m] = fit_rule(G[folds != k], yv[folds != k])(feats["val"][m])
    sc = StandardScaler().fit(E_tr)
    reg = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(sc.transform(E_tr), np.array([Y[tr][S[tr] == s, j].mean() for s in tr_subs]))
    report[t] = {}
    for Mw in MS:
        def per(key, subs):
            Yx, Sx = cohorts[key]
            V, L, Fm, res, tru, ab = [], [], [], [], [], []
            for s in subs:
                r = np.flatnonzero(Sx == s)
                v_, l_, f_, r_, a_ = [], [], [], [], []
                for d in range(N_DRAWS):
                    w = np.random.default_rng(SEED + 7 * d).choice(r, Mw, replace=False)
                    sc_ = np.setdiff1d(r, w)
                    v_.append(T[key][w, j].std())
                    l_.append(score[key][w].mean())
                    fw = np.nanmedian(E_win[key][w], 0) if Mw > 1 else E_win[key][w][0]
                    f_.append(np.where(np.isnan(fw), med, fw))
                    r_.append(T[key][sc_, j] - T[key][w, j].mean() - Yx[sc_, j])
                    if key == "calfree":
                        a_.append(np.abs(Acf[sc_, j] - Yx[sc_, j]).mean())
                V.append(v_), L.append(l_), Fm.append(f_), res.append(r_), tru.append(Yx[r, j].mean())
                ab.append(np.mean(a_) if a_ else np.nan)
            Fm = np.array(Fm)
            E = reg.predict(sc.transform(Fm.reshape(-1, Fm.shape[-1]))).reshape(Fm.shape[:2])
            return np.array(V), np.array(L), E, res, np.array(tru), np.array(ab)
        vs = np.array([s for s in vs_all if (Sv == s).sum() >= Mw + 20])
        V, L, E, _, tru, _ = per("val", vs)
        tau, e_mean = float(np.median(V)), float(E.mean())
        b = float(min(np.linspace(0, 1.5, 31), key=lambda b: np.abs(L + b * (E - e_mean) - tru[:, None])[V <= tau].mean()))
        V, L, E, res, tru, ab = per("calfree", ts)
        anc = L + np.where(V <= tau, b * (E - e_mean), 0.0)
        new = np.array([[np.abs(anc[i, d] + res[i][d]).mean() for d in range(N_DRAWS)] for i in range(len(ts))]).mean(1)
        cur = np.array([[np.abs(L[i, d] + res[i][d]).mean() for d in range(N_DRAWS)] for i in range(len(ts))]).mean(1)
        report[t][f"{Mw * 10}s"] = {"M": Mw, "tau": round(tau, 3), "b": round(b, 2), "n_val_used": int(len(vs)),
                                    "FLARE": ci(new), "no_gate": ci(cur), "absolute_same_windows": round(float(ab.mean()), 3),
                                    "FLARE_minus_no_gate": ci(new - cur)}
        print(t, Mw, json.dumps(report[t][f"{Mw * 10}s"]), flush=True)
(ROOT / "reports" / "s96_flare_warmup.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
