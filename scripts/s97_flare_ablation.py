"""s97: ablation of FLARE on CalFree (exploratory). Every configuration re-chooses its gate threshold and ECG weight
on the validation subjects with the same rules as s93; differences are paired per subject against full FLARE
(50 draws, subject bootstrap 2000). Positive change = higher error than FLARE.
Rows: full FLARE; no gate and no ECG (two-stage); ECG correction for every subject (no gate); population mean as the
level (gate and ECG kept); mean absolute-network output over the warm-up as the level; absolute network as the
tracker (same level, gate and ECG); absolute network alone; FLARE anchor alone (no tracking); oracle: true subject
mean as the level (non-deployable). A gate without the ECG term changes nothing (both branches equal), so it is not
a separate row.
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

M, N_DRAWS, N_BOOT, SEED = 20, 50, 2000, 42
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 1e4, 1e5)
f64 = lambda a: np.asarray(a, dtype=np.float64)  # noqa: E731
rng = np.random.default_rng(7)
Y, S = f64(np.load(C / "train" / "Y.npy")), np.load(C / "train" / "S.npy")
tr, va = np.load(C / "train" / "rows_train.npy"), np.load(C / "train" / "rows_validation.npy")
Yt, St = f64(np.load(C / "calfree_test" / "Y.npy")), np.load(C / "calfree_test" / "S.npy")
zc = np.load(R / "RAW_w_feat_cache.npz")
coh = {"val": (Y[va], S[va]), "cf": (Yt, St)}
TP = {"val": f64(np.load(R / "WS_PAIRED_val_pred_s89.npy")), "cf": f64(np.load(R / "WS_PAIRED_test_pred.npy"))}
TA = {"val": f64(np.load(R / "WS_ABS_val_pred_s89.npy")), "cf": f64(np.load(R / "WS_ABS_test_pred.npy"))}
feats = {"val": zc["val"], "cf": zc["test"]}
E_win = {"val": np.load(OUT93 / "ecg_window_features_validation.npy"), "cf": np.load(OUT93 / "ecg_window_features_calfree.npy")}
E_tr_win = np.load(OUT93 / "ecg_window_features_train_fit.npy")
tr_subs = np.array([s for s in np.unique(S[tr]) if (S[tr] == s).sum() >= 20])
med = np.nanmedian(E_tr_win, 0)
E_tr = np.stack([np.nanmedian(E_tr_win[i * 20:(i + 1) * 20], 0) for i in range(len(tr_subs))])
E_tr = np.where(np.isnan(E_tr), med, E_tr)
subs = {"val": np.array([s for s in np.unique(S[va]) if (S[va] == s).sum() >= 40]), "cf": np.unique(St)}
rows_of = {k: {int(s): np.flatnonzero(coh[k][1] == s) for s in subs[k]} for k in coh}
draws = {k: {s: [np.random.default_rng(SEED + 7 * d).choice(r, M, replace=False) for d in range(N_DRAWS)] for s, r in rows_of[k].items()} for k in coh}


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
    Yv, Sv = coh["val"]
    vs = subs["val"]
    G = np.stack([feats["val"][Sv == s].mean(0) for s in vs]).astype(np.float64)
    yv = np.array([Yv[Sv == s, j].mean() for s in vs])
    pop = float(Y[tr][:, j].mean())
    score = {"cf": fit_rule(G, yv)(feats["cf"]), "val": np.empty(len(Sv))}
    folds = np.random.default_rng(SEED).permutation(len(vs)) % 4
    for k in range(4):
        m = np.isin(Sv, vs[folds == k])
        score["val"][m] = fit_rule(G[folds != k], yv[folds != k])(feats["val"][m])
    sc = StandardScaler().fit(E_tr)
    reg = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(sc.transform(E_tr), np.array([Y[tr][S[tr] == s, j].mean() for s in tr_subs]))
    # per subject/draw quantities shared by all rows
    Q = {}
    for key in coh:
        Yx, Sx = coh[key]
        q = {"v": [], "L_ridge": [], "L_absmean": [], "E": [], "truth": [], "sc": [], "w": []}
        for s in subs[key]:
            r = rows_of[key][int(s)]
            for name in q:
                q[name].append([])
            for w in draws[key][int(s)]:
                q["v"][-1].append(TP[key][w, j].std())
                q["L_ridge"][-1].append(score[key][w].mean())
                q["L_absmean"][-1].append(TA[key][w, j].mean())
                fw = np.nanmedian(E_win[key][w], 0)
                q["E"][-1].append(np.where(np.isnan(fw), med, fw))
                q["w"][-1].append(w)
                q["sc"][-1].append(np.setdiff1d(r, w))
            q["truth"][-1] = Yx[r, j].mean()
        Fm = np.array(q["E"])
        q["E"] = reg.predict(sc.transform(Fm.reshape(-1, Fm.shape[-1]))).reshape(Fm.shape[:2])
        for name in ("v", "L_ridge", "L_absmean"):
            q[name] = np.array(q[name])
        q["truth"] = np.array(q["truth"])
        Q[key] = q

    def system(level, tracker, gate=True, ecg=True, ecg_all=False, track=True, oracle=False):
        """Returns per-subject CalFree MAE of one configuration (gate/ECG weights chosen on validation)."""
        def L_of(key):
            q = Q[key]
            if oracle:
                return np.repeat(q["truth"][:, None], N_DRAWS, 1)
            return {"ridge": q["L_ridge"], "absmean": q["L_absmean"], "pop": np.full(q["v"].shape, pop)}[level]
        qv = Q["val"]
        Lv = L_of("val")
        tau = float(np.median(qv["v"]))
        e_mean = float(qv["E"].mean())
        mask_v = np.ones_like(qv["v"], bool) if (ecg_all or not gate) else qv["v"] <= tau
        b = 0.0
        if ecg and not oracle:
            b = float(min(np.linspace(0, 1.5, 31), key=lambda b: np.abs(Lv + b * (qv["E"] - e_mean) - qv["truth"][:, None])[mask_v].mean()))
        q = Q["cf"]
        L = L_of("cf")
        mask = np.ones_like(q["v"], bool) if (ecg_all or not gate) else q["v"] <= tau
        anc = L + np.where(mask, b * (q["E"] - e_mean), 0.0)
        Tt = {"paired": TP["cf"], "abs": TA["cf"]}[tracker]
        out = []
        for i, s in enumerate(subs["cf"]):
            e = []
            for d in range(N_DRAWS):
                sc_, w = q["sc"][i][d], q["w"][i][d]
                pred = (Tt[sc_, j] - Tt[w, j].mean() + anc[i, d]) if track else np.full(len(sc_), anc[i, d])
                e.append(np.abs(pred - Yt[sc_, j]).mean())
            out.append(np.mean(e))
        return np.array(out), round(b, 2)
    full, bf = system("ridge", "paired")
    rows = {"FLARE (full)": (full, bf),
            "No gate, no ECG (two-stage)": system("ridge", "paired", gate=False, ecg=False),
            "ECG for every subject, no gate": system("ridge", "paired", ecg_all=True),
            "Population mean as level": system("pop", "paired"),
            "Mean absolute-network output as level": system("absmean", "paired"),
            "Absolute network as tracker": system("ridge", "abs"),
            "FLARE anchor alone, no tracking": system("ridge", "paired", track=False),
            "True subject mean as level (oracle)": system("ridge", "paired", ecg=False, oracle=True)}
    absn = []
    for i, s in enumerate(subs["cf"]):
        absn.append(np.mean([np.abs(TA["cf"][sc_, j] - Yt[sc_, j]).mean() for sc_ in Q["cf"]["sc"][i]]))
    rows["Absolute network alone"] = (np.array(absn), 0.0)
    report[t] = {k: {"MAE": round(float(v[0].mean()), 3), "change_vs_FLARE": ci(v[0] - full), "ecg_weight": v[1]} for k, v in rows.items()}
    print(t, json.dumps(report[t], indent=1), flush=True)
(ROOT / "reports" / "s97_flare_ablation.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
