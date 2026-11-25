"""s102: contribution of the hand-crafted ECG/PPG features to the level, in deployable form (no gate).
Level = stage-1 ridge level L + b * (E - mean E), for every subject and draw; E is a ridge level from a subset of
the 8 window features (median over the 20 warm-up windows) fitted on the training subjects; b is chosen on the
validation subject-draws (grid 0..1.5) and frozen. Feature sets: all 8; drop one; single feature. Score: change in
the two-stage MAE against the system without ECG (paired per subject, 50 draws, subject bootstrap 2000).
CalFree exploratory; AAMI (108 subjects) used for nothing else. Same windows, draws and rules as s93 --no-gate.
Standardized ridge coefficients of the 8-feature ECG level are also reported.
"""
import json
import sys
import types

import numpy as np

sys.argv.append("--no-gate")
import s93_fluctuation_gated_system as s93  # noqa: E402

ROOT, C, R = s93.ROOT, s93.C, s93.R
NAMES, M, ND, NB, SEED = s93.ECG_NAMES, s93.M, s93.N_DRAWS, s93.N_BOOT, s93.SEED


def main():
    sys.path.insert(0, str(ROOT / "scripts"))
    import bp_networks as q1
    sys.modules.setdefault("network_loader", types.SimpleNamespace(load_q1_module=lambda *a: q1))
    from s29_level_information import pick_alpha, ridge
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    f64 = lambda a: np.asarray(a, dtype=np.float64)  # noqa: E731
    rng = np.random.default_rng(7)
    Y, S = f64(np.load(C / "train" / "Y.npy")), np.load(C / "train" / "S.npy")
    tr, va = np.load(C / "train" / "rows_train.npy"), np.load(C / "train" / "rows_validation.npy")
    Yt, St = f64(np.load(C / "calfree_test" / "Y.npy")), np.load(C / "calfree_test" / "S.npy")
    Ya = np.concatenate([f64(np.load(C / p / "Y.npy")) for p in ("aami_cal", "aami_test")])
    Sa = np.concatenate([np.load(C / p / "S.npy") for p in ("aami_cal", "aami_test")])
    za, zc = np.load(R / "s78_aami_predictions.npz"), np.load(R / "RAW_w_feat_cache.npz")
    T = {"val": f64(np.load(R / "WS_PAIRED_val_pred_s89.npy")), "calfree": f64(np.load(R / "WS_PAIRED_test_pred.npy")),
         "aami": f64(za["WS_PAIRED"])}
    tp, cp = str(C / "train" / "X.npy"), str(C / "calfree_test" / "X.npy")
    ap = [str(C / "aami_cal" / "X.npy"), str(C / "aami_test" / "X.npy")]
    tr_subs = np.array([s for s in np.unique(S[tr]) if (S[tr] == s).sum() >= 20])
    tr_pick = np.concatenate([np.random.default_rng(SEED).choice(tr[S[tr] == s], 20, replace=False) for s in tr_subs])
    E_tr_win = s93.window_features("train_fit", [tp], tr_pick)
    E_win = {"val": s93.window_features("validation", [tp], va),
             "calfree": s93.window_features("calfree", [cp], np.arange(len(Yt))),
             "aami": s93.window_features("aami", ap, np.arange(len(Ya)))}
    med = np.nanmedian(E_tr_win, 0)
    E_tr = np.stack([np.nanmedian(E_tr_win[i * 20:(i + 1) * 20], 0) for i in range(len(tr_subs))])
    E_tr = np.where(np.isnan(E_tr), med, E_tr)
    cohorts = {"val": (Y[va], S[va]), "calfree": (Yt, St), "aami": (Ya, Sa)}
    subs = {k: np.array([s for s in np.unique(Sx) if (Sx == s).sum() >= (40 if k != "calfree" else 100)])
            for k, (Yx, Sx) in cohorts.items()}
    rows_of = {k: {int(s): np.flatnonzero(cohorts[k][1] == s) for s in subs[k]} for k in cohorts}
    draws = {k: {s: [np.random.default_rng(SEED + 7 * d).choice(r, M, replace=False) for d in range(ND)]
                 for s, r in rows_of[k].items()} for k in cohorts}
    # warm-up median features per subject and draw: [subject, draw, 8]
    Fd = {k: np.array([[np.where(np.isnan(m_ := np.nanmedian(E_win[k][w], 0)), med, m_) for w in draws[k][int(s)]]
                       for s in subs[k]]) for k in cohorts}
    sets = {"all": list(range(8))}
    sets.update({f"drop {n}": [i for i in range(8) if i != j] for j, n in enumerate(NAMES)})
    sets.update({f"only {n}": [j] for j, n in enumerate(NAMES)})
    report = {"protocol": __doc__}
    for j, t in ((0, "SBP"), (1, "DBP")):
        Yv, Sv = cohorts["val"]
        vs = subs["val"]
        G = np.stack([zc["val"][Sv == s].mean(0) for s in vs]).astype(np.float64)
        yv = np.array([Yv[Sv == s, j].mean() for s in vs])

        def fit(Gm, ym):
            mu, sd = Gm.mean(0), Gm.std(0)
            sd[sd < 1e-9] = 1.0
            a = pick_alpha((Gm - mu) / sd, ym, s93.ALPHAS, seed=SEED)
            cmu, w, b = ridge((Gm - mu) / sd, ym, a)
            return lambda F: ((np.asarray(F, np.float64) - mu) / sd - cmu) @ w + b
        rule = fit(G, yv)
        score = {"calfree": rule(zc["test"]), "aami": rule(za["RAW_w_feat"]), "val": np.empty(len(Sv))}
        folds = np.random.default_rng(SEED).permutation(len(vs)) % 4
        for k in range(4):
            m = np.isin(Sv, vs[folds == k])
            score["val"][m] = fit(G[folds != k], yv[folds != k])(zc["val"][m])
        ytr = np.array([Y[tr][S[tr] == s, j].mean() for s in tr_subs])
        base = {}
        for key in cohorts:
            Yx, Sx = cohorts[key]
            L, res, tru = [], [], []
            for s in subs[key]:
                l_, r_ = [], []
                for w in draws[key][int(s)]:
                    sc_ = np.setdiff1d(rows_of[key][int(s)], w)
                    l_.append(score[key][w].mean())
                    r_.append(T[key][sc_, j] - T[key][w, j].mean() - Yx[sc_, j])
                L.append(l_), res.append(r_), tru.append(Yx[rows_of[key][int(s)], j].mean())
            base[key] = (np.array(L), res, np.array(tru))
        cur = {k: np.array([[np.abs(base[k][0][i, d] + base[k][1][i][d]).mean() for d in range(ND)]
                            for i in range(len(base[k][1]))]).mean(1) for k in ("calfree", "aami")}
        rep_t = {}
        for name, cols in sets.items():
            sc = StandardScaler().fit(E_tr[:, cols])
            reg = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(sc.transform(E_tr[:, cols]), ytr)
            E = {k: reg.predict(sc.transform(Fd[k][..., cols].reshape(-1, len(cols)))).reshape(Fd[k].shape[:2])
                 for k in cohorts}
            em = float(E["val"].mean())
            Lv, _, tv = base["val"]
            b = float(min(np.linspace(0, 1.5, 31), key=lambda b: np.abs(Lv + b * (E["val"] - em) - tv[:, None]).mean()))
            row = {"b": round(b, 2)}
            if name == "all":
                row["standardized_coefficients"] = {NAMES[c]: round(float(w), 2) for c, w in zip(cols, reg.coef_)}
            for k in ("calfree", "aami"):
                L, res, _ = base[k]
                anc = L + b * (E[k] - em)
                new = np.array([[np.abs(anc[i, d] + res[i][d]).mean() for d in range(ND)] for i in range(len(res))]).mean(1)
                dd = new - cur[k]
                ii = rng.integers(0, len(dd), (NB, len(dd)))
                row[k] = [round(float(dd.mean()), 3), [round(float(x), 3) for x in np.percentile(dd[ii].mean(1), [2.5, 97.5])]]
            rep_t[name] = row
            print(t, name, json.dumps(row), flush=True)
        report[t] = rep_t
    (ROOT / "reports" / "s102_ecg_feature_contribution.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
