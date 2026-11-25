"""s104: FLARE (gated) by fluctuation branch, deployable, same rules, windows and draws as s93.
Per subject and draw: fluctuation index v (SD of the pairwise tracker over the 20 warm-up windows), gate v <= tau
(tau = median over validation subject-draws), anchor A = L + 1[low] * b * (E - mean E), b chosen on low-branch
validation subject-draws. Subjects are grouped by the branch they enter in most draws (mostly low / mostly high).
Per group: MAE of the absolute network, the two-stage core (b = 0) and FLARE, and paired differences with 2000
subject-bootstrap replicates. Also the fraction of draws in which a subject's branch agrees with its majority
branch, and the true within-subject SD by group. CalFree exploratory; AAMI used for nothing else.
"""
import json
import sys
import types

import numpy as np

sys.argv.append("--no-gate")
import s93_fluctuation_gated_system as s93  # noqa: E402

ROOT, C, R = s93.ROOT, s93.C, s93.R
M, ND, NB, SEED = s93.M, s93.N_DRAWS, s93.N_BOOT, s93.SEED


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
    ABS = {"calfree": f64(np.load(R / "WS_ABS_test_pred.npy")), "aami": f64(za["WS_ABS"])}
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

    def ci(x):
        ii = rng.integers(0, len(x), (NB, len(x)))
        return [round(float(x.mean()), 3), [round(float(v), 3) for v in np.percentile(x[ii].mean(1), [2.5, 97.5])]]

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
        sc = StandardScaler().fit(E_tr)
        reg = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(sc.transform(E_tr), ytr)
        per = {}
        for key in cohorts:
            Yx, Sx = cohorts[key]
            V, L, E, res, tru, absm, sdt = [], [], [], [], [], [], []
            for s in subs[key]:
                v_, l_, e_, r_, a_ = [], [], [], [], []
                for w in draws[key][int(s)]:
                    sc_ = np.setdiff1d(rows_of[key][int(s)], w)
                    tw = T[key][w, j]
                    v_.append(tw.std())
                    l_.append(score[key][w].mean())
                    fw = np.nanmedian(E_win[key][w], 0)
                    e_.append(reg.predict(sc.transform(np.where(np.isnan(fw), med, fw)[None]))[0])
                    r_.append(T[key][sc_, j] - tw.mean() - Yx[sc_, j])
                    if key in ABS:
                        a_.append(np.abs(ABS[key][sc_, j] - Yx[sc_, j]).mean())
                rr = rows_of[key][int(s)]
                V.append(v_), L.append(l_), E.append(e_), res.append(r_), tru.append(Yx[rr, j].mean())
                absm.append(np.mean(a_) if a_ else np.nan)
                sdt.append(Yx[rr, j].std())
            per[key] = (np.array(V), np.array(L), np.array(E), res, np.array(tru), np.array(absm), np.array(sdt))
        Vv, Lv, Ev, _, tv, _, _ = per["val"]
        tau = float(np.median(Vv))
        low_v = Vv <= tau
        em = float(Ev.mean())
        b = float(min(np.linspace(0, 1.5, 31), key=lambda b: np.abs((Lv + b * (Ev - em)) - tv[:, None])[low_v].mean()))
        rep_t = {"tau": round(tau, 3), "b": round(b, 2)}
        for key in ("calfree", "aami"):
            V, L, E, res, tru, absm, sdt = per[key]
            low = V <= tau
            core = np.array([[np.abs(L[i, d] + res[i][d]).mean() for d in range(ND)] for i in range(len(res))]).mean(1)
            anc = L + np.where(low, b * (E - em), 0.0)
            fl = np.array([[np.abs(anc[i, d] + res[i][d]).mean() for d in range(ND)] for i in range(len(res))]).mean(1)
            frac = low.mean(1)
            blk = {"n": int(len(res)), "FLARE": ci(fl), "core": ci(core), "absolute": ci(absm),
                   "FLARE_minus_absolute": ci(fl - absm), "FLARE_minus_core": ci(fl - core),
                   "core_minus_absolute": ci(core - absm)}
            for gname, m in (("mostly_low", frac >= 0.5), ("mostly_high", frac < 0.5)):
                blk[gname] = {"n": int(m.sum()), "true_SD_median": round(float(np.median(sdt[m])), 2),
                              "branch_agreement": round(float(np.mean(np.where(frac[m] >= 0.5, frac[m], 1 - frac[m]))), 3),
                              "absolute": ci(absm[m]), "core": ci(core[m]), "FLARE": ci(fl[m]),
                              "FLARE_minus_absolute": ci(fl[m] - absm[m]), "core_minus_absolute": ci(core[m] - absm[m]),
                              "FLARE_minus_core": ci(fl[m] - core[m])}
            rep_t[key] = blk
            np.savez(ROOT / "runs" / "s93" / f"s104_per_subject_{t}_{key}.npz", flare=fl, core=core, absolute=absm, frac_low=frac, true_sd=sdt)
            blk["n_lower_than_absolute"] = int((fl < absm).sum())
            print(t, key, json.dumps({k: blk[k] for k in ("FLARE", "core", "absolute", "FLARE_minus_core")}), flush=True)
            for g in ("mostly_low", "mostly_high"):
                print("  ", g, json.dumps(blk[g]), flush=True)
        report[t] = rep_t
    (ROOT / "reports" / "s104_flare_branches.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
