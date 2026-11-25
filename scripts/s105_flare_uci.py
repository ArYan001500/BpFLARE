"""s105: FLARE (gated) on the UCI records, frozen from VitalDB, same protocol as s69 (all four parts, records with at
least 30 windows, warm-up M in {5, 10, 20} windows, 50 draws, seed 42 + 7d, record bootstrap 2000).
Stage-1 level: the s69 rule (ridge on frozen RAW_w encoder features, validation subjects with >= 20 windows).
Gate: fluctuation index v = SD of the pairwise tracker over the warm-up; threshold tau, ECG weight b and the mean ECG
level are fitted on the VitalDB validation subjects for the same M with the s93/s104 rules; the ECG level is the
VitalDB-training ridge on the eight hand-crafted features (median over the warm-up windows). Nothing is refitted on
UCI. Reports: absolute network, two-stage core (b = 0), FLARE, paired differences; by majority branch.
"""
import json
import sys
import types

import numpy as np

sys.argv.append("--no-gate")
import s93_fluctuation_gated_system as s93  # noqa: E402

ROOT, C, R = s93.ROOT, s93.C, s93.R
ND, NB, SEED = 50, 2000, 42
M_EVAL, MIN_SCORED = (5, 10, 20), 10


def main():
    sys.path.insert(0, str(ROOT / "scripts"))
    import bp_networks as q1
    sys.modules.setdefault("network_loader", types.SimpleNamespace(load_q1_module=lambda *a: q1))
    from s29_level_information import pick_alpha, ridge
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    f64 = lambda a: np.asarray(a, dtype=np.float64)  # noqa: E731
    rng = np.random.default_rng(7)
    U = C / "uci_all"
    Yu, Su = f64(np.load(U / "Y.npy")), np.load(U / "S.npy", allow_pickle=True)
    zu = np.load(R / "s69_uci_cache_all.npz")
    feats_u, Pu, Au = f64(zu["feats"]), f64(zu["WS_PAIRED"]), f64(zu["WS_ABS"])
    E_u_win = s93.window_features("uci_all", [str(U / "X.npy")], np.arange(len(Yu)))
    print("UCI ECG features ready", E_u_win.shape, flush=True)

    Y, S = f64(np.load(C / "train" / "Y.npy")), np.load(C / "train" / "S.npy")
    tr, va = np.load(C / "train" / "rows_train.npy"), np.load(C / "train" / "rows_validation.npy")
    zc = np.load(R / "RAW_w_feat_cache.npz")
    Tv = f64(np.load(R / "WS_PAIRED_val_pred_s89.npy"))
    tp = str(C / "train" / "X.npy")
    tr_subs = np.array([s for s in np.unique(S[tr]) if (S[tr] == s).sum() >= 20])
    tr_pick = np.concatenate([np.random.default_rng(SEED).choice(tr[S[tr] == s], 20, replace=False) for s in tr_subs])
    E_tr_win = s93.window_features("train_fit", [tp], tr_pick)
    E_v_win = s93.window_features("validation", [tp], va)
    med = np.nanmedian(E_tr_win, 0)
    E_tr = np.stack([np.nanmedian(E_tr_win[i * 20:(i + 1) * 20], 0) for i in range(len(tr_subs))])
    E_tr = np.where(np.isnan(E_tr), med, E_tr)
    Yv, Sv = Y[va], S[va]
    fv = zc["val"].astype(np.float64)

    def ci(x):
        ii = rng.integers(0, len(x), (NB, len(x)))
        return [round(float(x.mean()), 3), [round(float(v), 3) for v in np.percentile(x[ii].mean(1), [2.5, 97.5])]]

    def ridge_rule(G, y):
        mu, sd = G.mean(0), G.std(0)
        sd[sd < 1e-9] = 1.0
        a = pick_alpha((G - mu) / sd, y, s93.ALPHAS, seed=SEED)
        cmu, w, b = ridge((G - mu) / sd, y, a)
        return lambda F: ((np.asarray(F, np.float64) - mu) / sd - cmu) @ w + b

    report = {"protocol": __doc__}
    for j, t in ((0, "SBP"), (1, "DBP")):
        ytr = np.array([Y[tr][S[tr] == s, j].mean() for s in tr_subs])
        sc = StandardScaler().fit(E_tr)
        reg = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(sc.transform(E_tr), ytr)
        ecg = lambda F: reg.predict(sc.transform(np.where(np.isnan(F), med, F)))  # noqa: E731
        # s69 stage-1 rule (validation subjects with >= 20 windows), applied to UCI windows
        fit_subs = np.array([g for g in np.unique(Sv) if (Sv == g).sum() >= 20])
        rule69 = ridge_rule(np.stack([fv[Sv == g].mean(0) for g in fit_subs]),
                            np.array([Yv[Sv == g, j].mean() for g in fit_subs]))
        Lu = rule69(feats_u)
        # s93 stage-1 rule with out-of-fold validation scores (for tau and b)
        vs = np.array([s for s in np.unique(Sv) if (Sv == s).sum() >= 40])
        G = np.stack([fv[Sv == s].mean(0) for s in vs])
        yv = np.array([Yv[Sv == s, j].mean() for s in vs])
        score_v = np.empty(len(Sv))
        folds = np.random.default_rng(SEED).permutation(len(vs)) % 4
        for k in range(4):
            m = np.isin(Sv, vs[folds == k])
            score_v[m] = ridge_rule(G[folds != k], yv[folds != k])(fv[m])
        rep_t = {}
        for M in M_EVAL:
            # ---- validation: tau, b, mean ECG level for this warm-up length
            Vv, Lv, Ev, tv = [], [], [], []
            for s in vs:
                rr = np.flatnonzero(Sv == s)
                v_, l_, e_ = [], [], []
                for d in range(ND):
                    w = np.random.default_rng(SEED + 7 * d).choice(rr, M, replace=False)
                    v_.append(Tv[w, j].std())
                    l_.append(score_v[w].mean())
                    e_.append(np.nanmedian(E_v_win[w], 0))
                Vv.append(v_), Lv.append(l_), Ev.append(ecg(np.array(e_))), tv.append(Yv[rr, j].mean())
            Vv, Lv, Ev, tv = np.array(Vv), np.array(Lv), np.array(Ev), np.array(tv)
            tau = float(np.median(Vv))
            em = float(Ev.mean())
            low_v = Vv <= tau
            b = float(min(np.linspace(0, 1.5, 31),
                          key=lambda b: np.abs((Lv + b * (Ev - em)) - tv[:, None])[low_v].mean()))
            # ---- UCI records
            need = max(M_EVAL) + MIN_SCORED
            recs = [r for r in np.unique(Su) if (Su == r).sum() >= need]
            core, fl, ab, frac, sdt = [], [], [], [], []
            for r in recs:
                rows = np.flatnonzero(Su == r)
                c_, f_, a_, lo_ = [], [], [], []
                for d in range(ND):
                    warm = np.random.default_rng(SEED + 7 * d).choice(rows, M, replace=False)
                    m_ = np.ones(len(rows), bool)
                    m_[np.searchsorted(rows, np.sort(warm))] = False
                    ev = rows[m_]
                    off = Lu[warm].mean() - Pu[warm, j].mean()
                    base = Pu[ev, j] + off - Yu[ev, j]
                    low = Pu[warm, j].std() <= tau
                    e_ = ecg(np.nanmedian(E_u_win[warm], 0)[None])[0]
                    corr = b * (e_ - em) if low else 0.0
                    c_.append(np.abs(base).mean())
                    f_.append(np.abs(base + corr).mean())
                    a_.append(np.abs(Au[ev, j] - Yu[ev, j]).mean())
                    lo_.append(low)
                core.append(np.mean(c_)), fl.append(np.mean(f_)), ab.append(np.mean(a_)), frac.append(np.mean(lo_))
                sdt.append(Yu[rows, j].std())
            core, fl, ab, frac, sdt = map(np.array, (core, fl, ab, frac, sdt))
            blk = {"n_records": len(recs), "tau": round(tau, 3), "b": round(b, 2), "fraction_low": round(float(frac.mean()), 3),
                   "absolute": ci(ab), "core": ci(core), "FLARE": ci(fl), "FLARE_minus_absolute": ci(fl - ab),
                   "core_minus_absolute": ci(core - ab), "FLARE_minus_core": ci(fl - core)}
            for g, m in (("mostly_low", frac >= 0.5), ("mostly_high", frac < 0.5)):
                if m.sum() > 1:
                    blk[g] = {"n": int(m.sum()), "true_SD_median": round(float(np.median(sdt[m])), 2),
                              "FLARE_minus_absolute": ci(fl[m] - ab[m]), "FLARE_minus_core": ci(fl[m] - core[m])}
            rep_t[f"M{M}"] = blk
            print(t, M, json.dumps(blk), flush=True)
        report[t] = rep_t
    (ROOT / "reports" / "s105_flare_uci.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote reports/s105_flare_uci.json")


if __name__ == "__main__":
    main()
