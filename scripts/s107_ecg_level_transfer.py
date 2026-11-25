"""s107: why does the ECG refinement not help on UCI? Per subject (VitalDB) or record (UCI): the ECG level E (ridge
fitted on the VitalDB training subjects, medians of the eight window features over all windows of the unit) against
the true mean SBP: Pearson r, mean bias of E, and the r of the correction direction (E - mean E) with the error of the
stage-1 level (true mean - L), which is what the refinement needs. Also the feature distributions (median) per cohort,
to see a shift in acquisition. Descriptive."""
import json
import sys
import types

import numpy as np

sys.argv.append("--no-gate")
import s93_fluctuation_gated_system as s93  # noqa: E402

ROOT, C, R = s93.ROOT, s93.C, s93.R


def main():
    import bp_networks as q1
    sys.modules.setdefault("network_loader", types.SimpleNamespace(load_q1_module=lambda *a: q1))
    from s29_level_information import pick_alpha, ridge
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    f64 = lambda a: np.asarray(a, dtype=np.float64)  # noqa: E731
    Y, S = f64(np.load(C / "train" / "Y.npy")), np.load(C / "train" / "S.npy")
    tr, va = np.load(C / "train" / "rows_train.npy"), np.load(C / "train" / "rows_validation.npy")
    tr_subs = np.array([s for s in np.unique(S[tr]) if (S[tr] == s).sum() >= 20])
    tr_pick = np.concatenate([np.random.default_rng(42).choice(tr[S[tr] == s], 20, replace=False) for s in tr_subs])
    E_tr_win = s93.window_features("train_fit", [str(C / "train" / "X.npy")], tr_pick)
    med = np.nanmedian(E_tr_win, 0)
    E_tr = np.where(np.isnan(np.stack([np.nanmedian(E_tr_win[i * 20:(i + 1) * 20], 0) for i in range(len(tr_subs))])),
                    med, np.stack([np.nanmedian(E_tr_win[i * 20:(i + 1) * 20], 0) for i in range(len(tr_subs))]))
    ytr = np.array([Y[tr][S[tr] == s, 0].mean() for s in tr_subs])
    sc = StandardScaler().fit(E_tr)
    reg = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(sc.transform(E_tr), ytr)
    # stage-1 level rule (s69 style) for the level error
    zc = np.load(R / "RAW_w_feat_cache.npz")
    fv, Sv, Yv = f64(zc["val"]), S[va], Y[va]
    fs = np.array([g for g in np.unique(Sv) if (Sv == g).sum() >= 20])
    G = np.stack([fv[Sv == g].mean(0) for g in fs])
    mu, sd = G.mean(0), G.std(0)
    sd[sd < 1e-9] = 1
    a = pick_alpha((G - mu) / sd, np.array([Yv[Sv == g, 0].mean() for g in fs]), s93.ALPHAS, seed=42)
    cmu, w, b = ridge((G - mu) / sd, np.array([Yv[Sv == g, 0].mean() for g in fs]), a)
    rule = lambda F: ((F - mu) / sd - cmu) @ w + b  # noqa: E731
    za = np.load(R / "s78_aami_predictions.npz")
    zu = np.load(R / "s69_uci_cache_all.npz")
    cohorts = {
        "VitalDB CalFree": (f64(np.load(C / "calfree_test" / "Y.npy")), np.load(C / "calfree_test" / "S.npy"),
                            s93.window_features("calfree", None, None), f64(zc["test"]), 100),
        "VitalDB AAMI": (np.concatenate([f64(np.load(C / p / "Y.npy")) for p in ("aami_cal", "aami_test")]),
                         np.concatenate([np.load(C / p / "S.npy") for p in ("aami_cal", "aami_test")]),
                         s93.window_features("aami", None, None), f64(za["RAW_w_feat"]), 40),
        "UCI": (f64(np.load(C / "uci_all" / "Y.npy")), np.load(C / "uci_all" / "S.npy", allow_pickle=True),
                s93.window_features("uci_all", None, None), f64(zu["feats"]), 30)}
    out = {"protocol": __doc__, "feature_names": s93.ECG_NAMES,
           "train_feature_median": [round(float(v), 3) for v in np.nanmedian(E_tr, 0)]}
    for name, (Yc, Sc, Ew, Fw, nmin) in cohorts.items():
        units = [u for u in np.unique(Sc) if (Sc == u).sum() >= nmin]
        E, L, T, Fm = [], [], [], []
        for u in units:
            m = Sc == u
            f = np.nanmedian(Ew[m], 0)
            f = np.where(np.isnan(f), med, f)
            Fm.append(f)
            E.append(reg.predict(sc.transform(f[None]))[0])
            L.append(rule(Fw[m]).mean())
            T.append(Yc[m, 0].mean())
        E, L, T, Fm = map(np.array, (E, L, T, Fm))
        out[name] = {"n_units": len(units), "r_E_true": round(float(np.corrcoef(E, T)[0, 1]), 3),
                     "r_L_true": round(float(np.corrcoef(L, T)[0, 1]), 3),
                     "r_correction_with_level_error": round(float(np.corrcoef(E - E.mean(), T - L)[0, 1]), 3),
                     "bias_E": round(float((E - T).mean()), 2), "true_mean_SBP": round(float(T.mean()), 1),
                     "true_SD_of_unit_means": round(float(T.std()), 1),
                     "feature_median": [round(float(v), 3) for v in np.median(Fm, 0)]}
        print(name, json.dumps({k: v for k, v in out[name].items() if k != "feature_median"}), flush=True)
        print("   features", dict(zip(s93.ECG_NAMES, out[name]["feature_median"])), flush=True)
    print("train features", dict(zip(s93.ECG_NAMES, out["train_feature_median"])))
    (ROOT / "reports" / "s107_ecg_level_transfer.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
