"""s94: FLARE (s93) for further tracker seeds and for a PPG-only system.

Variants
  seed 42 (reference), 44, 45 : WS_PAIRED_s{seed} tracker, WS_ABS_s{seed} comparator, stage-1 level from RAW_w
                                features; gate threshold and ECG weight refitted on validation for each seed.
  seed 43                     : only saved test predictions exist (no weights), so validation outputs cannot be
                                computed; threshold and ECG weight are taken from seed 42 (reported separately).
  PPG-only                    : tracker WS_PAIRED_ppgonly and comparator WS_ABS_ppgonly (ECG zeroed in training and
                                inference), level from the WS_ABS_ppgonly encoder; no ECG exists, so the low branch
                                has no ECG correction (gate only).
Protocol as in s93 (20 random warm-up windows, 50 draws, subject bootstrap 2000). CalFree is exploratory; AAMI
(seeds 44/45) has not been used to fit or choose anything.
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
import torch  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

M, N_DRAWS, N_BOOT, SEED = 20, 50, 2000, 42
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 1e4, 1e5)
f64 = lambda a: np.asarray(a, dtype=np.float64)  # noqa: E731
rng = np.random.default_rng(7)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Y, S = f64(np.load(C / "train" / "Y.npy")), np.load(C / "train" / "S.npy")
tr, va = np.load(C / "train" / "rows_train.npy"), np.load(C / "train" / "rows_validation.npy")
Yt, St = f64(np.load(C / "calfree_test" / "Y.npy")), np.load(C / "calfree_test" / "S.npy")
Ya = np.concatenate([f64(np.load(C / p / "Y.npy")) for p in ("aami_cal", "aami_test")])
Sa = np.concatenate([np.load(C / p / "S.npy") for p in ("aami_cal", "aami_test")])
za, z81 = np.load(R / "s78_aami_predictions.npz"), np.load(R / "s81_aami_predictions.npz")
cohorts = {"val": (Y[va], S[va]), "calfree": (Yt, St), "aami": (Ya, Sa)}
subs = {k: np.array([s for s in np.unique(Sx) if (Sx == s).sum() >= (100 if k == "calfree" else 40)]) for k, (Yx, Sx) in cohorts.items()}
rows_of = {k: {int(s): np.flatnonzero(cohorts[k][1] == s) for s in subs[k]} for k in cohorts}
draws = {k: {s: [np.random.default_rng(SEED + 7 * d).choice(r, M, replace=False) for d in range(N_DRAWS)]
             for s, r in rows_of[k].items()} for k in cohorts}
E_win = {k: np.load(OUT93 / f"ecg_window_features_{n}.npy") for k, n in (("val", "validation"), ("calfree", "calfree"), ("aami", "aami"))}
E_tr_win = np.load(OUT93 / "ecg_window_features_train_fit.npy")
tr_subs = np.array([s for s in np.unique(S[tr]) if (S[tr] == s).sum() >= 20])
med = np.nanmedian(E_tr_win, 0)
E_tr = np.stack([np.nanmedian(E_tr_win[i * 20:(i + 1) * 20], 0) for i in range(len(tr_subs))])
E_tr = np.where(np.isnan(E_tr), med, E_tr)


@torch.no_grad()
def predict(tag, X, rows, drop_ecg=False):
    ck = torch.load(R / f"{tag}_weights.pt", map_location="cpu", weights_only=False)
    m = q1.RawResNet1D(dropout=0.5)
    m.load_state_dict(ck["state_dict"])
    m = m.to(dev).eval()
    out = []
    for s in range(0, len(rows), 1024):
        x = (np.asarray(X[rows[s:s + 1024]], np.float32) - ck["signal_mean"][None, :, None]) / ck["signal_std"][None, :, None]
        if drop_ecg:
            x[:, 0] = 0.0
        out.append(m(torch.from_numpy(x).to(dev)).float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def val_pred(tag, drop_ecg=False):
    f = R / f"{tag}_val_pred_s94.npy"
    if not f.exists():
        np.save(f, predict(tag, np.load(C / "train" / "X.npy", mmap_mode="r"), va, drop_ecg))
    return f64(np.load(f))


def fit_rule(G, y):
    mu, sd = G.mean(0), G.std(0)
    sd[sd < 1e-9] = 1.0
    a = pick_alpha((G - mu) / sd, y, ALPHAS, seed=SEED)
    cmu, w, b = ridge((G - mu) / sd, y, a)
    return lambda F: ((np.asarray(F, np.float64) - mu) / sd - cmu) @ w + b


def ci(x):
    i = rng.integers(0, len(x), (N_BOOT, len(x)))
    return [round(float(x.mean()), 3), [round(float(v), 3) for v in np.percentile(x[i].mean(1), [2.5, 97.5])]]


def run(T, A, feats, use_ecg, fixed=None):
    """T, A: dict cohort -> tracker / comparator outputs (None if missing); feats: dict cohort -> level features."""
    out = {}
    for j, t in ((0, "SBP"), (1, "DBP")):
        Yv, Sv = cohorts["val"]
        vs = subs["val"]
        G = np.stack([feats["val"][Sv == s].mean(0) for s in vs]).astype(np.float64)
        yv = np.array([Yv[Sv == s, j].mean() for s in vs])
        rule = fit_rule(G, yv)
        score = {k: rule(feats[k]) for k in feats if k != "val"}
        score["val"] = np.empty(len(Sv))
        folds = np.random.default_rng(SEED).permutation(len(vs)) % 4
        for k in range(4):
            m = np.isin(Sv, vs[folds == k])
            score["val"][m] = fit_rule(G[folds != k], yv[folds != k])(feats["val"][m])
        ytr = np.array([Y[tr][S[tr] == s, j].mean() for s in tr_subs])
        sc = StandardScaler().fit(E_tr)
        reg = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(sc.transform(E_tr), ytr)

        def per_draw(key):
            Yx, Sx = cohorts[key]
            V, L, E, res, tru, ab = [], [], [], [], [], []
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
                    if A.get(key) is not None:
                        a_.append(np.abs(A[key][sc_, j] - Yx[sc_, j]).mean())
                V.append(v_), L.append(l_), E.append(e_), res.append(r_), tru.append(Yx[rows_of[key][int(s)], j].mean())
                ab.append(np.mean(a_) if a_ else np.nan)
            return np.array(V), np.array(L), np.array(E), res, np.array(tru), np.array(ab)
        if fixed is None:
            Vv, Lv, Ev, _, truev, _ = per_draw("val")
            tau, e_mean = float(np.median(Vv)), float(Ev.mean())
            b = float(min(np.linspace(0, 1.5, 31), key=lambda b: np.abs(Lv + b * (Ev - e_mean) - truev[:, None])[Vv <= tau].mean())) if use_ecg else 0.0
        else:
            tau, b, e_mean = fixed[t]
        blk = {"tau": round(tau, 3), "b": round(b, 2), "_fixed": (tau, b, e_mean)}
        for key in [k for k in ("calfree", "aami") if T.get(k) is not None]:
            V, L, E, res, tru, ab = per_draw(key)
            anc = L + np.where(V <= tau, b * (E - e_mean), 0.0)
            new = np.array([[np.abs(anc[i, d] + res[i][d]).mean() for d in range(N_DRAWS)] for i in range(len(res))]).mean(1)
            cur = np.array([[np.abs(L[i, d] + res[i][d]).mean() for d in range(N_DRAWS)] for i in range(len(res))]).mean(1)
            blk[key] = {"FLARE": ci(new), "no_gate": ci(cur), "FLARE_minus_no_gate": ci(new - cur)}
            if not np.isnan(ab).all():
                blk[key].update({"absolute": ci(ab), "FLARE_minus_absolute": ci(new - ab)})
        out[t] = blk
    return out


Xt = np.load(C / "calfree_test" / "X.npy", mmap_mode="r")
raw_feats = {"val": np.load(R / "RAW_w_feat_cache.npz")["val"], "calfree": np.load(R / "RAW_w_feat_cache.npz")["test"],
             "aami": za["RAW_w_feat"]}
load_t = lambda tag: f64(np.load(R / f"{tag}_test_pred.npy"))  # noqa: E731
report = {"protocol": __doc__}
base = None
for seed in (42, 44, 45):
    tagp = "WS_PAIRED" if seed == 42 else f"WS_PAIRED_s{seed}"
    taga = "WS_ABS" if seed == 42 else f"WS_ABS_s{seed}"
    T = {"val": f64(np.load(R / "WS_PAIRED_val_pred_s89.npy")) if seed == 42 else val_pred(tagp), "calfree": load_t(tagp),
         "aami": f64(za["WS_PAIRED"]) if seed == 42 else f64(z81[tagp])}
    A = {"calfree": load_t(taga), "aami": f64(za["WS_ABS"]) if seed == 42 else f64(z81[taga])}
    res = run(T, A, raw_feats, True)
    if seed == 42:
        base = {t: res[t]["_fixed"] for t in ("SBP", "DBP")}
    report[f"seed_{seed}"] = res
    print(seed, json.dumps({t: {k: v for k, v in res[t].items() if k != "_fixed"} for t in res}), flush=True)
T43 = {"val": None, "calfree": load_t("WS_PAIRED_s43"), "aami": None}
report["seed_43_threshold_from_seed42"] = run(T43, {"calfree": load_t("WS_ABS_s43")}, raw_feats, True, fixed=base)
print(43, json.dumps({t: {k: v for k, v in report["seed_43_threshold_from_seed42"][t].items() if k != "_fixed"} for t in ("SBP", "DBP")}), flush=True)
zp = np.load(R / "WS_ABS_ppgonly_feat_cache.npz")
Tp = {"val": val_pred("WS_PAIRED_ppgonly", drop_ecg=True), "calfree": load_t("WS_PAIRED_ppgonly"), "aami": None}
report["ppg_only"] = run(Tp, {"calfree": load_t("WS_ABS_ppgonly")}, {"val": zp["val"], "calfree": zp["test"]}, False)
print("ppg", json.dumps({t: {k: v for k, v in report["ppg_only"][t].items() if k != "_fixed"} for t in ("SBP", "DBP")}), flush=True)
for k in report:
    if isinstance(report[k], dict):
        for t in ("SBP", "DBP"):
            report[k].get(t, {}).pop("_fixed", None)
(ROOT / "reports" / "s94_flare_variants.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
