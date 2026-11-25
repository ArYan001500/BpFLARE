"""s93: fluctuation-gated two-stage system with an ECG-corrected anchor in the low-fluctuation branch.

Everything a deployed system would do, per subject and per warm-up draw (20 randomly sampled unlabeled 10-s windows,
50 draws, seed 42 + 7 d, warm-up windows excluded from scoring):
  1. Fluctuation index v = SD of the pairwise tracker output (WS_PAIRED) over the warm-up windows.
  2. Gate: v <= tau -> low-fluctuation branch, else high-fluctuation branch. tau = median of v over the validation
     subject-draws.
  3. Anchor: L = mean over the warm-up of the stage-1 ridge score (frozen RAW_w encoder features, ridge fitted on the
     194 validation subjects; out-of-fold for validation subjects). High branch: anchor = L. Low branch:
     anchor = L + b * (E - mean E), where E is a ridge level from 8 hand-crafted ECG features (median over the
     warm-up windows) fitted on the 1,099 training subjects, and b is chosen on the low-branch validation
     subject-draws (grid 0..1.5) and then frozen.
  4. Output for every scored window x: anchor + f(x) - mean of f over the warm-up.
Fitting: networks on the training subjects (already trained), ridge rules and b on validation, ECG ridge on
training. Evaluation only: CalFree (144 subjects, exploratory: used during development) and the AAMI subjects
(108 with >= 40 windows; not used to fit or choose anything). Reference: the current two-stage system on the same
windows and draws. Intervals: 2000 subject-level bootstrap replicates of paired per-subject differences.
"""
import json
import sys
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / "converted"
R = ROOT / "runs" / "s4_pat_control"
OUT = ROOT / "runs" / "s93"
OUT.mkdir(parents=True, exist_ok=True)
FS, M, N_DRAWS, N_BOOT, SEED = 125, 20, 50, 2000, 42
ECG_NAMES = ["HR", "RMSSD", "RR_SD", "PAT_peak", "PAT_foot", "R_width", "T_to_R", "PPG_interval"]
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 1e4, 1e5)
GATED = "--no-gate" not in sys.argv   # --no-gate: ECG correction of the anchor for every subject


def ecg_window_features(x):
    """Hand-crafted features of one 10-s window (rows: ECG, PPG; each scaled to [0, 1] by PulseDB)."""
    from scipy.signal import find_peaks
    ecg, ppg = x[0].astype(float), x[1].astype(float)
    cands = []
    for sgn in (1, -1):
        e = sgn * (ecg - np.median(ecg))
        p, _ = find_peaks(e, distance=40, prominence=0.25 * (e.max() - np.median(e)))
        ok = 4 <= len(p) <= 30
        spread = -np.std(np.diff(p)) if len(p) > 2 else -1e9
        cands.append(((ok, spread), p, e))
    _, rp, e = max(cands, key=lambda z: z[0])
    if len(rp) < 3:
        return np.full(len(ECG_NAMES), np.nan)
    rr = np.diff(rp) / FS
    pk, _ = find_peaks(ppg, distance=40, prominence=0.15)
    d1 = np.gradient(ppg)
    ft, _ = find_peaks(d1, distance=40, prominence=0.3 * d1.max())
    pat_p = [(pk[(pk > r) & (pk < r + FS)][0] - r) / FS for r in rp if ((pk > r) & (pk < r + FS)).any()]
    pat_f = [(ft[(ft > r) & (ft < r + int(0.8 * FS))][0] - r) / FS for r in rp if ((ft > r) & (ft < r + int(0.8 * FS))).any()]
    base = np.median(e)
    widths, post = [], []
    for r in rp:
        half = e[r] / 2 + base / 2
        lo = r
        while lo > 0 and e[lo] > half:
            lo -= 1
        hi = r
        while hi < len(e) - 1 and e[hi] > half:
            hi += 1
        widths.append((hi - lo) / FS)
        a, b = r + int(0.15 * FS), r + int(0.45 * FS)
        post.append(e[a:b].max() / e[r] if b < len(e) and e[r] > 0 else np.nan)
    return np.array([60 / np.median(rr), np.sqrt(np.mean(np.diff(rr) ** 2)) if len(rr) > 2 else np.nan, np.std(rr),
                     np.median(pat_p) if pat_p else np.nan, np.median(pat_f) if pat_f else np.nan,
                     np.median(widths), np.nanmedian(post), np.median(np.diff(pk)) / FS if len(pk) > 2 else np.nan])


def _feat_chunk(args):
    paths, rows = args
    arrays = [np.load(p, mmap_mode="r") for p in paths]
    sizes = np.cumsum([0] + [a.shape[0] for a in arrays])
    out = np.empty((len(rows), len(ECG_NAMES)), dtype=np.float32)
    for i, r in enumerate(rows):
        k = int(np.searchsorted(sizes, r, side="right") - 1)
        out[i] = ecg_window_features(np.asarray(arrays[k][r - sizes[k]]))
    return out


def window_features(name, paths, rows):
    """ECG features for the given (global) row indices of the concatenated arrays; cached."""
    f = OUT / f"ecg_window_features_{name}.npy"
    if f.exists():
        return np.load(f)
    chunks = [(paths, rows[i:i + 2000]) for i in range(0, len(rows), 2000)]
    with ProcessPoolExecutor() as ex:
        res = np.concatenate(list(ex.map(_feat_chunk, chunks)))
    np.save(f, res)
    return res


def main():
    sys.path.insert(0, str(ROOT / "scripts"))
    import bp_networks as q1
    sys.modules.setdefault("network_loader", types.SimpleNamespace(load_q1_module=lambda *a: q1))
    from s29_level_information import pick_alpha, ridge
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    import torch
    f64 = lambda a: np.asarray(a, dtype=np.float64)  # noqa: E731
    rng = np.random.default_rng(7)

    # ---------------- data
    Y, S = f64(np.load(C / "train" / "Y.npy")), np.load(C / "train" / "S.npy")
    tr, va = np.load(C / "train" / "rows_train.npy"), np.load(C / "train" / "rows_validation.npy")
    Yt, St = f64(np.load(C / "calfree_test" / "Y.npy")), np.load(C / "calfree_test" / "S.npy")
    Ya = np.concatenate([f64(np.load(C / p / "Y.npy")) for p in ("aami_cal", "aami_test")])
    Sa = np.concatenate([np.load(C / p / "S.npy") for p in ("aami_cal", "aami_test")])
    za = np.load(R / "s78_aami_predictions.npz")
    zc = np.load(R / "RAW_w_feat_cache.npz")

    # tracker outputs (validation: predicted here if not cached)
    fv = R / "WS_PAIRED_val_pred_s89.npy"
    if not fv.exists():
        ck = torch.load(R / "WS_PAIRED_weights.pt", map_location="cpu", weights_only=False)
        net = q1.RawResNet1D(dropout=0.5)
        net.load_state_dict(ck["state_dict"])
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = net.to(dev).eval()
        Xv = np.load(C / "train" / "X.npy", mmap_mode="r")
        with torch.no_grad():
            P = np.concatenate([net(torch.from_numpy((np.asarray(Xv[va[s:s + 1024]], np.float32) - ck["signal_mean"][None, :, None])
                                                     / ck["signal_std"][None, :, None]).to(dev)).float().cpu().numpy()
                                for s in range(0, len(va), 1024)])
        np.save(fv, P)
    T = {"val": f64(np.load(fv)), "calfree": f64(np.load(R / "WS_PAIRED_test_pred.npy")), "aami": f64(za["WS_PAIRED"])}

    # ---------------- ECG features per window
    train_path, cf_path = str(C / "train" / "X.npy"), str(C / "calfree_test" / "X.npy")
    aami_paths = [str(C / "aami_cal" / "X.npy"), str(C / "aami_test" / "X.npy")]
    tr_subs = np.array([s for s in np.unique(S[tr]) if (S[tr] == s).sum() >= 20])
    tr_pick = np.concatenate([np.random.default_rng(SEED).choice(tr[S[tr] == s], 20, replace=False) for s in tr_subs])
    E_tr_win = window_features("train_fit", [train_path], tr_pick)
    E_win = {"val": window_features("validation", [train_path], va),
             "calfree": window_features("calfree", [cf_path], np.arange(len(Yt))),
             "aami": window_features("aami", aami_paths, np.arange(len(Ya)))}
    med = np.nanmedian(E_tr_win, 0)
    E_tr = np.stack([np.nanmedian(E_tr_win[i * 20:(i + 1) * 20], 0) for i in range(len(tr_subs))])
    E_tr = np.where(np.isnan(E_tr), med, E_tr)

    cohorts = {"val": (Y[va], S[va]), "calfree": (Yt, St), "aami": (Ya, Sa)}
    ABS = {"calfree": f64(np.load(R / "WS_ABS_test_pred.npy")), "aami": f64(za["WS_ABS"])}
    subs = {k: np.array([s for s in np.unique(Sx) if (Sx == s).sum() >= (40 if k != "calfree" else 100)])
            for k, (Yx, Sx) in cohorts.items()}
    rows_of = {k: {int(s): np.flatnonzero(cohorts[k][1] == s) for s in subs[k]} for k in cohorts}
    draws = {k: {s: [np.random.default_rng(SEED + 7 * d).choice(r, M, replace=False) for d in range(N_DRAWS)]
                 for s, r in rows_of[k].items()} for k in cohorts}

    report = {"protocol": __doc__, "n_subjects": {k: int(len(v)) for k, v in subs.items()}, "n_training_subjects_ecg": int(len(tr_subs))}
    for j, t in ((0, "SBP"), (1, "DBP")):
        # stage-1 ridge (validation subjects), out-of-fold scores for validation windows
        Yv, Sv = cohorts["val"]
        vs = subs["val"]
        G = np.stack([zc["val"][Sv == s].mean(0) for s in vs]).astype(np.float64)
        yv = np.array([Yv[Sv == s, j].mean() for s in vs])

        def fit(Gm, ym):
            mu, sd = Gm.mean(0), Gm.std(0)
            sd[sd < 1e-9] = 1.0
            a = pick_alpha((Gm - mu) / sd, ym, ALPHAS, seed=SEED)
            cmu, w, b = ridge((Gm - mu) / sd, ym, a)
            return lambda F: ((np.asarray(F, np.float64) - mu) / sd - cmu) @ w + b
        rule = fit(G, yv)
        score = {"calfree": rule(zc["test"]), "aami": rule(za["RAW_w_feat"]), "val": np.empty(len(Sv))}
        folds = np.random.default_rng(SEED).permutation(len(vs)) % 4
        for k in range(4):
            m = np.isin(Sv, vs[folds == k])
            score["val"][m] = fit(G[folds != k], yv[folds != k])(zc["val"][m])
        # ECG ridge level (training subjects)
        ytr = np.array([Y[tr][S[tr] == s, j].mean() for s in tr_subs])
        sc = StandardScaler().fit(E_tr)
        ecg_reg = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(sc.transform(E_tr), ytr)

        def per_draw(key):
            """arrays [subject, draw]: index v, anchor L, ECG level E; plus residual vectors of the current system."""
            Yx, Sx = cohorts[key]
            V, L, E, res, tru, absm = [], [], [], [], [], []
            for s in subs[key]:
                v_, l_, e_, r_, a_ = [], [], [], [], []
                for w in draws[key][int(s)]:
                    sc_ = np.setdiff1d(rows_of[key][int(s)], w)
                    tw = T[key][w, j]
                    v_.append(tw.std())
                    l_.append(score[key][w].mean())
                    fw = np.nanmedian(E_win[key][w], 0)
                    e_.append(ecg_reg.predict(sc.transform(np.where(np.isnan(fw), med, fw)[None]))[0])
                    r_.append(T[key][sc_, j] - tw.mean() - Yx[sc_, j])
                    if key in ABS:
                        a_.append(np.abs(ABS[key][sc_, j] - Yx[sc_, j]).mean())
                V.append(v_), L.append(l_), E.append(e_), res.append(r_), tru.append(Yx[rows_of[key][int(s)], j].mean())
                absm.append(np.mean(a_) if a_ else np.nan)
            return np.array(V), np.array(L), np.array(E), res, np.array(tru), np.array(absm)
        pd = {k: per_draw(k) for k in cohorts}
        Vv, Lv, Ev, _, truev, _ = pd["val"]
        tau = float(np.median(Vv))
        low_v = (Vv <= tau) if GATED else np.ones_like(Vv, bool)
        e_mean = float(Ev.mean())
        b = float(min(np.linspace(0, 1.5, 31),
                      key=lambda b: np.abs((Lv + b * (Ev - e_mean)) - truev[:, None])[low_v].mean()))
        rep_t = {"tau": round(tau, 3), "b_low_branch": round(b, 2), "ecg_level_mean_validation": round(e_mean, 2)}
        for key in ("calfree", "aami"):
            V, L, E, res, tru, absm = pd[key]
            low = (V <= tau) if GATED else np.ones_like(V, bool)
            cur = np.array([[np.abs(L[i, d] + res[i][d]).mean() for d in range(N_DRAWS)] for i in range(len(res))])
            anc = L + np.where(low, b * (E - e_mean), 0.0)
            new = np.array([[np.abs(anc[i, d] + res[i][d]).mean() for d in range(N_DRAWS)] for i in range(len(res))])
            frac_low = low.mean(1)
            blk = {"n": int(len(res)), "fraction_of_draws_in_low_branch_mean": round(float(frac_low.mean()), 3),
                   "current_two_stage": round(float(cur.mean()), 3), "gated_system": round(float(new.mean()), 3)}
            def ci_mean(x):
                ii = rng.integers(0, len(x), (N_BOOT, len(x)))
                return [round(float(x.mean()), 3), [round(float(v), 3) for v in np.percentile(x[ii].mean(1), [2.5, 97.5])]]
            blk["MAE_with_CI"] = {"FLARE": ci_mean(new.mean(1)), "two_stage_without_gate": ci_mean(cur.mean(1)),
                                  "absolute_network": ci_mean(absm)}
            blk["FLARE_minus_absolute"] = ci_mean(new.mean(1) - absm)
            d_all = new.mean(1) - cur.mean(1)
            i = rng.integers(0, len(d_all), (N_BOOT, len(d_all)))
            blk["difference"] = [round(float(d_all.mean()), 3), [round(float(x), 3) for x in np.percentile(d_all[i].mean(1), [2.5, 97.5])]]
            for gname, m in (("mostly_low", frac_low >= 0.5), ("mostly_high", frac_low < 0.5)):
                dd = d_all[m]
                ii = rng.integers(0, len(dd), (N_BOOT, len(dd)))
                blk[gname] = {"n": int(m.sum()), "current": round(float(cur.mean(1)[m].mean()), 3),
                              "gated": round(float(new.mean(1)[m].mean()), 3),
                              "difference": [round(float(dd.mean()), 3), [round(float(x), 3) for x in np.percentile(dd[ii].mean(1), [2.5, 97.5])]]}
            rep_t[key] = blk
        report[t] = rep_t
        print(t, json.dumps(rep_t), flush=True)
    (ROOT / "reports" / ("s93_fluctuation_gated_system.json" if GATED else "s93_ecg_anchor_all_subjects.json")).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote reports/s93_fluctuation_gated_system.json")


if __name__ == "__main__":
    main()
