"""s89 (exploratory): variability-aware blending of the two-stage system and the absolute network.

Question. The pairwise tracker helps most when within-subject pressure varies strongly and hurts on
low-variability records (UCI). Can a variability index computed from the unlabeled warm-up windows
decide, per subject, how much to trust the two-stage system versus the absolute network?

Protocol (fixed before looking at test results):
  * Warm-up: M = 20 random windows per subject, 50 draws (seed 42 + 7 d), as in s70.
  * Two-stage prediction on scored windows: tracker + (mean level score over warm-up - mean tracker
    output over warm-up). Absolute prediction: WS_ABS output. Single tracker WS_PAIRED (seed 42).
  * Variability index v: standard deviation of the tracker output over the warm-up windows, per target.
    It uses no label.
  * Blend: y = w * two_stage + (1 - w) * absolute, with w = clip(a + b * v, 0, 1) (linear rule) or
    w = 1[v > t] (threshold rule). a, b, t are fitted on the 194 validation subjects only, by grid
    search on mean per-subject MAE. The level rule for validation subjects is out-of-fold (4 folds).
  * Evaluation: CalFree (144 subjects) and UCI (records with >= 30 windows). Paired differences
    against the two-stage system with 2000 subject-level bootstrap replicates.
Caveat: both networks were checkpoint-selected on the validation subjects, so their validation
predictions are not fully out-of-sample.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CONVERTED = ROOT / "converted"
RUNS = ROOT / "runs" / "s4_pat_control"
REPORTS = ROOT / "reports"
sys.path.insert(0, str(ROOT / "scripts"))
import bp_networks as q1  # noqa: E402

if "network_loader" not in sys.modules:   # legacy import of s29, not used here
    import types
    sys.modules["network_loader"] = types.SimpleNamespace(load_q1_module=lambda *a: q1)
from s29_level_information import pick_alpha, ridge  # noqa: E402

ALPHAS = (1.0, 10.0, 100.0, 1000.0, 1e4, 1e5)
SEED, M, N_DRAWS, N_BOOT = 42, 20, 50, 2000
WGRID = np.round(np.linspace(0, 1, 21), 2)
TARGETS = ((0, "SBP"), (1, "DBP"))


@torch.no_grad()
def predict(tag, X, rows, device, batch=1024):
    ck = torch.load(RUNS / f"{tag}_weights.pt", map_location="cpu", weights_only=False)
    model = q1.RawResNet1D(dropout=0.5)
    model.load_state_dict(ck["state_dict"])
    model = model.to(device).eval()
    mean, std = ck["signal_mean"], ck["signal_std"]
    out = np.empty((len(rows), 3), dtype=np.float32)
    for s in range(0, len(rows), batch):
        sel = rows[s: s + batch]
        x = (np.asarray(X[sel], dtype=np.float32) - mean[None, :, None]) / std[None, :, None]
        out[s: s + batch] = model(torch.from_numpy(x).to(device)).float().cpu().numpy()
    return out


def subj_feats(f, S, subs):
    return np.stack([f[S == g].mean(0) for g in subs]).astype(np.float64)


def fit_level(G, yv, j):
    mu, sd = G.mean(0), G.std(0)
    sd[sd < 1e-9] = 1.0
    a = pick_alpha((G - mu) / sd, yv, ALPHAS, seed=SEED)
    cmu, w, b = ridge((G - mu) / sd, yv, a)
    return lambda F: ((F.astype(np.float64) - mu) / sd - cmu) @ w + b


def tables(subs, rows_by, Y, track, base, lvl, j):
    """Per subject and draw: MAE for every blend weight on the grid, and the variability index."""
    E = np.empty((len(subs), N_DRAWS, len(WGRID)))
    V = np.empty((len(subs), N_DRAWS))
    for i, s in enumerate(subs):
        r = rows_by[int(s)]
        for d in range(N_DRAWS):
            w_rows = np.random.default_rng(SEED + 7 * d).choice(r, size=M, replace=False)
            sc = np.setdiff1d(r, w_rows)
            two = track[sc, j] + (lvl[w_rows].mean() - track[w_rows, j].mean())
            ab = base[sc, j]
            y = Y[sc, j]
            E[i, d] = np.abs(WGRID[:, None] * two[None] + (1 - WGRID[:, None]) * ab[None] - y[None]).mean(1)
            V[i, d] = track[w_rows, j].std()
    return E, V


def apply_rule(E, V, rule):
    w = np.clip(rule(V), 0, 1)
    k = np.rint(w / 0.05).astype(int)
    return np.take_along_axis(E, k[..., None], axis=2)[..., 0].mean(1)   # per-subject MAE


def ci(x, rng):
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pass
    # ---------- data
    Xv = np.load(CONVERTED / "train" / "X.npy", mmap_mode="r")
    Yv_all = np.asarray(np.load(CONVERTED / "train" / "Y.npy"), dtype=np.float64)
    Sv_all = np.load(CONVERTED / "train" / "S.npy")
    vrows = np.load(CONVERTED / "train" / "rows_validation.npy")
    Yv, Sv = Yv_all[vrows], Sv_all[vrows]
    Yt = np.asarray(np.load(CONVERTED / "calfree_test" / "Y.npy"), dtype=np.float64)
    St = np.load(CONVERTED / "calfree_test" / "S.npy")
    Yu = np.asarray(np.load(CONVERTED / "uci_all" / "Y.npy"), dtype=np.float64)
    Su = np.load(CONVERTED / "uci_all" / "S.npy")
    zc = np.load(RUNS / "RAW_w_feat_cache.npz")
    zu = np.load(RUNS / "s69_uci_cache_all.npz")

    # ---------- validation predictions (cached), with a sanity check against saved test predictions
    preds_v = {}
    for tag in ("WS_PAIRED", "WS_ABS"):
        f = RUNS / f"{tag}_val_pred_s89.npy"
        if not f.exists():
            np.save(f, predict(tag, Xv, vrows, device))
        preds_v[tag] = np.asarray(np.load(f), dtype=np.float64)
    Xt = np.load(CONVERTED / "calfree_test" / "X.npy", mmap_mode="r")
    chk = predict("WS_PAIRED", Xt, np.arange(2000), device)
    saved = np.load(RUNS / "WS_PAIRED_test_pred.npy")[:2000]
    sanity = float(np.abs(chk - saved).max())
    print(f"sanity: max |re-predicted - saved| on 2000 test windows = {sanity:.4f} mmHg", flush=True)

    track_t, base_t = (np.asarray(np.load(RUNS / f"{t}_test_pred.npy"), dtype=np.float64) for t in ("WS_PAIRED", "WS_ABS"))
    track_u, base_u = (np.asarray(zu[t], dtype=np.float64) for t in ("WS_PAIRED", "WS_ABS"))

    vsubs = np.array([g for g in np.unique(Sv) if (Sv == g).sum() >= 40])
    tsubs = np.array([g for g in np.unique(St) if (St == g).sum() >= 100])
    usubs = np.array([g for g in np.unique(Su) if (Su == g).sum() >= 30])
    rb = lambda S, subs: {int(g): np.flatnonzero(S == g) for g in subs}  # noqa: E731
    rv, rt, ru = rb(Sv, vsubs), rb(St, tsubs), rb(Su, usubs)
    print(f"validation {len(vsubs)}, CalFree {len(tsubs)}, UCI {len(usubs)}", flush=True)

    out = {"protocol": __doc__, "sanity_max_abs_diff_mmHg": sanity,
           "n": {"validation": len(vsubs), "calfree": len(tsubs), "uci": len(usubs)}}
    rng = np.random.default_rng(7)
    for j, t in TARGETS:
        # level scores: test/UCI from rule fitted on all validation subjects; validation out-of-fold
        Gv = subj_feats(zc["val"], Sv, vsubs)
        yv = np.array([Yv[Sv == g, j].mean() for g in vsubs])
        rule_all = fit_level(Gv, yv, j)
        lvl_t, lvl_u = rule_all(zc["test"]), rule_all(zu["feats"])
        lvl_v = np.empty(len(Sv))
        folds = np.random.default_rng(SEED).permutation(len(vsubs)) % 4
        for k in range(4):
            r_k = fit_level(Gv[folds != k], yv[folds != k], j)
            m = np.isin(Sv, vsubs[folds == k])
            lvl_v[m] = r_k(zc["val"][m])

        Ev, Vv = tables(vsubs, rv, Yv, preds_v["WS_PAIRED"], preds_v["WS_ABS"], lvl_v, j)
        Et, Vt = tables(tsubs, rt, Yt, track_t, base_t, lvl_t, j)
        Eu, Vu = tables(usubs, ru, Yu, track_u, base_u, lvl_u, j)

        # fit rules on validation only
        best_lin, best_thr = None, None
        for a in np.round(np.arange(-3, 3.01, 0.1), 2):
            for b in np.round(np.arange(-0.5, 0.501, 0.01), 3):
                mae = apply_rule(Ev, Vv, lambda V: a + b * V).mean()
                if best_lin is None or mae < best_lin[0]:
                    best_lin = (mae, a, b)
        for th in np.round(np.arange(0, 25.01, 0.25), 2):
            mae = apply_rule(Ev, Vv, lambda V: (V > th).astype(float)).mean()
            if best_thr is None or mae < best_thr[0]:
                best_thr = (mae, th)
        _, a, b = best_lin
        th = best_thr[1]
        rules = {"two_stage": lambda V: np.ones_like(V), "absolute": lambda V: np.zeros_like(V),
                 "linear_rule": lambda V: a + b * V, "threshold_rule": lambda V: (V > th).astype(float)}
        res = {"fitted_on_validation": {"linear": {"a": float(a), "b": float(b), "val_mae": float(best_lin[0])},
                                        "threshold": {"t": float(th), "val_mae": float(best_thr[0])}}}
        print(f"\n{t}: linear w = clip({a:+.2f} {b:+.3f} v); threshold v > {th}", flush=True)
        for name, E, V, Ytr, rows, subs in (("validation", Ev, Vv, Yv, rv, vsubs), ("calfree", Et, Vt, Yt, rt, tsubs),
                                            ("uci", Eu, Vu, Yu, ru, usubs)):
            per = {k: apply_rule(E, V, f) for k, f in rules.items()}
            oracle = np.minimum(per["two_stage"], per["absolute"])
            true_sd = np.array([Ytr[rows[int(g)], j].std() for g in subs])
            r_v = float(np.corrcoef(V.mean(1), true_sd)[0, 1])
            block = {"mae": {k: float(v.mean()) for k, v in per.items()},
                     "oracle_choice_mae": float(oracle.mean()),
                     "r_index_vs_true_sd": r_v,
                     "diff_vs_two_stage": {k: {"mean": float((per[k] - per["two_stage"]).mean()),
                                               "ci": ci(per[k] - per["two_stage"], rng)}
                                           for k in ("absolute", "linear_rule", "threshold_rule")},
                     "mean_weight_linear": float(np.clip(a + b * V, 0, 1).mean())}
            res[name] = block
            print(f"  {name:10s} two-stage {block['mae']['two_stage']:.2f} | abs {block['mae']['absolute']:.2f} | "
                  f"linear {block['mae']['linear_rule']:.2f} {block['diff_vs_two_stage']['linear_rule']['ci']} | "
                  f"threshold {block['mae']['threshold_rule']:.2f} | oracle {block['oracle_choice_mae']:.2f} | "
                  f"r(v,trueSD) {r_v:.2f} | mean w {block['mean_weight_linear']:.2f}", flush=True)
        out[t] = res
    (REPORTS / "s89_variability_gate.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\nwrote reports/s89_variability_gate.json")


if __name__ == "__main__":
    main()
