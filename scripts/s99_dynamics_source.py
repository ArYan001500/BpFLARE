"""s99: where does the preserved within-subject dynamics come from? CalFree (exploratory) and AAMI (untouched).
Models that differ in ONE factor: training objective (absolute vs within-subject pairwise) at fixed backbone
(ResNet seeds 42/44/45, InceptionTime, TCN). Metrics per subject: within-subject Pearson r between estimate and
reference over all windows; SD of the estimate vs SD of the reference (r across subjects, median ratio).
Paired differences PAIRED - ABS with subject bootstrap 2000. Encoder probe: ridge from within-subject-centred
frozen encoder features to within-subject-centred pressure, fitted on validation (lambda chosen on a validation
half), scored on each test cohort; PAIRED vs ABS encoder (ResNet, seed 42).
"""
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
C, R = ROOT / "converted", ROOT / "runs" / "s4_pat_control"
sys.path.insert(0, str(ROOT / "scripts"))
import bp_networks as q1  # noqa: E402
sys.modules.setdefault("network_loader", types.SimpleNamespace(load_q1_module=lambda *a: q1))
import s37_backbones as bb  # noqa: E402
import bb_moderntcn  # noqa: E402,F401  (registers the official ModernTCN backbone)

f64 = lambda a: np.asarray(a, dtype=np.float64)  # noqa: E731
rng = np.random.default_rng(7)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Ya = np.concatenate([f64(np.load(C / p / "Y.npy")) for p in ("aami_cal", "aami_test")])
Sa = np.concatenate([np.load(C / p / "S.npy") for p in ("aami_cal", "aami_test")])
Yt, St = f64(np.load(C / "calfree_test" / "Y.npy")), np.load(C / "calfree_test" / "S.npy")
za, z81 = np.load(R / "s78_aami_predictions.npz"), np.load(R / "s81_aami_predictions.npz")
XA = [np.load(C / p / "X.npy", mmap_mode="r") for p in ("aami_cal", "aami_test")]
NA0 = XA[0].shape[0]


def aami_x(rows):
    return np.concatenate([np.asarray(XA[0][rows[rows < NA0]]), np.asarray(XA[1][rows[rows >= NA0] - NA0])]) if len(rows) else None


def load_model(tag):
    ck = torch.load(R / f"{tag}_weights.pt", map_location="cpu", weights_only=False)
    m = bb.BACKBONES[ck["backbone"]](0.5) if "backbone" in ck else q1.RawResNet1D(dropout=0.5)
    m.load_state_dict(ck["state_dict"])
    return m.to(dev).eval(), ck["signal_mean"], ck["signal_std"]


@torch.no_grad()
def run(tag, getx, n, mode="pred"):
    m, mu, sd = load_model(tag)
    out = []
    for s in range(0, n, 1024):
        x = (getx(np.arange(s, min(n, s + 1024))).astype(np.float32) - mu[None, :, None]) / sd[None, :, None]
        xt = torch.from_numpy(x).to(dev)
        if mode == "pred":
            out.append(m(xt).float().cpu().numpy())
        else:
            e = m.encoder(xt)
            out.append(torch.cat([e.mean(-1), e.amax(-1)], 1).float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def aami_pred(tag):
    for z in (za, z81):
        if tag in z.files:
            return f64(z[tag])
    f = R / f"{tag}_aami_pred_s99.npy"
    if not f.exists():
        np.save(f, run(tag, aami_x, len(Ya)))
    return f64(np.load(f))


def metrics(P, Y, S, j):
    subs = [s for s in np.unique(S) if (S == s).sum() >= 40]
    wr, sp, st = [], [], []
    for s in subs:
        r = S == s
        wr.append(np.corrcoef(P[r, j], Y[r, j])[0, 1])
        sp.append(P[r, j].std())
        st.append(Y[r, j].std())
    return np.array(wr), np.array(sp), np.array(st)


def ci(x):
    i = rng.integers(0, len(x), (2000, len(x)))
    return [round(float(x.mean()), 3), [round(float(v), 3) for v in np.percentile(x[i].mean(1), [2.5, 97.5])]]


PAIRS = [("ResNet seed 42", "WS_ABS", "WS_PAIRED"), ("ResNet seed 44", "WS_ABS_s44", "WS_PAIRED_s44"),
         ("ResNet seed 45", "WS_ABS_s45", "WS_PAIRED_s45"), ("InceptionTime", "BB_INCEPTION_ABS", "BB_INCEPTION_PAIRED"),
         ("TCN", "BB_TCN_ABS", "BB_TCN_PAIRED"), ("ModernTCN (official code)", "BB_MODERNTCN_ABS", "BB_MODERNTCN_PAIRED")]
report = {"protocol": __doc__}
cohorts = {"calfree": (Yt, St, lambda tag: f64(np.load(R / f"{tag}_test_pred.npy"))), "aami": (Ya, Sa, aami_pred)}
for cname, (Y, S, get) in cohorts.items():
    report[cname] = {}
    for j, t in ((0, "SBP"), (1, "DBP")):
        rows = {}
        for name, a_tag, p_tag in PAIRS:
            wa, sa, st = metrics(get(a_tag), Y, S, j)
            wp, spp, _ = metrics(get(p_tag), Y, S, j)
            rows[name] = {"within_r_ABS": round(float(wa.mean()), 3), "within_r_PAIRED": round(float(wp.mean()), 3),
                          "within_r_PAIRED_minus_ABS": ci(wp - wa),
                          "SD_recovery_r_ABS": round(float(np.corrcoef(sa, st)[0, 1]), 3),
                          "SD_recovery_r_PAIRED": round(float(np.corrcoef(spp, st)[0, 1]), 3),
                          "amplitude_ratio_ABS": round(float(np.median(sa / st)), 2),
                          "amplitude_ratio_PAIRED": round(float(np.median(spp / st)), 2)}
        report[cname][t] = rows
        print(cname, t, json.dumps({k: (v["within_r_ABS"], v["within_r_PAIRED"], v["within_r_PAIRED_minus_ABS"], v["amplitude_ratio_ABS"], v["amplitude_ratio_PAIRED"]) for k, v in rows.items()}), flush=True)

# ---------------- encoder probe
va = np.load(C / "train" / "rows_validation.npy")
Xv = np.load(C / "train" / "X.npy", mmap_mode="r")
Yv, Sv = f64(np.load(C / "train" / "Y.npy"))[va], np.load(C / "train" / "S.npy")[va]
Xt = np.load(C / "calfree_test" / "X.npy", mmap_mode="r")


def centre(A, S):
    A = A.copy()
    for s in np.unique(S):
        m = S == s
        A[m] -= A[m].mean(0)
    return A


probe = {}
for tag in ("WS_ABS", "WS_PAIRED"):
    Fv = centre(run(tag, lambda r: np.asarray(Xv[va[r]]), len(va), "enc"), Sv)
    Fte = {"calfree": centre(run(tag, lambda r: np.asarray(Xt[r]), len(Yt), "enc"), St),
           "aami": centre(run(tag, aami_x, len(Ya), "enc"), Sa)}
    mu, sd = Fv.mean(0), Fv.std(0) + 1e-9
    Z = (Fv - mu) / sd
    vs = np.unique(Sv)
    half = np.isin(Sv, vs[np.random.default_rng(1).permutation(len(vs))[: len(vs) // 2]])
    probe[tag] = {}
    for j, t in ((0, "SBP"), (1, "DBP")):
        y = centre(Yv[:, [j]], Sv)[:, 0]
        fit = lambda m, lam: np.linalg.solve(Z[m].T @ Z[m] + lam * np.eye(Z.shape[1]), Z[m].T @ y[m])  # noqa: E731
        wr = lambda p, yy, S: np.mean([np.corrcoef(p[S == s], yy[S == s])[0, 1] for s in np.unique(S)])  # noqa: E731
        lam = max((1e1, 1e2, 1e3, 1e4, 1e5), key=lambda l: wr(Z[~half] @ fit(half, l), y[~half], Sv[~half]))
        w = fit(np.ones(len(y), bool), lam)
        probe[tag][t] = {}
        for cname, (Y, S, _) in cohorts.items():
            p = ((Fte[cname] - mu) / sd) @ w
            subs = [s for s in np.unique(S) if (S == s).sum() >= 40]
            probe[tag][t][cname] = round(float(np.mean([np.corrcoef(p[S == s], Y[S == s, j])[0, 1] for s in subs])), 3)
report["encoder_probe_within_r"] = probe
print("probe", json.dumps(probe), flush=True)
(ROOT / "reports" / "s99_dynamics_source.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
