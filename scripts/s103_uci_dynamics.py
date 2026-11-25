"""s103: external test of the dynamics result on the UCI cuffless dataset (all four parts; records are temporally
ordered; no subject identifiers, so the record is the unit). Prediction written before running: the gain of the
within-subject objective in within-record correlation is smaller than on VitalDB and may vanish for some
architectures, because UCI records vary little.
Same six pairs as s99 (absolute vs within-subject pairwise objective at a fixed architecture), all frozen from
VitalDB training, no refitting. Records with at least 40 windows. Per record: Pearson r between estimate and
reference over the record's windows (levels); Pearson r between consecutive window-to-window changes of estimate and
reference (temporal changes, possible because UCI windows are in time order); amplitude ratio SD(estimate)/SD(ref).
Paired differences PAIRED - ABS with 2000 record-bootstrap replicates. Also reported by tertile of the record's
reference SBP standard deviation (descriptive).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

import types  # noqa: E402

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "runs" / "s4_pat_control"
U = ROOT / "converted" / "uci_all"
sys.path.insert(0, str(ROOT / "scripts"))
import bp_networks as q1  # noqa: E402
sys.modules.setdefault("network_loader", types.SimpleNamespace(load_q1_module=lambda *a: q1))
import s37_backbones as bb  # noqa: E402
import bb_moderntcn  # noqa: E402,F401  (registers the official ModernTCN backbone)

f64 = lambda a: np.asarray(a, dtype=np.float64)  # noqa: E731
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAIRS = [("ResNet seed 42", "WS_ABS", "WS_PAIRED"), ("ResNet seed 44", "WS_ABS_s44", "WS_PAIRED_s44"),
         ("ResNet seed 45", "WS_ABS_s45", "WS_PAIRED_s45"), ("InceptionTime", "BB_INCEPTION_ABS", "BB_INCEPTION_PAIRED"),
         ("TCN", "BB_TCN_ABS", "BB_TCN_PAIRED"), ("ModernTCN (official code)", "BB_MODERNTCN_ABS", "BB_MODERNTCN_PAIRED")]


def load_model(tag):
    ck = torch.load(R / f"{tag}_weights.pt", map_location="cpu", weights_only=False)
    m = bb.BACKBONES[ck["backbone"]](0.5) if "backbone" in ck else q1.RawResNet1D(dropout=0.5)
    m.load_state_dict(ck["state_dict"])
    return m.to(dev).eval(), ck["signal_mean"], ck["signal_std"]


@torch.no_grad()
def run(tag, getx, n):
    m, mu, sd = load_model(tag)
    out = []
    for s in range(0, n, 1024):
        x = (getx(np.arange(s, min(n, s + 1024))).astype(np.float32) - mu[None, :, None]) / sd[None, :, None]
        out.append(m(torch.from_numpy(x).to(dev)).float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


rng = np.random.default_rng(7)
Y, S = f64(np.load(U / "Y.npy")), np.load(U / "S.npy", allow_pickle=True)
X = np.load(U / "X.npy", mmap_mode="r")


def pred(tag):
    for f in (R / f"UCI_{tag}_pred_all.npy", R / f"{tag}_uci_all_pred_s103.npy"):
        if f.exists():
            return f64(np.load(f))
    t0 = time.time()
    P = run(tag, lambda r: np.asarray(X[r]), len(Y))
    np.save(R / f"{tag}_uci_all_pred_s103.npy", P)
    print(f"  predicted {tag} in {time.time() - t0:.0f}s", flush=True)
    return P


u, c = np.unique(S, return_counts=True)
recs = u[c >= 40]
idx = [np.flatnonzero(S == s) for s in recs]
sd_ref = np.array([Y[i, 0].std() for i in idx])
terc = np.digitize(sd_ref, np.percentile(sd_ref, [100 / 3, 200 / 3]))


def per_record(P, j):
    lv, ch, amp = [], [], []
    for i in idx:
        p, y = P[i, j], Y[i, j]
        lv.append(np.corrcoef(p, y)[0, 1] if p.std() > 0 and y.std() > 0 else np.nan)
        dp, dy = np.diff(p), np.diff(y)
        ch.append(np.corrcoef(dp, dy)[0, 1] if dp.std() > 0 and dy.std() > 0 else np.nan)
        amp.append(p.std() / y.std() if y.std() > 0 else np.nan)
    return np.array(lv), np.array(ch), np.array(amp)


def ci(x):
    x = x[np.isfinite(x)]
    ii = rng.integers(0, len(x), (2000, len(x)))
    return [round(float(x.mean()), 3), [round(float(v), 3) for v in np.percentile(x[ii].mean(1), [2.5, 97.5])]]


report = {"protocol": __doc__, "n_records": int(len(recs)), "median_windows": float(np.median(c[c >= 40])),
          "median_record_SBP_SD": round(float(np.median(sd_ref)), 2)}
for j, t in ((0, "SBP"), (1, "DBP")):
    report[t] = {}
    for name, a_tag, p_tag in PAIRS:
        la, ca, aa = per_record(pred(a_tag), j)
        lp, cp, ap = per_record(pred(p_tag), j)
        row = {"within_r_ABS": round(float(np.nanmean(la)), 3), "within_r_PAIRED": round(float(np.nanmean(lp)), 3),
               "within_r_diff": ci(lp - la),
               "change_r_ABS": round(float(np.nanmean(ca)), 3), "change_r_PAIRED": round(float(np.nanmean(cp)), 3),
               "change_r_diff": ci(cp - ca),
               "amplitude_ABS": round(float(np.nanmedian(aa)), 2), "amplitude_PAIRED": round(float(np.nanmedian(ap)), 2),
               "within_r_diff_by_SBP_SD_tertile": [ci((lp - la)[terc == k]) for k in range(3)]}
        report[t][name] = row
        print(t, name, json.dumps({k: row[k] for k in ("within_r_ABS", "within_r_PAIRED", "within_r_diff", "change_r_diff", "amplitude_ABS", "amplitude_PAIRED")}), flush=True)
report["SBP_SD_tertile_edges"] = [round(float(v), 2) for v in np.percentile(sd_ref, [100 / 3, 200 / 3])]
(ROOT / "reports" / "s103_uci_dynamics.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
print("wrote reports/s103_uci_dynamics.json")
