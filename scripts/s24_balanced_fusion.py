"""Put the two branches on the same scale before fusing them, and see whether the phase survives.

The gated model was measured stage by stage. Zeroing the phase channels changes the image
encoder's output by 43 percent, the fused representation by 8 percent, and the final prediction by
0.14 percent. The gate is not the culprit: its mean weight is 0.5134, an even split. The culprit is
magnitude. The raw branch presents a vector of average size 0.345 to the fusion and the coherence
branch one of size 0.070, so a gate that looks balanced is mixing a loud voice with a quiet one and
the quiet one is five times quieter.

That makes every negative result about the phase in this project conditional. A branch whose
contribution is attenuated fivefold before it reaches the head cannot be said to have nothing to
offer; it can only be said not to have reached the head.

The fix is one line of architecture: normalise each branch to unit scale before the gate combines
them, so the gate actually decides the mixture. Two arms are trained, identical in every other
respect and to the manuscript recipe:

  BALANCED       LayerNorm on both branch outputs, phase channels present
  BALANCED_MAG   the same, with the phase channels zeroed

Their difference is the phase effect measured in an architecture that can express it. Everything
else follows the recipe fixed in stage two, at seed 42, so these are comparable to CWC3 and
MAGONLY directly.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
CONVERTED = ROOT / "converted"
CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
RUNS = ROOT / "runs" / "s4_pat_control"
from network_loader import load_q1_module  # noqa: E402

TARGETS = ("SBP", "DBP", "MAP")
TARGET_WEIGHTS = (2.0, 1.2, 0.8)
TEST_TAG = "calfree_test"
BATCH, LR, WEIGHT_DECAY, DROPOUT = 256, 1e-4, 1e-4, 0.5
MAX_EPOCHS, PATIENCE, SEED = 14, 4, 42
ARMS = ("BALANCED", "BALANCED_MAG")


class BalancedGatedNet(nn.Module):
    """The manuscript's gated network with both branch outputs normalised before the gate.

    Nothing else changes: same encoders, same gate, same head, same parameter count apart from the
    two normalisation layers, which add 512 parameters between them.
    """

    def __init__(self, q1, dropout=DROPOUT):
        super().__init__()
        self.base = q1.GatedRawCWCFusionNet(dropout=dropout)
        self.norm_raw = nn.LayerNorm(128)
        self.norm_cwc = nn.LayerNorm(128)

    def fuse(self, x, img):
        b = self.base
        e = b.raw_encoder(x)
        h_raw = self.norm_raw(b.raw_proj(torch.cat([e.mean(-1), e.amax(-1)], dim=1)))
        h_cwc = self.norm_cwc(b.cwc_proj(b.cwc_encoder(img).mean(dim=(2, 3))))
        g = b.gate(torch.cat([h_raw, h_cwc], dim=1))
        return g * h_raw + (1.0 - g) * h_cwc, g, h_raw, h_cwc

    def forward(self, x, img):
        fused, _, _, _ = self.fuse(x, img)
        return self.base.head(fused)


def metrics(pred, true, subjects):
    out = {}
    for j, t in enumerate(TARGETS):
        e = pred[:, j] - true[:, j]
        out[t] = {"mae": float(np.abs(e).mean()), "me": float(e.mean()),
                  "sd": float(e.std(ddof=1)),
                  "pearson_r": float(np.corrcoef(pred[:, j], true[:, j])[0, 1])}
    out["selection_score"] = float(
        1.5 * out["SBP"]["mae"] + out["DBP"]["mae"] + 0.75 * out["MAP"]["mae"])
    order = np.argsort(subjects, kind="stable")
    s, p, t_ = subjects[order], pred[order], true[order]
    uniq, starts = np.unique(s, return_index=True)
    ends = np.append(starts[1:], len(s))
    dec = {}
    for j, target in enumerate(TARGETS):
        bias, wmae, wr = [], [], []
        for a, b in zip(starts, ends):
            if b - a < 5:
                continue
            pv, tv = p[a:b, j], t_[a:b, j]
            bias.append(pv.mean() - tv.mean())
            pc, tc = pv - pv.mean(), tv - tv.mean()
            wmae.append(np.abs(pc - tc).mean())
            if pc.std() > 1e-9 and tc.std() > 1e-9:
                wr.append(np.corrcoef(pc, tc)[0, 1])
        dec[target] = {"between_subject_bias_mae": float(np.abs(bias).mean()),
                       "within_subject_mae": float(np.mean(wmae)),
                       "within_subject_r_mean": float(np.mean(wr))}
    out["error_decomposition"] = dec
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    args = ap.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q1 = load_q1_module()

    d, dt = CONVERTED / "train", CONVERTED / TEST_TAG
    X = np.load(d / "X.npy", mmap_mode="r")
    Y = np.asarray(np.load(d / "Y.npy"), dtype=np.float64)
    S = np.load(d / "S.npy")
    IM = np.load(CACHE / "train" / "images.npy", mmap_mode="r")
    tr, va = np.load(d / "rows_train.npy"), np.load(d / "rows_validation.npy")
    Xt = np.load(dt / "X.npy", mmap_mode="r")
    Yt = np.asarray(np.load(dt / "Y.npy"), dtype=np.float64)
    St = np.load(dt / "S.npy")
    IMt = np.load(CACHE / TEST_TAG / "images.npy", mmap_mode="r")
    assert not (set(json.loads((d / "subjects.json").read_text()))
                & set(json.loads((dt / "subjects.json").read_text()))), "subject leak"

    sub = tr[:: max(1, len(tr) // 40000)]
    xs = np.asarray(X[sub], dtype=np.float64)
    mean = xs.mean(axis=(0, 2)).astype(np.float32)
    std = np.maximum(xs.std(axis=(0, 2)), 1e-6).astype(np.float32)

    def make(Xa, IMa, idx, zero_phase):
        x = (np.asarray(Xa[idx], dtype=np.float32) - mean[None, :, None]) / std[None, :, None]
        img = np.asarray(IMa[idx, :3], dtype=np.float32)
        img[:, 0] /= 255.0
        img[:, 1:] = img[:, 1:] / 127.5 - 1.0
        if zero_phase:
            img[:, 1:] = 0.0
        return torch.from_numpy(x).to(device), torch.from_numpy(img).to(device)

    results = {}
    for arm in [a for a in args.arms if a in ARMS]:
        name = f"{arm}{args.suffix}"
        out_path = RUNS / f"{name}.json"
        if out_path.is_file():
            results[arm] = json.loads(out_path.read_text())
            print(f"  [{name}] already done", flush=True)
            continue
        zero_phase = arm.endswith("_MAG")

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = BalancedGatedNet(q1, DROPOUT).to(device)
        with torch.no_grad():
            model.base.head[-1].bias.copy_(torch.tensor(Y[tr].mean(axis=0), dtype=torch.float32))

        steps = len(tr) // BATCH
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.max_epochs * steps, eta_min=1e-6)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=4096.0)
        rng = np.random.default_rng(args.seed)
        best = {"score": float("inf"), "epoch": -1, "state": None}
        started = time.perf_counter()

        print(f"===== {name}  (phase {'zeroed' if zero_phase else 'present'}) =====", flush=True)
        for epoch in range(1, args.max_epochs + 1):
            model.train()
            order = rng.permutation(tr)
            running = 0.0
            for s in range(steps):
                idx = np.sort(order[s * BATCH: (s + 1) * BATCH])
                xb, img = make(X, IM, idx, zero_phase)
                yb = torch.from_numpy(Y[idx].astype(np.float32)).to(device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    loss = q1.weighted_huber_loss_custom(model(xb, img), yb, TARGET_WEIGHTS)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                running += float(loss.detach())

            model.eval()
            preds = []
            with torch.no_grad():
                for s in range(0, len(va), 512):
                    xb, img = make(X, IM, va[s: s + 512], zero_phase)
                    preds.append(model(xb, img).float().cpu().numpy())
            vm = metrics(np.concatenate(preds).astype(np.float64), Y[va], S[va])
            print(f"  ep{epoch:2d} train={running / steps:7.3f} val={vm['selection_score']:.4f} "
                  f"SBP={vm['SBP']['mae']:.3f}", flush=True)
            if vm["selection_score"] < best["score"]:
                best = {"score": vm["selection_score"], "epoch": epoch,
                        "state": {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}}
            if epoch - best["epoch"] >= args.patience:
                print(f"  [{name}] early stop at epoch {epoch}, best was {best['epoch']}",
                      flush=True)
                break

        model.load_state_dict(best["state"])
        model.eval()
        preds, gates, scales = [], [], []
        with torch.no_grad():
            for s in range(0, len(Yt), 512):
                idx = np.arange(s, min(s + 512, len(Yt)))
                xb, img = make(Xt, IMt, idx, zero_phase)
                fused, g, hr, hc = model.fuse(xb, img)
                preds.append(model.base.head(fused).float().cpu().numpy())
                gates.append(g.float().cpu().numpy())
                scales.append(np.stack([hr.abs().mean(1).float().cpu().numpy(),
                                        hc.abs().mean(1).float().cpu().numpy()], axis=1))
        pred = np.concatenate(preds).astype(np.float64)
        sc = np.concatenate(scales)

        results[arm] = {"arm": arm, "seed": args.seed, "phase_zeroed": zero_phase,
                        "best_epoch": best["epoch"], "validation_score": best["score"],
                        "mean_gate_toward_raw": float(np.concatenate(gates).mean()),
                        "branch_scale_raw": float(sc[:, 0].mean()),
                        "branch_scale_cwc": float(sc[:, 1].mean()),
                        "minutes": (time.perf_counter() - started) / 60,
                        "test": metrics(pred, Yt, St)}
        np.save(RUNS / f"{name}_test_pred.npy", pred)
        torch.save({"state_dict": best["state"], "arm": arm, "seed": args.seed,
                    "signal_mean": mean, "signal_std": std}, RUNS / f"{name}_weights.pt")
        out_path.write_text(json.dumps(results[arm], indent=2), encoding="utf-8")
        t = results[arm]["test"]
        print(f"  -> test SBP {t['SBP']['mae']:.4f}  DBP {t['DBP']['mae']:.4f}  "
              f"gate {results[arm]['mean_gate_toward_raw']:.4f}  "
              f"scales raw {results[arm]['branch_scale_raw']:.4f} "
              f"cwc {results[arm]['branch_scale_cwc']:.4f}  "
              f"({results[arm]['minutes']:.1f} min)", flush=True)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "s24_balanced_fusion.json").write_text(json.dumps(
        {"analysis_id": "s24_balanced_fusion",
         "created_utc": datetime.now(timezone.utc).isoformat(),
         "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
