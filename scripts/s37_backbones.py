"""Does the paired objective help because of the objective, or because of our particular network?

On the raw 1D ResNet, replacing the absolute pressure loss with a loss on within-subject pressure
differences moved the calibrated error from 8.84 to 7.25 mmHg. That is five times the selection
noise, but it is one architecture, and a single architecture cannot separate "this objective is
better" from "this objective happens to suit this network".

So the same comparison is run again on two backbones that share nothing with the ResNet beyond
being one-dimensional convolutional networks:

    INCEPTION   multi-scale inception blocks, three kernel widths in parallel per block
    TCN         a dilated convolution stack, receptive field grown by dilation rather than depth

Each is trained twice, once with the absolute loss and once with the paired loss, everything else
identical: same batches built by subject, same seed, same schedule, same selection rule, same test
set. If the gap survives on all three backbones, the claim is about the objective. If it appears
only on the ResNet, the claim is about our network and is much weaker.
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
REPORTS = ROOT / "reports"
RUNS = ROOT / "runs" / "s4_pat_control"
sys.path.insert(0, str(ROOT / "scripts"))
from s35_within_subject_objective import (  # noqa: E402
    SUBJ_PER_BATCH, WIN_PER_SUBJ, TARGET_WEIGHTS, paired_loss, tracking_scores, weighted_huber)

TEST_TAG = "calfree_test"
LR, WEIGHT_DECAY, DROPOUT, SEED = 1e-4, 1e-4, 0.5, 42
MAX_EPOCHS, CHECKS_PER_EPOCH, PATIENCE_CHECKS = 6, 4, 8


class Head(nn.Module):
    """Shared read-out so the backbones differ only in how they turn signal into features."""

    def __init__(self, channels, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * channels, 128), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(128, 3))

    def forward(self, e):
        return self.net(torch.cat([e.mean(-1), e.amax(-1)], dim=1))


class InceptionBlock(nn.Module):
    """Three kernel widths in parallel plus a pooled path, as in InceptionTime."""

    def __init__(self, cin, c=32):
        super().__init__()
        self.bottleneck = nn.Conv1d(cin, c, 1, bias=False) if cin > 1 else nn.Identity()
        cb = c if cin > 1 else cin
        self.convs = nn.ModuleList([nn.Conv1d(cb, c, k, padding=k // 2, bias=False)
                                    for k in (11, 21, 41)])
        self.pool = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1),
                                  nn.Conv1d(cin, c, 1, bias=False))
        self.bn = nn.BatchNorm1d(4 * c)
        self.act = nn.SiLU()

    def forward(self, x):
        b = self.bottleneck(x)
        return self.act(self.bn(torch.cat([f(b) for f in self.convs] + [self.pool(x)], dim=1)))


class InceptionTime1D(nn.Module):
    def __init__(self, dropout=DROPOUT, c=32, depth=6):
        super().__init__()
        blocks, cin = [], 2
        for i in range(depth):
            blocks.append(InceptionBlock(cin, c))
            cin = 4 * c
            if i % 2 == 1:
                blocks.append(nn.MaxPool1d(2))
        self.body = nn.Sequential(*blocks)
        self.head = Head(cin, dropout)

    def forward(self, x):
        return self.head(self.body(x))


class TCNBlock(nn.Module):
    def __init__(self, cin, cout, dilation, dropout):
        super().__init__()
        pad = 4 * dilation
        self.conv = nn.Sequential(
            nn.Conv1d(cin, cout, 9, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(cout), nn.SiLU(), nn.Dropout(dropout * 0.2),
            nn.Conv1d(cout, cout, 9, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(cout))
        self.skip = nn.Conv1d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


class TCN1D(nn.Module):
    def __init__(self, dropout=DROPOUT, c=64):
        super().__init__()
        layers, cin = [], 2
        for d in (1, 2, 4, 8, 16, 32):
            layers.append(TCNBlock(cin, c, d, dropout))
            cin = c
            layers.append(nn.MaxPool1d(2))
        self.body = nn.Sequential(*layers)
        self.head = Head(c, dropout)

    def forward(self, x):
        return self.head(self.body(x))


BACKBONES = {"INCEPTION": InceptionTime1D, "TCN": TCN1D}
OBJECTIVES = {"ABS": 0.0, "PAIRED": 1.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="*", default=list(BACKBONES))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d, dt = CONVERTED / "train", CONVERTED / TEST_TAG
    X = np.load(d / "X.npy", mmap_mode="r")
    Y = np.asarray(np.load(d / "Y.npy"), dtype=np.float64)
    S = np.load(d / "S.npy")
    tr, va = np.load(d / "rows_train.npy"), np.load(d / "rows_validation.npy")
    Xt = np.load(dt / "X.npy", mmap_mode="r")
    Yt = np.asarray(np.load(dt / "Y.npy"), dtype=np.float64)
    St = np.load(dt / "S.npy")

    rows_of = {int(s): tr[S[tr] == s] for s in np.unique(S[tr])
               if (S[tr] == s).sum() >= WIN_PER_SUBJ}
    train_subjects = np.array(sorted(rows_of))
    sub = tr[:: max(1, len(tr) // 40000)]
    xs = np.asarray(X[sub], dtype=np.float64)
    mean = xs.mean(axis=(0, 2)).astype(np.float32)
    std = np.maximum(xs.std(axis=(0, 2)), 1e-6).astype(np.float32)

    def signals(Xa, idx):
        x = (np.asarray(Xa[idx], dtype=np.float32) - mean[None, :, None]) / std[None, :, None]
        return torch.from_numpy(x).to(device)

    results = {}
    for bname in [b for b in args.backbones if b in BACKBONES]:
        for oname, pair_w in OBJECTIVES.items():
            tag = f"BB_{bname}_{oname}"
            out_path = RUNS / f"{tag}.json"
            if out_path.is_file():
                results[tag] = json.loads(out_path.read_text())
                print(f"  [{tag}] already done", flush=True)
                continue
            abs_w = 1.0 - pair_w

            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            model = BACKBONES[bname](DROPOUT).to(device)
            nparam = sum(p.numel() for p in model.parameters())
            with torch.no_grad():
                model.head.net[-1].bias.copy_(
                    torch.tensor(np.zeros(3) if abs_w == 0 else Y[tr].mean(axis=0),
                                 dtype=torch.float32))

            steps = len(tr) // (SUBJ_PER_BATCH * WIN_PER_SUBJ)
            check_at = {int(round(steps * f / CHECKS_PER_EPOCH)) - 1
                        for f in range(1, CHECKS_PER_EPOCH + 1)}
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=MAX_EPOCHS * steps, eta_min=1e-6)
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda",
                                          init_scale=4096.0)
            rng = np.random.default_rng(args.seed)
            best = {"score": float("inf"), "at": -1.0, "state": None}
            since_best, started, stop = 0, time.perf_counter(), False
            print(f"===== {tag}  ({nparam / 1e6:.2f}M parameters) =====", flush=True)

            @torch.no_grad()
            def validate():
                model.eval()
                p = []
                for s0 in range(0, len(va), 512):
                    p.append(model(signals(X, va[s0: s0 + 512])).float().cpu().numpy())
                model.train()
                return tracking_scores(np.concatenate(p).astype(np.float64), Y[va], S[va])

            for epoch in range(1, MAX_EPOCHS + 1):
                model.train()
                for s0 in range(steps):
                    picked = rng.choice(train_subjects, size=SUBJ_PER_BATCH, replace=False)
                    idx = np.concatenate([rng.choice(rows_of[int(p)], size=WIN_PER_SUBJ,
                                                     replace=False) for p in picked])
                    xb = signals(X, idx)
                    yt = torch.from_numpy(Y[idx].astype(np.float32)).to(device)
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                        pred = model(xb)
                        loss = pred.new_zeros(())
                        if abs_w > 0:
                            loss = loss + abs_w * weighted_huber(pred, yt, TARGET_WEIGHTS)
                        if pair_w > 0:
                            loss = loss + pair_w * paired_loss(pred, yt, SUBJ_PER_BATCH,
                                                               TARGET_WEIGHTS)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    sched.step()

                    if s0 in check_at:
                        at = epoch - 1 + (s0 + 1) / steps
                        v = validate()
                        score = 1.5 * v["SBP"]["tracked_mae"] + v["DBP"]["tracked_mae"]
                        flag = ""
                        if score < best["score"]:
                            best = {"score": score, "at": at,
                                    "state": {k: t.detach().cpu().clone()
                                              for k, t in model.state_dict().items()}}
                            since_best, flag = 0, "  *"
                        else:
                            since_best += 1
                        print(f"  {at:5.2f}ep  tracked SBP={v['SBP']['tracked_mae']:.4f} "
                              f"DBP={v['DBP']['tracked_mae']:.4f}{flag}", flush=True)
                        if since_best >= PATIENCE_CHECKS:
                            print(f"  [{tag}] stop at {at:.2f}, best {best['at']:.2f}", flush=True)
                            stop = True
                            break
                if stop:
                    break

            model.load_state_dict(best["state"])
            model.eval()
            preds = []
            with torch.no_grad():
                for s0 in range(0, len(Yt), 512):
                    preds.append(model(signals(Xt, np.arange(s0, min(s0 + 512, len(Yt)))))
                                 .float().cpu().numpy())
            pred = np.concatenate(preds).astype(np.float64)
            t = tracking_scores(pred, Yt, St)
            np.save(RUNS / f"{tag}_test_pred.npy", pred)
            torch.save({"state_dict": best["state"], "backbone": bname,
                        "objective": oname, "seed": args.seed,
                        "signal_mean": mean, "signal_std": std},
                       RUNS / f"{tag}_weights.pt")
            results[tag] = {"tag": tag, "backbone": bname, "objective": oname,
                            "parameters": int(nparam), "seed": args.seed,
                            "best_at_epoch": best["at"],
                            "minutes": (time.perf_counter() - started) / 60,
                            "tracked": {k: {"tracked_mae": t[k]["tracked_mae"],
                                            "within_r": t[k]["within_r"]}
                                        for k in ("SBP", "DBP")}}
            out_path.write_text(json.dumps(results[tag], indent=2), encoding="utf-8")
            print(f"  -> tracked SBP {t['SBP']['tracked_mae']:.4f} "
                  f"DBP {t['DBP']['tracked_mae']:.4f} r={t['SBP']['within_r']:.3f} "
                  f"({results[tag]['minutes']:.1f} min)\n", flush=True)

    print("=" * 76)
    print("%-12s %14s %14s %12s" % ("backbone", "ABS tracked", "PAIRED tracked", "gain"))
    subs = np.unique(St)
    boot = np.random.default_rng(7).integers(0, len(subs), size=(2000, len(subs)))
    rows = [("RESNET", "WS_ABS", "WS_PAIRED")] + [(b, f"BB_{b}_ABS", f"BB_{b}_PAIRED")
                                                  for b in BACKBONES]
    for name, fa, fb in rows:
        pa, pb = RUNS / f"{fa}_test_pred.npy", RUNS / f"{fb}_test_pred.npy"
        if not (pa.is_file() and pb.is_file()):
            continue
        ea = tracking_scores(np.load(pa), Yt, St)["SBP"]["per_subject"]
        eb = tracking_scores(np.load(pb), Yt, St)["SBP"]["per_subject"]
        dd = eb[boot].mean(1) - ea[boot].mean(1)
        lo, hi = float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))
        print("%-12s %14.4f %14.4f   %+.4f [%+.4f,%+.4f] %s"
              % (name, ea.mean(), eb.mean(), eb.mean() - ea.mean(), lo, hi,
                 "SIGNIFICANT" if lo * hi > 0 else "not significant"))

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "s37_backbones.json").write_text(json.dumps(
        {"analysis_id": "s37_backbones",
         "created_utc": datetime.now(timezone.utc).isoformat(), "results": results},
        indent=2), encoding="utf-8")
    print("\nsaved reports/s37_backbones.json")


if __name__ == "__main__":
    main()
