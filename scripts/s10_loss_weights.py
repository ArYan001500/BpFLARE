"""Choose the loss weights on validation, because the current ones were never chosen at all.

The manuscript trains with per-target weights of 2.0 on SBP, 1.2 on DBP and 0.8 on MAP. Nothing in
the manuscript justifies those numbers, and they are not neutral: they instruct the optimiser to
spend its capacity on SBP and to accept a worse DBP in exchange. Measured against the published
results on this benchmark, that is exactly where this work is behind, so it is worth asking what
the weights are costing.

The sweep runs on the raw arm alone. That is deliberate and follows the same reasoning stage 2
used when it fixed dropout: the cheap arm decides the recipe, and the recipe is then applied
unchanged everywhere, so that no later difference between arms can be blamed on tuning. The raw
arm costs about three minutes an epoch against roughly fourteen for a gated arm, which is what
makes a sweep affordable at all.

The selection rule is written down before the numbers exist, so that it cannot be bent afterwards:

    among the weightings whose validation SBP MAE is no worse than the manuscript weighting by
    more than 0.10 mmHg, take the one with the lowest validation DBP MAE.

The test set is not touched anywhere in this file.
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

ROOT = Path(__file__).resolve().parents[1]
CONVERTED = ROOT / "converted"
REPORTS = ROOT / "reports"
RUNS = ROOT / "runs" / "s10_loss_weights"
from network_loader import load_q1_module  # noqa: E402

TARGETS = ("SBP", "DBP", "MAP")
BATCH = 256
LR = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.5
MAX_EPOCHS = 14
PATIENCE = 4
SEED = 42
SBP_TOLERANCE = 0.10        # how much validation SBP may give up, decided in advance

WEIGHTINGS = {
    "manuscript":   (2.0, 1.2, 0.8),
    "dbp_equal":    (2.0, 2.0, 0.8),
    "dbp_favoured": (1.5, 2.0, 0.8),
    "dbp_heavy":    (1.2, 2.5, 0.8),
    "flat":         (1.0, 1.0, 1.0),
}


def metrics(pred, true):
    out = {}
    for j, t in enumerate(TARGETS):
        e = pred[:, j] - true[:, j]
        out[t] = float(np.abs(e).mean())
    out["selection_score"] = 1.5 * out["SBP"] + out["DBP"] + 0.75 * out["MAP"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weightings", nargs="*", default=list(WEIGHTINGS))
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    args = ap.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q1 = load_q1_module()

    d = CONVERTED / "train"
    X = np.load(d / "X.npy", mmap_mode="r")
    Y = np.asarray(np.load(d / "Y.npy"), dtype=np.float64)
    S = np.load(d / "S.npy")
    tr = np.load(d / "rows_train.npy")
    va = np.load(d / "rows_validation.npy")
    assert not (set(S[tr].tolist()) & set(S[va].tolist())), "subject leak"

    sub = tr[:: max(1, len(tr) // 40000)]
    xs = np.asarray(X[sub], dtype=np.float64)
    mean = xs.mean(axis=(0, 2)).astype(np.float32)
    std = np.maximum(xs.std(axis=(0, 2)), 1e-6).astype(np.float32)

    print(f"raw arm, validation only, {len(tr):,} training rows / {len(va):,} validation rows")
    print(f"selection rule fixed in advance: lowest validation DBP among weightings whose "
          f"validation SBP is within {SBP_TOLERANCE:.2f} mmHg of the manuscript weighting\n")

    @torch.no_grad()
    def predict(model, rows, batch=512):
        model.eval()
        out = []
        for s in range(0, len(rows), batch):
            idx = rows[s: s + batch]
            x = (np.asarray(X[idx], dtype=np.float32) - mean[None, :, None]) / std[None, :, None]
            out.append(model(torch.from_numpy(x).to(device)).float().cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    results = {}
    for name in [w for w in WEIGHTINGS if w in args.weightings]:
        out_path = RUNS / f"{name}.json"
        if out_path.is_file():
            results[name] = json.loads(out_path.read_text())
            print(f"  [{name}] already done", flush=True)
            continue

        weights = WEIGHTINGS[name]
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model = q1.RawResNet1D(dropout=DROPOUT).to(device)
        with torch.no_grad():
            model.head[-1].bias.copy_(torch.tensor(Y[tr].mean(axis=0), dtype=torch.float32))

        steps = len(tr) // BATCH
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.max_epochs * steps, eta_min=1e-6)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=4096.0)
        rng = np.random.default_rng(SEED)
        best = {"score": float("inf"), "epoch": -1, "val": None}
        started = time.perf_counter()

        print(f"===== {name}  weights {weights} =====", flush=True)
        for epoch in range(1, args.max_epochs + 1):
            model.train()
            order = rng.permutation(tr)
            running = 0.0
            for s in range(steps):
                idx = np.sort(order[s * BATCH: (s + 1) * BATCH])
                x = (np.asarray(X[idx], dtype=np.float32) - mean[None, :, None]) / std[None, :, None]
                xb = torch.from_numpy(x).to(device)
                yb = torch.from_numpy(Y[idx].astype(np.float32)).to(device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    loss = q1.weighted_huber_loss_custom(model(xb), yb, weights)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                running += float(loss.detach())

            vm = metrics(predict(model, va), Y[va])
            print(f"  ep{epoch:2d} train={running / steps:7.3f} SBP={vm['SBP']:.4f} "
                  f"DBP={vm['DBP']:.4f} MAP={vm['MAP']:.4f}", flush=True)
            # Selected on the score the whole project uses, so the weighting is judged on how well
            # it trains rather than on which epoch happens to suit it.
            if vm["selection_score"] < best["score"]:
                best = {"score": vm["selection_score"], "epoch": epoch, "val": vm}
            if epoch - best["epoch"] >= PATIENCE:
                print(f"  [{name}] early stop at epoch {epoch}, best was {best['epoch']}",
                      flush=True)
                break

        results[name] = {"name": name, "weights": list(weights),
                         "best_epoch": best["epoch"], "validation": best["val"],
                         "minutes": (time.perf_counter() - started) / 60}
        out_path.write_text(json.dumps(results[name], indent=2), encoding="utf-8")
        print(f"  -> validation SBP {best['val']['SBP']:.4f}  DBP {best['val']['DBP']:.4f}  "
              f"({results[name]['minutes']:.1f} min)", flush=True)

    if "manuscript" not in results:
        print("\nmanuscript weighting not run, cannot apply the selection rule")
        return

    ref = results["manuscript"]["validation"]
    eligible = {n: r for n, r in results.items()
                if r["validation"]["SBP"] <= ref["SBP"] + SBP_TOLERANCE}
    chosen = min(eligible, key=lambda n: eligible[n]["validation"]["DBP"])

    md = ["# Stage 10: choosing the loss weights on validation", "",
          f"*Generated {datetime.now(timezone.utc).isoformat()}*", "",
          "Raw arm, validation only. The test set is not touched in this stage.", "",
          f"Rule fixed before the numbers existed: among weightings whose validation SBP is "
          f"within {SBP_TOLERANCE:.2f} mmHg of the manuscript weighting, take the lowest "
          f"validation DBP.", "",
          "| weighting | SBP, DBP, MAP | val SBP | val DBP | val MAP | best epoch | eligible |",
          "|---|---|---:|---:|---:|---:|:--:|"]
    for n, r in sorted(results.items(), key=lambda kv: kv[1]["validation"]["DBP"]):
        v = r["validation"]
        md.append(f"| `{n}` | {tuple(r['weights'])} | {v['SBP']:.4f} | {v['DBP']:.4f} "
                  f"| {v['MAP']:.4f} | {r['best_epoch']} | {'yes' if n in eligible else 'no'} |")
    md += ["", f"Selected: **`{chosen}`** with weights {tuple(results[chosen]['weights'])}, "
               f"validation DBP {results[chosen]['validation']['DBP']:.4f} against "
               f"{ref['DBP']:.4f} for the manuscript weighting "
               f"({results[chosen]['validation']['DBP'] - ref['DBP']:+.4f}).", ""]
    (REPORTS / "s10_loss_weights.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (REPORTS / "s10_loss_weights.json").write_text(json.dumps(
        {"analysis_id": "s10_loss_weights", "sbp_tolerance": SBP_TOLERANCE,
         "selected": chosen, "selected_weights": results[chosen]["weights"],
         "results": results}, indent=2), encoding="utf-8")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
