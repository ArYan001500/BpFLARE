"""Train for the part of the error that is actually reachable.

The error splits in two. Four and a half millimetres of it is not knowing where a person's pressure
sits, and that part is not in the signal: the level is unpredictable from the waveform, from
demographics, and from an hour of that person's own unlabelled recording. One cuff reading fixes
it, and a plain linear head already extracts 96 percent of what a perfect cuff reading would give.
The other eight and a half millimetres is following that person's pressure as it moves, and that
part is learnable -- the current model tracks at a within-subject correlation of 0.63.

So the level is bought with a cuff and the tracking is where the remaining room is. Yet every model
here, and every model in the literature, is trained on absolute pressure, an objective whose
gradient is dominated by the component that cannot be learned. This asks the obvious question
nobody seems to have asked: if the objective is rewritten to be about movement inside a person,
does the movement get tracked better?

Four arms, same backbone, same seed, same number of gradient steps:

    ABS        absolute pressure, the control, exactly the current recipe
    CENTERED   each target has that subject's own training-set mean removed
    PAIRED     no absolute target at all; the loss is on the difference in pressure between two
               windows of the same person, which is invariant to that person's level
    ABSPAIRED  absolute loss plus the paired difference loss at half weight

Batches are built by subject rather than at random -- sixty-four subjects, four windows each -- so
that same-person pairs exist inside every batch. All four arms use that sampler, so the batch
composition is not a difference between them.

Selection and scoring are both on tracking, not on absolute error: each subject's own mean is
removed from the truth and from the prediction, which is exactly the 8.44 mmHg oracle-level
quantity. The decision rule is fixed before the run: an arm counts only if it beats the control's
tracked SBP error with a subject-level bootstrap interval that excludes zero.
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
CONVERTED = ROOT / "converted"
REPORTS = ROOT / "reports"
RUNS = ROOT / "runs" / "s4_pat_control"
sys.path.insert(0, str(ROOT / "scripts"))
from network_loader import load_q1_module  # noqa: E402

TARGET_WEIGHTS = (2.0, 1.2, 0.8)
TEST_TAG = "calfree_test"
SUBJ_PER_BATCH, WIN_PER_SUBJ = 64, 4          # 256 windows, arranged so pairs exist
LR, WEIGHT_DECAY, DROPOUT, SEED = 1e-4, 1e-4, 0.5, 42
MAX_EPOCHS, CHECKS_PER_EPOCH, PATIENCE_CHECKS = 6, 4, 8
MIN_TEST_WINDOWS = 5

# arm -> (centre the target, weight on the paired difference loss)
ARMS = {"ABS": (False, 0.0), "CENTERED": (True, 0.0),
        "PAIRED": (False, 1.0), "ABSPAIRED": (False, 0.5)}
ABSOLUTE_WEIGHT = {"ABS": 1.0, "CENTERED": 1.0, "PAIRED": 0.0, "ABSPAIRED": 1.0}


def weighted_huber(pred, target, weights, delta=5.0):
    w = pred.new_tensor(weights).view(1, -1)
    return (F.huber_loss(pred, target, reduction="none", delta=delta) * w).mean()


def paired_loss(pred, target, groups, weights):
    """Huber on the pressure difference between every pair of windows from the same person.

    pred and target arrive grouped: rows [g*m ... g*m+m-1] all belong to subject g. Reshaping to
    (groups, m, 3) makes every within-subject pair a simple broadcast difference, and the person's
    own level cancels out of both sides.
    """
    p = pred.view(groups, -1, pred.shape[-1])
    t = target.view(groups, -1, target.shape[-1])
    dp = p.unsqueeze(2) - p.unsqueeze(1)
    dt = t.unsqueeze(2) - t.unsqueeze(1)
    m = p.shape[1]
    iu = torch.triu_indices(m, m, offset=1, device=pred.device)
    return weighted_huber(dp[:, iu[0], iu[1]].reshape(-1, pred.shape[-1]),
                          dt[:, iu[0], iu[1]].reshape(-1, pred.shape[-1]), weights)


def tracking_scores(pred, true, subjects, min_windows=MIN_TEST_WINDOWS):
    """Per-subject error and correlation after each subject's own mean is removed from both."""
    out = {}
    for j, target in enumerate(("SBP", "DBP")):
        mae, corr = [], []
        for s in np.unique(subjects):
            rows = np.flatnonzero(subjects == s)
            if len(rows) < min_windows:
                continue
            dp = pred[rows, j] - pred[rows, j].mean()
            dt = true[rows, j] - true[rows, j].mean()
            mae.append(np.abs(dp - dt).mean())
            if dp.std() > 1e-9 and dt.std() > 1e-9:
                corr.append(np.corrcoef(dp, dt)[0, 1])
        out[target] = {"tracked_mae": float(np.mean(mae)),
                       "within_r": float(np.mean(corr)),
                       "per_subject": np.asarray(mae)}
    return out


def main() -> None:
    global SUBJ_PER_BATCH, WIN_PER_SUBJ
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--win-per-subj", type=int, default=WIN_PER_SUBJ,
                    help="windows each subject contributes to a batch; the pair count per "
                         "subject is this choose two")
    ap.add_argument("--batch-windows", type=int, default=SUBJ_PER_BATCH * WIN_PER_SUBJ,
                    help="total windows per batch, held fixed across the sweep")
    ap.add_argument("--drop-ecg", action="store_true",
                    help="zero the electrocardiogram channel, leaving the "
                         "photoplethysmogram as the only input")
    args = ap.parse_args()
    WIN_PER_SUBJ = args.win_per_subj
    SUBJ_PER_BATCH = args.batch_windows // WIN_PER_SUBJ

    RUNS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q1 = load_q1_module()

    d, dt = CONVERTED / "train", CONVERTED / TEST_TAG
    X = np.load(d / "X.npy", mmap_mode="r")
    Y = np.asarray(np.load(d / "Y.npy"), dtype=np.float64)
    S = np.load(d / "S.npy")
    tr, va = np.load(d / "rows_train.npy"), np.load(d / "rows_validation.npy")
    Xt = np.load(dt / "X.npy", mmap_mode="r")
    Yt = np.asarray(np.load(dt / "Y.npy"), dtype=np.float64)
    St = np.load(dt / "S.npy")

    rows_of = {}
    for s in np.unique(S[tr]):
        r = tr[S[tr] == s]
        if len(r) >= WIN_PER_SUBJ:
            rows_of[int(s)] = r
    train_subjects = np.array(sorted(rows_of))
    # each training subject's own mean, used only by the CENTERED arm and only for its own subjects
    subj_mean = {s: Y[rows_of[s]].mean(axis=0) for s in rows_of}
    print(f"{len(train_subjects):,} training subjects usable, "
          f"{sum(len(v) for v in rows_of.values()):,} windows; "
          f"batches of {SUBJ_PER_BATCH} subjects x {WIN_PER_SUBJ} windows\n", flush=True)

    sub = tr[:: max(1, len(tr) // 40000)]
    xs = np.asarray(X[sub], dtype=np.float64)
    mean = xs.mean(axis=(0, 2)).astype(np.float32)
    std = np.maximum(xs.std(axis=(0, 2)), 1e-6).astype(np.float32)

    def signals(Xa, idx):
        x = (np.asarray(Xa[idx], dtype=np.float32) - mean[None, :, None]) / std[None, :, None]
        if args.drop_ecg:
            x[:, 0] = 0.0
        return torch.from_numpy(x).to(device)

    results = {}
    for arm in [a for a in args.arms if a in ARMS]:
        centre, pair_w = ARMS[arm]
        abs_w = ABSOLUTE_WEIGHT[arm]
        out_path = RUNS / f"WS_{arm}{args.suffix}.json"
        if out_path.is_file():
            results[arm] = json.loads(out_path.read_text())
            print(f"  [{arm}] already done", flush=True)
            continue

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = q1.RawResNet1D(dropout=DROPOUT).to(device)
        with torch.no_grad():
            init = np.zeros(3) if centre else Y[tr].mean(axis=0)
            model.head[-1].bias.copy_(torch.tensor(init, dtype=torch.float32))

        steps = len(tr) // (SUBJ_PER_BATCH * WIN_PER_SUBJ)
        check_at = {int(round(steps * f / CHECKS_PER_EPOCH)) - 1
                    for f in range(1, CHECKS_PER_EPOCH + 1)}
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.max_epochs * steps, eta_min=1e-6)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=4096.0)
        rng = np.random.default_rng(args.seed)
        best = {"score": float("inf"), "at": -1.0, "state": None}
        curve, since_best, started, stop = [], 0, time.perf_counter(), False
        print(f"===== {arm}  (centred {centre}, paired weight {pair_w}) =====", flush=True)

        @torch.no_grad()
        def validate():
            model.eval()
            preds = []
            for s0 in range(0, len(va), 512):
                preds.append(model(signals(X, va[s0: s0 + 512])).float().cpu().numpy())
            model.train()
            p = np.concatenate(preds).astype(np.float64)
            return tracking_scores(p, Y[va], S[va])

        for epoch in range(1, args.max_epochs + 1):
            model.train()
            for s0 in range(steps):
                picked = rng.choice(train_subjects, size=SUBJ_PER_BATCH, replace=False)
                idx = np.concatenate([rng.choice(rows_of[int(p)], size=WIN_PER_SUBJ,
                                                 replace=False) for p in picked])
                xb = signals(X, idx)
                yb = Y[idx].astype(np.float32)
                if centre:
                    yb = yb - np.repeat(np.stack([subj_mean[int(p)] for p in picked]),
                                        WIN_PER_SUBJ, axis=0).astype(np.float32)
                yt = torch.from_numpy(yb).to(device)
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
                    curve.append({"at_epoch": round(at, 3), "score": score,
                                  "sbp": v["SBP"]["tracked_mae"],
                                  "dbp": v["DBP"]["tracked_mae"]})
                    flag = ""
                    if score < best["score"]:
                        best = {"score": score, "at": at,
                                "state": {k: t.detach().cpu().clone()
                                          for k, t in model.state_dict().items()}}
                        since_best, flag = 0, "  *"
                    else:
                        since_best += 1
                    print(f"  {at:5.2f}ep  tracked SBP={v['SBP']['tracked_mae']:.4f} "
                          f"DBP={v['DBP']['tracked_mae']:.4f} r={v['SBP']['within_r']:.3f}{flag}",
                          flush=True)
                    if since_best >= PATIENCE_CHECKS:
                        print(f"  [{arm}] stop at {at:.2f}, best was {best['at']:.2f}", flush=True)
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
        np.save(RUNS / f"WS_{arm}{args.suffix}_test_pred.npy", pred)
        torch.save({"state_dict": best["state"], "arm": arm, "seed": args.seed,
                    "signal_mean": mean, "signal_std": std},
                   RUNS / f"WS_{arm}{args.suffix}_weights.pt")
        results[arm] = {
            "arm": arm, "centred": centre, "paired_weight": pair_w, "seed": args.seed,
            "best_at_epoch": best["at"], "curve": curve,
            "minutes": (time.perf_counter() - started) / 60,
            "tracked": {k: {"tracked_mae": t[k]["tracked_mae"], "within_r": t[k]["within_r"]}
                        for k in ("SBP", "DBP")},
            "population_mae": {k: float(np.abs(pred[:, j] - Yt[:, j]).mean())
                               for j, k in enumerate(("SBP", "DBP"))} if abs_w > 0 and not centre
            else None}
        out_path.write_text(json.dumps(results[arm], indent=2), encoding="utf-8")
        print(f"  -> tracked SBP {t['SBP']['tracked_mae']:.4f} DBP {t['DBP']['tracked_mae']:.4f} "
              f"r={t['SBP']['within_r']:.3f}  ({results[arm]['minutes']:.1f} min)\n", flush=True)

    if "ABS" in results:
        print("=" * 80)
        print("tracking on the held-out test subjects, each subject's own mean removed")
        print("%-12s %8s %12s %12s %9s   %s"
              % ("arm", "best@ep", "SBP tracked", "DBP tracked", "within r",
                 "SBP change vs ABS, 95% CI"))
        base = np.load(RUNS / f"WS_ABS{args.suffix}_test_pred.npy")
        eb = tracking_scores(base, Yt, St)["SBP"]["per_subject"]
        boot = np.random.default_rng(7).integers(0, len(eb), size=(2000, len(eb)))
        for arm, r in results.items():
            ea = tracking_scores(np.load(RUNS / f"WS_{arm}{args.suffix}_test_pred.npy"),
                                 Yt, St)["SBP"]["per_subject"]
            dd = ea[boot].mean(1) - eb[boot].mean(1)
            lo, hi = float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))
            cell = ("" if arm == "ABS" else "%+.4f [%+.4f,%+.4f]  %s"
                    % (ea.mean() - eb.mean(), lo, hi,
                       "SIGNIFICANT" if lo * hi > 0 else "not significant"))
            print("%-12s %8.2f %12.4f %12.4f %9.3f   %s"
                  % (arm, r["best_at_epoch"], r["tracked"]["SBP"]["tracked_mae"],
                     r["tracked"]["DBP"]["tracked_mae"], r["tracked"]["SBP"]["within_r"], cell))
        print("\nfor reference, the absolute-loss model reported earlier tracked SBP at 8.4410")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"s35_within_subject_objective{args.suffix}.json").write_text(json.dumps(
        {"analysis_id": "s35_within_subject_objective",
         "created_utc": datetime.now(timezone.utc).isoformat(),
         "subjects_per_batch": SUBJ_PER_BATCH, "windows_per_subject": WIN_PER_SUBJ,
         "results": results}, indent=2), encoding="utf-8")
    print("\nsaved reports/s35_within_subject_objective.json")


if __name__ == "__main__":
    main()
