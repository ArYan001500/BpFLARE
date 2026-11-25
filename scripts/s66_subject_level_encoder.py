"""Train an encoder from scratch on the one quantity stage one actually needs.

Everything tried so far reads a subject's pressure level off a representation built for something
else. The three estimators in Section 4.1 and the attention head in s65 all sit on top of frozen
features from an encoder optimised for a single window's pressure, so if those features do not carry
the level, nothing placed on top of them can invent it.

This is the experiment that changes the features. The encoder is randomly initialised and its
gradient comes from one target only: the mean pressure of the person the windows came from. Each
training example is a set of windows from one subject, the encoder embeds each window, the set is
pooled, and the head predicts that subject's mean. Nothing in the objective mentions a single
window's pressure.

The honest risk is the label count. There are 1099 training subjects and therefore 1099 distinct
targets, which is very little for a 2.2 M parameter encoder, and resampling different window subsets
per step supplies variety but not new labels. If this overfits, that is a real answer about the
approach rather than a nuisance to tune away, so it is reported as measured.

Scoring is the strict protocol of s64: the warm-up windows set the level and the offset and are
excluded from the windows that are scored, so the numbers sit directly beside that run.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

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
RUNS = ROOT / "runs" / "s4_pat_control"
OUT = ROOT / "runs" / "s66_subject_level"
REPORTS = ROOT / "reports"
sys.path.insert(0, str(ROOT / "scripts"))
from network_loader import load_q1_module  # noqa: E402
from s29_level_information import pick_alpha, ridge  # noqa: E402

CAP = 120                      # windows kept per subject, so the arrays fit in memory
B, M_TRAIN = 16, 12            # subjects per step, windows per subject
STEPS, LR, WD, DROPOUT = 12000, 3e-4, 1e-3, 0.5
CHECK_EVERY, PATIENCE = 250, 12
M_EVAL, N_DRAWS = (10, 20), 50
MIN_TEST_WINDOWS, MIN_FIT_WINDOWS = 100, 20
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 1e4, 1e5)
SEED = 42


class SubjectLevelNet(nn.Module):
    """One encoder per window, pooled over the set, then a head that reads the subject's mean."""

    def __init__(self, q1, dropout=DROPOUT):
        super().__init__()
        self.encoder = q1.RawResNet1D(dropout=dropout).encoder
        self.score = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
        self.head = nn.Sequential(nn.Linear(1024, 128), nn.SiLU(), nn.Dropout(dropout),
                                  nn.Linear(128, 2))

    def embed(self, x):
        e = self.encoder(x)
        return torch.cat([e.mean(-1), e.amax(-1)], dim=1)

    def pool(self, f):                                   # f is (batch, M, 512)
        a = torch.softmax(self.score(f).squeeze(-1), dim=1)
        return self.head(torch.cat([(a.unsqueeze(-1) * f).sum(1), f.mean(1)], dim=1))

    def forward(self, x):                                # x is (batch, M, 2, 1250)
        b, m = x.shape[:2]
        return self.pool(self.embed(x.reshape(b * m, *x.shape[2:])).reshape(b, m, -1))


def load_sets(X, Y, S, rows, mean, std, cap, rng):
    """Windows and subject-mean targets, held in memory as normalised float16."""
    feats, targets = {}, {}
    for g in np.unique(S[rows]):
        idx = rows[S[rows] == g]
        if len(idx) < MIN_FIT_WINDOWS:
            continue
        targets[int(g)] = Y[idx, :2].mean(0)
        if len(idx) > cap:
            idx = np.sort(rng.choice(idx, size=cap, replace=False))
        x = (np.asarray(X[idx], dtype=np.float32) - mean[None, :, None]) / std[None, :, None]
        feats[int(g)] = x.astype(np.float16)
    return feats, targets


@torch.no_grad()
def embed_all(model, X, rows, mean, std, device, batch=512):
    model.eval()
    out = np.empty((len(rows), 512), dtype=np.float32)
    for s in range(0, len(rows), batch):
        sel = rows[s: s + batch]
        x = (np.asarray(X[sel], dtype=np.float32) - mean[None, :, None]) / std[None, :, None]
        out[s: s + batch] = model.embed(torch.from_numpy(x).to(device)).float().cpu().numpy()
    model.train()
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q1 = load_q1_module()
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    ck = torch.load(RUNS / "RAW_w_weights.pt", map_location="cpu", weights_only=False)
    mean, std = ck["signal_mean"], ck["signal_std"]

    X = np.load(CONVERTED / "train" / "X.npy", mmap_mode="r")
    Y = np.asarray(np.load(CONVERTED / "train" / "Y.npy"), dtype=np.float64)
    S = np.load(CONVERTED / "train" / "S.npy")
    tr_rows = np.load(CONVERTED / "train" / "rows_train.npy")
    val_rows = np.load(CONVERTED / "train" / "rows_validation.npy")

    print("loading window sets into memory ...", flush=True)
    tr_x, tr_y = load_sets(X, Y, S, tr_rows, mean, std, CAP, rng)
    va_x, va_y = load_sets(X, Y, S, val_rows, mean, std, CAP, rng)
    keys = list(tr_x)
    va_keys = list(va_x)
    va_target = np.stack([va_y[k] for k in va_keys])
    print(f"{len(keys)} training subjects, {len(va_keys)} validation subjects, "
          f"{sum(v.shape[0] for v in tr_x.values()):,} windows held\n", flush=True)

    torch.manual_seed(SEED)
    model = SubjectLevelNet(q1).to(device)
    with torch.no_grad():
        model.head[-1].weight.zero_()
        model.head[-1].bias.copy_(torch.tensor(np.stack(list(tr_y.values())).mean(0),
                                               dtype=torch.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    pop = np.abs(va_target - np.stack(list(tr_y.values())).mean(0)).mean(0)
    print("population-mean baseline on the validation subjects: "
          f"SBP {pop[0]:.3f}, DBP {pop[1]:.3f} mmHg\n", flush=True)

    best = {"score": np.inf, "state": None, "at": 0}
    since, started = 0, time.perf_counter()
    for step in range(1, STEPS + 1):
        ks = [keys[i] for i in rng.integers(0, len(keys), size=B)]
        xb = torch.from_numpy(np.stack(
            [tr_x[k][rng.choice(len(tr_x[k]), size=M_TRAIN, replace=False)]
             for k in ks]).astype(np.float32)).to(device)
        yb = torch.from_numpy(np.stack([tr_y[k] for k in ks]).astype(np.float32)).to(device)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            loss = nn.functional.huber_loss(model(xb), yb, delta=5.0)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step % CHECK_EVERY:
            continue
        model.eval()
        with torch.no_grad():
            errs = []
            for rep in range(4):
                r2 = np.random.default_rng(500 + rep)
                p = []
                for s0 in range(0, len(va_keys), 8):
                    part = va_keys[s0: s0 + 8]
                    xv = torch.from_numpy(np.stack(
                        [va_x[k][r2.choice(len(va_x[k]), size=20, replace=False)]
                         for k in part]).astype(np.float32)).to(device)
                    p.append(model(xv).float().cpu().numpy())
                errs.append(np.abs(np.concatenate(p) - va_target).mean(0))
        model.train()
        v = np.mean(errs, axis=0)
        flag = ""
        if v.sum() < best["score"] - 1e-4:
            best = {"score": float(v.sum()), "at": step,
                    "state": {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}}
            since, flag = 0, "  *"
        else:
            since += 1
        print("  step %5d  val level MAE  SBP %.3f  DBP %.3f%s" % (step, v[0], v[1], flag),
              flush=True)
        if since >= PATIENCE:
            print(f"  stopping at {step}, best was {best['at']}", flush=True)
            break

    model.load_state_dict(best["state"])
    model.eval()
    torch.save({"state_dict": best["state"], "signal_mean": mean, "signal_std": std,
                "best_step": best["at"]}, OUT / "encoder.pt")
    print(f"\ntrained in {(time.perf_counter() - started) / 60:.1f} min, "
          f"best step {best['at']}\n", flush=True)

    # ---- evaluation on the locked test subjects, strict protocol
    Yt = np.asarray(np.load(CONVERTED / "calfree_test" / "Y.npy"), dtype=np.float64)
    St = np.load(CONVERTED / "calfree_test" / "S.npy")
    Xt = np.load(CONVERTED / "calfree_test" / "X.npy", mmap_mode="r")
    subs = np.array([s for s in np.unique(St) if (St == s).sum() >= MIN_TEST_WINDOWS])
    rows_by = {int(s): np.flatnonzero(St == s) for s in subs}
    truth = np.stack([Yt[rows_by[int(s)], :2].mean(0) for s in subs])
    pred = np.asarray(np.load(RUNS / "WS_PAIRED_test_pred.npy"), dtype=np.float64)

    print("embedding the test and validation windows with the new encoder ...", flush=True)
    Ft = embed_all(model, Xt, np.arange(len(Yt)), mean, std, device)
    Fv = embed_all(model, X, val_rows, mean, std, device)
    model.eval()
    np.savez(OUT / "features.npz", test=Ft.astype(np.float16), val=Fv.astype(np.float16))

    # a ridge on the new representation, so the encoder and the pooling rule are separable
    gv = np.stack([Fv[S[val_rows] == g].mean(0) for g in va_keys])
    mu, sd = gv.mean(0), gv.std(0)
    sd[sd < 1e-9] = 1.0
    ridges = {j: ridge((gv - mu) / sd, va_target[:, j],
                       pick_alpha((gv - mu) / sd, va_target[:, j], ALPHAS, seed=SEED))
              for j in (0, 1)}
    old = np.load(RUNS / "RAW_w_feat_cache.npz")
    ov = np.stack([old["val"][S[val_rows] == g].mean(0) for g in va_keys]).astype(np.float64)
    omu, osd = ov.mean(0), ov.std(0)
    osd[osd < 1e-9] = 1.0
    oridges = {j: ridge((ov - omu) / osd, va_target[:, j],
                        pick_alpha((ov - omu) / osd, va_target[:, j], ALPHAS, seed=SEED))
               for j in (0, 1)}

    results = {}
    for M in M_EVAL:
        draws = {int(s): [np.random.default_rng(SEED + 7 * d).choice(
            rows_by[int(s)], size=M, replace=False) for d in range(N_DRAWS)] for s in subs}
        level = {n: np.zeros((N_DRAWS, len(subs), 2))
                 for n in ("old encoder, mean + ridge", "new encoder, mean + ridge",
                           "new encoder, learned pooling")}
        for i, s in enumerate(subs):
            sets = np.stack([Ft[w] for w in draws[int(s)]])
            with torch.no_grad():
                level["new encoder, learned pooling"][:, i] = model.pool(
                    torch.from_numpy(sets).to(device)).float().cpu().numpy()
            zs = (sets.mean(1) - mu) / sd
            zo = (np.stack([old["test"][w].astype(np.float64).mean(0)
                            for w in draws[int(s)]]) - omu) / osd
            for j in (0, 1):
                cmu, w_, b_ = ridges[j]
                level["new encoder, mean + ridge"][:, i, j] = (zs - cmu) @ w_ + b_
                cmu, w_, b_ = oridges[j]
                level["old encoder, mean + ridge"][:, i, j] = (zo - cmu) @ w_ + b_

        print("\n" + "=" * 88)
        print(f"--- warm-up of {M} windows, {M * 10 / 60:.1f} minutes ---")
        print("%-34s %6s %10s | %9s %9s" % ("level rule", "r", "level MAE", "SBP", "DBP"))
        for n, hat_all in level.items():
            row = {}
            for j, target in ((0, "SBP"), (1, "DBP")):
                hat = hat_all[:, :, j]
                r = float(np.mean([np.corrcoef(hat[d], truth[:, j])[0, 1]
                                   for d in range(N_DRAWS)]))
                e = np.zeros(len(subs))
                for i, s in enumerate(subs):
                    rows = rows_by[int(s)]
                    v = []
                    for d, warm in enumerate(draws[int(s)]):
                        m_ = np.ones(len(rows), dtype=bool)
                        m_[np.searchsorted(rows, np.sort(warm))] = False
                        ev = rows[m_]
                        off = hat[d, i] - pred[warm, j].mean()
                        v.append(np.abs(pred[ev, j] + off - Yt[ev, j]).mean())
                    e[i] = float(np.mean(v))
                row[target] = {"pearson_r": r,
                               "level_mae": float(np.abs(hat - truth[None, :, j]).mean()),
                               "two_stage_mae": float(e.mean())}
            print("%-34s %6.3f %10.3f | %9.4f %9.4f"
                  % (n, row["SBP"]["pearson_r"], row["SBP"]["level_mae"],
                     row["SBP"]["two_stage_mae"], row["DBP"]["two_stage_mae"]), flush=True)
            results.setdefault(str(M), {})[n] = row

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "s66_subject_level_encoder.json").write_text(json.dumps(
        {"analysis_id": "s66_subject_level_encoder",
         "created_utc": datetime.now(timezone.utc).isoformat(),
         "steps": STEPS, "best_step": best["at"], "subjects_train": len(keys),
         "draws": N_DRAWS, "results": results}, indent=2), encoding="utf-8")
    print("\nsaved reports/s66_subject_level_encoder.json")


if __name__ == "__main__":
    main()
