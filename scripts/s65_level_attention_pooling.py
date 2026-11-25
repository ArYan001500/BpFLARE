"""Can a better pooling rule recover more of a subject's pressure level than a plain average?

Stage one currently averages a frozen encoder's features over M windows and fits a ridge from that
average to the subject's mean pressure. The average weights every window equally, including the
noisy ones and the moments where the person is briefly unrepresentative, and the ridge that follows
is linear. Neither choice was ever compared against an alternative.

The alternative here is the usual one for a set-valued input: learn the pooling weights. A small
attention head scores each window, the scores become a softmax over the M windows, and the weighted
sum goes through an MLP. It is trained directly on the quantity stage one needs, a subject's mean
pressure, instead of borrowing a representation optimised for a single window's pressure.

Two fitting sets are tried, because the choice is not obvious. The validation subjects are
out-of-sample for the encoder and are what the ridge already uses, so that arm is a like-for-like
swap of the pooling rule. The training subjects give five times more data but the encoder has seen
them, so their features carry memorised identity and may not describe an unseen person. Reporting
both says which effect wins rather than assuming one.

Scoring uses the strict protocol of s64: the M warm-up windows set the level and the offset, and
they are excluded from the windows that are scored.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
CONVERTED = ROOT / "converted"
RUNS = ROOT / "runs" / "s4_pat_control"
REPORTS = ROOT / "reports"
sys.path.insert(0, str(ROOT / "scripts"))
from network_loader import load_q1_module  # noqa: E402
from s29_level_information import pick_alpha, ridge  # noqa: E402
from s63_cuffless_two_stage import encode  # noqa: E402

FEATURES = "RAW_w"
MAX_PER_TRAIN_SUBJECT = 200
M_TRAIN = (5, 10, 20, 40)
M_EVAL = (10, 20)
N_DRAWS = 50
MIN_TEST_WINDOWS = 100
MIN_FIT_WINDOWS = 20
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 1e4, 1e5)
STEPS, BATCH, LR, PATIENCE = 4000, 32, 3e-4, 25
CHECK_EVERY = 100
SEED = 42


class AttentionLevel(nn.Module):
    """Scores each window, pools by those scores, and reads a subject's mean pressure off the sum.

    The mean of the set is concatenated with the attended sum so the head can never do worse than
    plain averaging by construction; it has to learn only what the average is missing.
    """

    def __init__(self, dim=512, hidden=128, dropout=0.3):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.head = nn.Sequential(nn.Linear(2 * dim, hidden), nn.SiLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 2))

    def forward(self, x):                       # x is (batch, M, dim)
        a = torch.softmax(self.score(x).squeeze(-1), dim=1)
        return self.head(torch.cat([(a.unsqueeze(-1) * x).sum(1), x.mean(1)], dim=1))


def subject_features(cache_path, tag, X, rows, S, device, q1, cap=None, rng=None):
    """Per-window features grouped by subject, optionally capped so the cache stays small."""
    if cache_path.is_file():
        z = np.load(cache_path, allow_pickle=True)
        return {int(k): z[str(k)] for k in z.files}
    keep = []
    for g in np.unique(S[rows]):
        idx = rows[S[rows] == g]
        if len(idx) < MIN_FIT_WINDOWS:
            continue
        keep.append(idx if cap is None or len(idx) <= cap
                    else np.sort(rng.choice(idx, size=cap, replace=False)))
    flat = np.concatenate(keep)
    F = encode(tag, X, flat, device, q1).astype(np.float16)
    out, at = {}, 0
    for idx in keep:
        out[int(S[idx[0]])] = F[at: at + len(idx)]
        at += len(idx)
    np.savez(cache_path, **{str(k): v for k, v in out.items()})
    return out


def train_head(fit_feat, fit_y, mon_feat, mon_y, mu, sd, device, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = AttentionLevel().to(device)
    # Start the head at the population mean. Without this the output begins near zero against a
    # target near 120, where the Huber loss is linear and the model spends its whole budget
    # travelling rather than fitting.
    with torch.no_grad():
        model.head[-1].weight.zero_()
        model.head[-1].bias.copy_(torch.tensor(np.stack(list(fit_y.values()))[
            [i for i, k in enumerate(fit_y) if k in fit_feat]].mean(0), dtype=torch.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    keys = list(fit_feat)
    best = {"score": np.inf, "state": None}
    since = 0

    def draw(feat, ks, M, r):
        return torch.from_numpy(np.stack(
            [((feat[k][r.choice(len(feat[k]), size=min(M, len(feat[k])), replace=False)]
               .astype(np.float32)) - mu) / sd for k in ks])).to(device)

    for step in range(1, STEPS + 1):
        model.train()
        ks = [keys[i] for i in rng.integers(0, len(keys), size=BATCH)]
        xb = draw(fit_feat, ks, int(rng.choice(M_TRAIN)), rng)
        yb = torch.from_numpy(np.stack([fit_y[k] for k in ks]).astype(np.float32)).to(device)
        loss = nn.functional.huber_loss(model(xb), yb, delta=5.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % CHECK_EVERY:
            continue
        model.eval()
        mk = list(mon_feat)
        errs = []
        with torch.no_grad():
            for rep in range(8):
                r2 = np.random.default_rng(900 + rep)
                p = model(draw(mon_feat, mk, 20, r2)).cpu().numpy()
                errs.append(np.abs(p - np.stack([mon_y[k] for k in mk])).mean())
        v = float(np.mean(errs))
        if v < best["score"] - 1e-4:
            best = {"score": v, "state": {k: t.detach().cpu().clone()
                                          for k, t in model.state_dict().items()}}
            since = 0
        else:
            since += 1
            if since >= PATIENCE:
                break
    model.load_state_dict(best["state"])
    return model.eval(), best["score"]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q1 = load_q1_module()
    rng = np.random.default_rng(SEED)

    Yt = np.asarray(np.load(CONVERTED / "calfree_test" / "Y.npy"), dtype=np.float64)
    St = np.load(CONVERTED / "calfree_test" / "S.npy")
    Xt = np.load(CONVERTED / "calfree_test" / "X.npy", mmap_mode="r")
    Yv = np.asarray(np.load(CONVERTED / "train" / "Y.npy"), dtype=np.float64)
    Sv = np.load(CONVERTED / "train" / "S.npy")
    Xv = np.load(CONVERTED / "train" / "X.npy", mmap_mode="r")
    val_rows = np.load(CONVERTED / "train" / "rows_validation.npy")
    tr_rows = np.load(CONVERTED / "train" / "rows_train.npy")

    subs = np.array([s for s in np.unique(St) if (St == s).sum() >= MIN_TEST_WINDOWS])
    rows_by = {int(s): np.flatnonzero(St == s) for s in subs}
    pred = {a: np.asarray(np.load(RUNS / f"{a}_test_pred.npy"), dtype=np.float64)
            for a in ("WS_ABS", "WS_PAIRED")}

    print("collecting features ...", flush=True)
    feat_val = subject_features(RUNS / f"{FEATURES}_subj_val.npz", FEATURES, Xv, val_rows,
                                Sv, device, q1)
    feat_tr = subject_features(RUNS / f"{FEATURES}_subj_train.npz", FEATURES, Xv, tr_rows,
                               Sv, device, q1, cap=MAX_PER_TRAIN_SUBJECT, rng=rng)
    z = np.load(RUNS / f"{FEATURES}_feat_cache.npz")
    feat_test = {int(s): z["test"][rows_by[int(s)]].astype(np.float16) for s in subs}
    y_val = {int(g): Yv[val_rows][Sv[val_rows] == g, :2].mean(0) for g in feat_val}
    y_tr = {int(g): Yv[tr_rows][Sv[tr_rows] == g, :2].mean(0) for g in feat_tr}
    truth = np.stack([Yt[rows_by[int(s)], :2].mean(0) for s in subs])
    print(f"{len(feat_tr)} training, {len(feat_val)} validation, {len(subs)} test subjects\n",
          flush=True)

    stack_val = np.concatenate([v.astype(np.float32) for v in feat_val.values()])
    mu, sd = stack_val.mean(0), stack_val.std(0)
    sd[sd < 1e-9] = 1.0

    # the plain-average ridge, refit here so both rules see identical inputs
    Fv = np.stack([feat_val[k].astype(np.float32).mean(0) for k in feat_val])
    yv2 = np.stack([y_val[k] for k in feat_val])
    ridges = {}
    for j in (0, 1):
        a = pick_alpha((Fv - mu) / sd, yv2[:, j], ALPHAS, seed=SEED)
        ridges[j] = ridge((Fv - mu) / sd, yv2[:, j], a)

    inner = rng.permutation(list(feat_val))
    fit_k, mon_k = inner[:150], inner[150:]
    heads = {}
    for name, (ff, fy, mf, my) in (
            ("attention, fit on validation subjects",
             ({k: feat_val[k] for k in fit_k}, y_val, {k: feat_val[k] for k in mon_k}, y_val)),
            ("attention, fit on training subjects", (feat_tr, y_tr, feat_val, y_val))):
        m, sc = train_head(ff, fy, mf, my, mu, sd, device, SEED)
        heads[name] = m
        print(f"  {name}: monitor MAE {sc:.3f} over {len(ff)} subjects", flush=True)

    results = {}
    for M in M_EVAL:
        draws = {int(s): [np.random.default_rng(SEED + 7 * d).choice(
            rows_by[int(s)], size=M, replace=False) for d in range(N_DRAWS)] for s in subs}
        pos = {int(s): {int(r): i for i, r in enumerate(rows_by[int(s)])} for s in subs}

        level = {n: np.zeros((N_DRAWS, len(subs), 2)) for n in ("mean + ridge", *heads)}
        for i, s in enumerate(subs):
            sets = np.stack([feat_test[int(s)][[pos[int(s)][int(r)] for r in w]]
                             for w in draws[int(s)]]).astype(np.float32)
            Z = (sets - mu) / sd
            for j in (0, 1):
                cmu, w_, b_ = ridges[j]
                level["mean + ridge"][:, i, j] = (Z.mean(1) - cmu) @ w_ + b_
            with torch.no_grad():
                for n, m in heads.items():
                    level[n][:, i] = m(torch.from_numpy(Z).to(device)).cpu().numpy()

        print("\n" + "=" * 86)
        print(f"--- warm-up of {M} windows, {M * 10 / 60:.1f} minutes ---")
        print("%-38s %6s %9s | %9s %9s"
              % ("level rule", "r", "level MAE", "SBP", "DBP"))
        for n in level:
            row = {}
            for j, target in ((0, "SBP"), (1, "DBP")):
                hat = level[n][:, :, j]
                r = float(np.mean([np.corrcoef(hat[d], truth[:, j])[0, 1]
                                   for d in range(N_DRAWS)]))
                lmae = float(np.abs(hat - truth[None, :, j]).mean())
                e = np.zeros(len(subs))
                for i, s in enumerate(subs):
                    rows = rows_by[int(s)]
                    v = []
                    for d, warm in enumerate(draws[int(s)]):
                        m_ = np.ones(len(rows), dtype=bool)
                        m_[np.searchsorted(rows, np.sort(warm))] = False
                        ev = rows[m_]
                        off = hat[d, i] - pred["WS_PAIRED"][warm, j].mean()
                        v.append(np.abs(pred["WS_PAIRED"][ev, j] + off - Yt[ev, j]).mean())
                    e[i] = float(np.mean(v))
                row[target] = {"pearson_r": r, "level_mae": lmae, "two_stage_mae": float(e.mean())}
            print("%-38s %6.3f %9.3f | %9.4f %9.4f"
                  % (n, row["SBP"]["pearson_r"], row["SBP"]["level_mae"],
                     row["SBP"]["two_stage_mae"], row["DBP"]["two_stage_mae"]), flush=True)
            results.setdefault(str(M), {})[n] = row

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "s65_level_attention_pooling.json").write_text(json.dumps(
        {"analysis_id": "s65_level_attention_pooling",
         "created_utc": datetime.now(timezone.utc).isoformat(),
         "features": FEATURES, "n_test_subjects": int(len(subs)), "draws": N_DRAWS,
         "results": results}, indent=2), encoding="utf-8")
    print("\nsaved reports/s65_level_attention_pooling.json")


if __name__ == "__main__":
    main()
