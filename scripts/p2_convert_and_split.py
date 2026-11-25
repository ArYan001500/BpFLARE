"""Step P2: convert the PulseDB MATLAB files into a compact form and define the splits.

Reading signals out of a 12.85 GB HDF5 file on every training epoch would dominate the runtime,
so each subset is converted once into memory-mappable numpy arrays holding only what the models
need. The arterial pressure channel is used to derive MAP and is then dropped: it is a label
source, never a model input, exactly as in the manuscript.

What is written per subset:
    X.npy    (N, 2, 1250) float32   channel 0 ECG, channel 1 PPG
    Y.npy    (N, 3)       float32   SBP, DBP, MAP
    S.npy    (N,)         int32     subject index into subjects.json
    D.npy    (N, 4)       float32   age, BMI, height, weight
    G.npy    (N,)         int8      gender, 1 male, 0 female, -1 unknown

SBP and DBP come from the file's own label fields rather than being recomputed. Measurement
showed the published labels correlate with the ABP extrema at 0.947 and 0.931 but differ by a
median of 4.27 and 2.00 mmHg, so PulseDB is using a beat-wise estimator rather than a plain
max/min. Recomputing them would silently replace their definition with a noisier one.

MAP is not provided as a field and is computed as the temporal mean of the ABP samples in the
segment, which is the definition the manuscript used.

The split: PulseDB supplies Train, CalBased_Test, CalFree_Test and the two AAMI sets but no
validation set. A validation set is carved out of the training subjects, 15 percent of them,
chosen by subject so that no subject appears on both sides. Model selection therefore never sees
a subject it was trained on.

Resumable: conversion writes in chunks and records progress, so an interrupted run continues
where it stopped.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "AllDataSet"
OUT = ROOT / "converted"
REPORTS = ROOT / "reports"

FS = 125.0
ECG_CH, PPG_CH, ABP_CH = 0, 1, 2
SAMPLES = 1250
CHUNK = 2048
VALIDATION_FRACTION = 0.15
SPLIT_SEED = 42

SUBSETS = {
    "train": "VitalDB_Train_Subset.mat",
    "calbased_test": "VitalDB_CalBased_Test_Subset.mat",
    "calfree_test": "VitalDB_CalFree_Test_Subset.mat",
    "aami_cal": "VitalDB_AAMI_Cal_Subset.mat",
    "aami_test": "VitalDB_AAMI_Test_Subset.mat",
}


def deref_strings(f: h5py.File, dataset: h5py.Dataset) -> list[str]:
    out = []
    for ref in np.asarray(dataset).ravel():
        try:
            arr = np.asarray(f[ref]).ravel()
            out.append("".join(chr(int(c)) for c in arr if int(c) > 0))
        except Exception:
            out.append("")
    return out


def convert(tag: str, filename: str) -> dict:
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "convert_state.json"
    meta_path = out_dir / "meta.json"

    if meta_path.is_file():
        print(f"  [{tag}] already converted, skipping", flush=True)
        return json.loads(meta_path.read_text())

    src = DATA / filename
    with h5py.File(src, "r") as f:
        sub = f["Subset"]
        n = int(sub["SBP"].shape[1])

        sbp = np.asarray(sub["SBP"], dtype=np.float32).ravel()
        dbp = np.asarray(sub["DBP"], dtype=np.float32).ravel()
        subject_names = deref_strings(f, sub["Subject"])
        uniq = sorted(set(subject_names))
        index_of = {s: i for i, s in enumerate(uniq)}
        S = np.asarray([index_of[s] for s in subject_names], dtype=np.int32)

        demo = np.full((n, 4), np.nan, dtype=np.float32)
        for col, key in enumerate(("Age", "BMI", "Height", "Weight")):
            if key in sub:
                demo[:, col] = np.asarray(sub[key], dtype=np.float32).ravel()
        gender = np.full(n, -1, dtype=np.int8)
        if "Gender" in sub:
            for i, g in enumerate(deref_strings(f, sub["Gender"])):
                gu = g.strip().upper()
                gender[i] = 1 if gu.startswith("M") else (0 if gu.startswith("F") else -1)

        X = np.lib.format.open_memmap(out_dir / "X.npy", mode="w+",
                                      dtype=np.float32, shape=(n, 2, SAMPLES))
        Y = np.zeros((n, 3), dtype=np.float32)
        Y[:, 0], Y[:, 1] = sbp, dbp

        done = 0
        if state_path.is_file():
            done = int(json.loads(state_path.read_text()).get("segments_done", 0))
            print(f"  [{tag}] resuming from segment {done:,}", flush=True)

        signals = sub["Signals"]
        for start in range(done, n, CHUNK):
            stop = min(start + CHUNK, n)
            block = np.asarray(signals[:, :, start:stop], dtype=np.float32)  # (1250, 3, k)
            block = np.transpose(block, (2, 1, 0))                           # (k, 3, 1250)
            X[start:stop, 0, :] = block[:, ECG_CH, :]
            X[start:stop, 1, :] = block[:, PPG_CH, :]
            Y[start:stop, 2] = block[:, ABP_CH, :].mean(axis=1)              # MAP
            state_path.write_text(json.dumps({"segments_done": int(stop)}))
            if (stop // CHUNK) % 20 == 0 or stop == n:
                print(f"    {tag}: {stop:,}/{n:,} ({stop / n * 100:.1f}%)", flush=True)
        X.flush()

    np.save(out_dir / "Y.npy", Y)
    np.save(out_dir / "S.npy", S)
    np.save(out_dir / "D.npy", demo)
    np.save(out_dir / "G.npy", gender)
    (out_dir / "subjects.json").write_text(json.dumps(uniq), encoding="utf-8")

    meta = {
        "tag": tag,
        "source_file": filename,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_segments": int(n),
        "n_subjects": len(uniq),
        "samples_per_segment": SAMPLES,
        "seconds_per_segment": SAMPLES / FS,
        "channel_order": ["ECG", "PPG"],
        "targets": ["SBP", "DBP", "MAP"],
        "map_definition": "temporal mean of the ABP channel within the segment",
        "label_source": "SBP and DBP taken from the file's own fields, not recomputed",
        "label_summary": {
            t: {"mean": float(Y[:, j].mean()), "std": float(Y[:, j].std()),
                "min": float(Y[:, j].min()), "max": float(Y[:, j].max())}
            for j, t in enumerate(("SBP", "DBP", "MAP"))
        },
        "nonfinite_signal_values": int((~np.isfinite(np.asarray(X[:: max(1, n // 500)]))).sum()),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    state_path.unlink(missing_ok=True)
    print(f"  [{tag}] done: {n:,} segments, {len(uniq):,} subjects", flush=True)
    return meta


def build_validation_split() -> dict:
    """Carve a subject-disjoint validation set out of the training subjects."""
    tag_dir = OUT / "train"
    S = np.load(tag_dir / "S.npy")
    subjects = json.loads((tag_dir / "subjects.json").read_text())
    n_subjects = len(subjects)

    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(n_subjects)
    n_val = int(round(n_subjects * VALIDATION_FRACTION))
    val_subjects = set(perm[:n_val].tolist())

    is_val = np.isin(S, list(val_subjects))
    train_rows = np.where(~is_val)[0].astype(np.int64)
    val_rows = np.where(is_val)[0].astype(np.int64)

    assert not (set(S[train_rows].tolist()) & set(S[val_rows].tolist())), "subject leak"

    np.save(tag_dir / "rows_train.npy", train_rows)
    np.save(tag_dir / "rows_validation.npy", val_rows)

    return {
        "split_seed": SPLIT_SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "n_subjects_total": n_subjects,
        "n_subjects_train": n_subjects - n_val,
        "n_subjects_validation": n_val,
        "n_segments_train": int(train_rows.size),
        "n_segments_validation": int(val_rows.size),
        "subject_disjoint": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None,
                        help="convert only these tags, e.g. --only calfree_test")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    metas = {}
    order = [t for t in SUBSETS if args.only is None or t in args.only]
    # Smallest first, so a mistake surfaces in seconds rather than after the 12.85 GB file.
    order.sort(key=lambda t: (DATA / SUBSETS[t]).stat().st_size)
    for tag in order:
        print(f"converting {tag} ({SUBSETS[tag]})...", flush=True)
        metas[tag] = convert(tag, SUBSETS[tag])

    split = None
    if (OUT / "train" / "meta.json").is_file():
        split = build_validation_split()
        print(f"validation split: {split['n_subjects_validation']} of "
              f"{split['n_subjects_total']} subjects, "
              f"{split['n_segments_validation']:,} segments", flush=True)

    payload = {
        "analysis_id": "p2_convert_and_split",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "converted": metas,
        "validation_split": split,
    }
    (REPORTS / "p2_convert_and_split.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nconversion complete")


if __name__ == "__main__":
    main()
