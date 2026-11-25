"""Convert the UCI cuff-less blood pressure dataset into the same layout as the PulseDB sets.

This is for external validation only. Nothing here is ever trained on: the models are frozen after
training on PulseDB VitalDB and applied to this data unchanged, which is the point. VitalDB is
Korean surgical patients and this dataset is derived from MIMIC-II intensive care records, so the
shift is real and the numbers should be read as a transfer result rather than a second test score.

Two details decide whether the comparison is meaningful at all.

The channel order is reversed. A UCI record is (samples, 3) holding PPG, ABP, ECG, while our
arrays are (2, samples) holding ECG then PPG. Getting this backwards would silently feed the
network a PPG where it expects an ECG and produce a meaningless number that still looks plausible.

The normalisation must be the one PulseDB applied, not one of our own. PulseDB scales each segment
to [0, 1] individually, which was verified on the converted arrays: the per-window minimum and
maximum have zero variance. So the same per-window scaling is applied here, and the training set
statistics are left to the model script, exactly as for the PulseDB sets.

Labels are taken from the ABP channel of each window: the systolic value is the mean of the beat
peaks and the diastolic value the mean of the beat troughs, rather than the single maximum and
minimum, because one artefact sample would otherwise set the label.

Filtering is deliberately loose. Anything stricter would quietly select the windows our model
happens to find easy and flatter the transfer result.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve().parents[1] / "data" / "uci"
OUT = ROOT / "converted" / __import__("os").environ.get("UCI_TAG", "uci_external")

FS = 125
WINDOW = 1250                # ten seconds, the same as PulseDB
SBP_RANGE = (50.0, 220.0)
DBP_RANGE = (20.0, 150.0)
MIN_PULSE_PRESSURE = 10.0
MIN_BEATS = 5


def window_labels(abp: np.ndarray) -> tuple[float, float, float] | None:
    """Mean systolic peak, mean diastolic trough, and the temporal mean of the waveform."""
    peaks, _ = find_peaks(abp, distance=int(0.4 * FS))       # at most 150 bpm
    troughs, _ = find_peaks(-abp, distance=int(0.4 * FS))
    if len(peaks) < MIN_BEATS or len(troughs) < MIN_BEATS:
        return None
    sbp = float(np.mean(abp[peaks]))
    dbp = float(np.mean(abp[troughs]))
    return sbp, dbp, float(abp.mean())


def usable(sbp: float, dbp: float, ecg: np.ndarray, ppg: np.ndarray) -> bool:
    if not (SBP_RANGE[0] <= sbp <= SBP_RANGE[1]):
        return False
    if not (DBP_RANGE[0] <= dbp <= DBP_RANGE[1]):
        return False
    if sbp - dbp < MIN_PULSE_PRESSURE:
        return False
    if ecg.std() < 1e-6 or ppg.std() < 1e-6:
        return False
    return True


def scale01(v: np.ndarray) -> np.ndarray:
    lo, hi = float(v.min()), float(v.max())
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="*", default=["Part_1"])
    ap.add_argument("--max-windows-per-record", type=int, default=12,
                    help="caps how far one long record can dominate the set")
    ap.add_argument("--max-windows", type=int, default=60000)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    xs, ys, ss = [], [], []
    stats = {"records_seen": 0, "windows_examined": 0, "rejected_label": 0,
             "rejected_range": 0, "rejected_nonfinite": 0}
    record_id = 0

    for part in args.parts:
        path = SOURCE / f"{part}.mat"
        if not path.is_file():
            print(f"  {part}: missing, skipped")
            continue
        with h5py.File(path, "r") as f:
            refs = f[part]
            n_records = refs.shape[0]
            print(f"  {part}: {n_records:,} records", flush=True)
            for i in range(n_records):
                if len(ys) >= args.max_windows:
                    break
                rec = np.asarray(f[refs[i, 0]])
                stats["records_seen"] += 1
                record_id += 1
                if rec.ndim != 2 or rec.shape[1] != 3 or rec.shape[0] < WINDOW:
                    continue
                ppg_all, abp_all, ecg_all = rec[:, 0], rec[:, 1], rec[:, 2]
                kept_here = 0
                for start in range(0, rec.shape[0] - WINDOW + 1, WINDOW):
                    if kept_here >= args.max_windows_per_record:
                        break
                    if len(ys) >= args.max_windows:
                        break
                    sl = slice(start, start + WINDOW)
                    ppg, abp, ecg = ppg_all[sl], abp_all[sl], ecg_all[sl]
                    stats["windows_examined"] += 1
                    if not (np.isfinite(ppg).all() and np.isfinite(abp).all()
                            and np.isfinite(ecg).all()):
                        stats["rejected_nonfinite"] += 1
                        continue
                    lab = window_labels(abp)
                    if lab is None:
                        stats["rejected_label"] += 1
                        continue
                    sbp, dbp, mapv = lab
                    if not usable(sbp, dbp, ecg, ppg):
                        stats["rejected_range"] += 1
                        continue
                    # channel order flipped to match ours: ECG first, PPG second
                    xs.append(np.stack([scale01(ecg), scale01(ppg)]).astype(np.float32))
                    ys.append([sbp, dbp, mapv])
                    ss.append(record_id)
                    kept_here += 1
                if i % 500 == 0:
                    print(f"    record {i:,}/{n_records:,}  kept {len(ys):,}", flush=True)

    X = np.stack(xs).astype(np.float32)
    Y = np.asarray(ys, dtype=np.float32)
    S = np.asarray(ss, dtype=np.int32)
    np.save(OUT / "X.npy", X)
    np.save(OUT / "Y.npy", Y)
    np.save(OUT / "S.npy", S)

    meta = {
        "tag": "uci_external", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "UCI cuff-less blood pressure estimation dataset, Part files",
        "parts": args.parts, "sampling_rate_hz": FS, "window_samples": WINDOW,
        "channel_order": "0 = ECG, 1 = PPG, matching the PulseDB conversion",
        "per_window_scaling": "each channel scaled to [0, 1] within its own window, as PulseDB "
                              "does for its own segments",
        "label_definition": "SBP is the mean of ABP beat peaks in the window, DBP the mean of the "
                            "troughs, MAP the temporal mean of ABP",
        "grouping": "one group per source record; the dataset carries no subject identifiers, so "
                    "records are the finest grouping available and are used as the bootstrap unit",
        "filters": {"sbp_range": SBP_RANGE, "dbp_range": DBP_RANGE,
                    "min_pulse_pressure": MIN_PULSE_PRESSURE, "min_beats": MIN_BEATS},
        "counts": {**stats, "windows_kept": int(len(Y)), "records_kept": int(len(np.unique(S)))},
        "never_used_for_training": True,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print()
    print(f"kept {len(Y):,} windows from {len(np.unique(S)):,} records")
    print(f"  examined {stats['windows_examined']:,}, rejected "
          f"{stats['rejected_label']:,} for too few beats, "
          f"{stats['rejected_range']:,} out of range, "
          f"{stats['rejected_nonfinite']:,} non-finite")
    print(f"  SBP {Y[:, 0].min():.1f}-{Y[:, 0].max():.1f} mean {Y[:, 0].mean():.1f}")
    print(f"  DBP {Y[:, 1].min():.1f}-{Y[:, 1].max():.1f} mean {Y[:, 1].mean():.1f}")
    print(f"  X {X.shape}  range {X.min():.3f}..{X.max():.3f}")
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
