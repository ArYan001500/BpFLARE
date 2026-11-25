"""Step P1: inspect the PulseDB VitalDB subsets before anything is built on top of them.

Every downstream decision depends on facts that must be measured rather than assumed: how many
segments each file holds, which of the three signal channels is which, whether subject
identifiers are genuinely disjoint between the training and calibration-free test sets, and
whether the labels are what the manuscript's pipeline expects.

Two facts already differ from the manuscript's setup and will need handling:

  * Segments are 1250 samples (10 s), not the 1000 samples (8 s) used in the manuscript.
  * The files carry SBP and DBP but no MAP field, while the manuscript regresses three targets.
    If the arterial pressure waveform is one of the three channels, MAP can be derived from it
    the same way the manuscript did; this script checks whether that channel is present.

Channel identity is established by measurement, not by the documented order. The arterial
pressure channel is the one carrying physiological mmHg values; ECG is separated from PPG by its
broadband QRS energy. The same three checks used earlier on UCI and VitalDB are reused here so
the answer is comparable.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "AllDataSet"
REPORTS = ROOT / "reports"

FS = 125.0
FILES = [
    "VitalDB_Train_Subset.mat",
    "VitalDB_CalBased_Test_Subset.mat",
    "VitalDB_CalFree_Test_Subset.mat",
    "VitalDB_AAMI_Cal_Subset.mat",
    "VitalDB_AAMI_Test_Subset.mat",
]
SAMPLE_SEGMENTS = 400


def deref_strings(f: h5py.File, dataset: h5py.Dataset) -> list[str]:
    """MATLAB cell array of char -> list of Python strings."""
    out = []
    flat = np.asarray(dataset).ravel()
    for ref in flat:
        try:
            arr = np.asarray(f[ref]).ravel()
            out.append("".join(chr(int(c)) for c in arr if int(c) > 0))
        except Exception:
            out.append("")
    return out


def hf_fraction(x: np.ndarray, lo: float = 15.0) -> float:
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freq = np.fft.rfftfreq(len(x), 1.0 / FS)
    total = spec[freq > 0.3].sum()
    return float(spec[freq >= lo].sum() / max(total, 1e-12))


def channel_report(sig: np.ndarray) -> dict:
    """sig: (n_segments, 3, n_samples). Identify each channel by measured properties."""
    out = {}
    for c in range(sig.shape[1]):
        ch = sig[:, c, :]
        finite = ch[np.isfinite(ch)]
        out[c] = {
            "min": float(finite.min()),
            "max": float(finite.max()),
            "mean": float(finite.mean()),
            "std": float(finite.std()),
            "median_hf_fraction_above_15hz": float(np.median([hf_fraction(w) for w in ch[:60]])),
            "looks_like_pressure_mmHg": bool(40 <= finite.mean() <= 180 and finite.max() > 80),
        }
    pressure = [c for c, v in out.items() if v["looks_like_pressure_mmHg"]]
    non_pressure = [c for c in out if c not in pressure]
    ecg = max(non_pressure, key=lambda c: out[c]["median_hf_fraction_above_15hz"]) if non_pressure else None
    ppg = [c for c in non_pressure if c != ecg]
    return {
        "per_channel": out,
        "abp_channel": pressure[0] if len(pressure) == 1 else pressure,
        "ecg_channel": ecg,
        "ppg_channel": ppg[0] if len(ppg) == 1 else ppg,
    }


def inspect(path: Path) -> dict:
    info: dict = {"file": path.name, "bytes": path.stat().st_size}
    with h5py.File(path, "r") as f:
        sub = f["Subset"]
        fields = {k: {"shape": list(sub[k].shape), "dtype": str(sub[k].dtype)} for k in sub.keys()}
        info["fields"] = fields

        n = sub["SBP"].shape[1]
        info["n_segments"] = int(n)
        sig_shape = sub["Signals"].shape
        info["signals_shape_on_disk"] = list(sig_shape)
        info["samples_per_segment"] = int(sig_shape[0])
        info["n_channels"] = int(sig_shape[1])
        info["seconds_per_segment"] = float(sig_shape[0] / FS)

        # Labels, read in full: they are small even in the training file.
        sbp = np.asarray(sub["SBP"]).ravel()
        dbp = np.asarray(sub["DBP"]).ravel()
        info["labels"] = {
            "SBP": {"mean": float(sbp.mean()), "std": float(sbp.std()),
                    "min": float(sbp.min()), "max": float(sbp.max()),
                    "n_nonfinite": int((~np.isfinite(sbp)).sum())},
            "DBP": {"mean": float(dbp.mean()), "std": float(dbp.std()),
                    "min": float(dbp.min()), "max": float(dbp.max()),
                    "n_nonfinite": int((~np.isfinite(dbp)).sum())},
            "MAP_field_present": "MAP" in fields,
        }

        subjects = deref_strings(f, sub["Subject"])
        uniq = sorted(set(subjects))
        info["subjects"] = {
            "n_unique": len(uniq),
            "examples": uniq[:5],
            "segments_per_subject_min": int(min(subjects.count(u) for u in uniq[:50])) if uniq else 0,
        }
        counts = {}
        for s in subjects:
            counts[s] = counts.get(s, 0) + 1
        c = np.asarray(list(counts.values()))
        info["subjects"]["segments_per_subject"] = {
            "min": int(c.min()), "median": float(np.median(c)), "max": int(c.max())
        }
        info["_subject_list"] = uniq

        # Demographics
        for key in ("Age", "BMI", "Height", "Weight"):
            if key in fields:
                v = np.asarray(sub[key]).ravel()
                v = v[np.isfinite(v)]
                info.setdefault("demographics", {})[key] = {
                    "n_finite": int(v.size), "mean": float(v.mean()) if v.size else None}
        if "Gender" in fields:
            g = deref_strings(f, sub["Gender"])
            vals = {}
            for x in g:
                vals[x] = vals.get(x, 0) + 1
            info.setdefault("demographics", {})["Gender"] = vals

        # Signals: sample evenly across the file rather than reading 12.85 GB.
        idx = np.unique(np.linspace(0, n - 1, min(SAMPLE_SEGMENTS, n)).astype(int))
        block = np.stack([np.asarray(sub["Signals"][:, :, i]) for i in idx])  # (k, 1250, 3)
        block = np.transpose(block, (0, 2, 1))                                # (k, 3, 1250)
        info["channels"] = channel_report(block)
        info["signal_finite_fraction"] = float(np.isfinite(block).mean())
    return info


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in FILES:
        path = DATA / name
        if not path.is_file():
            print(f"MISSING: {name}", flush=True)
            continue
        print(f"inspecting {name} ({path.stat().st_size / 2**30:.2f} GiB)...", flush=True)
        results[name] = inspect(path)
        r = results[name]
        print(f"  segments {r['n_segments']:,} | subjects {r['subjects']['n_unique']:,} | "
              f"{r['samples_per_segment']} samples ({r['seconds_per_segment']:g} s) | "
              f"channels ECG={r['channels']['ecg_channel']} PPG={r['channels']['ppg_channel']} "
              f"ABP={r['channels']['abp_channel']}", flush=True)

    # The property the whole design depends on: are the split subject sets actually disjoint?
    overlaps = {}
    names = list(results.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = set(results[names[i]].get("_subject_list", []))
            b = set(results[names[j]].get("_subject_list", []))
            overlaps[f"{names[i]} vs {names[j]}"] = {
                "shared_subjects": len(a & b),
                "disjoint": len(a & b) == 0,
            }

    payload = {
        "analysis_id": "p1_inspect_pulsedb",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(DATA),
        "files": {k: {kk: vv for kk, vv in v.items() if kk != "_subject_list"}
                  for k, v in results.items()},
        "subject_overlap_between_files": overlaps,
    }
    (REPORTS / "p1_inspect_pulsedb.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = ["# PulseDB VitalDB subsets: what is actually in the files", "",
          f"*Generated {payload['created_utc']}*", "",
          "| File | GiB | segments | subjects | samples | seconds | ECG ch | PPG ch | ABP ch |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in results.items():
        c = r["channels"]
        md.append(f"| `{name.replace('VitalDB_', '').replace('_Subset.mat', '')}` "
                  f"| {r['bytes'] / 2**30:.2f} | {r['n_segments']:,} "
                  f"| {r['subjects']['n_unique']:,} | {r['samples_per_segment']} "
                  f"| {r['seconds_per_segment']:g} | {c['ecg_channel']} | {c['ppg_channel']} "
                  f"| {c['abp_channel']} |")

    md += ["", "## Labels", "",
           "| File | SBP mean | SBP std | DBP mean | DBP std | MAP field present |",
           "|---|---:|---:|---:|---:|---|"]
    for name, r in results.items():
        L = r["labels"]
        md.append(f"| `{name.replace('VitalDB_', '').replace('_Subset.mat', '')}` "
                  f"| {L['SBP']['mean']:.2f} | {L['SBP']['std']:.2f} "
                  f"| {L['DBP']['mean']:.2f} | {L['DBP']['std']:.2f} "
                  f"| {L['MAP_field_present']} |")

    md += ["", "## Subject disjointness between files", "",
           "| Pair | shared subjects | disjoint |", "|---|---:|---|"]
    for pair, v in overlaps.items():
        short = pair.replace("VitalDB_", "").replace("_Subset.mat", "")
        md.append(f"| {short} | {v['shared_subjects']} | {v['disjoint']} |")
    md.append("")
    (REPORTS / "p1_inspect_pulsedb.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
