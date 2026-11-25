"""Step P3: build the cross-wavelet representation cache, in parallel and resumably.

Design decisions and why each one is there.

**Random order with a fixed seed.** The converted files are ordered by subject, so building the
cache in file order would mean that after an hour the finished part covers only the first few
hundred subjects. Images are computed in a shuffled order instead, so whatever is finished at any
moment is a valid random sample across all subjects and can be trained on immediately. Each image
is still *stored* at its own row index; only the order of computation is shuffled.

**Completion bitmap.** A boolean array records which rows are finished. An interrupted run
resumes from exactly what is missing, with no rows recomputed and none skipped.

**Four channels.** Channels 0 to 2 are the manuscript's representation, verified bit for bit
against its implementation before any long job starts. Channel 3 is the group-delay map.

One cache serves every planned arm, because the other encodings are algebraic functions of what
is stored:

    corrected complex coherency   Re = sqrt(R^2) cos phi,  Im = sqrt(R^2) sin phi
    magnitude only                channels 1 and 2 zeroed
    band-limited 0.5 to 10 Hz     rows 0 to 97, since the frequency axis is not resized

so none of those require rebuilding.

**Preflight.** Disk space, legacy agreement and output sanity are all checked before the first
worker starts, because the failure that matters is the one discovered after an hour of compute.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONVERTED = ROOT / "converted"
CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import representation as rep  # noqa: E402

LEGACY_SOURCE = Path(__file__).resolve().parent / "bp_networks.py"
CHANNELS = 4
IMAGE = 128
SHUFFLE_SEED = 20260902
CHUNK = 256
SAFETY_MARGIN_GIB = 5.0

# Built in this order so that the parts needed to start training exist first.
BUILD_ORDER = ["train", "calfree_test", "calbased_test", "aami_test", "aami_cal",
               # external validation set, never trained on
               "uci_external"]

_WORKER_X = None


def _init_worker(x_path: str, shape: tuple):
    """np.load with mmap_mode parses the .npy header itself, so no offset has to be assumed."""
    global _WORKER_X
    _WORKER_X = np.load(x_path, mmap_mode="r")
    if _WORKER_X.shape != shape:
        raise RuntimeError(f"worker saw shape {_WORKER_X.shape}, expected {shape}")


def _compute_one(row: int) -> tuple[int, np.ndarray]:
    seg = np.asarray(_WORKER_X[row], dtype=np.float64)
    image = rep.compute_image(seg[0], seg[1], with_group_delay=True)
    return row, rep.quantize(image)


def load_legacy():
    import importlib.util

    spec = importlib.util.spec_from_file_location("q1_legacy", LEGACY_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preflight(tags: list[str]) -> dict:
    """Everything that can fail cheaply, checked before anything expensive starts."""
    checks: dict = {}

    total_rows = 0
    for tag in tags:
        meta = json.loads((CONVERTED / tag / "meta.json").read_text())
        total_rows += int(meta["n_segments"])
    needed = total_rows * CHANNELS * IMAGE * IMAGE
    free = shutil.disk_usage(str(ROOT))[2]
    checks["disk"] = {
        "rows": total_rows,
        "required_gib": needed / 2**30,
        "free_gib": free / 2**30,
        "margin_gib": SAFETY_MARGIN_GIB,
        "pass": bool(free - needed > SAFETY_MARGIN_GIB * 2**30),
    }

    legacy = load_legacy()
    X = np.load(CONVERTED / tags[0] / "X.npy", mmap_mode="r")
    idx = np.unique(np.linspace(0, len(X) - 1, 8).astype(int))
    diffs = []
    for i in idx:
        seg = np.asarray(X[i], dtype=np.float64)
        v = rep.verify_against_legacy(seg[0], seg[1], legacy)
        diffs.append(max(v["max_abs_difference_per_channel"]))
    checks["legacy_agreement"] = {
        "segments_checked": len(idx),
        "worst_difference": float(max(diffs)),
        "tolerance": 1e-5,
        "pass": bool(max(diffs) <= 1e-5),
    }

    seg = np.asarray(X[0], dtype=np.float64)
    img = rep.compute_image(seg[0], seg[1])
    packed = rep.quantize(img)
    restored = rep.dequantize(packed)
    checks["output_sanity"] = {
        "shape": list(img.shape),
        "finite": bool(np.isfinite(img).all()),
        "r2_in_range": bool(0.0 <= img[0].min() and img[0].max() <= 1.0),
        "sin_cos_in_range": bool(img[1:3].min() >= -1.0 and img[1:3].max() <= 1.0),
        "quantisation_max_error": float(np.abs(restored - img).max()),
        "pass": bool(
            img.shape == (CHANNELS, IMAGE, IMAGE)
            and np.isfinite(img).all()
            and np.abs(restored - img).max() < 0.01
        ),
    }

    checks["all_pass"] = all(v["pass"] for v in checks.values() if isinstance(v, dict))
    return checks


def build(tag: str, workers: int, report_every: int = 20000) -> dict:
    src = CONVERTED / tag
    dst = CACHE / tag
    dst.mkdir(parents=True, exist_ok=True)

    meta = json.loads((src / "meta.json").read_text())
    n = int(meta["n_segments"])
    x_path = src / "X.npy"
    x_shape = (n, 2, meta["samples_per_segment"])

    images_path = dst / "images.npy"
    bitmap_path = dst / "completion_bitmap.npy"

    if not images_path.exists():
        np.lib.format.open_memmap(images_path, mode="w+", dtype=np.uint8,
                                  shape=(n, CHANNELS, IMAGE, IMAGE))
    images = np.load(images_path, mmap_mode="r+")
    if images.shape != (n, CHANNELS, IMAGE, IMAGE):
        raise RuntimeError(f"{tag}: existing cache has shape {images.shape}, expected "
                           f"{(n, CHANNELS, IMAGE, IMAGE)}")

    done = np.load(bitmap_path) if bitmap_path.exists() else np.zeros(n, dtype=bool)
    if done.shape != (n,):
        raise RuntimeError(f"{tag}: bitmap length {done.shape} does not match {n} rows")

    order = np.random.default_rng(SHUFFLE_SEED).permutation(n)
    todo = order[~done[order]]
    if todo.size == 0:
        print(f"  [{tag}] already complete ({n:,} rows)", flush=True)
        return {"tag": tag, "rows": n, "computed_now": 0, "already_done": int(n)}

    print(f"  [{tag}] {todo.size:,} of {n:,} rows to build on {workers} workers", flush=True)
    started = time.perf_counter()
    completed = 0
    with Pool(processes=workers, initializer=_init_worker,
              initargs=(str(x_path), x_shape)) as pool:
        for row, packed in pool.imap_unordered(_compute_one, todo.tolist(), chunksize=CHUNK):
            images[row] = packed
            done[row] = True
            completed += 1
            if completed % report_every == 0:
                images.flush()
                np.save(bitmap_path, done)
                elapsed = time.perf_counter() - started
                rate = completed / elapsed
                print(f"    {tag}: {completed:,}/{todo.size:,} "
                      f"({rate:.0f} img/s, eta {(todo.size - completed) / rate / 60:.1f} min)",
                      flush=True)

    images.flush()
    np.save(bitmap_path, done)
    elapsed = time.perf_counter() - started
    print(f"  [{tag}] done: {completed:,} images in {elapsed / 60:.1f} min "
          f"({completed / elapsed:.0f} img/s)", flush=True)

    if not done.all():
        raise RuntimeError(f"{tag}: {int((~done).sum())} rows still missing after build")

    return {"tag": tag, "rows": n, "computed_now": int(completed),
            "minutes": elapsed / 60, "images_per_second": completed / elapsed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="*", default=BUILD_ORDER)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 6))
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    tags = [t for t in BUILD_ORDER if t in args.tags]

    if not args.skip_preflight:
        print("preflight...", flush=True)
        checks = preflight(tags)
        for name, c in checks.items():
            if isinstance(c, dict):
                print(f"  {name}: {'PASS' if c['pass'] else 'FAIL'}  "
                      f"{ {k: v for k, v in c.items() if k != 'pass'} }", flush=True)
        (REPORTS / "p3_preflight.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
        if not checks["all_pass"]:
            raise SystemExit("preflight failed; nothing was built")
        print("preflight passed\n", flush=True)

    results = {}
    for tag in tags:
        results[tag] = build(tag, args.workers)

    payload = {
        "analysis_id": "p3_build_cache",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "channels": ["R2", "sin_phi", "cos_phi", "group_delay_s"],
        "image_size": IMAGE,
        "shuffle_seed": SHUFFLE_SEED,
        "workers": args.workers,
        "representation_config": rep.RepresentationConfig().__dict__,
        "derivable_without_rebuild": {
            "corrected_complex_coherency": "Re = sqrt(R2)*cos_phi, Im = sqrt(R2)*sin_phi",
            "magnitude_only": "zero channels 1 and 2",
            "band_limited_0.5_to_10_Hz": "rows 0 to 97; the frequency axis is not resized",
        },
        "results": results,
    }
    (REPORTS / "p3_build_cache.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\ncache build complete")


if __name__ == "__main__":
    main()
