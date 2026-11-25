"""Downloads the public data, converts it, checks it against the data used in the paper, reruns the analyses from
the frozen weights and predictions, draws the figures, and compares every regenerated result file with
reports_reference/. Logs go to results/logs/, figures to results/figures/.

python reproduce.py                  download, conversion, hash check, analyses, figures, comparison
python reproduce.py --retrain        also retrains the networks first (GPU, many hours); results then differ
                                     slightly from the paper because GPU training is not bit-exact
python reproduce.py --from s93       restart the step list at the first step whose script name starts with s93
python reproduce.py --compare-only   only compare reports/ with reports_reference/"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
LOGS = ROOT / "results" / "logs"
FIGS = ROOT / "results" / "figures"
UCI_ALL = {"UCI_TAG": "uci_all"}
BIG = "1000000000"

DATA = [
    ("p1_inspect_pulsedb.py", [], {}),
    ("p2_convert_and_split.py", [], {}),
    ("p6_convert_uci.py", ["--parts", "Part_1", "Part_2", "Part_3", "Part_4",
                           "--max-windows-per-record", BIG, "--max-windows", BIG], UCI_ALL),
    ("p3_build_cache.py", ["--tags", "train", "calfree_test"], {}),   # image cache of the CWC networks
]

# networks trained in the paper with the scripts of this package (the frozen weights make this optional)
TRAIN = [
    ("s10_loss_weights.py", [], {}),
    ("s24_balanced_fusion.py", [], {}),
    ("s35_within_subject_objective.py", ["--arms", "ABS", "PAIRED", "CENTERED", "ABSPAIRED"], {}),
    ("s35_within_subject_objective.py", ["--arms", "ABS", "PAIRED", "--drop-ecg", "--suffix", "_ppgonly"], {}),
    ("s35_within_subject_objective.py", ["--arms", "ABS", "PAIRED", "--seed", "44", "--suffix", "_s44"], {}),
    ("s35_within_subject_objective.py", ["--arms", "ABS", "PAIRED", "--seed", "45", "--suffix", "_s45"], {}),
    ("s37_backbones.py", [], {}),
    ("s100_moderntcn.py", [], {}),
    ("s38_siamese_baseline.py", [], {}),
]

# analyses, in the order in which the result files of the paper were written
ANALYSES = [
    ("s29_level_information.py", [], {}),
    ("s36_calibrated_end_to_end.py", [], {}),
    ("s36_calibrated_end_to_end.py", ["--suffix", "_s43"], {}),
    ("s36_calibrated_end_to_end.py", ["--suffix", "_ppgonly"], {}),
    ("s38_siamese_baseline.py", ["--skip-train"], {}),
    ("s50_validation_ensemble.py", ["--criterion", "all"], {}),
    ("s63_cuffless_two_stage.py", [], {}),
    ("s65_level_attention_pooling.py", [], {}),
    ("s67_window_weighting.py", [], {}),
    ("s66_subject_level_encoder.py", [], {}),
    ("s68_level_ensemble.py", [], {}),
    ("s70_two_stage_headline.py", [], {}),
    ("s71_two_stage_bland_altman.py", [], {}),
    ("s72_two_stage_ablation.py", [], {}),
    ("s75_window_order.py", [], {}),
    ("s76_naive_level_baseline.py", [], {}),
    ("s77_true_level_single_tracker.py", [], {}),
    ("s78_aami_confirmatory.py", [], {}),
    ("s79_matched_single_calfree.py", [], {}),
    ("s80_map_results.py", [], {}),
    ("s81_seeds_and_comparators.py", [], {}),
    ("s82_ridge_cv_and_level_r.py", [], {}),
    ("s82b_dbp_alpha_sensitivity.py", [], {}),
    ("s70_two_stage_headline.py", [], {"BP_SINGLE": "1"}),
    ("s71_two_stage_bland_altman.py", [], {"BP_SINGLE": "1"}),
    ("s72_two_stage_ablation.py", [], {"BP_SINGLE": "1"}),
    ("s76_naive_level_baseline.py", [], {"BP_SINGLE": "1"}),
    ("s45_crossover.py", [], UCI_ALL),
    ("s54_variability_within_source.py", [], UCI_ALL),
    ("s69_two_stage_uci.py", [], {"UCI_TAG": "uci_all", "UCI_FIXED": "1"}),
    ("s83_sex_and_demographics.py", [], {}),
    ("s89_variability_gate.py", [], {}),
    ("s93_fluctuation_gated_system.py", [], {}),
    ("s94_flare_variants.py", [], {}),
    ("s95_flare_map_sex.py", [], {}),
    ("s96_flare_warmup.py", [], {}),
    ("s97_flare_ablation.py", [], {}),
    ("s93_fluctuation_gated_system.py", ["--no-gate"], {}),
    ("s99_dynamics_source.py", [], {}),
    ("s101_fluctuation_index_warmup.py", [], {}),
    ("s102_ecg_feature_contribution.py", [], {}),
    ("s103_uci_dynamics.py", [], {}),
    ("s104_flare_branches.py", [], {}),
    ("s106_flare_agreement.py", [], {}),
    ("s105_flare_uci.py", [], {}),
    ("s107_ecg_level_transfer.py", [], {}),
]

FIGURES = [
    ("fig1_data.py", [], {}),
    ("fig1_flare_overview.py", [str(FIGS / "Figure_1.pdf")], {}),
    ("fig2_per_subject.py", [str(FIGS / "Figure_2.pdf")], {}),
    ("fig3_warmup.py", [str(FIGS / "Figure_3.pdf")], {}),
    ("fig4_agreement.py", [str(FIGS / "Figure_4.pdf")], {}),
    ("fig5_crossover.py", [str(FIGS / "Figure_5.pdf")], {}),
]

# s10 and s24 are training runs, rerun only with --retrain; s50_validation_ensemble.json is an earlier output of
# s50 (the current script writes the _combined, _sbp and _dbp files)
NOT_RERUN = ["s10_loss_weights", "s24_balanced_fusion", "s50_validation_ensemble"]


def run(i, script, args, env):
    name = f"{i:02d}_{script[:-3]}{'_' + '_'.join(a.strip('-') for a in args if not a.startswith(str(ROOT))) if args else ''}"
    print(f"> {script} {' '.join(args)} {' '.join(f'{k}={v}' for k, v in env.items())}", flush=True)
    t0 = time.perf_counter()
    with open(LOGS / f"{name[:120]}.txt", "w", encoding="utf-8") as log:
        r = subprocess.run([PY, script, *args], cwd=ROOT / "scripts", stdout=log, stderr=subprocess.STDOUT,
                           env={**os.environ, **env})
    if r.returncode != 0:
        sys.exit(f"{script} failed after {time.perf_counter() - t0:.0f} s; see results/logs/{name[:120]}.txt")
    print(f"  done in {time.perf_counter() - t0:.0f} s", flush=True)


def check_hashes():
    bad, n = [], 0
    for line in (ROOT / "data" / "converted_sha256.txt").read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        h = hashlib.sha256()
        with open(ROOT / rel, "rb") as f:
            for blk in iter(lambda: f.read(1 << 24), b""):
                h.update(blk)
        n += 1
        if h.hexdigest() != digest:
            bad.append(rel)
    if bad:
        sys.exit("converted data differ from the data used in the paper: " + ", ".join(bad))
    print(f"all {n} converted arrays match the data used in the paper (SHA-256)")


def leaves(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{p}/{k}")
    elif isinstance(o, list):
        for k, v in enumerate(o):
            yield from leaves(v, f"{p}[{k}]")
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield p, float(o)


def compare(tol=0.005, retrained=False):
    ref_dir, new_dir = ROOT / "reports_reference", ROOT / "reports"
    lines, worst = [], 0.0
    for ref in sorted(ref_dir.glob("*.json")):
        new = new_dir / ref.name
        if ref.stem in NOT_RERUN and not (retrained and ref.stem != "s50_validation_ensemble"):
            lines.append(f"{ref.name}: stored only (not rerun)")
            continue
        if not new.is_file():
            lines.append(f"{ref.name}: MISSING")
            continue
        a = dict(leaves(json.loads(ref.read_text(encoding="utf-8"))))
        b = dict(leaves(json.loads(new.read_text(encoding="utf-8"))))
        keys = [k for k in a if k in b and "minutes" not in k and "seconds" not in k]
        diffs = [abs(a[k] - b[k]) for k in keys]
        off = sum(d > tol for d in diffs)
        m = max(diffs, default=0.0)
        worst = max(worst, m)
        lines.append(f"{ref.name}: {len(keys)} numbers compared, {off} differ by more than {tol}, largest {m:.4g}"
                     + (f", {len(a) - len(keys)} missing" if len(a) > len(keys) else ""))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "comparison.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"comparison written to results/comparison.txt (largest difference {worst:.4g})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--compare-only", action="store_true")
    a = ap.parse_args()
    if a.compare_only:
        compare()
        return
    LOGS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    steps = (TRAIN if a.retrain else []) + ANALYSES + FIGURES
    if a.start is None:
        subprocess.run([PY, str(ROOT / "data" / "download_data.py")], check=True)
        for i, (s, args, env) in enumerate(DATA):
            run(i, s, args, env)
        check_hashes()
    else:
        hit = [i for i, (s, _, _) in enumerate(steps) if s.startswith(a.start)]
        if not hit:
            sys.exit(f"no step starts with {a.start}")
        steps = steps[hit[0]:]
    for i, (s, args, env) in enumerate(steps, start=len(DATA)):
        run(i, s, args, env)
    compare(retrained=a.retrain)


if __name__ == "__main__":
    main()
