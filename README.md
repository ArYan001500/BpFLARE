# FLARE: calibration-free cuffless blood pressure estimation

Code, frozen model weights, subject lists, protocol, and result files for the paper "Fluctuation-Aware
Within-Patient Tracking with Unlabeled Anchoring for Calibration-Free Cuffless Blood Pressure Monitoring".

FLARE (Fluctuation-aware Level-Anchored Relative Estimation) estimates absolute blood pressure from
electrocardiogram (ECG) and photoplethysmogram (PPG) windows without any pressure reading from the patient. Stage 1
infers the patient's level from 20 unlabeled warm-up windows with a ridge rule on frozen encoder features. Stage 2 is
a tracker trained only on differences between windows of the same patient. A label-free fluctuation gate refines the
level with handcrafted ECG and PPG features for patients whose pressure moves little.

## Quick start

```
pip install -r requirements.txt
python smoke_test.py      # about 30 s, needs no dataset
python reproduce.py       # downloads the data and reproduces the results
```

`smoke_test.py` checks that every script compiles, that a stored network loads and runs, that the reference results
and the protocol are intact, and that every data source answers.

`reproduce.py` runs the whole chain:

1. downloads the data and the ModernTCN code (`data/download_data.py`);
2. converts the data and checks the converted arrays against the SHA-256 hashes of the arrays used in the paper
   (`data/converted_sha256.txt`);
3. builds the image cache of the coherence-image networks (`p3_build_cache.py`, about 33 GB);
4. reruns the analyses from the frozen weights and predictions;
5. draws Figures 1 to 5 into `results/figures/`;
6. compares every new result file with `reports_reference/` and writes the differences to `results/comparison.txt`.

`python reproduce.py --retrain` retrains the networks before the analyses. GPU training is not bit-exact, so
retrained results differ slightly from the paper. `python reproduce.py --from s93` restarts the run at a given step,
and `--compare-only` repeats the comparison.

## Data

The data are not redistributed. `data/download_data.py` fetches them; they can also be downloaded by hand:

- PulseDB version 1, VitalDB subsets (CC BY-NC-SA 4.0):
  https://www.kaggle.com/datasets/weinanwangrutgers/pulsedb-balanced-training-and-testing.
  The five `VitalDB_*_Subset.mat` files (about 19 GB) go to `AllDataSet/`.
- UCI Cuff-Less Blood Pressure Estimation (CC BY 4.0):
  https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation.
  `Part_1.mat` to `Part_4.mat` go to `data/uci/`.

The converted arrays are written to `converted/`. The subject lists in `splits/` show every split.

## Layout

| Folder | Content |
|---|---|
| `scripts/` | data conversion, training, analysis, and figure scripts |
| `runs/s4_pat_control/` | frozen weights of the networks behind the reported results (including the 21 candidates of the absolute ensemble) and their frozen test predictions; the seed-43 trackers have predictions only, because their weights were not kept |
| `reports_reference/` | the result files behind every number in the paper |
| `splits/` | subject identifiers of every PulseDB subset and of the training and validation division |
| `data/` | the download script and the hashes of the converted arrays |
| `protocol/` | the confirmatory protocol, fixed before the AAMI subjects were scored |
| `external/` | place for the official ModernTCN code (see `external/README.md`) |

## Scripts behind the paper

All scripts run from `scripts/`. `reproduce.py` lists their order, arguments, and environment variables.

| Paper item | Script |
|---|---|
| Absolute and pairwise residual networks (seeds 42 to 45, PPG only) | `s35_within_subject_objective.py` |
| InceptionTime-style and TCN backbones | `s37_backbones.py` |
| ModernTCN (official code, commit 56a9a2c) | `bb_moderntcn.py`, `s100_moderntcn.py` |
| Siamese baseline | `s38_siamese_baseline.py` |
| Absolute ensemble selected on validation from 21 candidate networks | `s50_validation_ensemble.py` |
| Target weights | `s10_loss_weights.py` |
| Image cache of the coherence-image networks | `p3_build_cache.py`, `representation.py` |
| Stage-1 level rule and level information | `s29_level_information.py`, `s82_ridge_cv_and_level_r.py`, `s82b_dbp_alpha_sensitivity.py` |
| Stage-1 alternatives | `s65` to `s68`, `s76_naive_level_baseline.py` |
| Window order in PulseDB | `s75_window_order.py` |
| Two-stage core: main results, agreement, ablation | `s63`, `s70` to `s72`, `s77`, `s79` |
| Confirmatory run on the AAMI subjects | `s78_aami_confirmatory.py` |
| Mean arterial pressure, seeds, and comparators | `s80`, `s81` |
| Reference-calibrated analysis and variability relation (Fig. 5) | `s36_calibrated_end_to_end.py`, `s45_crossover.py`, `s54_variability_within_source.py` |
| UCI records | `s69_two_stage_uci.py` |
| FLARE with the fluctuation gate (main results) | `s89_variability_gate.py`, `s93_fluctuation_gated_system.py` |
| FLARE seeds, PPG only, MAP, warm-up, and ablation | `s94` to `s97` |
| Fluctuation branches (Table 4) | `s104_flare_branches.py` |
| Descriptive agreement (Table 8, Fig. 4) | `s106_flare_agreement.py` |
| FLARE on the UCI records | `s105_flare_uci.py`, `s107_ecg_level_transfer.py` |
| Source of within-subject dynamics (Table 7) | `s99_dynamics_source.py` |
| Label-free fluctuation index | `s101_fluctuation_index_warmup.py` |
| ECG and PPG feature contribution | `s102_ecg_feature_contribution.py` |
| Within-record dynamics on UCI | `s103_uci_dynamics.py` |
| Figures 1 to 5 | `fig1_data.py`, `fig1_flare_overview.py`, `fig2_per_subject.py`, `fig3_warmup.py`, `fig4_agreement.py`, `fig5_crossover.py` |

`bp_networks.py` holds the network classes and the loss shared by these scripts, and `network_loader.py` loads it.

## Confirmatory protocol

`protocol/protocol_aami_confirmatory.md` was written and committed on 2 October 2026, before the AAMI subjects were
scored, and has not changed since. Its SHA-256 hash is
f381b33934819a8f7bf1348f5209a5783609fe589701aac28e7a6192636fc0cc.

## Requirements

Python 3.14 with the packages in `requirements.txt`. Training and inference used one NVIDIA RTX 4060 laptop GPU (5 to
20 minutes per network). The analysis scripts run on a CPU once the predictions are cached. Random draws use the
seed 42 + 7d for draw d, and bootstrap intervals use 2,000 replicates over subjects (over records on UCI).
