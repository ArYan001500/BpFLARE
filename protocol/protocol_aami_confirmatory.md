# Confirmatory evaluation on the PulseDB VitalDB AAMI subjects — protocol fixed before running

Written and committed before any two-stage prediction is computed on these subjects.
Nothing below may be changed after the run; any deviation is reported as a deviation.

## Why
CalFree (144 subjects) was reused during development, so its numbers are exploratory. The 116
VitalDB AAMI subjects (`VitalDB_AAMI_Cal_Subset` + `VitalDB_AAMI_Test_Subset`) share no subject with
the training/validation subjects or with CalFree (checked on subject identifiers: overlap 0 and 0),
and the two-stage system has never been run on them. Earlier runs s14/s46 evaluated calibrated
pairwise/absolute trackers on AAMI; no two-stage design choice used them.

## Cohort
- All windows of each AAMI subject: AAMI_Cal and AAMI_Test windows pooled per subject.
- Inclusion: subjects with at least 40 windows (20 warm-up + at least 20 scored). Reported: number
  included and excluded.
- Note: the AAMI subset was built by PulseDB to follow the AAMI pressure distribution, so its
  pressure range differs from CalFree; this is part of the test, not a reason to adapt anything.

## Frozen system (no refitting, no tuning, no selection on these subjects)
- Stage 1: ridge on averaged RAW_w encoder features, fitted on the same 194 validation subjects as
  s70, same alpha selection, same seed.
- Stage 2, primary: the single residual-network pairwise tracker `WS_PAIRED`.
- Stage 2, secondary: mean of the three trackers whose weights exist
  (`WS_PAIRED`, `BB_INCEPTION_PAIRED`, `BB_TCN_PAIRED`); `WS_PAIRED_s43` has no saved weights.
- Comparator: the absolute residual network `WS_ABS` (same backbone as the primary tracker), scored on
  exactly the same windows.
- Warm-up: M = 20 windows (200 s of signal) drawn at random, 50 draws per subject with the s70 seed
  scheme (SEED + 7 d); warm-up windows excluded from scoring; no pressure label of the subject used.

## Endpoints
- Primary: SBP, per-subject MAE averaged over draws, then averaged over subjects;
  paired difference (two-stage single tracker minus WS_ABS) with a 2000-replicate subject bootstrap.
  Confirmed if the upper limit of the 95% interval is below 0.
- Secondary: the same for DBP; the three-tracker ensemble; number of subjects whose SBP error fell.
- Descriptive: mean error and SD of error; MAE stratified by reference SBP below 90, 90–140 and
  above 140 mmHg (pooled windows, descriptive only).

## Reporting rule
All results are reported whatever their direction. If the primary endpoint fails, the paper says so.
