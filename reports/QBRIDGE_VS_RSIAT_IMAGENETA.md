# Q-BRIDGE versus original RSIAT on ImageNet-A

## Critical provenance note

The second result package was described as seeds **2015 and 2026**, but every
machine-readable record in that package identifies the seeds as **2015 and
2016**: `PACKAGE_MANIFEST.json`, `metadata/experiment_plan.json`, base-checkpoint
metadata, run paths, config directories, run manifests, and
`qbridge_summary.json`. No seed-2026 artifact is present.

This report preserves the measured provenance and therefore reports seeds
**1993, 2015, and 2016**. The 2016 results must not be relabeled as 2026. If
2026 was intended, that seed still needs to be run or its correct artifact must
be supplied.

## Executive summary

Four variants have complete results on all three available seeds: matched
legacy-QKSR R0, uniform-margin-plus-guard R3, raw-PQK Q-BRIDGE R5, and
calibrated-PQK Q-BRIDGE R6. Results below are descriptive mean +/- sample
standard deviation over seeds 1993/2015/2016.

| Run | AIA | Final top-1 | Final old | Final new | Forgetting |
|---|---:|---:|---:|---:|---:|
| R0 | 74.693 +/- 1.327 | 66.447 +/- 1.540 | 66.047 +/- 1.816 | 70.600 +/- 3.110 | 10.907 +/- 1.419 |
| R3 | 74.677 +/- 1.319 | 66.467 +/- 1.102 | 66.073 +/- 1.327 | 70.600 +/- 3.110 | 10.904 +/- 0.976 |
| R5 | 74.671 +/- 1.252 | 66.603 +/- 1.332 | 66.220 +/- 1.518 | 70.653 +/- 3.475 | **10.659 +/- 1.035** |
| R6 | **74.743 +/- 1.180** | **66.757 +/- 1.147** | **66.343 +/- 1.233** | **71.200 +/- 3.549** | 10.663 +/- 1.041 |

R6 has the highest mean AIA and final accuracy, while R5 has the lowest mean
forgetting by 0.004 pp. Paired against R0, R5 changes AIA by -0.022 +/- 0.098
pp, final top-1 by +0.157 +/- 0.214 pp, and forgetting by -0.248 +/- 0.389 pp.
R6 changes the same metrics by +0.050 +/- 0.148, +0.310 +/- 0.471, and
-0.244 +/- 0.397 pp, respectively. These effects are small relative to their
between-seed variability. With only three seeds, they do not establish
statistical significance or quantum advantage.

## Evidence and protocol

### Result packages

- Seed 1993 package: `C:\Users\Asus\Downloads\qbridge_imageneta_results.zip`
  - SHA-256: `35d12b0fe7c686682bd8a75acb772b1c094209118a166dfbfe112313cf462c11`
  - Variants: R0--R10.
- Seeds 2015/2016 package:
  `C:\Users\Asus\Downloads\qbridge_imageneta_results (1).zip`
  - SHA-256: `40825c53397c7435ad7faaab354f102f843d6da31445276fe4cd9fb5fbd29a8c`
  - Variants: R0, R3, R5, and R6.
- Historical comparator:
  `logs/adapter/imageneta/0/20/all_1993_pretrained_vit_b16_224_in21k_adapter.log`.

All 19 packaged run manifests are `completed`, end at task 9, and no run is
listed under `skipped_incomplete`. Repository fixture tests are recorded as
passed, but these tests are not benchmark evidence.

### Shared protocol

- Dataset: ImageNet-A, 200 classes, 20 initial classes, increments of 20; 10
  tasks total.
- Data split: deterministic per-class 80/20 split with seed 67, producing 6,000
  train and 1,500 test images.
- Model/training: ViT-B/16 adapter, 20 base epochs, 15 incremental epochs, 10
  classifier-alignment epochs, batch size 48, and SGD.
- Pretrained model SHA-256 in both packages:
  `27807c6d60f64fe471761985e35fe5256fb4e1ab840277724f8ed2e67876fa3a`.
- Within each seed, R0/R3/R5/R6 share the same base-checkpoint hash, class order,
  and protocol hash. Different seeds correctly have different base checkpoints,
  class orders, and protocol hashes.

| Seed | Base top-1 | Shared base-checkpoint SHA-256 | Protocol SHA-256 |
|---:|---:|---|---|
| 1993 | 87.71 | `a7df36ac17bbb1ec5e92683fce3e6ceb085d524121329e306d99c64207729e87` | `b2239cded4e3b09169babc18aaf801621836aed263d4766dad54d32ccedb34fd` |
| 2015 | 85.71 | `4f95781aacdd31a727b3da5f2278a2a0cccbcbe5254112ef9cf820825c861d2a` | `eeb95a3e7d1cafb4a1637331584220590256f2f5d2b961fb506be34e3ebb6365` |
| 2016 | 88.18 | `18c6355f12af9a88ee5728d56d9b0a500b398dbc417e20094787316c8b869e6c` | `eaab03916125d258fd1b66f10f1397af53cbdf20cd526ecf57374f96bee03fc9` |

Mean base top-1 is 87.200 +/- 1.312 pp. Both packages mark the source dataset
tree as unhashed. Matching split metadata and counts therefore strongly align
the runs, but do not cryptographically prove that the underlying image trees
are identical. Both experiment-plan files also retain the stale field
`benchmark_status: not_started`; completion is instead established from the run
manifests and measured summaries.

## Variant definitions

| Run | Incremental mechanism | Available seeds |
|---|---|---|
| R0 | Legacy QKSR `old_proj` separation; Q-BRIDGE disabled | 1993, 2015, 2016 |
| R1 | All-seen CosFace, alignment retained, no legacy separation | 1993 only |
| R2 | Uniform cross-task margin; guard off | 1993 only |
| R3 | Uniform cross-task margin plus old-anchor guard | 1993, 2015, 2016 |
| R4 | Cosine risk weighting | 1993 only |
| R5 | Raw-PQK risk weighting; full MVP | 1993, 2015, 2016 |
| R6 | Monotone calibrated-PQK risk weighting; full v1 | 1993, 2015, 2016 |
| R7 | Projected-RBF classical risk selector | 1993 only |
| R8 | Small-MLP classical risk selector | 1993 only |
| R9 | Active-pair shuffled raw-PQK risk | 1993 only |
| R10 | Active-pair mean-constant raw-PQK risk | 1993 only |

R7 and R8 are conditional selector controls initialized from the learned base
PQK projector, not independently matched joint-from-scratch base-training
controls.

## Per-seed results for the shared variants

All values are percentage points. Lower forgetting is better.

| Seed | Run | Base | AIA | Final | Final old | Final new | Forgetting |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1993 | R0 | 87.71 | 75.828 | 67.27 | 67.27 | 67.19 | 9.903 |
| 1993 | R3 | 87.71 | 75.832 | 67.00 | 66.98 | 67.19 | 10.341 |
| 1993 | R5 | 87.71 | 75.693 | 67.27 | 67.27 | 67.19 | 9.880 |
| 1993 | R6 | 87.71 | 75.762 | 67.13 | 67.06 | 67.97 | 10.023 |
| 2015 | R0 | 85.71 | 73.234 | 64.67 | 63.96 | 71.33 | 12.531 |
| 2015 | R3 | 85.71 | 73.239 | 65.20 | 64.55 | 71.33 | 12.031 |
| 2015 | R5 | 85.71 | 73.275 | 65.07 | 64.48 | 70.63 | 11.833 |
| 2015 | R6 | 85.71 | 73.450 | 65.47 | 64.92 | 70.63 | 11.864 |
| 2016 | R0 | 88.18 | 75.018 | 67.40 | 66.91 | 73.28 | 10.287 |
| 2016 | R3 | 88.18 | 74.960 | 67.20 | 66.69 | 73.28 | 10.340 |
| 2016 | R5 | 88.18 | 75.045 | 67.47 | 66.91 | 74.14 | 10.263 |
| 2016 | R6 | 88.18 | 75.017 | 67.67 | 67.05 | 75.00 | 10.101 |

R5 reduces forgetting relative to R0 in all three seeds, with paired deltas of
-0.023, -0.698, and -0.023 pp. Its final-accuracy deltas are 0.00, +0.40, and
+0.07 pp, while its AIA deltas have mixed signs. R6 improves final accuracy in
two of three seeds and forgetting in two of three seeds; it has mixed AIA
effects. The seed-level pattern is encouraging for final performance, but is
not a consistent win across all metrics.

## Three-seed paired contrasts

Entries are paired mean delta +/- sample standard deviation. Positive AIA and
final deltas are better; negative forgetting deltas are better.

| Contrast | Delta AIA | Delta final | Delta forgetting |
|---|---:|---:|---:|
| R3 - R0 | -0.016 +/- 0.036 | +0.020 +/- 0.443 | -0.003 +/- 0.471 |
| R5 - R0 | -0.022 +/- 0.098 | +0.157 +/- 0.214 | -0.248 +/- 0.389 |
| R6 - R0 | +0.050 +/- 0.148 | +0.310 +/- 0.471 | -0.244 +/- 0.397 |
| R5 - R3 | -0.006 +/- 0.118 | +0.137 +/- 0.231 | -0.245 +/- 0.197 |
| R6 - R5 | +0.072 +/- 0.102 | +0.153 +/- 0.273 | +0.004 +/- 0.155 |

R3 is effectively neutral relative to R0. R5 and R6 improve mean final accuracy
and forgetting, but the gains are only a few tenths of a point and their paired
standard deviations are comparable to or larger than the mean effects.
Calibration (R6 versus R5) raises mean AIA and final accuracy slightly but
leaves mean forgetting essentially unchanged.

## New-seed accuracy trajectories

| Seed/run | T0 | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2015/R0 | 85.71 | 76.74 | 77.31 | 76.44 | 73.70 | 71.20 | 69.34 | 69.14 | 68.09 | 64.67 |
| 2015/R3 | 85.71 | 77.41 | 77.73 | 75.98 | 73.08 | 71.31 | 69.43 | 68.89 | 67.65 | 65.20 |
| 2015/R5 | 85.71 | 76.74 | 77.31 | 76.29 | 73.45 | 71.52 | 69.34 | 69.38 | 67.94 | 65.07 |
| 2015/R6 | 85.71 | 77.08 | 77.52 | 76.29 | 73.82 | 71.41 | 69.81 | 69.30 | 68.09 | 65.47 |
| 2016/R0 | 88.18 | 83.85 | 78.97 | 74.66 | 73.32 | 72.46 | 72.72 | 70.27 | 68.35 | 67.40 |
| 2016/R3 | 88.18 | 83.85 | 78.97 | 74.50 | 73.05 | 72.46 | 73.00 | 70.11 | 68.28 | 67.20 |
| 2016/R5 | 88.18 | 83.85 | 79.40 | 74.50 | 73.45 | 72.46 | 72.81 | 70.19 | 68.14 | 67.47 |
| 2016/R6 | 88.18 | 83.85 | 78.97 | 74.66 | 73.32 | 72.35 | 73.00 | 70.11 | 68.06 | 67.67 |

The curves cross repeatedly; no Q-BRIDGE variant remains above R0 throughout
the sequence. The aggregate gains arise from small task-specific differences,
not from uniform dominance.

## Seed-1993 extended ablation

Only seed 1993 contains R1/R2/R4/R7/R8/R9/R10. These rows must not be compared
as three-seed estimates.

| Run | AIA | Delta AIA vs R0 | Final | Delta final | Final old | Final new | Forgetting | Delta forgetting |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | 75.828 | 0.000 | **67.27** | 0.00 | **67.27** | 67.19 | 9.903 | 0.000 |
| R1 | 74.237 | -1.591 | 63.80 | -3.47 | 62.76 | **75.00** | 17.049 | +7.146 |
| R2 | 75.751 | -0.077 | **67.27** | 0.00 | **67.27** | 67.19 | 9.823 | -0.080 |
| R3 | **75.832** | +0.004 | 67.00 | -0.27 | 66.98 | 67.19 | 10.341 | +0.438 |
| R4 | 75.716 | -0.112 | 67.00 | -0.27 | 66.98 | 67.19 | 10.454 | +0.551 |
| R5 | 75.693 | -0.135 | **67.27** | 0.00 | **67.27** | 67.19 | 9.880 | -0.023 |
| R6 | 75.762 | -0.066 | 67.13 | -0.14 | 67.06 | 67.97 | 10.023 | +0.120 |
| R7 | 75.709 | -0.119 | 67.13 | -0.14 | 67.13 | 67.19 | **9.767** | -0.137 |
| R8 | 75.805 | -0.023 | 67.13 | -0.14 | 67.13 | 67.19 | 10.030 | +0.127 |
| R9 | 75.757 | -0.071 | 67.13 | -0.14 | 67.13 | 67.19 | 10.219 | +0.316 |
| R10 | 75.827 | -0.001 | 66.93 | -0.34 | 66.91 | 67.19 | 10.349 | +0.446 |

The classical projected-RBF control R7 has the lowest forgetting on seed 1993.
Because the classical controls were not repeated for 2015/2016 and were not
jointly matched from base training, the current study cannot isolate a
quantum-specific benefit.

## Historical RSIAT comparison: seed 1993 only

The historical RSIAT log is available only for seed 1993. It records base
top-1 86.86, AIA 74.891, final top-1 66.23, final-old 66.02, and final-new
68.55. Its dataset tree, split, base checkpoint, and protocol are not hashed, so
this is an aligned historical comparison rather than a fully paired control.

| Contrast | Delta base | Delta AIA | Delta final | Delta final old | Delta final new |
|---|---:|---:|---:|---:|---:|
| R0 - RSIAT | +0.85 | +0.937 | +1.04 | +1.25 | -1.36 |
| R5 - RSIAT | +0.85 | +0.802 | +1.04 | +1.25 | -1.36 |
| R6 - RSIAT | +0.85 | +0.871 | +0.90 | +1.04 | -0.58 |

The base advantage is already comparable in magnitude to the apparent final
improvement. These historical deltas must not be attributed causally to
Q-BRIDGE.

## Calibrator diagnostics

R6 calibrators were fitted using base-train-only features. The table reports
held-out pair evaluation; the positive rate varies substantially with the
seed/class order.

| Seed | Pairs | Positive rate | Raw AUROC | Raw AUPRC | Calibrated NLL | Constant NLL | Calibrated Brier | Constant Brier | Calibrated ECE | Constant ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1993 | 2,983 | 0.536% | 0.8268 | 0.0320 | 0.0383 | **0.0336** | 0.00545 | **0.00534** | 0.01095 | **0.00181** |
| 2015 | 2,527 | 2.058% | 0.5384 | 0.0231 | **0.1005** | 0.1058 | **0.02016** | 0.02029 | **0.00331** | 0.01158 |
| 2016 | 1,938 | 0.671% | 0.5947 | 0.0110 | 0.0559 | **0.0542** | 0.00734 | **0.00723** | 0.02601 | **0.02383** |

Calibration beats the constant-prevalence baseline on NLL/Brier/ECE only for
seed 2015. Raw ranking quality also varies widely: AUROC falls from 0.8268 on
1993 to 0.5384 on 2015 and 0.5947 on 2016, with low AUPRC under strong class
imbalance. Although all three artifacts have `gate_passed: true`, that internal
gate does not establish stable calibration quality or downstream superiority.

## Evidence gates and conclusion

| Gate | Status | Consequence |
|---|---|---|
| R0/R3/R5/R6 complete on three artifact seeds | **Pass** | Three-seed descriptive paired summaries are available |
| Shared base checkpoint/protocol within each seed | **Pass** | Within-seed paired contrasts are supported |
| Requested seed 2026 present | **Fail** | Artifact contains 2016; 2026 cannot be reported |
| Exact pairing with historical RSIAT | **Fail** | Historical deltas cannot isolate method effects |
| At least three paired seeds | **Pass** | Mean and sample SD can be reported |
| Adequate power/significance analysis | **Not established** | Three seeds are insufficient for a strong significance claim |
| Classical controls repeated on all seeds | **Fail** | No multi-seed quantum-specific comparison |
| Q-BRIDGE uniformly beats R0 across metrics/seeds | **Fail** | No dominance claim is supported |
| Quantum advantage | **Fail** | Required matched controls and resource evidence are incomplete |

The updated evidence is more favorable than the seed-1993 result alone: R5
reduces forgetting relative to R0 in all three measured seeds, and R6 improves
mean final accuracy by 0.310 pp. However, the effects remain small, vary by
metric and seed, and are not backed by multi-seed classical controls. The
defensible conclusion is a **promising but unconfirmed incremental improvement**,
not a validated Q-BRIDGE or quantum advantage claim.

Minimum next evidence: resolve the 2016-versus-2026 seed mismatch; run the
intended seed if necessary; repeat R7/R8 and preferably no-CNOT/cosine controls
on every seed with matched base-training and tuning budgets; preserve dataset
tree hashes; and report paired confidence intervals plus compute/resource cost.
