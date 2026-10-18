# Q-BRIDGE versus original RSIAT on ImageNet-A

## Evidence used

- Original RSIAT log: `logs/adapter/imageneta/0/20/all_1993_pretrained_vit_b16_224_in21k_adapter.log`
- Q-BRIDGE package: `qbridge_imageneta_results.zip`
- Dataset: ImageNet-A, 200 classes, 20 initial classes, increments of 20
- Seed and class-order seed: 1993
- Shared training settings recorded in both runs: ViT-B/16 adapter, 20 base epochs,
  15 incremental epochs, 10 classifier-alignment epochs, batch size 48, and the
  same optimizer and learning-rate settings.

The Q-BRIDGE package records a deterministic 80/20 per-class split with seed 67
(6,000 train and 1,500 test images). The historical RSIAT log records the same
class order and training settings but does not record a dataset-tree or split
hash. The comparison is therefore strongly aligned, but it is not a fully
hash-verified paired experiment.

## Results

| Run | Meaning | Base top-1 | AIA | Final top-1 | Final old | Final new | Forgetting |
|---|---|---:|---:|---:|---:|---:|---:|
| RSIAT | Original repository log | 86.86 | 74.891 | 66.23 | 66.02 | 68.55 | unavailable |
| R0 | Legacy QKSR control on the Q-BRIDGE base | 87.71 | 75.828 | 67.27 | 67.27 | 67.19 | 9.903 |
| R5 | Raw-PQK Q-BRIDGE pilot | 87.71 | 75.693 | 67.27 | 67.27 | 67.19 | 9.880 |
| R6 | Full calibrated Q-BRIDGE | 87.71 | 75.762 | 67.13 | 67.06 | 67.97 | 10.023 |

Compared with the original RSIAT log, R6 changes AIA by **+0.871 percentage
points** and final top-1 by **+0.90 points**. Its final old-class accuracy rises
by **+1.04 points**, while new-class accuracy falls by **-0.58 points**. R5
changes AIA by **+0.802 points** and final top-1 by **+1.04 points**, with
**+1.25 points** on old classes and **-1.36 points** on new classes.

## Interpretation

Q-BRIDGE R5/R6 are numerically above the historical RSIAT log for AIA and final
top-1. This does not isolate a Q-BRIDGE gain because their shared task-0 QKSR
checkpoint already scores 87.71 versus RSIAT's 86.86. Under the controlled
Q-BRIDGE protocol, the R0 legacy-QKSR control remains stronger than full R6:
R6 is -0.066 points in AIA, -0.14 points in final top-1, and +0.120 points in
forgetting relative to R0. R5 ties R0 on final top-1 but is -0.135 points in AIA;
its forgetting improvement is only 0.023 points.

The defensible conclusion from this single seed is: the QKSR/Q-BRIDGE pipeline
is about 0.8--1.0 points above the available historical RSIAT log, but the new
Q-BRIDGE mechanism has not beaten its matched legacy-QKSR control. Additional
seeds and a fresh RSIAT run on the exact hashed split are required before making
a contribution or significance claim.
