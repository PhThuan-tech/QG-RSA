# SCF ablation (mean over 3 seeds)

Kiểm tra: SCF tính lại từ score đã lưu lệch tối đa 0.27 điểm so với curve trong log (score lưu float16).

## A_B (final accuracy)

| Variant | CIFAR-100 | ImageNet-R | ImageNet-A | CUB-200 | Mean |
|---|---:|---:|---:|---:|---:|
| Linear head (RSIAT) | 92.03 | 82.55 | 67.02 | 88.47 | 82.52 |
| Ridge alone | 91.41 | 80.02 | 65.71 | 87.40 | 81.14 |
| PGM alone | 91.32 | 80.11 | 65.06 | 85.90 | 80.60 |
| Linear + ridge, no κ (κ=1, w=1) | 92.11 | 82.63 | 67.38 | 88.52 | 82.66 |
| Linear + PGM, no κ (κ=1, w=1) | 92.24 | 82.98 | 67.49 | 88.56 | 82.82 |
| SCF ridge, w=0.25 (frozen) | 92.41 | 83.03 | 67.69 | 88.85 | 83.00 |
| SCF ridge, w=1 | 92.11 | 82.09 | 67.71 | 88.51 | 82.61 |
| SCF ridge, w=4 | 91.69 | 80.93 | 66.51 | 87.90 | 81.76 |
| SCF PGM, w=0.5 (frozen) | 92.34 | 83.07 | 67.65 | 88.57 | 82.91 |
| SCF PGM, w=1 | 92.34 | 82.96 | 67.67 | 88.55 | 82.88 |
| SCF PGM, w=4 | 91.90 | 81.83 | 66.62 | 87.77 | 82.03 |

## Ā (average incremental accuracy)

| Variant | CIFAR-100 | ImageNet-R | ImageNet-A | CUB-200 | Mean |
|---|---:|---:|---:|---:|---:|
| Linear head (RSIAT) | 95.02 | 87.11 | 74.26 | 93.53 | 87.48 |
| Ridge alone | 94.82 | 85.56 | 73.18 | 92.84 | 86.60 |
| PGM alone | 94.71 | 85.63 | 73.52 | 92.03 | 86.47 |
| Linear + ridge, no κ (κ=1, w=1) | 95.06 | 87.17 | 74.54 | 93.55 | 87.58 |
| Linear + PGM, no κ (κ=1, w=1) | 95.14 | 87.38 | 74.74 | 93.50 | 87.69 |
| SCF ridge, w=0.25 (frozen) | 95.22 | 87.42 | 75.12 | 93.54 | 87.82 |
| SCF ridge, w=1 | 95.16 | 87.01 | 75.04 | 93.39 | 87.65 |
| SCF ridge, w=4 | 94.98 | 86.19 | 74.01 | 93.07 | 87.06 |
| SCF PGM, w=0.5 (frozen) | 95.23 | 87.50 | 74.89 | 93.50 | 87.78 |
| SCF PGM, w=1 | 95.26 | 87.48 | 75.08 | 93.43 | 87.81 |
| SCF PGM, w=4 | 95.05 | 86.82 | 74.62 | 93.03 | 87.38 |
