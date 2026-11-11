# W07: lượng tử lúc train × SCF (ImageNet-R, seed 1993)

Run hoàn chỉnh: qscf_qksr_rep0, qscf_qksr_rep1, qscf_qksr_rep2, qscf_rsiat_rep0, qscf_rsiat_rep1, qscf_rsiat_rep2

## Bảng 2×2, Ā (mean ± SD, n = số replicate)

| Train \ Head | linear head | SCF ridge | SCF PGM |
|---|---:|---:|---:|
| RSIAT | 87.39 ± 0.08 (n=3) | 87.76 ± 0.07 (n=3) | 87.76 ± 0.10 (n=3) |
| QKSR | 87.41 ± 0.02 (n=3) | 87.85 ± 0.04 (n=3) | 87.87 ± 0.07 (n=3) |

## Bảng 2×2, A_B (mean ± SD, n = số replicate)

| Train \ Head | linear head | SCF ridge | SCF PGM |
|---|---:|---:|---:|
| RSIAT | 82.50 ± 0.24 (n=3) | 82.75 ± 0.09 (n=3) | 82.85 ± 0.23 (n=3) |
| QKSR | 82.64 ± 0.04 (n=3) | 82.82 ± 0.15 (n=3) | 82.91 ± 0.05 (n=3) |

## So sánh ghép cặp (theo replicate)

| So sánh | Metric | Δ | CI 95% | n |
|---|---|---:|---|---:|
| SCF ridge − linear, mạng RSIAT | Ā | 0.375 | [0.295, 0.456] | 3 |
| SCF ridge − linear, mạng QKSR | Ā | 0.443 | [0.364, 0.523] | 3 |
| SCF PGM − linear, mạng RSIAT | Ā | 0.370 | [0.280, 0.460] | 3 |
| SCF PGM − linear, mạng QKSR | Ā | 0.464 | [0.276, 0.652] | 3 |
| QKSR − RSIAT, linear head | Ā | 0.021 | [-0.231, 0.273] | 3 |
| (QKSR + SCF ridge) − (RSIAT + SCF ridge) | Ā | 0.089 | [-0.082, 0.260] | 3 |
| tương tác: gain SCF ridge trên QKSR − trên RSIAT | Ā | 0.068 | [-0.019, 0.155] | 3 |
| (QKSR + SCF PGM) − (RSIAT + SCF PGM) | Ā | 0.115 | [0.010, 0.220] | 3 |
| tương tác: gain SCF PGM trên QKSR − trên RSIAT | Ā | 0.094 | [-0.053, 0.241] | 3 |
| SCF ridge − linear, mạng RSIAT | A_B | 0.253 | [-0.148, 0.655] | 3 |
| SCF ridge − linear, mạng QKSR | A_B | 0.180 | [-0.145, 0.505] | 3 |
| SCF PGM − linear, mạng RSIAT | A_B | 0.350 | [0.212, 0.488] | 3 |
| SCF PGM − linear, mạng QKSR | A_B | 0.270 | [0.053, 0.487] | 3 |
| QKSR − RSIAT, linear head | A_B | 0.140 | [-0.391, 0.671] | 3 |
| (QKSR + SCF ridge) − (RSIAT + SCF ridge) | A_B | 0.067 | [-0.319, 0.453] | 3 |
| tương tác: gain SCF ridge trên QKSR − trên RSIAT | A_B | -0.073 | [-0.359, 0.212] | 3 |
| (QKSR + SCF PGM) − (RSIAT + SCF PGM) | A_B | 0.060 | [-0.591, 0.711] | 3 |
| tương tác: gain SCF PGM trên QKSR − trên RSIAT | A_B | -0.080 | [-0.409, 0.249] | 3 |

## Kiểm định chốt trước

(QKSR + SCF ridge) − (RSIAT + SCF ridge), Ā: 0.089 [-0.082, 0.260] (n=3)

**KHÔNG ĐẠT: QKSR không thêm gì đáng kể khi đã có SCF (CI 95% của ΔĀ chứa 0 hoặc âm)**
