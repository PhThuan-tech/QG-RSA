# SCF trên giao thức 3 seed (1993, 2015, 2026)

Trọng số fusion khoá từ vòng A2: ridge 0.25, PGM 0.5, PGM phân cấp 1.

## Bảng chính (mean ± SD trên seed)

| Dataset | n | Phương pháp | AIA | Final | Forgetting ↓ |
|---|---:|---|---:|---:|---:|
| ImageNet-R | 3 | RSIAT | 87.11 ± 0.49 | 82.55 ± 0.05 | 5.74 ± 0.26 |
| ImageNet-R | 3 | SCF ridge | 87.42 ± 0.46 | 83.03 ± 0.24 | 5.97 ± 0.11 |
| ImageNet-R | 3 | SCF PGM | 87.50 ± 0.43 | 83.07 ± 0.42 | 5.79 ± 0.37 |
| ImageNet-R | 3 | SCF PGM hier | 87.56 ± 0.32 | 83.27 ± 0.31 | 5.88 ± 0.40 |
| ImageNet-R | 3 | ridge alone | 85.56 ± 0.67 | 80.02 ± 0.46 | 8.19 ± 0.63 |
| ImageNet-R | 3 | PGM alone | 85.63 ± 0.95 | 80.11 ± 0.54 | 7.32 ± 0.70 |
| ImageNet-A | 3 | RSIAT | 74.26 ± 1.11 | 67.02 ± 1.13 | 10.44 ± 0.99 |
| ImageNet-A | 3 | SCF ridge | 75.12 ± 1.20 | 67.69 ± 0.91 | 11.46 ± 1.13 |
| ImageNet-A | 3 | SCF PGM | 74.89 ± 1.01 | 67.65 ± 1.34 | 10.94 ± 1.09 |
| ImageNet-A | 3 | SCF PGM hier | 74.94 ± 1.10 | 67.40 ± 1.07 | 11.39 ± 0.81 |
| ImageNet-A | 3 | ridge alone | 73.18 ± 1.66 | 65.71 ± 0.44 | 12.44 ± 0.70 |
| ImageNet-A | 3 | PGM alone | 73.52 ± 1.00 | 65.06 ± 0.64 | 12.81 ± 1.61 |
| CIFAR-100 | 3 | RSIAT | 95.02 ± 0.21 | 92.03 ± 0.19 | 3.39 ± 0.21 |
| CIFAR-100 | 3 | SCF ridge | 95.22 ± 0.20 | 92.41 ± 0.21 | 3.26 ± 0.28 |
| CIFAR-100 | 3 | SCF PGM | 95.23 ± 0.22 | 92.34 ± 0.16 | 3.40 ± 0.28 |
| CIFAR-100 | 3 | SCF PGM hier | 95.30 ± 0.27 | 92.44 ± 0.21 | 3.46 ± 0.32 |
| CIFAR-100 | 3 | ridge alone | 94.82 ± 0.26 | 91.41 ± 0.23 | 4.11 ± 0.41 |
| CIFAR-100 | 3 | PGM alone | 94.71 ± 0.28 | 91.32 ± 0.15 | 4.08 ± 0.33 |
| CUB-200 | 3 | RSIAT | 93.53 ± 0.79 | 88.47 ± 0.31 | 6.77 ± 0.18 |
| CUB-200 | 3 | SCF ridge | 93.54 ± 0.88 | 88.85 ± 0.22 | 6.20 ± 0.19 |
| CUB-200 | 3 | SCF PGM | 93.50 ± 0.84 | 88.57 ± 0.30 | 6.60 ± 0.17 |
| CUB-200 | 3 | SCF PGM hier | 93.43 ± 0.89 | 88.43 ± 0.39 | 6.89 ± 0.38 |
| CUB-200 | 3 | ridge alone | 92.84 ± 0.97 | 87.40 ± 0.06 | 6.81 ± 0.25 |
| CUB-200 | 3 | PGM alone | 92.03 ± 1.23 | 85.90 ± 0.11 | 7.83 ± 0.17 |

## Chênh lệch ghép cặp so với RSIAT (cùng run), CI 95%

| Dataset | Phương pháp | Δ AIA | Δ Final | Δ Forgetting |
|---|---|---|---|---|
| ImageNet-R | SCF ridge | 0.315 [0.124, 0.507] | 0.477 [0.016, 0.938] | 0.228 [-0.124, 0.580] |
| ImageNet-R | SCF PGM | 0.390 [0.251, 0.529] | 0.520 [-0.408, 1.448] | 0.051 [-0.877, 0.980] |
| ImageNet-R | SCF PGM hier | 0.452 [0.034, 0.871] | 0.717 [0.072, 1.362] | 0.136 [-0.854, 1.127] |
| ImageNet-A | SCF ridge | 0.851 [0.622, 1.080] | 0.667 [-0.202, 1.535] | 1.027 [0.282, 1.771] |
| ImageNet-A | SCF PGM | 0.628 [-0.101, 1.356] | 0.623 [-0.077, 1.324] | 0.499 [0.241, 0.756] |
| ImageNet-A | SCF PGM hier | 0.679 [-0.336, 1.693] | 0.377 [-0.036, 0.790] | 0.956 [0.356, 1.557] |
| CIFAR-100 | SCF ridge | 0.199 [0.119, 0.280] | 0.383 [0.075, 0.692] | -0.130 [-0.787, 0.527] |
| CIFAR-100 | SCF PGM | 0.216 [0.152, 0.281] | 0.307 [0.182, 0.432] | 0.011 [-0.423, 0.445] |
| CIFAR-100 | SCF PGM hier | 0.279 [0.114, 0.444] | 0.410 [0.344, 0.476] | 0.067 [-0.419, 0.552] |
| CUB-200 | SCF ridge | 0.018 [-0.238, 0.274] | 0.383 [0.158, 0.609] | -0.566 [-0.786, -0.346] |
| CUB-200 | SCF PGM | -0.028 [-0.176, 0.119] | 0.103 [0.046, 0.161] | -0.166 [-0.450, 0.117] |
| CUB-200 | SCF PGM hier | -0.095 [-0.367, 0.177] | -0.037 [-0.268, 0.194] | 0.115 [-0.533, 0.764] |

## Quyết định (quy tắc đặt trước)

- **Chuyển giao sang dataset mới** `fuse_ridge_stats_b0p25`: gộp 0.109 [-0.018, 0.235] (n=6); CIFAR-100 0.199 (n=3), CUB-200 0.018 (n=3) → **KHÔNG ĐẠT**
- **Chuyển giao sang dataset mới** `fuse_pgm_stats_b0p5`: gộp 0.094 [-0.053, 0.241] (n=6); CIFAR-100 0.216 (n=3), CUB-200 -0.028 (n=3) → **KHÔNG ĐẠT**
- **Đóng góp lượng tử** (SCF PGM − SCF ridge, mọi run, n=12): -0.044 [-0.169, 0.081] → **không giữ được**

**Kiểm tra hồi quy (head linear = CNN curve):** ImageNet-R 1993: =; ImageNet-R 2015: =; ImageNet-R 2026: =; ImageNet-A 1993: =; ImageNet-A 2015: =; ImageNet-A 2026: =; CIFAR-100 1993: =; CIFAR-100 2015: =; CIFAR-100 2026: =; CUB-200 1993: =; CUB-200 2015: =; CUB-200 2026: =

**Còn thiếu:** không

