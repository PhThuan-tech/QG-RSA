# QSD-CIL vòng A2

Tuning: imagenetr seed 1993 (3 run). Held-out: 8 run.

## imageneta | 5 run (mọi seed)

| Head | AIA | Final | Task 0 | Dự đoán task (%) | Δ AIA vs linear | CI 95% |
|---|---:|---:|---:|---:|---:|---|
| linear | 74.97 | 67.37 | 64.60 | 70.07 |  |  |
| ridge_stats | 73.97 | 66.00 | 60.68 | 69.24 | -0.993 | [-1.587, -0.399] |
| pgm_channel | 73.99 | 64.91 | 60.97 | 68.09 | -0.972 | [-1.403, -0.541] |
| pgm_channel_hier | 74.35 | 66.51 | 63.84 | 70.11 | -0.618 | [-1.416, 0.181] |
| pgm_fuse | 75.40 | 67.76 | 64.63 | 70.55 | 0.435 | [0.260, 0.610] |
| pgm_stats_stale | 73.86 | 64.49 | 59.36 | 67.75 | -1.109 | [-1.590, -0.628] |
| pgm_stats | 74.01 | 65.12 | 62.20 | 68.30 | -0.954 | [-1.348, -0.559] |
| pgm_stats_hier | 74.02 | 65.75 | 61.89 | 69.37 | -0.947 | [-1.768, -0.127] |
| pgm_stats_oracle | 74.30 | 65.92 | 67.05 | 69.10 | -0.665 | [-1.009, -0.321] |
| fuse_ridge_stats_b0p25 | 75.78 | 67.99 | 63.78 | 70.81 | 0.814 | [0.710, 0.917] |
| fuse_pgm_channel_b0p5 | 75.49 | 67.69 | 64.28 | 70.56 | 0.525 | [0.259, 0.790] |
| fuse_pgm_channel_hier_b1 | 75.54 | 67.91 | 64.44 | 70.76 | 0.576 | [0.085, 1.067] |
| fuse_pgm_stats_b0p5 | 75.49 | 67.83 | 64.50 | 70.72 | 0.525 | [0.210, 0.841] |
| fuse_pgm_stats_hier_b1 | 75.45 | 67.67 | 63.66 | 70.55 | 0.486 | [-0.030, 1.001] |

## imagenetr | 6 run (mọi seed)

| Head | AIA | Final | Task 0 | Dự đoán task (%) | Δ AIA vs linear | CI 95% |
|---|---:|---:|---:|---:|---:|---|
| linear | 87.25 | 82.64 | 79.10 | 83.91 |  |  |
| ridge_stats | 85.79 | 80.05 | 77.92 | 81.87 | -1.466 | [-1.776, -1.156] |
| pgm_channel | 85.81 | 79.87 | 79.57 | 81.72 | -1.444 | [-1.932, -0.956] |
| pgm_channel_hier | 86.58 | 81.74 | 81.78 | 83.44 | -0.679 | [-0.872, -0.487] |
| pgm_fuse | 87.49 | 83.03 | 79.85 | 84.30 | 0.236 | [0.183, 0.289] |
| pgm_stats_stale | 85.81 | 79.75 | 80.26 | 81.57 | -1.446 | [-1.983, -0.908] |
| pgm_stats | 85.92 | 80.11 | 81.93 | 81.91 | -1.336 | [-1.898, -0.773] |
| pgm_stats_hier | 86.52 | 81.30 | 79.88 | 82.97 | -0.732 | [-0.962, -0.502] |
| pgm_stats_oracle | 86.00 | 80.62 | 86.35 | 82.41 | -1.255 | [-1.740, -0.770] |
| fuse_ridge_stats_b0p25 | 87.59 | 83.05 | 80.53 | 84.36 | 0.335 | [0.269, 0.402] |
| fuse_pgm_channel_b0p5 | 87.65 | 83.09 | 80.64 | 84.40 | 0.392 | [0.317, 0.467] |
| fuse_pgm_channel_hier_b1 | 87.71 | 83.26 | 81.47 | 84.61 | 0.459 | [0.325, 0.592] |
| fuse_pgm_stats_b0p5 | 87.63 | 83.10 | 81.23 | 84.41 | 0.377 | [0.294, 0.460] |
| fuse_pgm_stats_hier_b1 | 87.64 | 83.09 | 80.41 | 84.43 | 0.384 | [0.233, 0.535] |

## Chọn trọng số fusion (chỉ trên nhóm tuning)

| Nguồn | w=0.25 | w=0.5 | w=1 | w=2 | w=4 | Chọn |
|---|---:|---:|---:|---:|---:|---|
| ridge_stats | 0.377 | 0.288 | -0.027 | -0.330 | -0.622 | w=0.25 |
| pgm_channel | 0.301 | 0.412 | 0.377 | 0.202 | -0.094 | w=0.5 |
| pgm_channel_hier | 0.260 | 0.401 | 0.448 | 0.339 | 0.125 | w=1 |
| pgm_stats | 0.290 | 0.403 | 0.396 | 0.269 | 0.007 | w=0.5 |
| pgm_stats_hier | 0.260 | 0.370 | 0.378 | 0.259 | 0.028 | w=1 |

## Held-out (trọng số cố định), Δ AIA so với linear

| Nguồn | Head | Tuning | imageneta | imagenetr | Gộp | CI 95% gộp |
|---|---|---:|---:|---:|---:|---|
| ridge_stats | fuse_ridge_stats_b0p25 | 0.377 | 0.814 (n=5) | 0.294 (n=3) | 0.619 | [0.386, 0.851] |
| pgm_channel | fuse_pgm_channel_b0p5 | 0.412 | 0.525 (n=5) | 0.372 (n=3) | 0.467 | [0.310, 0.625] |
| pgm_channel_hier | fuse_pgm_channel_hier_b1 | 0.448 | 0.576 (n=5) | 0.469 (n=3) | 0.536 | [0.267, 0.805] |
| pgm_stats | fuse_pgm_stats_b0p5 | 0.403 | 0.525 (n=5) | 0.350 (n=3) | 0.459 | [0.275, 0.644] |
| pgm_stats_hier | fuse_pgm_stats_hier_b1 | 0.378 | 0.486 (n=5) | 0.389 (n=3) | 0.450 | [0.165, 0.734] |

## Quyết định

- **Phương pháp** (`pgm_stats` fusion, held-out gộp: Δ ≥ 0.3, CI > 0, mọi dataset > 0): **ĐẠT**
- **Đóng góp của phép đo lượng tử** (fusion `pgm_stats` − fusion `ridge_stats`, held-out): Δ -0.159 [-0.392, 0.074] → **không giữ được**
- Trạng thái theo mean SSCA tốt hơn mean cũ (`pgm_stats − pgm_stats_stale`, mọi run, n=11): Δ 0.130 [0.084, 0.177] → có
- Phép đo PGM tốt hơn ridge trên cùng thống kê (`pgm_stats − ridge_stats`, mọi run, n=11): Δ 0.089 [-0.225, 0.403] → không
- Trạng thái từ thống kê so với từ feature từng mẫu (`pgm_stats − pgm_channel`, mọi run, n=11): Δ 0.068 [-0.020, 0.156] → không
- Phân cấp task → lớp có lợi (`pgm_stats_hier − pgm_stats`, mọi run, n=11): Δ 0.332 [0.020, 0.643] → có
- Trần của vận chuyển trạng thái (final, oracle − thống kê) (`pgm_stats_oracle − pgm_stats (final)`, mọi run, n=11): Δ 0.639 [0.476, 0.802] → không

**Kiểm tra hồi quy:** imagenetr seed 1993 rep 0: linear = CNN; head giai đoạn A trùng hết; imagenetr seed 1993 rep 1: linear = CNN; head giai đoạn A trùng hết; imagenetr seed 1993 rep 2: linear = CNN; head giai đoạn A trùng hết; imageneta seed 1993 rep 0: linear = CNN; head giai đoạn A trùng hết; imageneta seed 1993 rep 1: linear = CNN; head giai đoạn A trùng hết; imageneta seed 1993 rep 2: linear = CNN; head giai đoạn A trùng hết; imagenetr seed 2042 rep 0: linear = CNN; không có run giai đoạn A; imageneta seed 2015 rep 0: linear = CNN; head giai đoạn A trùng hết; imageneta seed 2026 rep 0: linear = CNN; head giai đoạn A trùng hết; imagenetr seed 2015 rep 0: linear = CNN; head giai đoạn A trùng hết; imagenetr seed 2026 rep 0: linear = CNN; head giai đoạn A trùng hết

