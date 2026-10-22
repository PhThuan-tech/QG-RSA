# QSD-CIL giai đoạn A: so sánh trong cùng run

## imageneta | qsd | seed 1993 | 3 run

| Head | AIA | Final | Task 0 | Task 1 | Dự đoán task (%) | Tỉ số nhầm khác/trong task |
|---|---:|---:|---:|---:|---:|---:|
| linear | 75.86 | 67.62 | 66.67 | 65.08 | 70.49 | 1.088 |
| linear_rerun | 75.98 | 67.60 | 67.04 | 64.90 | 70.40 | 1.117 |
| linear_noscale | 75.95 | 67.29 | 65.18 | 64.20 | 70.04 | 1.148 |
| ridge_stats | 75.12 | 66.29 | 64.06 | 62.96 | 69.58 | 0.977 |
| pgm_stale | 74.59 | 64.76 | 62.57 | 60.85 | 67.98 | 1.050 |
| pgm_channel | 74.67 | 64.91 | 63.13 | 61.73 | 67.98 | 1.103 |
| pgm_channel_hier | 74.89 | 66.53 | 64.06 | 61.20 | 70.49 | 0.788 |
| pgm_fuse | 76.21 | 67.91 | 66.29 | 65.26 | 70.89 | 1.032 |
| pgm_nonlinear | 68.97 | 59.78 | 56.42 | 56.79 | 63.98 | 0.909 |

### Phương pháp (so trong cùng run)

| Head | n | Δ AIA vs linear | CI 95% | Δ AIA vs ridge_stats | CI 95% | Δ final | Δ task 0 | Kết luận |
|---|---:|---:|---|---:|---|---:|---:|---|
| pgm_channel | 3 | -1.188 | [-1.830, -0.545] | -0.451 | [-0.581, -0.321] | -2.713 | -3.537 | KÉM HƠN linear |
| pgm_channel_hier | 3 | -0.973 | [-2.027, 0.081] | -0.237 | [-0.750, 0.276] | -1.090 | -2.603 | không kết luận |
| pgm_fuse | 3 | 0.345 | [0.190, 0.499] | 1.081 | [0.531, 1.631] | 0.287 | -0.373 | ĐẠT: thắng linear (CI > 0, Δ ≥ 0.3) và thắng ridge_stats |

### Các khẳng định thành phần (Δ AIA)

| So sánh | Nội dung | n | Δ | CI 95% | Giữ được? |
|---|---|---:|---:|---|---|
| pgm_channel − pgm_stale | Kênh vận chuyển có tác dụng | 3 | 0.085 | [-0.138, 0.308] | không |
| pgm_nonlinear − pgm_channel | Mã hoá phi tuyến (hai bản sao) có lợi | 3 | -5.704 | [-6.072, -5.336] | không |
| linear_noscale − linear_rerun | Bỏ heuristic co mean của CA có lợi | 3 | -0.029 | [-0.299, 0.240] | không |
| ridge_stats − linear | Ridge giải tích trên thống kê có lợi | 3 | -0.736 | [-1.430, -0.042] | không |
| linear_rerun − linear | Nhiễu lấy mẫu của chính CA (thông tin) | 3 | 0.117 | [-0.057, 0.292] |  |

**Kiểm tra hồi quy:** seed 1993 rep 0: head linear = CNN curve; trùng baseline: diag_legacy; seed 1993 rep 1: head linear = CNN curve; trùng baseline: diag_legacy; seed 1993 rep 2: head linear = CNN curve; trùng baseline: diag_legacy

## imageneta | qsd | seed 2015 | 1 run

| Head | AIA | Final | Task 0 | Task 1 | Dự đoán task (%) | Tỉ số nhầm khác/trong task |
|---|---:|---:|---:|---:|---:|---:|
| linear | 73.62 | 65.87 | 60.39 | 48.98 | 68.93 | 1.069 |
| linear_rerun | 73.64 | 65.80 | 60.39 | 49.66 | 68.67 | 1.154 |
| linear_noscale | 73.36 | 65.53 | 59.74 | 48.30 | 68.47 | 1.135 |
| ridge_stats | 72.61 | 65.20 | 55.84 | 46.94 | 68.53 | 0.996 |
| pgm_stale | 72.77 | 64.27 | 53.25 | 42.18 | 67.73 | 0.982 |
| pgm_channel | 72.96 | 64.33 | 56.49 | 44.22 | 68.20 | 0.868 |
| pgm_channel_hier | 73.12 | 66.20 | 62.99 | 48.98 | 69.80 | 0.885 |
| pgm_fuse | 74.12 | 66.40 | 61.69 | 48.30 | 69.33 | 1.104 |
| pgm_nonlinear | 65.83 | 58.93 | 49.35 | 42.86 | 62.93 | 0.978 |

### Phương pháp (so trong cùng run)

| Head | n | Δ AIA vs linear | CI 95% | Δ AIA vs ridge_stats | CI 95% | Δ final | Δ task 0 | Kết luận |
|---|---:|---:|---|---:|---|---:|---:|---|
| pgm_channel | 1 | -0.665 |  | 0.348 |  | -1.540 | -3.900 | chưa đủ run (cần ≥ 3) |
| pgm_channel_hier | 1 | -0.502 |  | 0.511 |  | 0.330 | 2.600 | chưa đủ run (cần ≥ 3) |
| pgm_fuse | 1 | 0.501 |  | 1.514 |  | 0.530 | 1.300 | chưa đủ run (cần ≥ 3) |

### Các khẳng định thành phần (Δ AIA)

| So sánh | Nội dung | n | Δ | CI 95% | Giữ được? |
|---|---|---:|---:|---|---|
| pgm_channel − pgm_stale | Kênh vận chuyển có tác dụng | 1 | 0.184 |  | không |
| pgm_nonlinear − pgm_channel | Mã hoá phi tuyến (hai bản sao) có lợi | 1 | -7.131 |  | không |
| linear_noscale − linear_rerun | Bỏ heuristic co mean của CA có lợi | 1 | -0.283 |  | không |
| ridge_stats − linear | Ridge giải tích trên thống kê có lợi | 1 | -1.013 |  | không |
| linear_rerun − linear | Nhiễu lấy mẫu của chính CA (thông tin) | 1 | 0.022 |  |  |

**Kiểm tra hồi quy:** seed 2015 rep 0: head linear = CNN curve; trùng baseline: không có

## imageneta | qsd | seed 2026 | 1 run

| Head | AIA | Final | Task 0 | Task 1 | Dự đoán task (%) | Tỉ số nhầm khác/trong task |
|---|---:|---:|---:|---:|---:|---:|
| linear | 73.63 | 68.13 | 62.59 | 73.17 | 69.93 | 1.763 |
| linear_rerun | 73.76 | 68.33 | 64.03 | 72.56 | 70.13 | 1.751 |
| linear_noscale | 73.61 | 68.33 | 62.59 | 70.73 | 70.27 | 1.623 |
| ridge_stats | 71.89 | 65.93 | 55.40 | 61.59 | 68.93 | 1.093 |
| pgm_stale | 73.03 | 65.00 | 53.24 | 57.32 | 67.67 | 1.280 |
| pgm_channel | 72.99 | 65.47 | 58.99 | 59.76 | 68.33 | 1.166 |
| pgm_channel_hier | 73.96 | 66.73 | 64.03 | 65.24 | 69.27 | 1.281 |
| pgm_fuse | 74.27 | 68.67 | 62.59 | 73.17 | 70.73 | 1.495 |
| pgm_nonlinear | 64.76 | 58.87 | 43.88 | 55.49 | 62.27 | 1.171 |

### Phương pháp (so trong cùng run)

| Head | n | Δ AIA vs linear | CI 95% | Δ AIA vs ridge_stats | CI 95% | Δ final | Δ task 0 | Kết luận |
|---|---:|---:|---|---:|---|---:|---:|---|
| pgm_channel | 1 | -0.634 |  | 1.108 |  | -2.660 | -3.600 | chưa đủ run (cần ≥ 3) |
| pgm_channel_hier | 1 | 0.333 |  | 2.075 |  | -1.400 | 1.440 | chưa đủ run (cần ≥ 3) |
| pgm_fuse | 1 | 0.642 |  | 2.384 |  | 0.540 | 0.000 | chưa đủ run (cần ≥ 3) |

### Các khẳng định thành phần (Δ AIA)

| So sánh | Nội dung | n | Δ | CI 95% | Giữ được? |
|---|---|---:|---:|---|---|
| pgm_channel − pgm_stale | Kênh vận chuyển có tác dụng | 1 | -0.033 |  | không |
| pgm_nonlinear − pgm_channel | Mã hoá phi tuyến (hai bản sao) có lợi | 1 | -8.234 |  | không |
| linear_noscale − linear_rerun | Bỏ heuristic co mean của CA có lợi | 1 | -0.144 |  | không |
| ridge_stats − linear | Ridge giải tích trên thống kê có lợi | 1 | -1.742 |  | không |
| linear_rerun − linear | Nhiễu lấy mẫu của chính CA (thông tin) | 1 | 0.129 |  |  |

**Kiểm tra hồi quy:** seed 2026 rep 0: head linear = CNN curve; trùng baseline: không có

## imagenetr | qsd | seed 1993 | 3 run

| Head | AIA | Final | Task 0 | Task 1 | Dự đoán task (%) | Tỉ số nhầm khác/trong task |
|---|---:|---:|---:|---:|---:|---:|
| linear | 87.41 | 82.67 | 80.08 | 80.63 | 83.89 | 1.391 |
| linear_rerun | 87.37 | 82.66 | 80.13 | 80.32 | 83.89 | 1.387 |
| linear_noscale | 87.35 | 82.59 | 79.61 | 79.91 | 83.82 | 1.400 |
| ridge_stats | 86.21 | 80.03 | 78.15 | 77.74 | 81.94 | 1.000 |
| pgm_stale | 86.42 | 79.77 | 80.60 | 76.56 | 81.76 | 0.968 |
| pgm_channel | 86.38 | 79.90 | 79.43 | 78.47 | 81.89 | 0.961 |
| pgm_channel_hier | 86.88 | 81.68 | 82.06 | 79.85 | 83.47 | 0.974 |
| pgm_fuse | 87.62 | 82.97 | 81.07 | 80.32 | 84.21 | 1.354 |
| pgm_nonlinear | 82.63 | 75.78 | 74.86 | 72.38 | 77.97 | 1.059 |

### Phương pháp (so trong cùng run)

| Head | n | Δ AIA vs linear | CI 95% | Δ AIA vs ridge_stats | CI 95% | Δ final | Δ task 0 | Kết luận |
|---|---:|---:|---|---:|---|---:|---:|---|
| pgm_channel | 3 | -1.028 | [-1.151, -0.906] | 0.169 | [0.054, 0.285] | -2.767 | -0.657 | KÉM HƠN linear |
| pgm_channel_hier | 3 | -0.530 | [-0.828, -0.233] | 0.667 | [0.389, 0.945] | -0.987 | 1.977 | KÉM HƠN linear |
| pgm_fuse | 3 | 0.216 | [0.168, 0.263] | 1.413 | [1.340, 1.487] | 0.300 | 0.987 | không kết luận |

### Các khẳng định thành phần (Δ AIA)

| So sánh | Nội dung | n | Δ | CI 95% | Giữ được? |
|---|---|---:|---:|---|---|
| pgm_channel − pgm_stale | Kênh vận chuyển có tác dụng | 3 | -0.037 | [-0.122, 0.048] | không |
| pgm_nonlinear − pgm_channel | Mã hoá phi tuyến (hai bản sao) có lợi | 3 | -3.747 | [-3.791, -3.702] | không |
| linear_noscale − linear_rerun | Bỏ heuristic co mean của CA có lợi | 3 | -0.018 | [-0.038, 0.002] | không |
| ridge_stats − linear | Ridge giải tích trên thống kê có lợi | 3 | -1.198 | [-1.227, -1.168] | không |
| linear_rerun − linear | Nhiễu lấy mẫu của chính CA (thông tin) | 3 | -0.035 | [-0.104, 0.034] |  |

**Kiểm tra hồi quy:** seed 1993 rep 0: head linear = CNN curve; trùng baseline: noise; seed 1993 rep 1: head linear = CNN curve; trùng baseline: noise; seed 1993 rep 2: head linear = CNN curve; trùng baseline: noise

## imagenetr | qsd | seed 2015 | 1 run

| Head | AIA | Final | Task 0 | Task 1 | Dự đoán task (%) | Tỉ số nhầm khác/trong task |
|---|---:|---:|---:|---:|---:|---:|
| linear | 87.45 | 82.59 | 78.12 | 79.11 | 83.99 | 1.210 |
| linear_rerun | 87.48 | 82.74 | 78.12 | 79.11 | 84.15 | 1.183 |
| linear_noscale | 87.44 | 82.70 | 77.68 | 78.90 | 84.07 | 1.232 |
| ridge_stats | 85.77 | 80.45 | 75.80 | 79.11 | 82.36 | 0.976 |
| pgm_stale | 85.69 | 80.37 | 78.99 | 74.65 | 82.23 | 1.010 |
| pgm_channel | 85.49 | 80.17 | 76.81 | 76.06 | 82.14 | 0.955 |
| pgm_channel_hier | 86.68 | 82.24 | 80.29 | 80.73 | 83.87 | 1.046 |
| pgm_fuse | 87.72 | 83.15 | 78.12 | 80.12 | 84.54 | 1.167 |
| pgm_nonlinear | 82.07 | 75.90 | 70.43 | 73.83 | 78.41 | 0.906 |

### Phương pháp (so trong cùng run)

| Head | n | Δ AIA vs linear | CI 95% | Δ AIA vs ridge_stats | CI 95% | Δ final | Δ task 0 | Kết luận |
|---|---:|---:|---|---:|---|---:|---:|---|
| pgm_channel | 1 | -1.955 |  | -0.271 |  | -2.420 | -1.310 | chưa đủ run (cần ≥ 3) |
| pgm_channel_hier | 1 | -0.768 |  | 0.916 |  | -0.350 | 2.170 | chưa đủ run (cần ≥ 3) |
| pgm_fuse | 1 | 0.273 |  | 1.957 |  | 0.560 | 0.000 | chưa đủ run (cần ≥ 3) |

### Các khẳng định thành phần (Δ AIA)

| So sánh | Nội dung | n | Δ | CI 95% | Giữ được? |
|---|---|---:|---:|---|---|
| pgm_channel − pgm_stale | Kênh vận chuyển có tác dụng | 1 | -0.200 |  | không |
| pgm_nonlinear − pgm_channel | Mã hoá phi tuyến (hai bản sao) có lợi | 1 | -3.428 |  | không |
| linear_noscale − linear_rerun | Bỏ heuristic co mean của CA có lợi | 1 | -0.045 |  | không |
| ridge_stats − linear | Ridge giải tích trên thống kê có lợi | 1 | -1.684 |  | không |
| linear_rerun − linear | Nhiễu lấy mẫu của chính CA (thông tin) | 1 | 0.031 |  |  |

**Kiểm tra hồi quy:** seed 2015 rep 0: head linear = CNN curve; trùng baseline: không có

## imagenetr | qsd | seed 2026 | 1 run

| Head | AIA | Final | Task 0 | Task 1 | Dự đoán task (%) | Tỉ số nhầm khác/trong task |
|---|---:|---:|---:|---:|---:|---:|
| linear | 86.55 | 82.57 | 77.97 | 77.37 | 83.79 | 1.406 |
| linear_rerun | 86.57 | 82.52 | 77.97 | 77.21 | 83.72 | 1.432 |
| linear_noscale | 86.53 | 82.41 | 77.80 | 76.57 | 83.61 | 1.442 |
| ridge_stats | 84.81 | 79.53 | 76.78 | 68.54 | 81.08 | 1.292 |
| pgm_stale | 84.46 | 79.07 | 78.81 | 67.26 | 80.65 | 1.294 |
| pgm_channel | 84.63 | 79.37 | 80.85 | 68.38 | 80.96 | 1.260 |
| pgm_channel_hier | 85.71 | 81.39 | 82.71 | 73.84 | 83.00 | 1.114 |
| pgm_fuse | 86.86 | 82.97 | 78.64 | 77.21 | 84.20 | 1.353 |
| pgm_nonlinear | 81.14 | 75.70 | 76.27 | 68.22 | 77.77 | 1.133 |

### Phương pháp (so trong cùng run)

| Head | n | Δ AIA vs linear | CI 95% | Δ AIA vs ridge_stats | CI 95% | Δ final | Δ task 0 | Kết luận |
|---|---:|---:|---|---:|---|---:|---:|---|
| pgm_channel | 1 | -1.921 |  | -0.187 |  | -3.200 | 2.880 | chưa đủ run (cần ≥ 3) |
| pgm_channel_hier | 1 | -0.842 |  | 0.892 |  | -1.180 | 4.740 | chưa đủ run (cần ≥ 3) |
| pgm_fuse | 1 | 0.315 |  | 2.049 |  | 0.400 | 0.670 | chưa đủ run (cần ≥ 3) |

### Các khẳng định thành phần (Δ AIA)

| So sánh | Nội dung | n | Δ | CI 95% | Giữ được? |
|---|---|---:|---:|---|---|
| pgm_channel − pgm_stale | Kênh vận chuyển có tác dụng | 1 | 0.168 |  | không |
| pgm_nonlinear − pgm_channel | Mã hoá phi tuyến (hai bản sao) có lợi | 1 | -3.488 |  | không |
| linear_noscale − linear_rerun | Bỏ heuristic co mean của CA có lợi | 1 | -0.044 |  | không |
| ridge_stats − linear | Ridge giải tích trên thống kê có lợi | 1 | -1.734 |  | không |
| linear_rerun − linear | Nhiễu lấy mẫu của chính CA (thông tin) | 1 | 0.021 |  |  |

**Kiểm tra hồi quy:** seed 2026 rep 0: head linear = CNN curve; trùng baseline: không có

## imageneta | qsd | gộp mọi seed | 5 run (xác nhận)

| Head | AIA | Final | Task 0 | Task 1 | Dự đoán task (%) | Tỉ số nhầm khác/trong task |
|---|---:|---:|---:|---:|---:|---:|
| linear | 74.97 | 67.37 | 64.60 | 63.48 | 70.07 | 1.219 |
| linear_rerun | 75.07 | 67.39 | 65.11 | 63.39 | 70.00 | 1.251 |
| linear_noscale | 74.96 | 67.15 | 63.57 | 62.32 | 69.77 | 1.240 |
| ridge_stats | 73.97 | 66.00 | 60.68 | 59.48 | 69.24 | 1.004 |
| pgm_stale | 73.91 | 64.71 | 58.84 | 56.41 | 67.87 | 1.082 |
| pgm_channel | 73.99 | 64.91 | 60.97 | 57.83 | 68.09 | 1.069 |
| pgm_channel_hier | 74.35 | 66.51 | 63.84 | 59.56 | 70.11 | 0.906 |
| pgm_fuse | 75.40 | 67.76 | 64.63 | 63.45 | 70.55 | 1.139 |
| pgm_nonlinear | 67.50 | 59.43 | 52.50 | 53.74 | 63.43 | 0.976 |

### Phương pháp (so trong cùng run)

| Head | n | Δ AIA vs linear | CI 95% | Δ AIA vs ridge_stats | CI 95% | Δ final | Δ task 0 | Kết luận |
|---|---:|---:|---|---:|---|---:|---:|---|
| pgm_channel | 5 | -0.972 | [-1.403, -0.541] | 0.020 | [-0.849, 0.890] | -2.468 | -3.622 | KÉM HƠN linear |
| pgm_channel_hier | 5 | -0.618 | [-1.416, 0.181] | 0.375 | [-0.884, 1.635] | -0.868 | -0.754 | không kết luận |
| pgm_fuse | 5 | 0.435 | [0.260, 0.610] | 1.428 | [0.699, 2.158] | 0.386 | 0.036 | ĐẠT: thắng linear (CI > 0, Δ ≥ 0.3) và thắng ridge_stats |

### Các khẳng định thành phần (Δ AIA)

| So sánh | Nội dung | n | Δ | CI 95% | Giữ được? |
|---|---|---:|---:|---|---|
| pgm_channel − pgm_stale | Kênh vận chuyển có tác dụng | 5 | 0.081 | [-0.042, 0.205] | không |
| pgm_nonlinear − pgm_channel | Mã hoá phi tuyến (hai bản sao) có lợi | 5 | -6.496 | [-7.931, -5.060] | không |
| linear_noscale − linear_rerun | Bỏ heuristic co mean của CA có lợi | 5 | -0.103 | [-0.272, 0.066] | không |
| ridge_stats − linear | Ridge giải tích trên thống kê có lợi | 5 | -0.993 | [-1.587, -0.399] | không |
| linear_rerun − linear | Nhiễu lấy mẫu của chính CA (thông tin) | 5 | 0.101 | [0.018, 0.183] |  |

**Kiểm tra hồi quy:** seed 1993 rep 0: head linear = CNN curve; trùng baseline: diag_legacy; seed 1993 rep 1: head linear = CNN curve; trùng baseline: diag_legacy; seed 1993 rep 2: head linear = CNN curve; trùng baseline: diag_legacy; seed 2015 rep 0: head linear = CNN curve; trùng baseline: không có; seed 2026 rep 0: head linear = CNN curve; trùng baseline: không có

## imagenetr | qsd | gộp mọi seed | 5 run (xác nhận)

| Head | AIA | Final | Task 0 | Task 1 | Dự đoán task (%) | Tỉ số nhầm khác/trong task |
|---|---:|---:|---:|---:|---:|---:|
| linear | 87.24 | 82.63 | 79.27 | 79.67 | 83.89 | 1.358 |
| linear_rerun | 87.23 | 82.65 | 79.30 | 79.46 | 83.91 | 1.355 |
| linear_noscale | 87.20 | 82.58 | 78.86 | 79.04 | 83.83 | 1.375 |
| ridge_stats | 85.84 | 80.02 | 77.41 | 76.18 | 81.85 | 1.054 |
| pgm_stale | 85.88 | 79.75 | 79.92 | 74.32 | 81.63 | 1.042 |
| pgm_channel | 85.85 | 79.85 | 79.19 | 75.97 | 81.76 | 1.020 |
| pgm_channel_hier | 86.60 | 81.74 | 81.84 | 78.83 | 83.46 | 1.016 |
| pgm_fuse | 87.49 | 83.01 | 79.99 | 79.66 | 84.27 | 1.316 |
| pgm_nonlinear | 82.22 | 75.79 | 74.26 | 71.84 | 78.02 | 1.043 |

### Phương pháp (so trong cùng run)

| Head | n | Δ AIA vs linear | CI 95% | Δ AIA vs ridge_stats | CI 95% | Δ final | Δ task 0 | Kết luận |
|---|---:|---:|---|---:|---|---:|---:|---|
| pgm_channel | 5 | -1.392 | [-2.012, -0.772] | 0.010 | [-0.266, 0.286] | -2.784 | -0.080 | KÉM HƠN linear |
| pgm_channel_hier | 5 | -0.640 | [-0.857, -0.423] | 0.762 | [0.573, 0.951] | -0.898 | 2.568 | KÉM HƠN linear |
| pgm_fuse | 5 | 0.247 | [0.188, 0.306] | 1.649 | [1.245, 2.053] | 0.372 | 0.726 | không kết luận |

### Các khẳng định thành phần (Δ AIA)

| So sánh | Nội dung | n | Δ | CI 95% | Giữ được? |
|---|---|---:|---:|---|---|
| pgm_channel − pgm_stale | Kênh vận chuyển có tác dụng | 5 | -0.029 | [-0.194, 0.136] | không |
| pgm_nonlinear − pgm_channel | Mã hoá phi tuyến (hai bản sao) có lợi | 5 | -3.631 | [-3.830, -3.433] | không |
| linear_noscale − linear_rerun | Bỏ heuristic co mean của CA có lợi | 5 | -0.029 | [-0.048, -0.010] | không |
| ridge_stats − linear | Ridge giải tích trên thống kê có lợi | 5 | -1.402 | [-1.751, -1.054] | không |
| linear_rerun − linear | Nhiễu lấy mẫu của chính CA (thông tin) | 5 | -0.011 | [-0.059, 0.038] |  |

**Kiểm tra hồi quy:** seed 1993 rep 0: head linear = CNN curve; trùng baseline: noise; seed 1993 rep 1: head linear = CNN curve; trùng baseline: noise; seed 1993 rep 2: head linear = CNN curve; trùng baseline: noise; seed 2015 rep 0: head linear = CNN curve; trùng baseline: không có; seed 2026 rep 0: head linear = CNN curve; trùng baseline: không có

