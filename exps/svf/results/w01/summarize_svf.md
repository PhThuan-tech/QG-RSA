# SVF: fusion view plastic + snapshot task 0

Trọng số chọn trên run tuning ImageNet-R 1993: `fuse_snap_ridge_b0p25`, `fuse3_snap_b0p25`

## Từng run

| Dataset | Seed | Phương pháp | AIA | Final | Forgetting ↓ | Task 0 | Task 1 | Pair ratio |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ImageNet-R | 1993 (tuning) | RSIAT | 87.25 | 82.42 | 5.39 | 79.8 | 79.6 | 1.44 |
| ImageNet-R | 1993 (tuning) | SCF ridge | 87.66 | 82.64 | 5.77 | 80.5 | 79.9 | 1.38 |
| ImageNet-R | 1993 (tuning) | snapshot alone | 83.58 | 75.72 | 5.64 | 87.6 | 77.7 | 1.03 |
| ImageNet-R | 1993 (tuning) | SVF | 87.62 | 82.83 | 4.84 | 85.0 | 81.0 | 1.25 |
| ImageNet-R | 1993 (tuning) | SVF + SCF | 87.63 | 82.75 | 5.29 | 85.0 | 80.7 | 1.14 |

## Luật dừng sàng lọc (run tuning, lạc quan)

- Δ final SVF+SCF − SCF ridge tại `fuse3_snap_b0p25`: 0.11 (ngưỡng +0.3) → **DỪNG, không chạy thêm seed**
- Theo trọng số: fuse3_snap_b0p25 0.11, fuse3_snap_b0p5 -0.05, fuse3_snap_b1 -0.99, fuse3_snap_b2 -2.42, fuse3_snap_b4 -4.21

## Quyết định (quy tắc đặt trước, chỉ run held-out)

- Δ final SVF − RSIAT:   (n=0); ImageNet-R  (n=0), ImageNet-A  (n=0)
- Δ final SVF+SCF − SCF ridge:   (n=0)
- **Rule S** (gain ≥ 1 với CI > 0, và phần thêm so với SCF có CI > 0): **chưa đủ run (0/5)**

**Kiểm tra hồi quy (head linear = CNN curve):** ImageNet-R 1993: =

