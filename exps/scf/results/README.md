# SCF vòng A3: kết quả (2026-10-22)

- Đặc tả: `../README.md`.
- Bảng đầy đủ: `summary.md` và `summary.json`, sinh bằng `scripts/summarize_scf.py`. Bảng gộp sẵn ImageNet-R/A
  của vòng A2, replicate 0, seed 1993/2015/2026.
- Run mới:
  - `kaggle1`: CIFAR-100 seed 1993, 2015.
  - `kaggle2`: CIFAR-100 seed 2026 và CUB-200 seed 1993, 2015, 2026.

Cả 6 run mới đều chạy xong, không có lỗi. CUB dùng split chính thức (5 794 ảnh test, 200 lớp).

**Kiểm tra hồi quy: đạt.** Ở cả 12 run của bảng 3 seed, `linear` trùng CNN curve.

## Bảng 4 dataset × 3 seed (mean ± SD; trọng số khoá từ A2: ridge 0.25, PGM 0.5)

| Dataset (ảnh/lớp) | Phương pháp | AIA | Final | Forgetting ↓ |
|---|---|---:|---:|---:|
| ImageNet-R (~120) | RSIAT | 87.11 ± 0.49 | 82.55 ± 0.05 | 5.74 ± 0.26 |
| | SCF ridge | 87.42 ± 0.46 | 83.03 ± 0.24 | 5.97 ± 0.11 |
| | SCF PGM | 87.50 ± 0.43 | 83.07 ± 0.42 | 5.79 ± 0.37 |
| ImageNet-A (~30) | RSIAT | 74.26 ± 1.11 | 67.02 ± 1.13 | 10.44 ± 0.99 |
| | SCF ridge | 75.12 ± 1.20 | 67.69 ± 0.91 | 11.46 ± 1.13 |
| | SCF PGM | 74.89 ± 1.01 | 67.65 ± 1.34 | 10.94 ± 1.09 |
| CIFAR-100 (500) | RSIAT | 95.02 ± 0.21 | 92.03 ± 0.19 | 3.39 ± 0.21 |
| | SCF ridge | 95.22 ± 0.20 | 92.41 ± 0.21 | 3.26 ± 0.28 |
| | SCF PGM | 95.23 ± 0.22 | 92.34 ± 0.16 | 3.40 ± 0.28 |
| CUB-200 (~30) | RSIAT | 93.53 ± 0.79 | 88.47 ± 0.31 | 6.77 ± 0.18 |
| | SCF ridge | 93.54 ± 0.88 | 88.85 ± 0.22 | 6.20 ± 0.19 |
| | SCF PGM | 93.50 ± 0.84 | 88.57 ± 0.30 | 6.60 ± 0.17 |

Chênh lệch ghép cặp so với RSIAT, n = 3, CI 95%:

| Dataset | Δ AIA ridge | Δ AIA PGM | Δ Final ridge | Δ Forgetting ridge |
|---|---|---|---|---|
| ImageNet-R | +0.32 [+0.12, +0.51] | +0.39 [+0.25, +0.53] | +0.48 [+0.02, +0.94] | +0.23 [−0.12, +0.58] |
| ImageNet-A | +0.85 [+0.62, +1.08] | +0.63 [−0.10, +1.36] | +0.67 [−0.20, +1.54] | +1.03 [+0.28, +1.77] |
| CIFAR-100 | +0.20 [+0.12, +0.28] | +0.22 [+0.15, +0.28] | +0.38 [+0.08, +0.69] | −0.13 [−0.79, +0.53] |
| CUB-200 | +0.02 [−0.24, +0.27] | −0.03 [−0.18, +0.12] | **+0.38 [+0.16, +0.61]** | **−0.57 [−0.79, −0.35]** |

## Quyết định theo quy tắc đặt trước

| Quy tắc | Kết quả | Kết luận |
|---|---|---|
| **T.** Chuyển giao, SCF ridge | 6 run mới: +0.11 [−0.02, +0.24]; CIFAR +0.20, CUB +0.02 | **KHÔNG ĐẠT**: CI chạm 0 |
| **T.** Chuyển giao, SCF PGM | +0.09 [−0.05, +0.24]; CIFAR +0.22, CUB −0.03 | **KHÔNG ĐẠT** |
| **Q.** SCF PGM − SCF ridge (12 run) | −0.04 [−0.17, +0.08] | Không có đóng góp lượng tử, xác nhận lại A2 |
| **D.** Số ảnh mỗi lớp (mô tả) | CUB (~30 ảnh/lớp) gần 0; CIFAR (500 ảnh/lớp) +0.20 | **Giả thuyết bị bác**: gain không do số ảnh/lớp |

## Diễn giải

1. **Gain trên AIA phụ thuộc dataset.**
   - Gain dương và chắc chắn trên 3/4 dataset: ImageNet-R, ImageNet-A, CIFAR. Riêng CIFAR có CI > 0 dù gain nhỏ.
   - Trên CUB, gain về AIA bằng 0.
   - Quy tắc T đặt ra cho "cả hai dataset mới" nên không đạt. Kết luận đúng là: **SCF không phải cải tiến phổ quát cho AIA.**
2. **Final accuracy tăng ở cả 4 dataset**, từ +0.38 đến +0.67 với SCF ridge. CI > 0 ở 3/4 dataset; ImageNet-A có
   CI chứa 0 với n=3.
   - Đây là kết quả mô tả, không phải quy tắc đặt trước. Dù vậy, nó nhất quán.
   - Trên CUB, gain chỉ xuất hiện ở 3 giai đoạn cuối (+0.22, +0.31, +0.38). Ở các giai đoạn giữa, Δ dao động
     quanh 0 (−0.16 đến +0.12). Vì AIA lấy trung bình 20 giai đoạn, gain cuối bị pha loãng.
3. **Không phải do trọng số chuyển giao kém.** Trên CUB, không trọng số nào trong lưới cho AIA dương đáng kể:
   - ridge: tốt nhất +0.02, ở w = 0.25;
   - PGM: tốt nhất −0.01.

   Nghĩa là thống kê lớp của CUB không bổ sung thêm cho head CA, chứ không phải trọng số được chọn sai.
4. **Forgetting không có hướng chung.** Forgetting tăng trên ImageNet-A, giảm trên CUB (−0.57 với ridge) và gần
   như không đổi trên R và CIFAR. Chỉ số này đi theo việc gain dồn vào lớp mới hay lớp cũ ở từng dataset.
5. **Lượng tử:** kết luận của A2 được xác nhận lại trên 12 run và 4 dataset. SCF PGM không hơn SCF ridge.

## Hệ quả cho paper

- Claim chính phải **thu hẹp**: SCF cải thiện AIA trên ImageNet-R, ImageNet-A và CIFAR-100, nhưng không trên CUB-200.
  Final accuracy tăng ở cả 4 dataset.
- Phần hạn chế phải nêu kết quả CUB và việc giả thuyết "số ảnh/lớp" bị bác.
- Bảng chính chuyển sang quy ước 4 dataset × 3 seed.
