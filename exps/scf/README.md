# SCF vòng A3: kiểm định chuyển giao trên CIFAR-100 và CUB-200 (giao thức 3 seed)

Đặc tả này được viết **trước khi chạy**, ngày 2026-10-22. Quy tắc bên dưới là cố định. Mọi thay đổi sau khi
thấy kết quả phải ghi vào mục "Nhật ký thay đổi".

- Phương pháp và kết quả trên ImageNet-R/A: `exps/qsd/README_A2.md`, `exps/qsd/results/a2/README.md`.
- Quy ước seed: `exps/README.md`.

## 1. Mục tiêu

SCF (statistics–classifier fusion) đã vượt RSIAT trên mọi run của ImageNet-R và ImageNet-A. Vòng này trả lời
ba câu hỏi:

1. **Chuyển giao:** với trọng số đã **khoá** từ vòng A2, SCF có vượt RSIAT trên hai dataset hoàn toàn mới không?
2. **Giả thuyết số ảnh mỗi lớp:**
   - Gain lớn nhất ở ImageNet-A, nơi có 30 ảnh/lớp và covariance có hạng ≤ 29.
   - CUB-200 có khoảng 30 ảnh/lớp, CIFAR-100 có 500 ảnh/lớp.
   - Dự đoán: gain trên CUB lớn hơn trên CIFAR.
3. **Lượng tử:** trên cả 4 dataset, SCF PGM có hơn SCF ridge không?

## 2. Thiết lập (không chỉnh gì)

- **Config RSIAT gốc:** `exps/rsiat/cifar224.json` (B0Inc10, 10 task) và `exps/rsiat/cub.json` (B0Inc10, **20 task**).
- **Dữ liệu:**
  - CIFAR-100: `cifar-100-python`, chuẩn 50 000 / 10 000 ảnh.
  - CUB-200-2011: **split chính thức** 5 994 / 5 794, dựng từ `train_test_split.txt` (`data.data.prepare_cub_split`).
- **Overlay `scf.json`:**
  - Head: `linear`, `ridge_stats`, `pgm_stats`, `pgm_stats_hier`, cùng lưới fusion của A2 và lưu điểm số.
  - Bỏ `pgm_stats_oracle` (quá tốn trên CIFAR) và các head per-sample.
  - So với `qsd2_rep0`, overlay chỉ khác các khoá `qsd_*`, `prefix` và `offline`. Huấn luyện vì vậy **giống hệt**
    RSIAT gốc với cùng seed.
- **Seed:** 1993, 2015, 2026, với replicate 0.

**Phương pháp được kiểm định, trọng số khoá từ A2, không chọn lại:**

| Tên | Head |
|---|---|
| SCF ridge | `fuse_ridge_stats_b0p25` |
| SCF PGM | `fuse_pgm_stats_b0p5` |
| SCF PGM phân cấp (phụ) | `fuse_pgm_stats_hier_b1` |

Các trọng số khác trong lưới chỉ được báo cáo thăm dò, không được dùng để chọn.

## 3. Quy tắc quyết định (cài trong `scripts/summarize_scf.py`)

1. **T. Chuyển giao.** Xét riêng SCF ridge và SCF PGM, mỗi phương pháp ĐẠT nếu thoả đủ hai điều kiện:
   - trên 6 run mới (CIFAR ×3, CUB ×3), Δ AIA so với RSIAT có cận dưới CI 95% > 0;
   - mean Δ > 0 trên **từng** dataset.
2. **Q. Lượng tử.** `SCF PGM − SCF ridge` trên mọi run của bảng 3 seed (4 dataset × 3 seed = 12) có CI > 0.
   Nếu không, giữ kết luận của A2: phép đo lượng tử không đóng góp.
3. **D. Số ảnh mỗi lớp (mô tả, không quyết định).** So sánh gain trên CUB và ImageNet-A (khoảng 30 ảnh/lớp)
   với gain trên CIFAR (500 ảnh/lớp).
4. **Hồi quy.** Ở mọi run, head `linear` trùng CNN curve.
5. **Báo cáo bắt buộc:** AIA, Final và Forgetting, kể cả khi forgetting tăng (như ở A2).

**Hệ quả:**

| Kết quả | Bước tiếp |
|---|---|
| T đạt | Bảng chính của paper gồm 4 dataset × 3 seed; SCF là đóng góp chính |
| T đạt trên CUB, không đạt trên CIFAR | Viết rõ phạm vi: SCF giúp khi thống kê lớp ước lượng từ ít ảnh. Đây là phát hiện có giá trị, không giấu |
| T không đạt | Thu hẹp claim về ImageNet-R/A và nêu trong phần hạn chế |

## 4. Kế hoạch chạy

Notebook `QGRSA_EXPERIMENTS.ipynb`, cell 1: chỉ đổi `PLAN`.

| `PLAN` | Run | Input cần thêm | Thời gian ước tính |
|---|---|---|---|
| `kaggle1` | CIFAR-100 seed 1993, 2015 | CIFAR-100 (thư mục `cifar-100-python`) | ~4 giờ (chưa đo) |
| `kaggle2` | CIFAR-100 seed 2026 + CUB-200 seed 1993, 2015, 2026 | CIFAR-100 và CUB-200-2011 (bản gốc có `train_test_split.txt`) | ~4 giờ |

CIFAR có 50 000 ảnh train, gấp khoảng 2 lần ImageNet-R tính theo số mẫu × epoch, nên thời gian chỉ là ước
lượng. Nếu một run vượt giới hạn 12 giờ của Kaggle, notebook bỏ qua run đã xong khi chạy lại.

## 5. Tổng hợp ở máy local

Giải nén kết quả vào `exps/scf/results/kaggle{1,2}/`, rồi chạy (kèm A2 để có bảng 4 dataset):

```bash
python scripts/summarize_scf.py \
  --root exps/scf/results/kaggle1/logs --root exps/scf/results/kaggle2/logs \
  --root exps/qsd/results/a2/kaggle1/logs --root exps/qsd/results/a2/kaggle2/logs --root exps/qsd/results/a2/kaggle3/logs \
  --output exps/scf/results/summary.md --json exps/scf/results/summary.json
```

## Nhật ký thay đổi

- (trống)
