# W07: lượng tử lúc train × SCF, kết quả

Wave `W07_qscf`, Kaggle, 10/11/2026 (base `qgrsa-base-c7bc4fd`). ImageNet-R B0Inc20, seed 1993, replicate 0–2.
Đủ 6/6 run, console không có lỗi. Bảng đầy đủ: [summary.md](summary.md); log: `kaggle1/logs/`
(thư mục `_stats/`, khoảng 184 MB, chỉ có ở máy local).

## Bảng 2×2 (Ā / A_B, mean trên 3 replicate)

| Train \ Head | linear head | SCF ridge | SCF PGM |
|---|---:|---:|---:|
| RSIAT | 87.39 / 82.50 | 87.76 / 82.75 | 87.76 / 82.85 |
| QKSR | 87.41 / 82.64 | 87.85 / 82.82 | 87.87 / 82.91 |

## Kết luận

- **Kiểm định chốt trước: KHÔNG ĐẠT.** (QKSR + SCF ridge) − (RSIAT + SCF ridge), Ā = +0.09 [−0.08, +0.26].
- **SCF tăng trên cả hai mạng:** Ā +0.38 [0.30, 0.46] trên RSIAT, +0.44 [0.36, 0.52] trên QKSR.
- **QKSR không có tác dụng:** QKSR − RSIAT với linear head là +0.02 [−0.23, +0.27]. Kết quả này khớp với QKSR-v2 (+0.05).
- **Tương tác không có ý nghĩa thống kê:** +0.07 [−0.02, +0.16]. SCF cộng thêm được trên mạng train bằng lượng tử, nhưng lượng tử không làm SCF tốt hơn.
- **Không được dùng làm kết quả chính:**
  - Dòng PGM, (QKSR + SCF PGM) − (RSIAT + SCF PGM) = +0.12 [0.01, 0.22], không phải kiểm định chốt trước.
  - A_B: mọi so sánh QKSR − RSIAT đều có CI chứa 0.
- **Kiểm tra độ tin cậy:** RSIAT linear Ā = 87.39 ± 0.08, nằm trong dải nhiễu đã đo trước đây (87.35 ± 0.14).

**Dùng trong paper:** Bảng `tab:qscf`, mục 7.3 (C3). Lượng tử lúc train chỉ đổi lớp cũ lấy lớp mới; phần tăng đến từ fusion lúc dự đoán.
