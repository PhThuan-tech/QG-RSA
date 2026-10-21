# Kết quả QRF-CA trên ImageNet-A (seed 1993, 3 replicate)

Quy tắc và thiết kế được đặt trước trong `../README.md`. Bảng đầy đủ nằm ở `summary.md`,
sinh bằng `scripts/summarize_ca_heads.py`.

**Kiểm tra hồi quy: đạt.** CNN curve của cả 3 run trùng từng chữ số với `diag_legacy`
và `_rep1`, `_rep2` (AIA 75.546 / 75.751 / 76.285). Việc thêm head không thay đổi
huấn luyện, nên các so sánh giữa head là chính xác.

| Head | AIA | Final | Δ AIA so với linear | CI 95% |
|---|---:|---:|---:|---|
| linear (RSIAT) | 75.861 | 67.62 | — | — |
| rp_ridge | 74.623 | 66.34 | −1.24 | [−2.29, −0.19] |
| rff_ridge | 74.447 | 67.13 | −1.41 | [−2.52, −0.31] |
| qda | 68.534 | 60.13 | −7.33 | [−8.61, −6.05] |
| pqk_noent_ridge | 65.457 | 54.78 | −10.40 | [−11.10, −9.71] |
| pqk_ridge | 58.532 | 48.20 | −17.33 | [−19.07, −15.59] |

**Kết luận theo quy tắc:**
- **Không head phi tuyến nào thắng `linear`.** Mọi head đều kém hơn với CI nằm hoàn
  toàn dưới 0. Giả thuyết "head phi tuyến ở bước CA" bị bác bỏ trên ImageNet-A.
- `pqk_ridge` kém mọi đối chứng, nên không có đóng góp lượng tử.

## Cơ chế

1. **Mẫu giả bị suy biến, nên head linh hoạt học thuộc chúng.**
   - ImageNet-A chỉ có 30 ảnh train mỗi lớp, nên covariance 768 chiều của mỗi lớp có
     hạng ≤ 29, cộng thêm 1e-3·I.
   - Mẫu Gaussian vì vậy nằm gần như trên các không gian con 29 chiều riêng của từng
     lớp, và các lớp tách nhau một cách tầm thường.
   - Hệ quả: RFF và RP đạt **100%** trên mẫu giả để riêng nhưng chỉ khoảng 67% trên
     test. Bước chọn λ trên mẫu giả cũng mất tác dụng, vì λ nhỏ nhất luôn thắng.
   - QDA hỏng vì cùng lý do: Mahalanobis với covariance gần suy biến.
   - Head cosine tuyến tính, học bằng SGD 10 epoch với chuẩn hoá cosine, được điều
     chuẩn ngầm nên khái quát hoá tốt hơn.
2. **Feature map PQK làm mất thông tin, và entanglement làm tệ hơn.**
   - Ngay trên mẫu giả để riêng (bài toán dễ), PQK chỉ đạt 78–91%, bỏ CNOT đạt 88–96%,
     trong khi RFF và RP đạt 100%.
   - Mỗi mạch chỉ nhận 8 phép chiếu qua tanh, rồi chỉ đọc RDM cục bộ. Sau các lớp CNOT
     với góc ngẫu nhiên, thông tin lan sang tương quan không cục bộ, và RDM cục bộ trở
     về gần trạng thái hỗn hợp. Đây là dạng concentration được mô tả trong Thanasilp
     et al. 2024.
   - Kernel–target alignment ở task 9: PQK 0.336 < RFF 0.369, RP 0.376, PQK-noent 0.383.
   - Geometric difference g(K_rff ‖ K_pqk) = 14.5: kernel PQK **khác** RFF nhiều,
     nhưng khác theo hướng không chứa thông tin nhãn.
3. **Chi phí:** khoảng 79 giây mỗi task cho cả 6 head trên RTX PRO 6000.
