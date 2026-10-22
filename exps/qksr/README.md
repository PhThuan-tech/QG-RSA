# Phát triển QKSR: tách lớp cũ/mới trên feature hiện tại

## Căn cứ

Kết quả vòng `UPPER` (`exps/upper/README.md`) trên ImageNet-R:

- So với huấn luyện chung, học tăng dần mất 3.9 điểm final. Mất mát dồn vào các lớp
  cũ nhất: task 0 thấp hơn 9.0 điểm, task 1 thấp hơn 8.3 điểm.
- Thống kê lớp cũ chính xác tuyệt đối chỉ lấy lại được +0.5 final. Phần còn lại là chồng
  lấn biểu diễn giữa các task: tính trên mỗi cặp lớp, tỉ số nhầm khác task / trong task
  là 1.38 ở CIL so với 0.90 ở huấn luyện chung.

## Lỗ hổng trong thiết kế cũ

Trong `models/RSIAT_adapter.py::_inc_loss`, cả loss tách lớp gốc của RSIAT (`loss_orth`,
cosine) lẫn QKSR cũ (`q_inc_pair = "old_proj"`) đều so prototype lớp cũ với
`old_ae(features_old)`. Mà `features_old` đến từ **mạng cũ đã đóng băng**, nên gradient
của loss tách lớp chỉ cập nhật `old_ae`, **không bao giờ đến adapter**. Nói cách khác,
loss tách lớp chưa từng định hình biểu diễn, trong khi biểu diễn mới là nút thắt. Đây
là một lý do hợp lý cho việc QKSR và Q-BRIDGE không tạo khác biệt. Q-BRIDGE còn có thêm
một lý do: loss của nó có trọng số nhỏ (λ = 0.1, margin 0.1 trên điểm cosine).

## Ứng viên

| Family | Thay đổi | Vai trò |
|---|---|---|
| `qksr_old` | QKSR base + loss tách lớp cũ (`q_inc_pair = old_proj`) | Tham chiếu: QKSR như các run trước |
| `qksr_cur` | Như trên, nhưng `q_inc_pair = current`: so `old_ae(μ_c)` với feature **hiện tại** `z` | Ý tưởng mới: gradient của kernel PQK (đóng băng) đi vào adapter |

Hai family chỉ khác nhau đúng một điểm là `q_inc_pair`. Các thiết lập còn lại giống
nhau: `inc_loss_mode = mean`, trọng số `gamma = 1.5` như cấu hình RSIAT, PQK 8 qubit
được huấn luyện ở task 0 rồi đóng băng. Mỗi family có 3 replicate (`rng_replicate`
0, 1, 2), seed 1993, ImageNet-R.

Kiểm tra trên CPU: với PQK đóng băng, `relu(k(old_ae(μ), z) − m)` vẫn truyền gradient
về `z` (norm khoảng 1.5e-2 với dữ liệu ngẫu nhiên), còn tham số của PQK không được cập
nhật.

## Quy tắc đánh giá (đặt trước khi chạy)

So sánh ghép cặp theo replicate bằng `scripts/run_level_stats.py compare`, với n = 3:

1. **Ý tưởng mới:** so `qksr_cur` với `qksr_old`. Hai family dùng chung base task 0,
   nên ghép cặp rất chặt.
   - Đạt: cận dưới CI 95% của Δ AIA lớn hơn 0.
   - Bổ sung: so độ chính xác cuối của task 0–1, vì đây là nơi giả thuyết dự đoán sẽ
     cải thiện.
2. **So với RSIAT:** so `qksr_cur` với baseline `noise`. Base task 0 khác nhau (QKSR so
   với cosine), nên ghép cặp lỏng hơn. Đây chính là kết luận mà bài báo cần.
3. **Kiểm chứng lại kết quả cũ:** so `qksr_old` với `noise`, tức câu hỏi "QKSR có vượt
   RSIAT không", được kiểm theo protocol ghép cặp.

Nếu `qksr_cur` đạt ở bước 1, thì trước khi viết rằng đóng góp đến từ thành phần lượng
tử, cần một đối chứng cổ điển: loss tách lớp bằng cosine trên feature hiện tại (chưa
cài đặt), cùng bước xác nhận với seed 2015/2026 và ImageNet-A, theo
`exps/measurement/README.md`.

## Lệnh so sánh (ở máy local, khi đã có kết quả)

```bash
python scripts/run_level_stats.py compare --root exps/measurement/results/logs \
  --root exps/qksr/results/logs --baseline qksr_old --candidate qksr_cur
python scripts/run_level_stats.py compare --root exps/measurement/results/logs \
  --root exps/qksr/results/logs --baseline noise --candidate qksr_cur
```
