# Kết quả: QKSR tách lớp trên feature hiện tại (`qksr_cur`) so với QKSR cũ (`qksr_old`)

Thiết kế và quy tắc đánh giá được đặt trước trong `exps/develop/README.md`. ImageNet-R
(Kaggle `zaphat206/imagenet-r`), seed 1993, 3 replicate mỗi family, RTX PRO 6000
Blackwell, 3 run song song. Baseline RSIAT là `noise` trong `exps/measurement/results/`.

| Run | AIA | Final | Forgetting |
|---|---:|---:|---:|
| `qksr_old` rep 0 / 1 / 2 | 87.423 / 87.451 / 87.516 | 82.69 / 82.74 / 82.77 | 4.97 / 4.94 / 4.81 |
| `qksr_cur` rep 0 / 1 / 2 | 87.422 / 87.413 / 87.503 | 82.70 / 82.67 / 82.80 | 4.96 / 5.01 / 4.73 |
| RSIAT `noise` (4 run) | 87.347 ± 0.144 | 82.59 ± 0.23 | 5.25 ± 0.19 |

## So sánh ghép cặp (n = 3)

| So sánh | Δ AIA | CI 95% | Δ final | Δ forgetting |
|---|---:|---|---:|---:|
| `qksr_cur` − `qksr_old` | −0.017 | [−0.064, +0.030] | −0.01 | −0.01 |
| `qksr_cur` − RSIAT | +0.039 | [−0.105, +0.184] | +0.05 | −0.32 |
| `qksr_old` − RSIAT | +0.057 | [−0.069, +0.182] | +0.06 | −0.31 |

Cả ba so sánh đều cho kết quả "không kết luận". Tuy vậy, CI rất hẹp, nên có thể nói
mạnh hơn "không thấy hiệu ứng":

- Hiệu ứng của việc đổi `q_inc_pair` lên AIA nằm trong khoảng ±0.065.
- Hiệu ứng của QKSR so với RSIAT nằm dưới +0.19 AIA.

## Cơ chế

- **Lớp cũ nhất:** độ chính xác task 0 là 81.6 (`cur`) và 81.5 (`old`), so với 80.1 của
  RSIAT. Task 1 khoảng 80.6 ở cả ba. Phần tăng nhỏ ở task 0 đến từ base QKSR (task 0
  đạt 96.33 so với 96.05), không đến từ thay đổi ở các task tăng dần.
- **Nhầm giữa các task:** tỉ số trên mỗi cặp lớp là 1.36 (`cur`), 1.38 (`old`), 1.38
  (RSIAT); mô hình huấn luyện chung là 0.90. Chồng lấn giữa các task không giảm.
- **Loss tách lớp không được tối ưu.** `Losses_rt` cuối mỗi task gần như trùng nhau
  giữa `cur` và `old`: task 1 là 1.744 so với 1.744, task 9 là 1.715 so với 1.738.
  Thành phần PQK chiếm khoảng 0.45 trong đó (RSIAT là 1.18–1.28). Kernel trung bình
  khoảng 0.5–0.6, tức PQK 8 qubit đóng băng thấy các cặp prototype/feature đều "khá
  giống nhau". Dù gradient đã đi vào adapter, adapter **không làm giảm** độ tương đồng
  PQK. Giải thích hợp lý là alignment (β = 3, ép z ≈ old_ae(z⁻)) cùng với phép chiếu
  768 → 8 chiều qua tanh khiến áp lực tách lớp quá yếu so với các loss khác.

## Đối chiếu quy tắc đặt trước

Kết quả rơi vào **nhánh B** (không kết luận) với dấu hiệu "loss tách lớp không tác
động". Quy tắc cho phép **đúng một vòng nữa** với áp lực tách lớp mạnh hơn. Nếu vòng
đó vẫn không có hiệu ứng, hoặc gây hại, thì dừng hướng dùng PQK làm metric tách lớp
và chuyển sang bước 3: bài báo theo hướng phân tích và đo lường.
