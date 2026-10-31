# QKSR-v2 P0: kết quả đo nguyên nhân (W04, ImageNet-R 1993)

Nguồn: `kaggle/waves/w04_qksr_p0` (base `c7bc4fd` + patch), chạy ngày 2026-10-31. RTX PRO 6000 Blackwell, torch 2.11.0+cu128.

File gốc nằm trong `p0/`. `console/` và `_stats/` chỉ giữ ở máy. Bảng do `scripts/summarize_qksr_probe.py` sinh ra.

Spec: `exps/qksr2/README.md` mục P0. Probe chỉ đọc: test trùng từng bit đã đạt, và `p0_rsiat` trùng `hp_diag` của W03 từng số.

## Tái lập

| Run | Đường accuracy | Ghi chú |
|---|---|---|
| `p0_rsiat` | **trùng** `hp_diag` (W03), AIA 87.246 | probe không đổi huấn luyện, môi trường Kaggle ổn định |
| `p0_qksr_cur` | **khác** `qksr_cur` rep 0 cũ (lệch tối đa 0.28, AIA 87.402 so với 87.422) | do môi trường: task 0 ra 96.05, so với 96.33 ở môi trường cũ |

Vì vậy P1 phải so `p1_qksr_cur` chạy lại trong cùng môi trường, không dùng 3 run cũ. W05 đã làm như vậy.

## Kết quả theo luật P0.3 (trung vị task ≥ 1)

| Luật | Số đo | Quyết định |
|---|---|---|
| **W6 xung đột** | cos(∇CosFace, ∇sep) = **−0.002**, 50% bước âm | **gần vuông góc: bỏ P4** |
| **W1 nút cổ chai** | AUC PQK **0.59** so với Euclid trên z 768 chiều 0.85 (cosine trên z: 0.95) | **đạt → chạy P2** |
| **W1 hướng** | tán xạ giữa lớp trong row space = **3.9×** mức ngẫu nhiên (ngưỡng 5×) | **đạt → P2** |
| **W2 bão hoà** | **43.5%** góc có \|θ\| > 0.9π | **đạt → scale trước tanh (có trong P2)** |
| W7 tương đương | AUC PQK − RBF(θ) = **−0.128**, CKA = 0.18 | **không tương đương: PQK kém hơn RBF trên cùng góc** |
| **W4 cặp khó** | kernel top-5 0.85 so với toàn bộ 0.56 (+0.30) | **đạt → P3 đáng thử** |

## Theo từng task (`p0_qksr_cur`)

| Task | AUC PQK | AUC RBF(θ) | AUC z | Góc bão hoà | Tán xạ giữa lớp trong row space | Kernel cùng lớp / khác lớp | Kernel loss: mọi cặp / top-5 |
|---|---:|---:|---:|---:|---:|---|---|
| 0 | 0.759 | 0.857 | 0.894 | 0.48 | 0.090 | 0.73 / 0.54 | — |
| 1 | 0.656 | 0.779 | 0.861 | 0.43 | 0.061 | 0.68 / 0.57 | 0.55 / 0.75 |
| 3 | 0.597 | 0.720 | 0.843 | 0.42 | 0.043 | 0.66 / 0.59 | 0.57 / 0.83 |
| 5 | 0.581 | 0.720 | 0.842 | 0.44 | 0.040 | 0.65 / 0.59 | 0.56 / 0.85 |
| 9 | 0.568 | 0.714 | 0.850 | 0.45 | 0.035 | 0.64 / 0.60 | 0.55 / 0.88 |

Gradient trên adapter, epoch cuối của task:

| Task | cos(c, kd) | cos(c, sep) | \|sep\| / \|c\| | \|kd\| / \|c\| |
|---|---:|---:|---:|---:|
| 1 | −0.16 | −0.02 | 0.25 | 0.20 |
| 5 | −0.33 | −0.00 | 0.23 | 0.14 |
| 9 | −0.17 | +0.04 | 0.33 | 0.19 |

## Đọc kết quả

1. **Phép chiếu là gốc vấn đề.**
   - Feature 768 chiều phân biệt lớp tốt (AUC 0.85 Euclid, 0.95 cosine).
   - Qua phép chiếu 768 → 8, chỉ còn 0.72 (RBF trên θ).
   - Qua mạch PQK, còn 0.59, gần mức ngẫu nhiên 0.5.
   - Phép chiếu học ở task 0 rồi đóng băng nên **thoái hoá dần**: tán xạ giữa lớp nằm trong row space giảm từ 9.0× xuống 3.5× mức ngẫu nhiên, AUC PQK giảm từ 0.76 xuống 0.57. Điều này xác nhận W3.
   - Gần một nửa số góc bị tanh bão hoà (W2).
2. **Mạch lượng tử làm mất thêm thông tin.** Trên cùng góc θ, PQK (2 lớp biến phân + vòng CNOT, đọc reduced state 1 qubit) kém RBF 0.13 AUC, và Gram của nó gần như không giống RBF (CKA 0.18). P1 (`p1_qksr_nocnot`) sẽ cho biết phần mất này do CNOT hay do cách đọc reduced state.
3. **Không có xung đột gradient giữa tách lớp và phân loại.** Gradient `sep` gần như vuông góc với gradient CosFace, và chỉ bằng khoảng 25% về độ lớn. Nó đẩy adapter theo hướng *không liên quan* đến phân loại. Điều này phù hợp với việc QKSR không đổi AIA, và cho thấy gradient surgery không có gì để sửa.
4. **Quan sát sau khi xem số, không thuộc luật:** gradient **KD** ngược chiều CosFace ở **100%** bước (cos −0.14 đến −0.33), ở cả QKSR lẫn RSIAT (`p0_rsiat`: −0.15). Nếu đường đánh đổi cũ–mới có nguồn gốc ở gradient, ứng viên là KD chứ không phải loss tách lớp. Kiểm chứng điều này cần một spec mới, ví dụ chiếu gradient KD hoặc quét β, và chưa nằm trong kế hoạch QKSR-v2.
5. **Có cặp khó:** kernel trung bình trên top-5 prototype gần nhất (0.85) cao hơn hẳn trung bình mọi cặp (0.56), và khoảng cách này tăng theo task. Loss lấy trung bình đều nên làm loãng tín hiệu từ đúng những cặp hay nhầm.

## Quyết định cho P1–P5

| Bước | Quyết định |
|---|---|
| P1 (đang chạy, W05) | giữ nguyên. Dự đoán cập nhật: PQK ≤ RBF, không còn là "tương đương" |
| **P2** (W06) | **chạy sau W05**: luật W1 và W2 kích hoạt |
| **P3** (hard-pair margin) | **đáng làm**: luật W4 kích hoạt, cần code |
| P4 (gradient surgery trên sep) | **bỏ** theo luật W6 |
| P5 (16 qubit) | chờ P2 |

# QKSR-v2 P1: PQK so với bản đối chứng cổ điển (W05, ImageNet-R 1993, n = 3)

Nguồn: `kaggle/waves/w05_qksr_p1`, chạy ngày 2026-10-31, cùng môi trường với W04. File gốc nằm trong `p1/`. Bảng do `scripts/summarize_qksr_p1.py` sinh ra (gộp log `p0/` và `p1/`).

| Họ | AIA (3 rep) | Final | Forgetting | Old (task 0–4) | New (task 5–9) | Task 0–1 | Task 0 sau base |
|---|---:|---:|---:|---:|---:|---:|---:|
| RSIAT (`p0_rsiat`, chỉ rep 0) | 87.246 | 82.42 | — | 80.47 | 84.49 | 79.70 | 96.05 |
| `p1_qksr_cur` (PQK) | 87.427 | 82.82 | 4.67 | 81.28 | 84.40 | 81.16 | 96.05 |
| `p1_qksr_rbf` (RBF trên θ) | 87.354 | 82.60 | 5.07 | 80.91 | 84.39 | 80.30 | 96.05 |
| `p1_qksr_nocnot` (bỏ CNOT) | 87.384 | 82.64 | 5.08 | 80.87 | 84.51 | 80.31 | 96.33 |

## Theo luật P1.3 (Δ = A − B, CI 95%)

| So sánh | Δ AIA | Δ final | Δ forgetting | Δ old | Δ new | Kết quả |
|---|---|---|---|---|---|---|
| PQK − RBF(θ) (T2) | +0.074 [−0.007, +0.154] | +0.22 [+0.05, +0.39] | −0.40 [−0.78, −0.03] | +0.37 [+0.24, +0.50] | +0.01 [−0.27, +0.29] | **không kết luận** |
| PQK − bỏ CNOT (T4) | +0.043 [+0.036, +0.050] | +0.18 [+0.01, +0.35] | −0.41 [−0.67, −0.15] | +0.41 [+0.01, +0.81] | −0.12 [−0.40, +0.17] | **PQK hơn** (CI > 0) |
| Đọc kèm: QKSR − RSIAT (rep 0) | +0.156 | +0.33 | −0.61 | +0.70 | −0.14 | chỉ 1 cặp |

## Đọc kết quả

1. **Dự đoán "tương đương" trượt, và dự đoán cập nhật sau P0 ("RBF ≥ PQK") cũng trượt.**
   - PQK giữ task cũ tốt hơn cả hai bản đối chứng: old +0.37 và +0.41 (CI > 0), forgetting −0.4, final +0.2 (CI > 0).
   - Task mới không đổi rõ rệt.
   - Về AIA, lợi thế chỉ +0.04 đến +0.07. Riêng T4 qua luật, nhưng **mức hiệu ứng rất nhỏ**: CI hẹp chỉ vì 3 cặp chênh gần như đều nhau (+0.040, +0.045, +0.044).
2. **Mâu thuẫn với P0 cần giải thích.** P0 cho thấy khoảng cách PQK *kém* chọn lọc hơn RBF(θ) (AUC 0.59 so với 0.72), vậy mà PQK lại giữ task cũ tốt hơn. Đây là giả thuyết, chưa kiểm:
   - kernel ít chọn lọc tạo gradient tách lớp "dàn đều" hơn trên mọi lớp cũ;
   - kernel chọn lọc thì chỉ đẩy vài cặp gần.

   Phần chênh nằm ở task cũ nhất (task 0–1: PQK 81.2, RBF 80.3, bỏ CNOT 80.3).
3. **T4 bị nhiễu bởi base task.** Bỏ CNOT làm task 0 ra 96.33, so với 96.05 ở PQK và RBF, tức loss base ở task 0 cũng khác. Vì vậy Δ của T4 gộp cả ảnh hưởng ở task 0 lẫn ở các task sau.
4. **So với RSIAT cùng môi trường:** chỉ có một cặp (rep 0), QKSR +0.16 AIA, old +0.70, new −0.14. Ở môi trường cũ, QKSR − RSIAT là +0.04 [−0.11, +0.18] với n = 3. W06 sẽ thêm RSIAT rep 1 và 2 cùng môi trường để có n = 3.

## Hệ quả

- Phần lượng tử của QKSR **có** một tác động nhỏ, nhất quán, lên độ giữ task cũ so với hai bản cổ điển gần nhất. Tác động lên AIA rất nhỏ: +0.04 đến +0.07.
- Phù hợp với phần A của `exps/qbound`: QKSR vẫn nằm trên hoặc gần đường đánh đổi cũ–mới.
- P2 (W06) giữ nguyên. Thêm 2 run RSIAT cùng môi trường; W06 thành 11 run, khoảng 5 giờ.

# QKSR-v2 P2: phép chiếu LDA + bandwidth mỗi task (W06, n = 3)

Nguồn: `kaggle/waves/w06_qksr_p2` (8 run), chạy ngày 2026-10-31, cùng môi trường với W04 và W05. File gốc nằm trong `p2/`. Đã hoãn `p2_gamma`.

| Họ | AIA (rep 0/1/2) | Trung bình AIA | Final | Forgetting |
|---|---|---:|---:|---:|
| RSIAT (`p0_rsiat`, `p1_rsiat_rep1/2`) | 87.246 / 87.399 / 87.487 | 87.377 | 82.64 | 5.20 |
| QKSR (`p1_qksr_cur`) | 87.402 / 87.398 / 87.482 | 87.427 | 82.82 | 4.67 |
| `p2_lda` | 87.320 / 87.346 / 87.402 | 87.356 | 82.69 | 4.88 |
| `p2_lda_rbf` | 87.347 / 87.402 / 87.432 | 87.394 | 82.65 | 4.96 |

## Theo luật P2.3

| So sánh | Δ AIA | Δ final | Δ forgetting | Δ old | Δ new | Kết quả |
|---|---|---|---|---|---|---|
| **`p2_lda` − QKSR** | **−0.071 [−0.113, −0.030]** | −0.13 [−0.30, +0.04] | +0.22 [+0.11, +0.33] | −0.05 | −0.18 | **KÉM QKSR; không vượt đường đánh đổi** |
| `p2_lda` − `p2_lda_rbf` | −0.038 [−0.077, +0.002] | +0.04 | −0.08 | +0.06 | +0.02 | tương đương |
| Đọc kèm: QKSR − RSIAT | +0.050 [−0.178, +0.278] | +0.19 [−0.14, +0.52] | **−0.54 [−0.69, −0.38]** | **+0.52 [+0.14, +0.90]** | −0.25 [−0.61, +0.12] | không kết luận |
| Đọc kèm: `p2_lda` − RSIAT | −0.021 [−0.230, +0.188] | +0.06 | −0.32 [−0.58, −0.06] | +0.48 | −0.42 | không kết luận |

## Đọc kết quả

1. **Cách sửa đã đúng về mặt cơ chế nhưng không giúp.** Theo `QKSR refit` trong log:
   - phép chiếu được fit lại mỗi task (8 hướng LDA giữ 41% Fisher ở task 1, 17% ở task 9);
   - kernel lúc train hết dồn cục: trung bình 0.06–0.11, có đuôi tới 0.7–0.86, so với 0.55 của QKSR.

   Dù vậy, AIA **giảm** 0.07 so với QKSR (CI < 0), forgetting tăng.
2. **Kernel chọn lọc hơn thì loss tách lớp tập trung vào ít cặp gần nhau.** Cơ chế này cũng là cơ chế mà P3 (hard-pair margin) nhắm tới, và kết quả không tốt hơn. Kết hợp với P1 (PQK "mù" giữ task cũ tốt hơn RBF), dữ liệu gợi ý: tác dụng nhỏ của QKSR đến từ một lực đẩy **dàn đều, không chọn lọc**, chứ không đến từ việc tách đúng các cặp hay nhầm.
3. **Phần lượng tử không còn khác biệt khi đã có LDA:** PQK − RBF tương đương.
4. **So với RSIAT cùng môi trường (n = 3), kết quả tái lập:**
   - QKSR không khác RSIAT về AIA (+0.05 [−0.18, +0.28]); ở môi trường cũ là +0.04 [−0.11, +0.18].
   - QKSR **giảm forgetting** (−0.54, CI < 0) và **tăng task cũ** (+0.52, CI > 0), đổi lại task mới −0.25.

   Đây đúng là đường đánh đổi của phần A, tỉ giá khoảng 0.5.

## Kết luận cho QKSR-v2

| Bước | Trạng thái |
|---|---|
| P0 | xong: nguyên nhân kỹ thuật là W1, W2, W3, W4; không có W6 |
| P1 | xong: PQK giữ task cũ tốt hơn bản cổ điển, nhưng AIA chỉ +0.04 đến +0.07 |
| P2 | xong: **sửa W1–W3 làm kết quả tệ hơn** |
| P3 (hard-pair) | **đề xuất bỏ**: P2 đã đẩy loss về phía cặp gần và không giúp |
| P4 | đã bỏ theo P0 |
| P5 (16 qubit) | **bỏ**: P2 cho thấy làm phép chiếu mạnh hơn không giúp |

Sửa điểm yếu kỹ thuật của kernel không đưa QKSR vượt RSIAT về AIA. Hướng "loss tách lớp trên adapter dùng chung" dừng ở đây. Nó có đủ bằng chứng (P0 đo cơ chế, P1 so bản cổ điển, P2 sửa nhưng thất bại) để làm một mục kết quả âm có kiểm soát trong paper.
