# QKSR-v2, bước P0: đo nguyên nhân trước khi sửa

Đặc tả viết **trước khi chạy**, ngày 2026-10-31. Luật đọc ở mục P0.3 là cố định.

Căn cứ:
- `reports/Khắc phục điểm yếu QKSR.md` (điểm yếu W1–W7, thứ tự P0–P5);
- `exps/qksr/results/README.md` (QKSR không khác RSIAT: +0.04–0.06 AIA);
- `exps/qbound/results/README.md` (mọi can thiệp lúc huấn luyện, kể cả 6 run QKSR, chỉ đổi task mới lấy task cũ).

## P0.1. Mục đích

Báo cáo đề xuất bốn cách sửa (projector theo thống kê, bandwidth mỗi task, hard-pair margin, gradient surgery) và một bộ đối chứng cổ điển. Mỗi cách sửa đều dựa trên một giả thuyết về nguyên nhân, và **chưa giả thuyết nào được đo**. P0 đo các nguyên nhân đó trên đúng cấu hình QKSR cũ, **không đổi huấn luyện**, rồi dùng luật cố định để quyết định bước P1–P5 nào đáng chạy.

Các run cũ đặt `save_checkpoints false`, nên không có mạng để đo lại. P0 cần chạy lại hai run có bật hook đo.

## P0.2. Run và phép đo

| Run | Overlay | Bản sao của | Mục đích |
|---|---|---|---|
| `p0_qksr_cur` | `qksr_cur_rep0` + probe | `qksr_cur` rep 0 | đo W1–W7 trên QKSR |
| `p0_rsiat` | `noise_rep0` + probe | RSIAT (trùng `hp_diag` của W03 nếu môi trường giống) | tham chiếu: xung đột KD–CosFace và độ chọn lọc của các đại lượng cổ điển khi không có QKSR |

Probe nằm trong `utils/qksr_probe.py` và chỉ đọc:
- gradient lấy bằng `autograd.grad`;
- feature trích dưới `no_grad` + `preserve_rng` ở chế độ eval.

`tests/test_qksr_probe.py` kiểm tra trọng số, `old_ae` và thống kê lớp **trùng từng bit** khi bật hoặc tắt probe, với cả RSIAT lẫn QKSR.

**Gradient** trên tham số adapter đang train, mỗi 10 bước, tổng hợp theo epoch:
- chuẩn và cosine từng cặp giữa ba loss: CosFace `c`, căn chỉnh KD `kd` (β·MSE) và tách lớp `sep` (γ·PQK);
- cosine giữa `c` và tổng `kd + sep`;
- phần `sep` còn lại sau phép chiếu kiểu PCGrad;
- ở task 0: cosine giữa `c` và loss base `base`.

**Cuối mỗi task**, trên 20 ảnh train mỗi lớp đã thấy (mạng hiện tại):

| Đại lượng | Điểm yếu |
|---|---|
| Tỉ lệ tán xạ giữa lớp / trong lớp (sau LayerNorm) nằm trong row space của projector 768 → 8, so với mức ngẫu nhiên 8/768 | W1 |
| Tỉ lệ góc bão hoà \|θ\| > 0.9π, độ lệch chuẩn θ theo qubit | W2 |
| AUC "cặp khác lớp xa hơn cặp cùng lớp" cho khoảng cách PQK, so với các bản cổ điển: Euclid trên θ, Euclid và cosine trên z 768 chiều. Có tách cặp khác lớp **khác task** và **cùng task** | W1, W7 |
| Kernel trung bình trong lớp / giữa lớp ở γ đã học | W2 |
| CKA giữa Gram PQK và Gram RBF (bandwidth median) trên θ và trên z | W7 |
| Các cặp mà loss thật sự dùng (prototype cũ qua `old_ae` với feature lớp mới): kernel trung bình trên mọi cặp so với trên top-5 prototype gần nhất | W4 |

Đọc kèm, không phải luật: cùng các đại lượng ở task 0 (W5) và độ chính xác cuối của hai run.

## P0.3. Luật đọc (cố định, `scripts/summarize_qksr_probe.py`)

Mọi luật dùng trung vị qua các task ≥ 1 của `p0_qksr_cur`.

| Luật | Điều kiện | Quyết định |
|---|---|---|
| **W6 xung đột** | median cos(c, sep) < −0.2 **và** ≥ 60% bước âm | chạy P4 (gradient surgery) |
| **W6 không xung đột** | \|median cos(c, sep)\| < 0.1 | **bỏ P4**: đường đánh đổi không do xung đột gradient |
| W6 vùng giữa | còn lại | P4 không ưu tiên |
| **W1 nút cổ chai** | AUC PQK thấp hơn AUC Euclid trên z 768 chiều quá 0.05 | ưu tiên P2 (projector LDA từ thống kê) |
| **W1 hướng** | tán xạ giữa lớp trong row space < 5 × mức ngẫu nhiên | như trên |
| **W2 bão hoà** | > 20% góc có \|θ\| > 0.9π | thêm scale trước tanh vào P2 |
| **W7 tương đương** | \|AUC PQK − AUC RBF(θ)\| < 0.02, hoặc CKA(PQK, RBF θ) > 0.9 | dự đoán P1 cho kết quả null lượng tử; vẫn chạy P1 để có số |
| **W4 cặp khó** | kernel top-5 − kernel toàn bộ > 0.1 | chạy P3 (hard-pair margin) |

**Kiểm tra tái lập** (điều kiện để tin các phép đo):
- `p0_qksr_cur` phải có cùng đường accuracy với `qksr_cur` rep 0;
- `p0_rsiat` phải có cùng đường accuracy với `hp_diag` của W03.

Nếu khác, ghi độ lệch. Nguyên nhân có thể là môi trường Kaggle khác (torch, GPU), không phải do probe, vì test trùng bit đã đạt. Phép đo vẫn dùng được, vì chúng mô tả chính run đó.

## P0.4. Dự đoán ghi trước

- **W6:** cos(c, sep) gần 0 hoặc âm nhẹ, vì `sep` nhỏ so với `c` (Losses_rt của QKSR gần như không đổi).
- **W1:** AUC PQK thấp hơn rõ so với z 768 chiều.
- **W7:** PQK gần RBF trên θ.
- **W4:** có khoảng cách top-k so với toàn bộ, vì nhầm lẫn dồn vào vài cặp task.

## P0.5. Wave

`kaggle/waves/w04_qksr_p0`: ImageNet-R 1993, 2 run song song, base `c7bc4fd` + patch. Ước tính khoảng 2 giờ: probe làm mỗi bước đo chậm thêm khoảng 3 lượt backward của adapter, nên toàn run chậm thêm khoảng 25%.

# Bước P1: PQK so với bản đối chứng cổ điển trên cùng phép chiếu

Viết trước khi chạy, ngày 2026-10-31. P1 luôn được chạy, bất kể kết quả P0 (P0.3 chỉ dự đoán trước kết quả).

## P1.1. Run

ImageNet-R 1993, `rng_replicate` 0–2, tổng 9 run. Ba họ giống hệt `qksr_cur`, chỉ khác `q_kernel_type`. Kernel này dùng cho **cả** loss base ở task 0 lẫn loss tách lớp ở task ≥ 1, nên đổi nó là thay được toàn bộ phần lượng tử.

| Họ | `q_kernel_type` | Vai trò |
|---|---|---|
| `p1_qksr_cur` | `pqk` | QKSR như cũ, chạy lại cùng môi trường |
| `p1_qksr_rbf` | `rbf_proj` | **T2:** RBF trên cùng góc θ = π·tanh(W·LN(z)), cùng cách hiệu chỉnh γ |
| `p1_qksr_nocnot` | `pqk_no_cnot` | **T4:** PQK bỏ vòng CNOT, không còn entanglement |

`p1_qksr_cur` được chạy lại thay vì dùng 3 run `qksr_cur` cũ, để cả ba họ cùng một môi trường. Nếu P0 cho thấy môi trường tái lập đúng run cũ, phép so sánh với run cũ sẽ được đọc kèm. `scripts/summarize_qksr_p1.py` tự kiểm tra điều này khi có log cũ.

Bản T3 (kernel trên z 768 chiều, không qua phép chiếu) cần code mới và đã được đo ở mức độ chọn lọc trong P0. P1 không chạy T3.

## P1.2. Thước đo

So sánh ghép cặp theo replicate (`scripts/run_level_stats.py`), với Δ = PQK − bản đối chứng cho AIA, final, forgetting, cùng old (task 0–4) và new (task 5–9) sau task cuối.

## P1.3. Luật đọc (cố định)

| Kết quả | Điều kiện trên Δ AIA (n = 3, CI 95% t) |
|---|---|
| PQK hơn bản cổ điển | cận dưới CI > 0 |
| PQK kém bản cổ điển | cận trên CI < 0 |
| Tương đương | CI nằm trong ±0.1 |
| Không kết luận | còn lại |

- **PQK − RBF (T2):** trả lời câu "phần lượng tử có đóng góp không" ở đúng cấu hình QKSR.
- **PQK − bỏ CNOT (T4):** trả lời câu "entanglement có đóng góp không".

**Dự đoán ghi trước:** tương đương ở cả hai so sánh. Dựa trên Flórez-Ablan et al., P0, và kết quả QKSR không khác RSIAT.

Vì `qksr_cur` không khác RSIAT, một kết quả "tương đương" ở P1 **không** nói gì về việc QKSR có ích. Nó chỉ cho biết: nếu QKSR-v2 (P2–P4) có ích, phần ích đó có thể giữ nguyên khi thay PQK bằng RBF. Vì vậy P2 trở đi luôn chạy kèm bản RBF.

## P1.4. Wave

`kaggle/waves/w05_qksr_p1`: 9 run, max_parallel 3, khoảng 4 giờ. Chạy song song với W04 được, nếu quota GPU cho phép.

# Bước P2: phép chiếu fit lại từ thống kê và bandwidth mỗi task (code sẵn, **chỉ chạy nếu P0 xác nhận W1 hoặc W2**)

Viết trước khi chạy, ngày 2026-10-31.

## P2.1. Thay đổi (mặc định tắt, `models/RSIAT_adapter.py::_qksr_v2_refit`)

Ở đầu mỗi task t ≥ 1, trước khi train, chỉ dùng:
- thống kê đã lưu của lớp cũ (sau SSCA);
- lượt trích feature bắt buộc trên ảnh của task mới, bằng mạng của task trước.

Không đọc ảnh cũ, không dùng RNG. Task 0 giữ nguyên, nên trùng từng bit với QKSR.

| Khoá | Giá trị | Tác dụng |
|---|---|---|
| `q_proj_refit` | `lda` | Hàng của projector là n hướng Fisher-LDA trên mọi lớp đã thấy: S_W là trung bình covariance lớp, S_B là covariance của các mean, cộng ridge 1e-3·tr(S_W)/D. Mỗi hàng được scale để θ có độ lệch chuẩn khoảng `q_angle_std` = 0.8 rad (chống bão hoà, W2), và lấy tâm ở trung bình các mean. Từ đó bỏ LayerNorm, vì thống kê lưu cho feature thô; `old_ae` là residual nên cùng thang |
| `q_gamma_refit` | `target` | γ được đặt sao cho kernel trung vị giữa mean lớp cũ và feature lớp mới bằng `q_gamma_target` = 0.2 (chống dồn cục, W2). Phần học được của γ được đặt về 0 |

## P2.2. Run (9, ImageNet-R 1993, `rng_replicate` 0–2)

| Họ | Kernel | Refit |
|---|---|---|
| `p2_lda` | PQK | LDA + γ |
| `p2_lda_rbf` | RBF trên θ | LDA + γ (bản cổ điển của `p2_lda`) |
| `p2_gamma` | PQK | chỉ γ: tách tác dụng của bandwidth khỏi phép chiếu |

Vế so sánh là `p1_qksr_cur` (W05). P2 phải chạy **cùng môi trường** với W05, và việc này được kiểm bằng phần so khớp môi trường của P0/P1.

## P2.3. Luật (cố định, `scripts/summarize_qksr_p1.py`)

| Kết quả | Điều kiện |
|---|---|
| **Vượt đường đánh đổi** | `p2_lda` − `p1_qksr_cur`: Δold > 0, Δnew > 0 **và** cận dưới CI 95% của Δ AIA > 0 |
| Phần lượng tử | `p2_lda` − `p2_lda_rbf`, đọc như P1.3 |
| Phép chiếu so với bandwidth | `p2_lda` − `p1_qksr_cur` đặt cạnh `p2_gamma` − `p1_qksr_cur` (đọc kèm) |

**Dự đoán ghi trước:** γ một mình không vượt đường đánh đổi, vì nó chỉ làm loss mạnh hơn nên đẩy điểm đi xa hơn dọc đường, giống `holevo_spread`. LDA có thể giúp phần old, nhưng kỳ vọng vượt đường thấp, theo phần A của `exps/qbound`. Nếu cả hai không vượt, đây là bằng chứng cho giới hạn của loss tách lớp trên adapter dùng chung (xem mục "nếu mọi giả thuyết sai").

## P2.4. Wave

`kaggle/waves/w06_qksr_p2`: 9 run, khoảng 4 giờ. **Chỉ chạy khi** P0 kích hoạt luật W1 hoặc W2 bão hoà, và W05 đã xong.

## Nhật ký thay đổi

- 2026-10-31: viết đặc tả P0 và code probe. 5 test đạt, trong đó có kiểm tra trùng từng bit với RSIAT và QKSR.
- 2026-10-31: thêm P1 (9 overlay, `scripts/summarize_qksr_p1.py`, W05) và P2 (code `_qksr_v2_refit`, `refit_projector_lda`, `set_gamma_target`, `tests/test_qksr_v2.py`, 9 overlay, W06). Build lại W04 cùng phiên bản code; kết quả P0 không đổi vì P2 mặc định tắt. 157 test đạt.
- 2026-10-31: P0 xong (`results/README.md`). Kích hoạt W1, W2, W4; W6 không xung đột nên bỏ P4; PQK kém RBF(θ) về độ chọn lọc. `p0_qksr_cur` khác `qksr_cur` cũ do môi trường (task 0 96.05 so với 96.33), nên P1 dùng `p1_qksr_cur` cùng wave. Quan sát ngoài luật: KD ngược chiều CosFace ở mọi bước.
- 2026-10-31: P1 xong (`results/README.md`). T2 (PQK − RBF) không kết luận (+0.074 [−0.007, +0.154]); T4 (PQK − bỏ CNOT) qua luật nhưng nhỏ (+0.043 [+0.036, +0.050]). PQK giữ old tốt hơn hai bản cổ điển (+0.4, CI > 0). Thêm `p1_rsiat_rep1/rep2` vào W06 (RSIAT cùng môi trường; rep 0 = `p0_rsiat`), cùng hai dòng so với RSIAT trong `scripts/summarize_qksr_p1.py`.
- 2026-10-31: W06 rút còn 8 run: `p2_lda` ×3, `p2_lda_rbf` ×3, `p1_rsiat_rep1/2`. `p2_gamma` hoãn, chỉ chạy nếu `p2_lda` có tác dụng.
- 2026-10-31: P2 xong (`results/README.md`). `p2_lda` kém QKSR (Δ AIA −0.071 [−0.113, −0.030]), không vượt đường đánh đổi; PQK ≈ RBF khi có LDA. QKSR − RSIAT cùng môi trường n = 3: +0.05 [−0.18, +0.28] AIA, forgetting −0.54 (CI < 0). Đề xuất dừng QKSR-v2 (bỏ P3, P5).
