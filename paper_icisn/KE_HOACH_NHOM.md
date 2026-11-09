# Kế hoạch viết paper nộp ICISN 2027 (1 tuần)

> Phân công cho Nhóm 4: **Thuận, Khánh, Đức Anh, Phát**. Cập nhật 09/11/2026.
> Mục tiêu: **xong bản nháp đầy đủ ≤ 10 trang vào Chủ nhật 15/11/2026.**
> Từ 16/11 đến hạn nộp (25/12) chỉ review và chỉnh sửa, không viết mới.

---

## 1. Hội nghị

| Mục | Thông tin (theo icisn.com) |
|---|---|
| Hội nghị | ICISN 2027, 20–21/03/2027, Đại học Hải Dương. Kỷ yếu Springer LNNS (series được Scopus index) |
| **Hạn nộp** | **25/12/2026** qua Microsoft CMT |
| Thông báo / camera-ready | 07/02/2027 / 15/02/2027 |
| Định dạng | Template Springer (LaTeX), tiếng Anh, **tối đa 10 trang** (phí cơ bản 6 trang, +$15/trang) |
| Lưu ý | Similarity ≤ 15%. Phải trình bày tại hội nghị. Không nộp song song nơi khác. |

**Thuận gửi email** (icisnconf@gmail.com) hỏi: review single- hay double-blind? 10 trang có tính References không?

---

## 2. Novelty của paper: 3 đóng góp, mọi section phải phục vụ chúng

**Tên gợi ý:** *Statistics-Calibrated Fusion for Exemplar-Free Class-Incremental Learning: A Quantum Measurement and Its Classical Twin*

**Một câu tóm tắt:** Các phương pháp exemplar-free đã lưu sẵn thống kê lớp (μ, Σ) nhưng chỉ dùng chúng để train lại classifier. Chúng tôi chỉ ra rằng chính các thống kê đó, khi được đọc như một phép đo lượng tử hoặc như hồi quy ridge, sai ở những ảnh khác với classifier. Một phép kết hợp có hiệu chỉnh, không cần train và không cần nhãn, tận dụng được điều này.

| | Đóng góp | Điểm mới so với các công trình trước | Bằng chứng trong paper |
|---|---|---|---|
| **C1** | **SCF**: kết hợp điểm của linear head với điểm tính từ thống kê lớp, có hệ số κ tự hiệu chỉnh | FeCAM, RanPAC và NCM **thay** classifier bằng một bộ phân loại khác. SLCA dùng thống kê để **train lại** classifier. SCF **giữ nguyên** classifier và **cộng thêm** một nguồn điểm thứ hai: không train, không lưu thêm, không cần nhãn. κ làm w bất biến với phép co giãn thang điểm (Mệnh đề 3), nên **w chọn một lần trên ImageNet-R rồi khoá, vẫn tăng A_B trên cả 3 dataset chưa dùng để chọn**. | A_B tăng trên 4/4 dataset; ablation κ và w; đo chi phí |
| **C2** | **Góc nhìn lượng tử có kiểm chứng**: dựng ma trận mật độ của lớp từ (μ, Σ), phân loại bằng PGM | Các bài QML thường báo cáo mức tăng nhưng không có bản cổ điển để so. Chúng tôi chứng minh (Mệnh đề 1–2) rằng mạch lượng tử đặt sau bước mã hoá và phép co trạng thái **không thể** thay đổi kết quả đo. Kèm sự tương đương giữa PGM và least-squares, điều này **dự đoán** rằng ridge sẽ ngang PGM, và thực nghiệm xác nhận. | Bảng lượng tử vs twin; SCF-PGM − SCF-ridge = −0.04 [−0.17, +0.08] |
| **C3** | **Vì sao SCF hiệu quả**, phân tích có kiểm soát nhiễu | Linear head và điểm thống kê **sai ở ảnh khác nhau** (bất đồng 14% trên ImageNet-R). Mọi thay đổi lúc train mà chúng tôi thử (kể cả lượng tử) chỉ **đổi** accuracy lớp mới lấy lớp cũ, còn SCF tăng **cả hai**. Bảng 2×2 cho thấy SCF **cộng được** lên mạng train bằng lượng tử. | Phân tích lỗi; hình trade-off cũ/mới; bảng 2×2 W07 |

**Quy tắc khi viết novelty:**
- Chỉ khẳng định điều số liệu chứng minh được.
- **Không** viết "quantum advantage" và **không** viết "SOTA". Số đã công bố dùng cách chia dữ liệu khác, nên chỉ dùng để định vị kết quả.
- Mỗi section mở đầu hoặc kết thúc bằng **một câu nói rõ nó phục vụ C1, C2 hay C3**.

**Bỏ khỏi bản này:** chi tiết Holevo/POVM (giữ 1 dòng trong bảng twin), probe QKSR-v2, phân tích joint training, bảng phương pháp có exemplar, appendix.

---

## 3. Cấu trúc bài và yêu cầu novelty từng section

| Section | Trang | Người viết | Phải có để làm rõ novelty |
|---|---|---|---|
| Abstract | 0.25 | Thuận | 1 câu bài toán, 1 câu khoảng trống, C1–C3 mỗi ý một câu kèm **số cụ thể**, 1 câu kết luận về lượng tử |
| 1. Introduction | 1.25 | Thuận | Khoảng trống: thống kê lớp được lưu nhưng chưa được dùng lúc dự đoán. Câu hỏi về lượng tử thường không có bản cổ điển để so. Danh sách **3 đóng góp đánh số C1–C3**, mỗi ý trỏ tới section và bảng chứng minh. |
| 2. Related Work | 0.8 | Khánh | 3 đoạn: (a) CIL với mô hình pre-trained; (b) **classifier xây từ thống kê lớp: FeCAM, RanPAC, SLCA, NCM** (đoạn quan trọng nhất); (c) QML và benchmark với classical twin. Mỗi đoạn kết thúc bằng một câu "khác với các công trình này, chúng tôi…" |
| 3. Background | 0.5 | Khánh | Bài toán CIL, metric, RSIAT rút gọn. Nêu rõ memory (μ, Σ) là thứ **có sẵn**, SCF không thêm gì. |
| 4. Method | 2.4 | Đức Anh | **Theo thứ tự pipeline: lúc train → lúc dự đoán.** Đoạn mở đầu nối hai phần. 4.1 PQK lúc train: base task (steering) và incremental task (QKSR), ~0.4 tr, chỉ định nghĩa, kết bằng câu chuyển ý "lớp cũ chỉ còn lại (μ, Σ)". 4.2 Ma trận mật độ từ (μ, Σ). 4.3 PGM. 4.4 Ridge twin. 4.5 SCF và κ. 4.6 Ba mệnh đề: bất biến unitary, bất biến khi co trạng thái, bất biến thang điểm của SCF. **Mỗi mệnh đề kèm một câu "ý nghĩa thực tế".** Trọng tâm là 4.2–4.6. |
| 5. Experimental Setup | 0.5 | Phát | Dataset, seed, quy trình so cặp, w khoá trên tuning group (thể hiện C1: w khoá trước, không chỉnh trên tập test) |
| 6. Results | 1.75 | Phát | 6.1 Bảng chính SCF vs RSIAT kèm số đã công bố. 6.2 Bảng lượng tử vs twin (C2). 6.3 Hình gain. |
| 7. Analysis | 1.05 | Khánh + Phát | 7.1 Vì sao SCF giúp: bất đồng 14%, hình trade-off cũ/mới thu nhỏ (C3) – **Khánh**. 7.2 Ablation κ và w + chi phí – **Phát**. 7.3 Bảng 2×2 W07 – **Phát**. |
| 8. Conclusion + Limitations | 0.5 | Khánh | Nhắc lại C1–C3. Hạn chế: một baseline, cách chia dữ liệu khác, mạch nhỏ được mô phỏng. |
| References (~25 mục) | 1.0 | Khánh | |

### Bảng định vị cho Related Work (Khánh hoàn thiện và đối chiếu với bài gốc)

| Phương pháp | Dùng thống kê lớp để làm gì | Thay classifier? | Cần train thêm? |
|---|---|---|---|
| NCM / prototype (SimpleCIL, APER) | Phân loại theo mean gần nhất | Có | Không |
| FeCAM (NeurIPS 2023) | Bộ phân loại Bayes/Mahalanobis từ prototype và covariance | Có | Không |
| RanPAC (NeurIPS 2023) | Chiếu ngẫu nhiên + head ridge dạng đóng | Có | Không (cộng dồn từ feature) |
| SLCA (ICCV 2023), RSIAT | Lấy mẫu Gaussian để train lại classifier | Không | Có |
| **SCF (của nhóm)** | **Nguồn điểm thứ hai, kết hợp với classifier có sẵn, có κ** | **Không** | **Không** |

Mô tả trong bảng là tóm tắt ban đầu. **Khánh phải đọc lại bài gốc** của từng phương pháp để kiểm tra trước khi đưa vào paper.

---

## 4. Thí nghiệm (đều ngắn)

> Script và notebook đã chuẩn bị xong; bảng ablation đã có số (`paper_icisn/results/scf_ablation.md`).
> Hướng dẫn từng bước cho mỗi người: `paper_icisn/HUONG_DAN_THANH_VIEN.md`.

### 4a. Ablation: 1 bảng, khoảng 0.4 trang (Phát)

Số đã có trong `exps/scf/results/summary.json` (head `linear`, `ridge_stats`, `pgm_stats`, `fuse_*_b0p25` … `fuse_*_b4`).

Bảng A_B, trung bình 3 seed, cột cuối là trung bình 4 dataset:

| Biến thể | CIFAR-100 | ImageNet-R | ImageNet-A | CUB-200 | TB |
|---|---|---|---|---|---|
| Linear head (RSIAT) | | | | | |
| Ridge riêng / PGM riêng | | | | | |
| Cộng thẳng, không có κ | | | | | |
| SCF, w đã khoá (ridge 0.25, PGM 0.5) | | | | | |
| SCF, w = 1 / w = 4 | | | | | |

- Dòng "không có κ" hiện mới có cho ImageNet-R và ImageNet-A (+0.25 / +0.44). Nếu chưa tính được cho CIFAR và CUB trước thứ Tư thì chỉ báo 2 dataset này và ghi rõ trong chú thích.
- **Chi phí:** đo ms/ảnh và bộ nhớ của linear, ridge, PGM trên 1 GPU, mất khoảng 1 giờ. Đưa thành 1 câu hoặc 1 dòng trong bảng.

### 4b. Wave W07: lượng tử lúc train + SCF (Thuận chạy, Phát viết)

Thiết kế 2×2, ImageNet-R seed 1993, 3 replicate, **cùng một wave**:

| | Linear head | SCF (ridge, PGM) |
|---|---|---|
| **Train RSIAT** (3 run) | baseline | SCF |
| **Train QKSR** (3 run): PQK ở base + QKSR ở incremental | QKSR | **QKSR + SCF** |

- Config QKSR có sẵn (`exps/qksr2/p1_qksr_cur_rep*.json`), chỉ thêm các khoá `qsd_*` như trong `exps/scf/scf.json`.
- **Quy tắc chốt trước khi chạy:** w giữ nguyên. "Kết hợp có ích" khi Δ Ā của (QKSR + SCF) − (RSIAT + SCF) có cận dưới khoảng tin cậy 95% lớn hơn 0.
- **Pilot đã có** (1 run, mạng Holevo/POVM + SCF): SCF vẫn tăng Ā khoảng +0.4; phần lượng tử lúc train chỉ thêm +0.05 đến +0.10, trong dải nhiễu.
- Kết quả nào cũng phục vụ C3: SCF cộng được lên mạng train bằng lượng tử.

---

## 5. Phân công

Mỗi người viết khoảng 2.5 trang. Ai có phần thực nghiệm thì viết ít chữ hơn.

| Người | Viết | Việc khác | Đọc chéo |
|---|---|---|---|
| **Thuận** (trưởng nhóm) | Abstract, Introduction (1.5 tr) | Upload khung lên Overleaf; chạy W07; email ban tổ chức; ghép bài và giữ ≤ 10 trang; kiểm tra cuối | Đức Anh |
| **Khánh** | Related Work, Background, Analysis 7.1, Conclusion + Limitations (2.3 tr) | Bảng định vị; rà soát toàn bộ trích dẫn (đối chiếu tác giả, năm, hội nghị; thêm FeCAM) | Thuận |
| **Đức Anh** | Method 4.1–4.6, theo thứ tự train → dự đoán (2.4 tr) | Kiểm tra phần Method của khung biên dịch đúng; thống nhất ký hiệu toàn bài; viết lại Mệnh đề 1–3 cho gọn | Phát |
| **Phát** | Setup, Results, Analysis 7.2–7.3 (2.75 tr) | Bảng ablation, đo chi phí, bảng 2×2 W07, hình gain | Khánh |

**Nguồn để viết:** bản ACL `paper_acl/sections/*.tex`. **Viết lại bằng lời của mình**, vì yêu cầu similarity ≤ 15%. Mọi con số phải khớp với file kết quả trong repo.

---

## 6. Lịch 7 ngày

| Ngày | Thuận | Khánh | Đức Anh | Phát |
|---|---|---|---|---|
| **T2 09/11** | Họp 30 phút: chốt C1–C3 và tên bài. **Upload khung `dist/paper_icisn_overleaf.zip` lên Overleaf, chia sẻ cho nhóm.** **Chạy W07.** Gửi email BTC. | Đọc FeCAM, RanPAC, SLCA; bắt đầu bảng định vị | Đọc phần Method trong khung; chốt bảng ký hiệu | Đọc bảng ablation đã có (`paper_icisn/results/scf_ablation.md`); chạy `scf_cost.py` |
| **T3 10/11** | Viết Introduction (danh sách C1–C3); theo dõi W07 | Viết Related Work | Viết đoạn mở đầu Method, 4.1 (PQK lúc train), 4.2 (ma trận mật độ) | Viết Setup; kiểm tra bảng chính và bảng twin trong khung |
| **T4 11/11** | Tải kết quả W07, tổng hợp bảng 2×2 → gửi Phát | Viết Background; rà soát trích dẫn | Viết 4.3–4.5 (PGM, ridge, SCF) | Viết 7.2 (ablation + chi phí) |
| **T5 12/11** | Viết Abstract; ghép bài lần 1, đếm trang | Viết 7.1 (vì sao SCF giúp) + Conclusion + Limitations | Viết 4.6 (ba mệnh đề + chứng minh ngắn) | Viết Results, 7.3; chèn bảng 2×2 và hình gain |
| **T6 13/11** | Đọc chéo phần Đức Anh | Đọc chéo phần Thuận | Đọc chéo phần Phát | Đọc chéo phần Khánh |
| **T7 14/11** | Ghép bài lần 2, **cắt về ≤ 10 trang** | Sửa theo góp ý | Sửa theo góp ý | Sửa theo góp ý; đối chiếu lại mọi số |
| **CN 15/11** | **Bản nháp v1.** Họp 1 tiếng đọc toàn bài, theo checklist mục 7 | Cả nhóm | Cả nhóm | Cả nhóm |

**Sau 15/11** (mỗi tuần 1 buổi):
- **16/11–06/12:** mỗi người đọc toàn bài trong vai reviewer, ghi 3 điểm yếu; sửa.
- **07–20/12:** kiểm tra similarity, tiếng Anh, số liệu, trích dẫn.
- **Nộp CMT chậm nhất 24/12.**

---

## 7. Checklist novelty khi họp Chủ nhật

- [ ] Abstract và Introduction nêu đủ **C1–C3**, mỗi ý có số cụ thể.
- [ ] Related Work có **FeCAM, RanPAC, SLCA, NCM** và câu nói rõ SCF khác ở đâu (giữ classifier, kết hợp, κ không cần nhãn).
- [ ] Method có đủ **3 mệnh đề**, mỗi mệnh đề kèm ý nghĩa thực tế.
- [ ] Ablation báo đúng kết quả: κ **cần cho nguồn ridge** (với PGM giúp ít hơn); w vừa phải (0.25–1) ổn định, **w lớn làm hại**.
- [ ] Bảng 2×2 W07 (hoặc pilot nếu W07 lỗi) cho thấy SCF cộng được lên mạng train bằng lượng tử.
- [ ] Không có câu "quantum advantage" hay "SOTA". Số đã công bố luôn kèm chú thích về cách chia dữ liệu.
- [ ] Mỗi bảng và hình được nhắc trong text. Tổng ≤ 10 trang.

---

## 8. Nếu bị trễ hoặc vượt trang

| Tình huống | Cách xử lý |
|---|---|
| Vượt 10 trang | Cắt theo thứ tự: chứng minh mệnh đề (giữ phát biểu) → dòng Holevo/POVM trong bảng twin → thu nhỏ hình gain → bớt đoạn (a) của Related Work |
| W07 lỗi hoặc chưa xong trước T5 12/11 | Dùng số pilot Holevo/POVM + SCF, ghi rõ là pilot 1 run; đưa QKSR + SCF vào Future work |
| Chưa có dòng "không có κ" cho CIFAR/CUB | Chỉ báo ImageNet-R và ImageNet-A, ghi rõ trong chú thích |
| Một phần trễ hạn ngày | Người đọc chéo hỗ trợ hôm sau; Thuận điều phối lại |
| Review double-blind | Xoá tên tác giả, thay link repo bằng link ẩn danh |

**Để dành cho sau:** chạy SCF trên cách chia dữ liệu của tác giả RSIAT; áp SCF lên baseline khác; chạy W07 trên thêm dataset và seed.

---

## 9. Tài nguyên trong repo

| Cần gì | Ở đâu |
|---|---|
| Bản ACL (nguồn chữ và số liệu) | `paper_acl/sections/*.tex`, `paper_acl/references.bib` |
| **Khung paper ICISN** (template Springer chính thức, TODO theo tên) | `paper_icisn/main.tex`; zip cho Overleaf: `dist/paper_icisn_overleaf.zip` |
| Giải thích pipeline cho người mới | `paper_acl/PIPELINE_cho_nhom.md` |
| Kết quả SCF (4 dataset × 3 seed, mọi giá trị w) | `exps/scf/results/summary.json`, `exps/scf/README.md` |
| Pilot lượng tử lúc train + SCF | `exps/holevo_povm/results/kaggle1/logs/` (các dòng `QSD curve head=`) |
| Hình đã có | `paper_acl/figures/` (`slide_scf_gain.png`, `slide_scf_accuracy.png`, `frontier.pdf`, `pipeline.pdf`) |
| Script vẽ hình | `paper_acl/make_figures.py`, `paper_acl/make_slide_figures.py` |
