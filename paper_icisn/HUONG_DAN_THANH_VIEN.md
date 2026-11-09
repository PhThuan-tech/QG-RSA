# Hướng dẫn công việc cho từng thành viên

> Đi kèm `KE_HOACH_NHOM.md` (lịch và phân công). File này nói **làm thế nào**: đọc gì trước, từng bước,
> nộp gì, thế nào là xong. Mỗi người đọc phần **0** và phần của mình.

---

## 0. Ai cũng cần biết (đọc 20 phút)

1. Đọc `paper_acl/PIPELINE_cho_nhom.md`: pipeline RSIAT, chỗ gắn lượng tử, SCF là gì.
2. Paper nói gì, trong 3 câu:
   - **C1, SCF:** giữ nguyên classifier của RSIAT, cộng thêm điểm tính từ thống kê lớp (μ, Σ) đã lưu sẵn,
     có hệ số κ để đưa hai thang điểm về cùng độ lớn. Không train, không lưu thêm. A_B tăng +0.38 đến +0.67 trên 4 dataset.
   - **C2, lượng tử có kiểm chứng:** nguồn điểm lượng tử (PGM) và nguồn cổ điển (ridge) cho kết quả như nhau.
     Lý thuyết dự đoán trước điều này.
   - **C3, vì sao hiệu quả:** classifier và điểm thống kê sai ở ảnh khác nhau; SCF tăng cả lớp cũ lẫn lớp mới;
     SCF cộng được lên mạng train bằng lượng tử (W07).
3. **Quy tắc chung khi viết:**
   - Nguồn chữ và số: `paper_acl/sections/*.tex`. **Viết lại bằng lời của mình**, vì yêu cầu similarity ≤ 15%.
   - Mọi con số phải có trong file kết quả của repo. Không chắc thì để `\todo{...}` và hỏi Thuận.
   - Không dùng "quantum advantage" và "state-of-the-art".
   - Ký hiệu thống nhất theo bảng của Đức Anh (có từ thứ Hai).
   - Cả nhóm viết chung **một file `main.tex`** trên Overleaf (khung có sẵn: `paper_icisn/main.tex`,
     zip upload: `dist/paper_icisn_overleaf.zip`). Mỗi chỗ cần viết có TODO màu ghi tên người:
     đỏ = Thuận, xanh dương = Khánh, xanh lục = Đức Anh, tím = Phát. Viết xong thì xoá TODO của mình.

**Ký hiệu chính** (Đức Anh chốt bản cuối):

| Ký hiệu | Nghĩa |
|---|---|
| z ∈ ℝ⁷⁶⁸ | feature sau ViT + adapter |
| ℓ_c(z) | logit của linear head cho lớp c |
| μ_c, Σ_c | mean, covariance của lớp c (memory của RSIAT) |
| ρ_c | ma trận mật độ của lớp c |
| ψ(z) | trạng thái lượng tử của ảnh (mã hoá biên độ, 769 chiều) |
| g_c(z) | điểm thống kê: log p_PGM(c\|z) hoặc g^R_c(z) của ridge |
| κ, w | hệ số hiệu chỉnh thang điểm, trọng số kết hợp |
| Ā, A_B | accuracy trung bình qua các task, accuracy sau task cuối |

---

## 1. Thuận: chạy W07, viết Abstract + Introduction, ghép bài

### 1a. Chạy wave W07 (thứ Hai)

**Mục đích:** lấp ô "QKSR + SCF" của bảng 2×2 cho C3. Spec và quy tắc chốt trước: `exps/qscf/README.md`.

| Bước | Làm gì |
|---|---|
| 1 | Notebook đã build sẵn: `kaggle/waves/w07_qscf/W07_qscf_kaggle.ipynb`. Nếu sửa code thì build lại: `python scripts/build_wave.py kaggle/waves/w07_qscf/wave.json` |
| 2 | Kaggle → *New Notebook* → *File → Import Notebook* → chọn file trên |
| 3 | *Add Input*: dataset `qgrsa-base-c7bc4fd`, dataset `zaphat206/imagenet-r`, model ViT-B/16 224 in21k (giống W04–W06) |
| 4 | *Settings*: Accelerator **GPU RTX PRO 6000**, Internet tắt cũng được (config có `offline: true`) |
| 5 | *Save Version → Save & Run All (Commit)*. 6 run, 3 song song, khoảng 3–4 giờ |
| 6 | Runner tự dừng nếu sai base, sai hash patch, test lỗi, thiếu GPU/dữ liệu. Nếu dừng, xem cell output và `runs/console/` |
| 7 | Xong: tải `results_W07_qscf.zip` → giải nén vào `exps/qscf/results/kaggle1/` |
| 8 | Chạy tổng hợp ở máy: |

```bash
python scripts/summarize_qscf.py --root exps/qscf/results/kaggle1/logs \
    --output exps/qscf/results/summary.md --json exps/qscf/results/summary.json
```

| Bước | Làm gì |
|---|---|
| 9 | Mở `exps/qscf/results/summary.md`: có bảng 2×2 (Ā, A_B), các so sánh ghép cặp, và dòng kết luận **ĐẠT / KHÔNG ĐẠT** |
| 10 | Gửi file đó cho Phát trước tối thứ Tư |

**Nếu có run lỗi:** chạy lại đúng notebook, runner bỏ qua run đã xong. Nếu đến tối thứ Tư vẫn chưa đủ 3 cặp,
báo Phát dùng phương án dự phòng (pilot Holevo/POVM, xem mục 4b).

**Không làm:** không đổi w, không thêm run sau khi thấy kết quả để "đẹp" hơn.

### 1b. Viết

| Phần | Nội dung bắt buộc | Nguồn |
|---|---|---|
| Introduction (1.25 tr) | Đoạn 1: CIL exemplar-free, memory (μ, Σ) có sẵn nhưng chỉ dùng để train lại classifier. Đoạn 2: QML thường thiếu bản cổ điển để so. Đoạn 3: ý tưởng SCF trong 3 câu. Đoạn 4: **danh sách C1–C3**, mỗi ý có số và trỏ tới section/bảng | `paper_acl/sections/introduction.tex` |
| Abstract (≤ 200 từ) | Viết **cuối cùng**, thứ Năm: bài toán → khoảng trống → C1 (số) → C2 (số) → C3 → kết luận | Bản nháp các section |

### 1c. Ghép bài (thứ Năm, thứ Bảy)

- Đếm trang. Vượt 10 thì cắt theo thứ tự trong `KE_HOACH_NHOM.md` mục 8.
- Kiểm tra mọi bảng và hình đều được nhắc trong text.
- Gửi email ban tổ chức (icisnconf@gmail.com) ngay thứ Hai: hỏi single- hay double-blind, và 10 trang có tính References không.

**Xong khi:** W07 có `summary.md`; Intro + Abstract trên Overleaf; bản ghép ≤ 10 trang biên dịch không lỗi.

---

## 2. Khánh: Related Work, Background, Conclusion + Limitations, trích dẫn

### 2a. Đọc trước (thứ Hai)

| Bài | Đọc để biết | Ghi vào bảng định vị |
|---|---|---|
| FeCAM (NeurIPS 2023) | Dùng prototype + covariance để làm gì | Thay hay giữ classifier? Có train không? |
| RanPAC (NeurIPS 2023) | Head ridge dạng đóng | Như trên |
| SLCA (ICCV 2023) | Classifier alignment bằng mẫu Gaussian | Như trên |
| SimpleCIL / APER (IJCV) | Prototype classifier với ViT pre-trained | Như trên |
| RSIAT (CVPR 2026) | Baseline của nhóm | Memory gồm những gì |
| Bowles et al. 2024 | Lập luận "phải so với classical baseline" trong QML | — |

Bảng định vị mẫu nằm trong `KE_HOACH_NHOM.md` mục 3. **Kiểm tra lại từng ô với bài gốc**; ô nào không chắc thì đánh dấu.

### 2b. Viết

| Phần | Nội dung bắt buộc |
|---|---|
| Related Work (0.8 tr) | 3 đoạn: (a) CIL với mô hình pre-trained; (b) **classifier xây từ thống kê lớp** (FeCAM, RanPAC, SLCA, NCM): đoạn quan trọng nhất; (c) QML trong continual learning và benchmark với classical twin. **Mỗi đoạn kết thúc bằng một câu "Unlike these works, we …"** |
| Background (0.5 tr) | Định nghĩa bài toán (task, lớp, không exemplar), metric Ā, A_B, F (công thức), RSIAT rút gọn: ViT đóng băng + adapter chung + cosine classifier + memory (μ, Σ) + classifier alignment. Kết bằng câu: memory là thứ **có sẵn**, SCF không thêm gì |
| Analysis 7.1: vì sao SCF giúp (0.4 tr) | Trên ImageNet-R, PGM và linear head bất đồng ở 14% ảnh test: linear đúng 5.8%, PGM đúng 3.2%, cả hai sai 5.4% (`paper_acl/sections/analysis.tex`, mục Error Analysis). Hình trade-off cũ/mới (`figures/frontier.pdf`, thu nhỏ): mọi thay đổi lúc train (kể cả lượng tử) chỉ đổi lớp mới lấy lớp cũ, fusion tăng cả hai. **Bỏ câu cũ** "uncalibrated sum gains only +0.25 / +0.44" (khác định nghĩa với bảng ablation mới; phần κ do Phát viết) |
| Conclusion + Limitations (0.5 tr) | Nhắc lại C1–C3 bằng số. Hạn chế: một baseline; cách chia dữ liệu khác số công bố; mạch nhỏ được mô phỏng; W07 chỉ 1 dataset 1 seed |

**Câu novelty gợi ý cho đoạn (b)** (viết lại bằng lời của mình): *Unlike FeCAM, RanPAC and NCM, which replace the
classifier with a statistics-based one, and SLCA, which uses the statistics to retrain it, SCF keeps the trained
classifier and adds the statistics-based score at inference, calibrated without labels.*

### 2c. Trích dẫn

- File bib của khung: `references.bib` trên Overleaf (bản sao của `paper_acl/references.bib` + FeCAM).
  BibTeX chỉ in mục được `\cite`, nên không cần xoá mục thừa; chỉ cần kiểm tra mục được dùng (khoảng 20–25).
- Với mỗi mục: đối chiếu tác giả, năm, hội nghị hoặc tạp chí, trang với bản gốc (Google Scholar, DBLP).
  Ưu tiên kiểm tra các mục có "and others", Jiang 2022, Wu 2021, MEMO.
- Thêm FeCAM (và bài nào Khánh dùng thêm), lấy BibTeX từ DBLP.
- Style `splncs04` đã cài sẵn trong khung.

**Xong khi:** Related Work, Background, 7.1, Conclusion trên Overleaf; bảng định vị đã kiểm tra; `references.bib` sạch.

---

## 3. Đức Anh: Method (4.1–4.6), ký hiệu

### 3a. Thứ Hai

- Khung Overleaf đã có sẵn (template Springer `llncs` chính thức, phương trình Method đã chép từ bản ACL).
  Thuận upload và chia sẻ; Đức Anh kiểm tra phần Method biên dịch đúng.
- Chốt bảng ký hiệu (mục 0), gửi nhóm.

### 3b. Mạch của Method: theo pipeline, lúc train → lúc dự đoán

Method kể đúng thứ tự dữ liệu đi qua mô hình, để người đọc thấy một mạch liền:

```
Đoạn mở đầu  : lượng tử vào lúc train (4.1) → lớp cũ chỉ còn (μ, Σ) → đọc (μ, Σ) như trạng thái lượng tử lúc dự đoán (4.2–4.6)
4.1  PQK lúc train : base task (steering) → incremental tasks (QKSR)          [ngắn, ~0.4 tr]
     câu chuyển ý  : "dù train thế nào, lớp cũ chỉ còn lại (μ_c, Σ_c)"
4.2  Lớp → trạng thái ρ_c ;  ảnh → ψ(z)                                     [trọng tâm từ đây]
4.3  PGM: đo ψ(z) với các ρ_c → p(c|z)
4.4  Ridge twin: cùng (μ_c, Σ_c), bản cổ điển
4.5  SCF: linear head + w·κ·(PGM hoặc ridge)
4.6  Ba mệnh đề: giải thích vì sao PGM ≈ ridge
```

### 3c. Từng mục

| Mục | Nội dung | Nguồn trong bản ACL |
|---|---|---|
| Đoạn mở đầu | 3–4 câu nối phần train với phần dự đoán (xem sơ đồ trên) | — |
| 4.1 PQK lúc train (~0.4 tr) | θ(z) = π·tanh(W·LN(z) + b), 8 qubit, R_y + vòng CNOT, kernel trên RDM (Eq. pqk, đã có). **Base task:** PQK thay cosine trong steering loss. **Incremental tasks (QKSR):** phạt trung bình kernel giữa mean lớp cũ (qua AE) và feature mới. Twin: RBF trên cùng 8 góc; bỏ CNOT. "QKSR" = bật cả hai (config `p1_qksr_cur`). Kết bằng câu chuyển ý. **Không phải đóng góp chính.** | `method.tex` mục 4.2; `PIPELINE_cho_nhom.md` mục 4 |
| 4.2 Lớp như trạng thái lượng tử | Mã hoá v(z), ψ(z); ρ_c dựng từ (μ_c, Σ_c), **không cần lưu feature** | `method.tex`, Eq. lift, state |
| 4.3 PGM | ρ̄_δ, E_c, xác suất Born p(c\|z) | Eq. pgm |
| 4.4 Ridge twin | Công thức dạng đóng, điểm g^R_c | Eq. ridge |
| 4.5 SCF | ŷ = argmax ℓ_c + w κ g_c; κ = tỉ số độ phân tán, tính trên ảnh train của task hiện tại, **không cần nhãn** | Eq. fusion |
| 4.6 Ba mệnh đề | (1) mạch unitary sau mã hoá không đổi kết quả đo; (2) co trạng thái không đổi argmax; (3) SCF bất biến khi co giãn hoặc dịch g. Chứng minh ngắn 2–3 dòng mỗi cái | `appendix.tex` mục A |

**Mỗi mệnh đề kèm 1 câu "ý nghĩa thực tế":**
- (1): train thêm mạch lượng tử sau bước mã hoá là vô ích.
- (2): kỹ thuật co trạng thái thường dùng không có tác dụng.
- (3): w có cùng ý nghĩa cho mọi nguồn điểm và mọi dataset, nên khoá được w.

**Chú ý:** 4.1 dùng PQK (kernel 8 qubit), 4.2–4.3 dùng mã hoá biên độ 10 qubit. Hai phần lượng tử khác nhau;
ghi rõ để người đọc không nhầm.

**Xong khi:** Method 4.1–4.6 viết xong, biên dịch không lỗi; bảng ký hiệu đã gửi.

---

## 4. Phát: Setup, Results, Ablation, chi phí, bảng 2×2

### 4a. Bảng ablation (đã có số)

Đã chạy sẵn: `paper_icisn/results/scf_ablation.md`. Chạy lại được bằng:

```bash
python scripts/scf_ablation.py --output paper_icisn/results/scf_ablation.md --json paper_icisn/results/scf_ablation.json
```

Script đọc 12 run SCF và score đã lưu, không cần GPU, chạy vài giây. Dòng đầu file in độ lệch khi tính lại SCF từ
score (0.27 điểm, do score lưu float16). Đây là kiểm tra, không đưa vào paper.

**Kết quả A_B (trung bình 4 dataset) và cách đọc trung thực:**

| Biến thể | A_B TB | Đọc |
|---|---:|---|
| Linear head (RSIAT) | 82.52 | mốc |
| Ridge riêng / PGM riêng | 81.14 / 80.60 | mỗi nguồn riêng **kém** linear head |
| Linear + ridge, không κ | 82.66 | **gần như không tăng**: độ phân tán điểm ridge nhỏ hơn logit 37–80 lần (κ ridge trong log) |
| Linear + PGM, không κ | 82.82 | tăng được, vì log-xác suất PGM vốn gần thang logit hơn (κ PGM chỉ 2.4–5.7) |
| SCF ridge w = 0.25 (khoá) | **83.00** | κ cần thiết với ridge |
| SCF PGM w = 0.5 (khoá) | 82.91 | |
| SCF ridge w = 1 / w = 4 | 82.61 / 81.76 | ridge **nhạy với w**: w = 1 đã về mức baseline |
| SCF PGM w = 1 / w = 4 | 82.88 / 82.03 | PGM ổn định ở 0.5–1, hỏng ở w = 4 |

**Viết 3–4 câu (khoảng 0.4 trang) theo đúng số, không nói quá:**
1. Kết hợp tốt hơn từng nguồn riêng, dù mỗi nguồn riêng kém linear head (lỗi bù nhau).
2. κ cần thiết khi thang điểm hai nguồn khác xa nhau (ridge); với PGM, κ giúp ít hơn.
3. w nhỏ đến vừa (0.25–1) cho kết quả tốt; w lớn để nguồn yếu hơn lấn át và làm giảm accuracy.
   w được khoá trên tuning group, không chọn trên tập test.

**Không viết** "w không nhạy" hay "κ luôn cần thiết": số không cho phép.

### 4b. Bảng 2×2 W07 (thứ Năm)

- Nhận `exps/qscf/results/summary.md` từ Thuận (tối thứ Tư).
- Đưa vào paper: bảng 2×2 (Ā và A_B, mean ± SD) + 2–3 câu: SCF có tăng trên cả hai mạng không; kiểm định chính
  ĐẠT hay KHÔNG ĐẠT, ghi đúng như file.
- **Phương án dự phòng** nếu W07 chưa xong: dùng pilot 1 run trên ImageNet-R seed 1993 (Ā / A_B), ghi rõ "pilot, 1 run":

| Mạng | Linear | + SCF ridge | + SCF PGM |
|---|---|---|---|
| RSIAT (`hp_diag`) | 87.25 / 82.42 | 87.66 / 82.64 | 87.60 / 82.65 |
| Holevo + POVM (`hp_full`) | 87.31 / 82.70 | 87.75 / 82.67 | 87.63 / 82.79 |
| POVM (`hp_povm`) | 87.30 / 82.80 | 87.71 / 82.83 | 87.69 / 82.87 |

Nguồn: dòng `QSD curve head=` trong `exps/holevo_povm/results/kaggle1/logs/adapter/imagenetr/0/20/hp_*.log`.

### 4c. Đo chi phí (thứ Tư, khoảng 30 phút)

```bash
python scripts/scf_cost.py --classes 100 200 --output paper_icisn/results/scf_cost.md
```

- Chạy được trên CPU (máy cá nhân). Nếu có GPU (Kaggle/Colab: clone repo rồi chạy lệnh trên) thì dùng GPU và ghi
  tên GPU, vì paper chạy trên GPU.
- Đo 3 head (linear, ridge, PGM): thời gian chuẩn bị mỗi task, thời gian chấm điểm mỗi ảnh, bộ nhớ. Không tính ViT.
- Số thử trên CPU, C = 100: ridge thêm ~0.002 ms/ảnh và 0.6 MB; PGM thêm ~0.9 ms/ảnh và ~230 MB (lưu ma trận
  769×769 cho mỗi lớp). **Đưa đúng số này vào paper** (1 câu hoặc 1 dòng bảng): ridge gần như miễn phí,
  PGM đắt hơn đáng kể về bộ nhớ. Đây là thêm một lý do để chọn bản cổ điển.

### 4d. Viết

| Phần | Nội dung | Nguồn |
|---|---|---|
| Setup (0.5 tr) | 4 dataset (số task, số lớp), 3 seed, ViT-B/16 IN-21K, GPU, so cặp + CI 95%, w khoá trên ImageNet-R seed 1993 | `paper_acl/sections/setup.tex`, `task.tex` |
| Results 6.1 | Bảng chính: RSIAT, SCF ridge, SCF PGM, Δ có CI, kèm số đã công bố (ghi chú khác cách chia dữ liệu) | `exps/scf/results/summary.md`, `paper_acl/sections/results.tex` |
| Results 6.2 | Bảng lượng tử vs twin, rút gọn còn 5–6 dòng quan trọng | Bảng 1 của bản ACL |
| Results 6.3 | Hình gain: `paper_acl/figures/slide_scf_gain.png` (hoặc vẽ lại bản PDF bằng `paper_acl/make_slide_figures.py`) | |
| 7.2 Ablation + chi phí | Mục 4a và 4c | |
| 7.3 Bảng 2×2 | Mục 4b | |

**Xong khi:** mọi bảng có số khớp file kết quả; ablation viết đúng 3 ý ở 4a; bảng 2×2 (hoặc pilot) đã chèn.

---

## 5. Đọc chéo (thứ Sáu)

Thuận → Đức Anh → Phát → Khánh → Thuận. Người đọc kiểm tra:
- Section có nói rõ nó phục vụ C1, C2 hay C3 không.
- Mỗi con số có nguồn trong repo không (hỏi người viết nếu không thấy).
- Có câu nào nói quá số liệu không ("significantly", "state-of-the-art", "always").
- Ký hiệu có khớp bảng chung không.
- Có đoạn nào chép gần nguyên văn bản ACL không (similarity).

Ghi góp ý bằng comment trên Overleaf, không sửa thẳng vào phần của người khác.

---

## 6. File liên quan

| File | Dùng cho |
|---|---|
| `paper_icisn/KE_HOACH_NHOM.md` | Lịch, phân công, novelty C1–C3, checklist |
| `paper_icisn/results/scf_ablation.md` | Bảng ablation (Phát) |
| `exps/qscf/README.md` | Spec W07 + quy tắc chốt trước (Thuận, Phát) |
| `kaggle/waves/w07_qscf/` | Notebook W07 (Thuận) |
| `scripts/summarize_qscf.py`, `scripts/scf_ablation.py`, `scripts/scf_cost.py` | Script tổng hợp và đo |
| `paper_acl/sections/*.tex` | Nguồn chữ và số |
| `paper_acl/PIPELINE_cho_nhom.md` | Giải thích pipeline |
