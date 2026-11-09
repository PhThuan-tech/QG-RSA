# Hướng quantum cho Class-Incremental Learning: đọc trong 10 phút

> Tài liệu cho thành viên nhóm. Đọc xong bạn sẽ biết: bài toán là gì, mô hình chạy
> thế nào qua từng bước, lượng tử được gắn vào đâu, kết quả ra sao, và chỗ nào
> cần hỗ trợ. Paper đầy đủ: `paper_acl/` (tiếng Anh, format ACL).

---

## 1. Câu hỏi của paper (1 câu)

**Gắn các thành phần lượng tử vào một mô hình học liên tục mạnh có giúp tăng
accuracy thật không, khi so công bằng với phiên bản cổ điển tương ứng?**

Câu trả lời ngắn: **không đáng kể**. Thứ thật sự giúp là một phép *fusion điểm số*
và bản cổ điển của nó làm tốt ngang bản lượng tử.

---

## 2. Bài toán: Exemplar-free Class-Incremental Learning (CIL)

- Mô hình học **10 task nối tiếp nhau**. Mỗi task mang đến một nhóm **lớp mới**
  (ví dụ ImageNet-R: 200 lớp, chia 10 task × 20 lớp).
- Khi học task mới, **không được giữ lại ảnh của task cũ** (exemplar-free).
  Cái duy nhất được giữ về lớp cũ là **thống kê**: vector trung bình μ và ma trận
  hiệp phương sai Σ của feature mỗi lớp.
- Sau mỗi task, mô hình phải phân loại **tất cả các lớp đã gặp**.
- Khó khăn chính: **quên** lớp cũ và **nhầm lẫn giữa lớp cũ với lớp mới**.

**Thước đo**

| Ký hiệu | Ý nghĩa |
|---|---|
| Ā | Accuracy trung bình sau mỗi task (càng cao càng tốt) |
| A_B | Accuracy sau task cuối cùng |
| F | Mức quên trung bình |
| old / new half | Accuracy cuối trên 5 task đầu / 5 task sau, để xem đánh đổi |

**Dữ liệu:** CIFAR-100, ImageNet-R, ImageNet-A, CUB-200; mỗi dataset chạy 3 seed
(1993, 2015, 2026).

---

## 3. Mô hình gốc: RSIAT (baseline, không có lượng tử)

Ba khối chính:

| Khối | Vai trò | Có train không? |
|---|---|---|
| **ViT-B/16** pre-trained | Trích feature từ ảnh | Không (đóng băng) |
| **Adapter** (dùng chung cho mọi task) | Lớp nhỏ gắn vào ViT để thích nghi | Có |
| **Cosine classifier** ("linear head") | Cho điểm mỗi lớp từ feature `z` (768 chiều) | Có |
| **Memory** `(μ_c, Σ_c)` | Thống kê của mỗi lớp đã học | Tính lại sau mỗi task |
| **Autoencoder (AE)** | Nối feature mạng cũ sang mạng mới để giữ kiến thức | Có |

### 3.1. Task đầu tiên (base task)

```
 ảnh ──► ViT (đóng băng) + Adapter (train) ──► feature z ──► Cosine classifier ──► CosFace loss
                                                   │
                                                   └──► Steering loss (giữ cấu trúc feature)
 Học xong:
   • tính (μ_c, Σ_c) cho mỗi lớp  ──► lưu vào Memory
   • lấy mẫu Gaussian từ Memory   ──► train lại classifier ("classifier alignment")
```

### 3.2. Các task sau (incremental task)

```
 ảnh lớp MỚI ─┬─► ViT + Adapter hiện tại (train) ──► z ──► CosFace loss (lớp mới)
              │                                      │
              └─► ViT + Adapter task trước (đóng băng) ──► z_old ──► AE ──► so khớp với z
                                                                         (alignment loss:
                                                                          chống quên)
 Học xong:
   • tính (μ_c, Σ_c) cho lớp mới, dịch chuyển μ của lớp cũ theo mạng mới (drift correction)
   • lấy mẫu Gaussian từ toàn bộ Memory ──► train lại classifier cho TẤT CẢ lớp
```

### 3.3. Dự đoán (inference)

```
 ảnh test ──► ViT + Adapter ──► z ──► Cosine classifier ──► lớp dự đoán
```

---

## 4. Lượng tử được gắn vào đâu?

Mọi mạch lượng tử đều **mô phỏng trên GPU** (≤ 11 qubit). Ta không hỏi về
"lợi thế lượng tử" mà hỏi: *cách mô hình hoá kiểu lượng tử có giúp ích không?*

Hai công cụ lượng tử được dùng:

- **PQK (projected quantum kernel):** nén `z` thành 8 góc → đưa vào mạch 8 qubit
  (xoay R_y + CNOT) → đo trạng thái từng qubit → ra một **độ tương tự** giữa hai feature.
- **PGM (pretty-good measurement):** biến mỗi lớp thành một **ma trận mật độ** ρ_c
  (dựng từ μ_c, Σ_c), biến ảnh thành trạng thái lượng tử ψ, rồi "đo" để ra
  **xác suất thuộc mỗi lớp**.

Có **7 vị trí** gắn thử, chia hai nhóm. Mỗi lần chạy chỉ bật **một** vị trí.

### Nhóm A: lúc train (thay đổi adapter, nên phải train lại cả mô hình)

Luồng chung: `ảnh → ViT + Adapter → feature z → loss của RSIAT + loss lượng tử → cập nhật Adapter`

- **① PQK làm độ đo cho steering loss** (chỉ ở task đầu)
  - Thay chỗ: độ tương tự cosine trong steering loss của RSIAT.
  - Luồng: `z → PQK → độ tương tự giữa các feature → steering loss`
- **② QKSR: PQK làm regulariser tách lớp** (các task sau)
  - Mục đích: đẩy feature lớp mới ra xa các lớp cũ.
  - Luồng: `μ lớp cũ (Memory) → AE → PQK(μ cũ, z mới) → phạt nếu giống nhau`
  - Bản sửa lỗi **QKSR-v2**: mỗi task tính lại phép chiếu 8 góc từ Memory (Fisher-LDA).
- **④ Holevo χ làm loss**
  - Mục đích: làm các lớp "phân biệt được" theo nghĩa lượng tử.
  - Luồng: `Memory + feature lớp mới → ma trận mật độ ρ_c → χ → cộng −λ·χ vào loss`
- **⑤ POVM loss**
  - Mục đích: buộc lớp mới nằm ở phần không gian mà lớp cũ chưa chiếm.
  - Luồng: `z lớp mới → phép đo POVM (dựng từ ρ lớp cũ) → loss phân loại`

### Nhóm B: lúc dự đoán (không train lại, dùng chung một mạng đã train)

Luồng chung: `ảnh test → ViT + Adapter (đóng băng) → feature z → một head mới → lớp dự đoán`

- **③ PQK làm feature map**
  - Luồng: `z → PQK → feature lượng tử → ridge head → lớp dự đoán`
- **⑥ PGM đứng một mình**
  - Luồng: `Memory (μ_c, Σ_c) → ρ_c` và `z → ψ` → `phép đo PGM → xác suất mỗi lớp → lớp dự đoán`
- **⑦ PGM fusion (SCF)**
  - Luồng: `điểm của cosine classifier + điểm PGM → cộng có hiệu chỉnh → lớp dự đoán`
  - Xem chi tiết ở mục 5.

### Quy tắc "classical twin" (quan trọng nhất)

Mỗi thành phần lượng tử được so với một **bản cổ điển nhận đúng cùng đầu vào**:

| Thành phần lượng tử | Bản cổ điển (twin) |
|---|---|
| PQK | Kernel RBF trên cùng 8 góc; cùng mạch nhưng bỏ CNOT |
| Holevo χ | Coding rate của MCR² |
| POVM loss | Bộ phân loại Gaussian |
| PGM | Hồi quy ridge trên cùng (μ_c, Σ_c) |

Lý do: nếu chỉ so với baseline, mọi cải thiện đều có thể đến từ *thông tin thêm*
(ví dụ dùng Memory) chứ không phải từ *lượng tử*. So với twin mới tách được hai thứ.

---

## 5. Thành phần duy nhất giúp ích: SCF (fusion)

Luồng: `ảnh test → z → [điểm classifier] + [điểm thống kê] → cộng có hiệu chỉnh → argmax → lớp dự đoán`

Các thành phần:

- **Điểm classifier ℓ_c(z):** điểm của cosine classifier RSIAT cho từng lớp c.
- **Điểm thống kê g_c(z):** tính từ Memory (μ_c, Σ_c), chọn một trong hai nguồn:
  - **PGM** (lượng tử): log xác suất Born của lớp c.
  - **Ridge** (cổ điển, twin): hồi quy ridge trên cùng (μ_c, Σ_c).
- **Hệ số κ:** đưa hai điểm về cùng thang, tự tính, không cần nhãn. Thiếu κ thì fusion gần như vô dụng.
- **Trọng số w:** chọn một lần trên ImageNet-R seed 1993 (ridge 0.25, PGM 0.5) rồi giữ nguyên cho mọi dataset.
- **Công thức:** `lớp dự đoán = argmax_c [ ℓ_c(z) + w · κ · g_c(z) ]`

Vì sao giúp được:

- **Không train thêm, không lưu thêm gì.** Chỉ cộng hai điểm số lúc dự đoán.
- Classifier và điểm thống kê **sai ở những ảnh khác nhau** (bất đồng 14% ảnh trên
  ImageNet-R), nên gộp lại sửa được một phần lỗi.

---

## 6. Kết quả chính

**Lượng tử vs twin (Δ Ā, khoảng tin cậy 95%)**

| Vị trí | Kết quả |
|---|---|
| ① PQK steering | +0.06 [−0.07, +0.18], không rõ |
| ② QKSR vs RBF / vs bỏ CNOT | +0.07 (không rõ) / **+0.04** (duy nhất > 0, rất nhỏ) |
| ② QKSR-v2 (bản sửa lỗi) | −0.07 so với QKSR, tệ hơn |
| ③ PQK feature map | −17.3 so với classifier, rất tệ |
| ④ ⑤ Holevo, POVM | Dừng ở vòng thử, không đạt ngưỡng |
| ⑥ PGM vs ridge | +0.09 [−0.23, +0.40], ngang nhau |
| ⑦ Fusion: PGM vs ridge | −0.04 [−0.17, +0.08], ngang nhau |

**Fusion so với RSIAT (accuracy cuối A_B, trung bình 3 seed)**

| | CIFAR-100 | ImageNet-R | ImageNet-A | CUB-200 |
|---|---|---|---|---|
| RSIAT | 92.03 | 82.55 | 67.02 | 88.47 |
| SCF, ridge (cổ điển) | **92.41** | 83.03 | **67.69** | **88.85** |
| SCF, PGM (lượng tử) | 92.34 | **83.07** | 67.65 | 88.57 |

**Ba bài học**

1. **Đo nhiễu trước khi tuyên bố cải thiện.** Một kết quả cũ +0.36 của nhóm còn
   +0.06 khi chạy lặp lại. Chạy lại chính RSIAT, Ā đã dao động 0.14–0.38 điểm (SD),
   nên so một run với một run không phân biệt được chênh lệch dưới ~0.5 điểm.
2. **Luôn so với classical twin.** PGM fusion thắng RSIAT nhưng không thắng ridge.
3. **Mọi thay đổi lúc train đều là đánh đổi.** Trên 61 run đã lưu: tăng lớp cũ thì
   giảm lớp mới (≈ 0.49 điểm mới mất cho mỗi điểm cũ được). Chỉ fusion lúc dự đoán
   tăng được cả hai.

---

## 7. Thuật ngữ nhanh

| Thuật ngữ | Giải thích ngắn |
|---|---|
| Adapter | Vài lớp nhỏ gắn vào ViT, chỉ phần này được train |
| CosFace | Loss phân loại dùng cosine, có margin |
| Classifier alignment | Train lại classifier bằng feature giả lấy mẫu từ Gaussian (μ, Σ) |
| Drift correction | Dịch chuyển μ lớp cũ cho khớp với adapter đã thay đổi |
| Ma trận mật độ ρ | Cách biểu diễn một "trạng thái lượng tử hỗn hợp", ở đây = tóm tắt một lớp |
| Xác suất Born | Xác suất ra kết quả c khi đo trạng thái ψ |
| Paired difference | So hai phương pháp trên cùng seed/lần chạy rồi lấy hiệu, giảm nhiễu |
| Pre-registered | Ghi tiêu chí thành công *trước* khi chạy, tránh chọn kết quả đẹp |

---

## 8. Nhóm có thể hỗ trợ gì

Các việc còn mở trong paper (đánh dấu `\todo` màu đỏ):

- [ ] **Ví dụ định tính:** lấy một số ảnh test ImageNet-R mà classifier và fusion
      dự đoán khác nhau (truy lại ảnh theo test index từ split cố định).
- [ ] **Tổng số giờ GPU** của các lần chạy, tính từ log.
- [ ] **Kiểm tra trích dẫn:** các mục có "and others" trong `references.bib`,
      đặc biệt Jiang 2022, Wu 2021, và năm/hội nghị của MEMO.
- [ ] **Biên dịch trên Overleaf** và kiểm tra phần thân ≤ 8 trang.
- [ ] Đọc góp ý văn phong tiếng Anh cho Introduction và Results.

Muốn đọc sâu hơn: `paper_acl/pipeline.md` (sơ đồ chi tiết từng tensor),
`exps/README.md` (danh mục mọi thí nghiệm và kết quả).
