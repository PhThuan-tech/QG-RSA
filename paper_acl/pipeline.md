# Pipeline của hướng quantum (paper ACL)

Mô tả dạng văn bản của Figure 1, tách theo base task (t = 0) và incremental
task (t >= 1). Ký hiệu:

```
[ ... ]   thành phần (module)          (frozen)    không cập nhật trọng số
( ... )   dữ liệu / tensor             (train)     được cập nhật trong task này
  -->     luồng dữ liệu                 ==>        tín hiệu loss quay về adapter
  {Q}     vị trí đặt thành phần quantum (mỗi lần chạy chỉ bật một vị trí)
  {C}     classical twin của vị trí đó (cùng đầu vào, không có lượng tử)
```

---

## 1. Base task (t = 0)

```
(ảnh x, nhãn y ∈ Y_0)
        |
        v
+-------------------------------+
| ViT-B/16, pre-trained IN-21K  |  (frozen)
|   + shared adapter            |  (train)
+-------------------------------+
        |
        v
(feature z ∈ R^768)
        |
        +------------------------------+
        |                              |
        v                              v
+------------------+        +--------------------------------------+
| cosine classifier|        | steering loss (RSIAT)                |
| l_c = s·cos(w_c,z)|       |  độ tương tự giữa các feature:       |
+------------------+        |   - baseline: cosine                 |
        |                   |   - {Q} PQK kernel  (Eq. 1)          |
        v                   |   - {C} RBF trên cùng 8 góc          |
  CosFace loss              +--------------------------------------+
        |                              |
        +==============+===============+
                       ||
                       vv
              cập nhật adapter (SGD)

Sau khi train xong task 0:
  (feature z của Y_0) --> [tính thống kê lớp] --> memory M_0 = {(mu_c, Sigma_c)}
  (feature z của Y_0) --> [m_0, s_0]            (dùng để chuẩn hoá cho PGM, cố định từ đây)
  [PQK projector + circuit]                      --> đóng băng (frozen) cho các task sau
  memory M_0 --> [Gaussian sampling] --> (pseudo-feature) --> [classifier alignment]
                                                              (train lại cosine classifier)
```

---

## 2. Incremental task (t >= 1)

Chỉ có ảnh của các lớp mới Y_t; lớp cũ chỉ còn tồn tại dưới dạng memory M_{t-1}.

```
(ảnh x, nhãn y ∈ Y_t)  -- chỉ lớp mới, không exemplar
        |
        +---------------------------------------------+
        |                                             |
        v                                             v
+-------------------------------+        +-------------------------------+
| ViT-B/16 + adapter hiện tại   |        | ViT-B/16 + adapter task t-1   |
|  (adapter: train)             |        |  (frozen copy)                |
+-------------------------------+        +-------------------------------+
        |                                             |
        v                                             v
   (feature z)                                  (feature z_old)
        |                                             |
        |                                             v
        |                                   +-------------------+
        |                                   | autoencoder AE    |
        |                                   +-------------------+
        |                                             |
        |                                             v
        |                                      (AE(z_old))
        |                                             |
        +------------------+                          |
        |                  |                          |
        v                  v                          v
+--------------+   alignment loss  beta·|| z - AE(z_old) ||^2
| cosine head  |                  |
| (lớp mới)    |                  |
+--------------+                  |
        |                         |
        v                         |
  CosFace loss                    |
        |                         |
        +============+============+============+
                     ||                        ||
                     ||        +-------------------------------------------+
                     ||        | (tuỳ chọn) MỘT regulariser training-time: |
                     ||        |                                           |
                     ||        |  {Q} QKSR: mean k_PQK( AE(mu_c^old), z )  |
                     ||        |      {C} RBF trên 8 góc / circuit no-CNOT |
                     ||        |      QKSR-v2: refit projector bằng LDA    |
                     ||        |      từ M_{t-1} ở đầu mỗi task            |
                     ||        |                                           |
                     ||        |  {Q} Holevo: -lambda·chi (Eq. Holevo)     |
                     ||        |      {C} coding rate của MCR^2            |
                     ||        |                                           |
                     ||        |  {Q} residual-POVM loss                   |
                     ||        |      {C} Gaussian log-likelihood          |
                     ||        |                                           |
                     ||        |  đầu vào: z, memory M_{t-1} (lớp cũ),     |
                     ||        |  feature queue (lớp mới)                  |
                     ||        +-------------------------------------------+
                     ||                        ||
                     vv                        vv
                  cập nhật adapter (SGD) <=======

Sau khi train xong task t:
  (feature z của Y_t)  --> [tính thống kê lớp mới] --> (mu_c, Sigma_c), c ∈ Y_t
  memory M_{t-1}       --> [drift correction mean lớp cũ] --> M_{t-1} đã dịch chuyển
                       --> gộp --> memory M_t = {(mu_c, Sigma_c)}, c ∈ C_t
  memory M_t --> [Gaussian sampling] --> (pseudo-feature) --> [classifier alignment]
                                                              (train lại cosine head
                                                               cho tất cả C_t lớp)
```

---

## 3. Inference sau mỗi task (mọi head dùng chung một mạng)

Không train gì thêm, không lưu thêm gì ngoài memory M_t. Tất cả các head
được đánh giá trên cùng test feature z.

```
(ảnh test x, lớp bất kỳ trong C_t)
        |
        v
+-------------------------------+
| ViT-B/16 + adapter (frozen)   |
+-------------------------------+
        |
        v
   (feature z) ------------------------------+------------------------------+
        |                                    |                              |
        v                                    v                              v
+------------------+        +------------------------------+   +--------------------------+
| linear head      |        | {Q} PGM                      |   | {C} ridge twin           |
| (cosine, RSIAT)  |        |  z --> v=[(z-m_0)/s_0; 1]    |   |  M_t --> giải ridge      |
|  --> l_c(z)      |        |    --> psi = v/||v||         |   |  (Eq. ridge) --> W       |
+------------------+        |    (769 chiều, 10 qubit)     |   |  g^R_c(z) = z^T W_c      |
        |                   |  M_t --> rho_c (Eq. state)   |   +--------------------------+
        |                   |    --> rho_bar + delta·I     |                |
        |                   |    --> E_c  (Eq. PGM)        |                |
        |                   |  g_c(z) = log p(c|z)         |                |
        |                   +------------------------------+                |
        |                                    |                              |
        |                                    +---------- chọn MỘT ----------+
        |                                                 |
        |                                                 v
        |                                            (g_c(z))
        |                                                 |
        +---------------------+---------------------------+
                              |
                              v
             +-----------------------------------------+
             | SCF: statistics-calibrated fusion       |
             |  y = argmax_c  l_c(z) + w·kappa·g_c(z)  |
             |  kappa = sd(l) / sd(g)   (không nhãn)   |
             |  w = 0.25 (ridge), 0.5 (PGM)            |
             +-----------------------------------------+
                              |
                              v
                     (nhãn dự đoán y_hat)

Vị trí inference-time khác (không dùng trong SCF):
  {Q} PQK làm feature map:  z --> [PQK, frozen] --> [ridge head] --> y_hat
      {C} random Fourier features / random projection / circuit no-CNOT
```

---

## 4. Chi tiết hai khối quantum

### PQK (dùng cho steering, QKSR, feature map)

```
(z, 768) --> [LayerNorm] --> [Linear 768 -> 8] --> [pi·tanh] --> (theta, 8 góc)
         --> [encoding R_y(theta_j) trên 8 qubit]
         --> [2 lớp: R_y huấn luyện được + vòng CNOT]
         --> [RDM 1 qubit rho_j(z), j = 1..8]
         --> k(z, z') = exp( -gamma · sum_j || rho_j(z) - rho_j(z') ||_F^2 )   (Eq. 1)

QKSR-v2: bỏ LayerNorm; hàng của Linear = hướng Fisher-LDA từ M_{t-1};
         độ lệch góc 0.8 rad; gamma sao cho median kernel = 0.2.
```

### PGM (dùng cho inference và SCF)

```
memory M_t = {(mu_c, Sigma_c)}
    --> chuẩn hoá bằng (m_0, s_0) --> (mu~_c, Sigma~_c)
    --> rho_c = E_c[v v^T] / E_c||v||^2                                       (Eq. state)
    --> rho_bar = (1/C_t) sum_c rho_c ;  rho_bar_delta = rho_bar + delta·I
    --> E_c = (1/C_t) rho_bar_delta^(-1/2) rho_c rho_bar_delta^(-1/2)       (Eq. PGM)

(z) --> psi(z) --> p(c|z) = psi^T E_c psi / sum_c' psi^T E_c' psi       (xác suất Born)
```

---

## 5. Tóm tắt: thành phần nào thay đổi gì

| Vị trí | Khi nào | Thay đổi gì | Twin | Kết quả |
|---|---|---|---|---|
| PQK steering | base task | loss của adapter | cosine / RBF | +0.06 [−0.07, +0.18] so với cosine |
| QKSR (PQK separation) | incremental | loss của adapter | RBF / no-CNOT | chỉ thắng no-CNOT (+0.04) |
| QKSR-v2 | incremental | loss + refit projector | RBF | kém QKSR −0.07 |
| Holevo / residual POVM | incremental | loss của adapter | MCR² / Gaussian | dừng ở vòng screening |
| PQK feature map | inference | head mới | RFF / random proj. | −17.3 so với linear head |
| PGM | inference | head mới | ridge | ngang ridge |
| SCF (PGM hoặc ridge) | inference | fusion với linear head | ridge ↔ PGM | +0.38 đến +0.67 A_B, nguồn nào cũng như nhau |
