# Holevo-POVM giai đoạn A: bộ mã hoá Holevo + bộ giải mã Residual-POVM

Đặc tả này được viết **trước khi code và trước khi chạy**, ngày 2026-10-30. Quy tắc bên dưới là cố định. Mọi
thay đổi sau khi thấy kết quả phải ghi vào mục "Nhật ký thay đổi".

- Quy ước seed: `exps/README.md`.
- **Phụ thuộc:** `exps/holevo/README.md` (hướng A). Spec này **dùng lại nguyên văn** từ đó các phần sau:
  - bộ mã hoá v(z), D = 769 (mục 2.1);
  - trạng thái lớp cũ từ thống kê và queue lớp mới (mục 2.2);
  - χ, cách tính entropy và surrogate (mục 2.3–2.4);
  - hiệu chỉnh λ theo tỉ số gradient (mục 3.3);
  - bước trước khi chạy và ngưỡng δ_M (mục 3.5);
  - chẩn đoán χ_oracle và luật G (mục 4.1, 4.3).

  Mục nào không nhắc lại ở đây thì theo spec A.
- Hạ tầng dùng lại: head `pgm_stats`, `ridge_stats` và fusion của `utils/qsd.py` (vòng A2, `exps/qsd/README_A2.md`).
- Kiểm tính mới: `research_notes/Tính mới Residual POVM cho CIL/`.
  - Chưa tìm thấy công trình nào **phát triển POVM tăng dần trong phần dư I − Σ E_old** để train bộ trích đặc
    trưng cho EFCIL.
  - Gần nhất: Born-rule loss (CNM, Guidotti & Ferrara); classifier PGM hoặc Helstrom (nhóm Sergioli); FeCAM
    (Gaussian logit, chỉ lúc suy luận); IL2A (softmax trên Gaussian lúc train); Adam-NSCL (null space theo
    covariance cũ, áp lên gradient).

## 1. Vì sao ghép A với B

**Bằng chứng nội bộ** (chi tiết ở spec A, mục 1): mất mát nằm ở biểu diễn, là chồng lấn giữa các task (pair ratio
1.38 so với joint 0.90, task 0–1 thấp hơn joint 9.0 và 8.3). Loss lúc train lại không nhìn thấy lớp cũ (F5).

**Hai lý do để ghép, từ review của spec A** (nhật ký spec A, 2026-10-30):
1. **Cận lý thuyết chỉ chặt khi bộ giải mã là phép đo lượng tử.**
   - Cận Holevo + Fano, P_err ≥ (log C − χ − log 2)/log(C − 1), chỉ áp dụng cho POVM trên một bản sao.
   - Với A một mình (bộ giải mã CA), χ chỉ là proxy.
   - Khi bộ giải mã là POVM thì χ thật sự giới hạn lỗi của chính bộ giải mã được huấn luyện. Khung "CIL là truyền
     tin lượng tử tăng dần" khi đó khép kín: bộ mã hoá tăng χ, bộ giải mã là phép đo.
2. **Tính mới.**
   - Bản classical tương đương của A (rate reduction tăng dần với covariance lưu, Wu et al. arXiv 2011.14593) đã
     được công bố.
   - Phần chưa ai làm là bộ giải mã Residual-POVM: một phép đo tăng dần, đầy đủ theo cấu trúc, dành phần dư cho
     lớp mới.

**Khác Q-BRIDGE R1 và CAS** (đều đã thất bại: R1 cho −3.47 final trên ImageNet-A):
- Lớp cũ đi vào qua phần tử POVM dựng từ mean + cov, ở mức phân bố, không qua hàng classifier.
- Xác suất của lớp cũ được **chuẩn hoá chung** với lớp mới nhờ ràng buộc đầy đủ. Không có margin cứng.
- Phần dư mềm R chỉ phạt mẫu mới rơi vào các hướng mà lớp cũ thật sự chiếm.

## 2. Công thức

### 2.1. Trạng thái

Theo spec A, mục 2.1–2.2:
- v(z) = [(z − m₀)/s₀ ; 1] (D = 769, không chuẩn hoá).
- ρ_c lớp cũ = `stats_states` (hằng số trong task t, detach).
- ρ_c lớp mới từ queue FIFO, trọng số ‖v‖².

Mỗi mẫu x có **trạng thái thuần** ρ_x = ψψᵀ, với ψ = v(z)/‖v(z)‖ (mang gradient).

### 2.2. Phần tử POVM lớp cũ (cố định trong task t)

Dùng PGM có điều chuẩn δ trên lớp cũ, với prior đều p_c = 1/K:

```
ρ̄_old = (1/K) Σ_{c<K} ρ_c
E_c   = (ρ̄_old + δI)^{-1/2} (ρ_c / K) (ρ̄_old + δI)^{-1/2}          c < K
Σ_{c<K} E_c = ρ̄_old (ρ̄_old + δI)^{-1}                               (giao hoán)
```

**Phần dư mềm:**
```
R = I − Σ_{c<K} E_c = δ (ρ̄_old + δI)^{-1}  ⪰ 0
```
- Trị riêng của R là δ/(λ_i + δ):
  - gần 0 ở các hướng lớp cũ chiếm mạnh (λ_i ≫ δ);
  - gần 1 ở các hướng lớp cũ chưa chiếm (λ_i ≪ δ).
- R **không** phải phép chiếu. Paper phải gọi là "soft residual", không gọi là orthogonal complement.
- R và R^{1/2} = √δ (ρ̄_old + δI)^{-1/2} chỉ cần **một** `eigh` float64 769×769 mỗi task.

**Chọn δ** (ngân sách phần dư): δ = τ / D, với τ = `hp_tau` = 1, tức trị riêng trung bình của ρ̄_old (vết 1).
- τ lớn thì phần dư lớn: lớp mới dễ chiếm chỗ, lớp cũ được bảo vệ ít hơn.
- Không quét τ ở wave sàng lọc. Lưới τ ∈ {0.3, 3} chỉ chạy khi luật M đạt mà luật O trượt.

### 2.3. Phần tử POVM lớp mới (trong phần dư, đầy đủ theo cấu trúc)

Với lớp mới j ≥ K, prior đều q_j = 1/n_new, và ρ_j từ queue:

```
σ_j = R^{1/2} ρ_j R^{1/2}                                     (trạng thái lớp mới nhìn qua phần dư)
Ω   = Σ_j q_j σ_j + ε_Ω I
M_j = Ω^{-1/2} (q_j σ_j) Ω^{-1/2}
E_j = R^{1/2} M_j R^{1/2}
```

- Σ_j M_j = I − ε_Ω Ω^{-1}. Lượng thiếu ε_Ω Ω^{-1} được gom thành **một phần tử "không kết luận"** E_∅ = R^{1/2}
  ε_Ω Ω^{-1} R^{1/2}. Như vậy Σ_{c<K} E_c + Σ_j E_j + E_∅ = I **chính xác** ở mọi bước.
- E_∅ không ứng với lớp nào. Xác suất rơi vào E_∅ được log để theo dõi; nó được tính vào mẫu số, nên làm giảm
  p(y|x).
- **Gradient:** Ω^{-1/2} được tính bằng `eigh` mỗi bước rồi **detach** (stop-gradient), giống detach thống kê
  chuẩn hoá.
  - Gradient đi qua σ_j (queue, phần tử của batch) và qua ρ_x.
  - Ràng buộc đầy đủ vẫn đúng chính xác, vì Ω được dựng từ đúng các σ_j của bước đó.
  - Lý do: đạo hàm qua `eigh` khi trị riêng suy biến cho NaN; các σ_j lớp mới có hạng thấp.
- ε_Ω = `hp_eps_omega` = 1e-6 · tr(Ω₀)/D.

### 2.4. Xác suất đo và loss B

Với mẫu mới (x, y), y ≥ K:

```
p(c | x) = Tr(E_c ρ_x) = ψᵀ E_c ψ                       với mọi c < K, mọi j ≥ K và ∅
Σ_c p(c | x) = 1                                         (đúng theo cấu trúc)
L_B = − log( p(y | x) + 1e-12 )
```

- Cơ chế bảo vệ lớp cũ:
  - Khối lượng Σ_{c<K} p(c|x) = ψᵀ ρ̄_old(ρ̄_old + δI)^{-1} ψ bị phạt **tự động** qua mẫu số, không cần margin.
  - Gradient đẩy ψ vào các hướng mà R lớn, tức nơi lớp cũ chưa chiếm.
  - Đây là bản "cứng hơn" của gradient −log ρ̄ trong hướng A (spec A, mục 2.3).
- **Lưu ý cho reviewer:** với ρ_x hạng một, ψᵀ E_c ψ là một dạng toàn phương đã chuẩn hoá. Điểm khác FeCAM và
  Mahalanobis là:
  1. ràng buộc đầy đủ dành phần dư theo cấu trúc;
  2. đưa vào **loss lúc train**, không chỉ lúc suy luận.

  Bản classical tương đương ở mục 2.6 đo đúng điểm này.

### 2.5. Loss tổng

Chỉ áp dụng ở task t ≥ 1. **Task 0 giữ nguyên RSIAT.**

```
L = CosFace_new + β·MSE(f, AE(f_old)) + γ·orth             (RSIAT, giữ nguyên)
    − λ_A · w(e) · χ̃                                        (hướng A, spec A mục 2.3)
    + λ_B · w(e) · L_B                                       (hướng B, mục 2.4)
w(e) = min(1, e / warmup_epoch)
```

- Giữ CosFace_new và fc, vì CA sau mỗi task vẫn cần fc.
- **χ và L_B dùng chung trạng thái** (cùng v, cùng queue, cùng ρ_c lớp cũ). Chỉ cần một bộ queue, và các `eigh`
  được chia sẻ khi có thể.
- **Hiệu chỉnh λ** (cùng quy tắc với spec A, mục 3.3, ở epoch 0 của task 1 với w = 0):
  - Overlay có hai objective: mỗi objective được hiệu chỉnh tới r* = 0.25.
  - Overlay có một objective: r* = 0.5.
  - Như vậy **tổng ngân sách gradient phụ bằng nhau** (0.5) giữa các overlay.

### 2.6. Bộ giải mã lúc suy luận

Báo cả hai, cùng một run:

| Head | Định nghĩa | Vai trò |
|---|---|---|
| `linear` | CA của RSIAT (fc sau stage 2) | metric chính, so với baseline |
| `pgm_stats` | PGM trên mọi lớp, dựng từ thống kê đã lưu (code A2). Trong code, δ_qsd = `qsd_delta`/1024 (kích thước thanh ghi danh nghĩa của 769 chiều). Để khớp δ = τ/769 của mục 2.2, đặt `qsd_delta` = τ·1024/769 ≈ **1.33** | bộ giải mã lượng tử; cận Holevo + Fano áp dụng ở đây |
| `fuse_pgm_stats_b0p5` | fusion linear + PGM với trọng số đã khoá từ A2 | so với SCF của paper cũ |
| `ridge_stats`, `fuse_ridge_stats_b0p25` | đối chứng cổ điển của A2 | luật Q ở bộ giải mã |

Overlay bật `qsd_heads` như `exps/scf/scf.json`: lưu score và **không** đổi huấn luyện.

### 2.7. Lý thuyết dùng cho paper (phạm vi chính xác)

1. **Ràng buộc đầy đủ theo cấu trúc** (mệnh đề cần chứng minh trong paper): với mọi trạng thái lớp mới và mọi
   δ > 0, các phần tử E_c (c < K), E_j và E_∅ đều nửa xác định dương và có tổng bằng I. Chứng minh:
   - R ⪰ 0 vì δ/(λ + δ) ∈ (0, 1];
   - M_j ⪰ 0, và Σ M_j + ε_Ω Ω^{-1} = I.
2. **Cận Holevo + Fano** áp dụng cho bộ giải mã `pgm_stats` và cho phép đo dùng lúc train:
   ```
   P_err ≥ (log C − χ − log 2) / log(C − 1)
   ```
   **Lưu ý về độ chính xác:** cận áp dụng cho ensemble trạng thái thật ρ_c^true = E_{z|c}[ψψᵀ]. Trạng thái từ
   thống kê là E[vvᵀ]/E‖v‖², tức có trọng số ‖v‖², và chỉ trùng ρ_c^true khi ‖v‖ không đổi trong lớp.
   - Phải đo và báo khoảng cách vết ½‖ρ_c − ρ_c^true‖₁ trên lớp mới (có mẫu) ở mỗi task.
   - Nếu trung bình > 0.1, cận chỉ được trình bày là cận xấp xỉ.
3. **CA tuyến tính** (`linear`): χ chỉ là proxy, như spec A mục 2.3. Không được viết rằng χ ràng buộc CA.
4. **Bất biến unitary:** χ và phân bố đo của PGM đều bất biến dưới phép quay unitary chung của bộ mã hoá. Gain
   chỉ có thể đến từ adapter.

## 3. Thiết lập

### 3.1. Overlay (wave `W03_hp`, ImageNet-R 1993)

| Overlay | λ_A | λ_B | Câu hỏi |
|---|---|---|---|
| `hp_full` | auto (r* 0.25) | auto (r* 0.25) | **phương pháp chính**: A + B |
| `hp_povm` | 0 | auto (r* 0.5) | B một mình có đủ không? A có thêm gì? |
| `hp_twin` | MCR² auto (r* 0.25) | Gaussian auto (r* 0.25) | bản classical tương đương của A + B (**luật Q_AB**) |
| `hp_diag` | 0 | 0 | baseline chẩn đoán: huấn luyện = RSIAT, cùng head PGM δ, cùng log I_orc (mục 3.4) |

**Bản classical tương đương `hp_twin`:**
- **Bộ mã hoá:** coding rate R_ε thay cho χ, như overlay `mcr2` ở spec A.
- **Bộ giải mã:** softmax trên log-likelihood Gaussian, dạng IL2A/FeCAM, đưa vào loss lúc train.
  - Lớp cũ: N(μ_c, Σ_c + ηI) đóng băng, từ thống kê.
  - Lớp mới: N(μ_j, Σ_j + ηI) từ cùng queue, mang gradient qua phần tử của batch.
  - η = τ · tr(Σ̄_old)/768 với cùng τ, để mức điều chuẩn tương ứng với δ.
  - Prior đều, không có phần tử không kết luận.
  - Loss: −log softmax_y(log N(z; μ_c, Σ_c + ηI)) trên mọi lớp đã thấy.
- Như vậy `hp_twin` có cùng thông tin (mean + cov lớp cũ, queue lớp mới), cùng ngân sách gradient, chỉ khác:
  1. entropy von Neumann so với log-det;
  2. phép đo đầy đủ với phần dư so với softmax Gaussian.

**Ưu tiên khi GPU hạn chế:** `hp_full` → `hp_diag` → `hp_twin` → `hp_povm`.

### 3.2. Khoá mới (mặc định tắt, `hp_mode: off` là bit-identical với RSIAT)

| Khoá | Mặc định | W03 | Ghi chú |
|---|---|---|---|
| `hp_mode` | `off` | `full` / `povm` / `twin` | |
| `hp_tau` | 1.0 | 1.0 | δ = τ/D; η = τ·tr(Σ̄_old)/768 |
| `hp_eps_omega` | 1e-6 | 1e-6 | tương đối, theo tr(Ω₀)/D |
| `hp_grad_ratio` | 0.5 | 0.5 | tổng r*, chia đều khi có hai objective |
| các khoá `holevo_*` | như spec A | như spec A | queue, eps, calib_steps, oracle_diag |
| `qsd_heads` | — | `linear, pgm_stats, ridge_stats` + fuse | `qsd_delta` = 1.33 (khớp δ = 1/769, mục 2.6) |

**Code dự kiến:**
- `utils/holevo.py`: thêm `residual_povm` (R, E_old, M_j, E_∅, xác suất) và `gauss_twin`.
- Mở rộng nhánh `_compute_rt_loss`.
- `scripts/summarize_holevo.py` đọc cả `hp_*`.
- `tests/test_holevo_povm.py`.

### 3.3. Thứ tự với spec A

- W03 chạy **sau** W02 (spec A), vì dùng chung code, queue và kiểm tra hồi quy.
- **Nếu W02 kích hoạt luật G** (χ bị lách): bỏ `hp_full`, chỉ chạy `hp_povm` và bản tương đương của nó
  `hp_twin_povm` (λ_A = 0, chỉ softmax Gaussian). Lý do: thành phần A đã được chứng minh là bị lách.
- **Nếu W02 trượt luật P** (mất plasticity): **giảm r*** của W03 xuống 0.25 tổng. Ghi vào nhật ký trước khi chạy.
- **Nếu W02 đạt hết:** chạy W03 như bảng 3.1.
- Kết quả W02 **không** được dùng để chọn τ hay bất kỳ siêu tham số nào khác ngoài hai quy tắc trên.

### 3.4. Baseline

- Trung bình 4 run `noise` (ImageNet-R 1993: final 82.59 ± 0.23, AIA 87.347 ± 0.144). Ngưỡng δ_M lấy từ spec A,
  mục 3.5.
- **Head PGM của baseline:** **không** dùng `pgm_stats` của `qsd2_rep0`, vì run đó dùng `qsd_delta` = 0.1, khác
  δ của spec này. Thay vào đó dùng run chẩn đoán `hp_diag`: `hp_mode: diag`, λ_A = λ_B = 0, cùng `qsd_heads` và
  `qsd_delta` = 1.33.
  - Huấn luyện của run này bit-identical với RSIAT, nên đây là biểu diễn RSIAT với cùng bộ giải mã.
  - Run này cũng cho baseline của I_orc (mục 4.3) và kiêm kiểm tra hồi quy.
  - `hp_diag` vì vậy là run **bắt buộc** của W03. Nếu GPU hạn chế, thứ tự là `hp_full` → `hp_diag` → `hp_twin` →
    `hp_povm`.

## 4. Quy tắc quyết định (cố định)

### 4.1. Sàng lọc (W03, ImageNet-R 1993; `hp_full` là chính, `hp_povm` là phụ)

| Luật | Điều kiện (trên head `linear`, trừ khi ghi khác) | Nếu trượt |
|---|---|---|
| **R** | task 0 trùng `noise` rep0 | lỗi code |
| **G** | median Δχ_oracle > 0 (spec A, mục 4.3), **và** tỉ lệ lấn oracle I_orc (mục 4.3) ở task cuối thấp hơn baseline | dừng overlay đó |
| **M** | pair ratio ở task cuối ≤ mean_PR − δ_M | dừng |
| **O** | Δ final (`hp_full` − noise) ≥ +0.5 | nếu M đạt: lưới τ ∈ {0.3, 3}; nếu không: dừng |
| **P** | acc trung bình 3 task cuối giảm ≤ 1.0 | dừng |
| **D** (bộ giải mã) | final của `pgm_stats` trên `hp_full` − final của `pgm_stats` trên `hp_diag` ≥ +0.5 (cùng `qsd_delta`) | không dừng; ghi là "biểu diễn không giúp riêng bộ giải mã lượng tử" |

**Đọc kết quả:**
- `hp_povm` chỉ được dùng để đọc vai trò của A. Nó **không** thay `hp_full` để vượt luật O. Chọn overlay tốt hơn
  sau khi thấy kết quả là chọn lạc quan.
- Đi tiếp sang held-out khi R, G, M, O, P của `hp_full` cùng đạt.
- Độ nhạy của luật O như spec A, mục 4.1.

### 4.2. Held-out (chỉ khi sàng lọc đạt)

- **Run:** ImageNet-R 2015, 2026 và ImageNet-A 1993, 2015, 2026 (n = 5). Mỗi run gồm `hp_full` và `hp_twin`, ghép
  cặp với RSIAT cùng seed (`qsd2_rep0`).
- **Luật H:** Δ final (`hp_full` − RSIAT) có mean ≥ +0.5, CI 95% > 0.
- **Luật Q_AB (đóng góp lượng tử của cả khung):** Δ final (`hp_full` − `hp_twin`) có CI 95% > 0.
- **Luật Q_dec (bộ giải mã):** Δ final (`pgm_stats` − `ridge_stats`) trên biểu diễn `hp_full` có CI 95% > 0. Ở A2,
  hiệu số này là −0.04 trên biểu diễn RSIAT.
- **Đọc kết quả:**
  - H đạt, Q_AB đạt: đóng góp lượng tử đứng được. Viết theo khung "CIL là truyền tin lượng tử tăng dần".
  - H đạt, Q_AB trượt: viết là "khung lý thuyết thông tin lượng tử dẫn ra objective và phép đo mới, tương đương
    cổ điển về hiệu năng". Phần mới là cơ chế phần dư đầy đủ theo cấu trúc, phải có ablation ở mục 4.4.
  - H trượt: dừng, ghi kết quả âm.

### 4.3. Đọc kèm

Mọi chẩn đoán dưới đây dùng `fork_rng` và không đổi huấn luyện.

- **Khối lượng xác suất của mẫu mới** theo epoch: Σ_{c<K} p(c|x) (lấn sang lớp cũ), p(∅|x), p(y|x).
- **Tỉ lệ lấn oracle I_orc** (chẩn đoán, đọc ảnh lớp cũ, như χ_oracle): với ảnh train thật của lớp cũ ở cuối task
  t, I_orc = trung bình Σ_{j≥K} p(j|x), dùng phép đo của bước cuối.
  - Đây là đo trực tiếp intrusion mà repo đã xác định là nút thắt.
  - Baseline: lấy từ run `hp_diag` (mục 3.4). Với λ = 0, phép đo vẫn được dựng và log mỗi bước, nhưng không
    vào loss.
- **Khoảng cách vết** ½‖ρ_c − ρ_c^true‖₁ trên lớp mới (mục 2.7).
- Trị riêng của R: phân bố δ/(λ_i + δ); tỉ lệ hướng có R > 0.5.
- χ, χ_oracle, drift, r theo task, chi phí thời gian (như spec A).

### 4.4. Ablation (chỉ chạy ở held-out nếu H đạt; R 2015 và A 1993)

| Ablation | Thay đổi | Câu hỏi |
|---|---|---|
| `hp_noresid` | E_old = PGM trên mọi lớp (old + new) dựng lại mỗi bước, không có R cố định | Phần dư cố định có cần thiết không, hay chỉ là PGM lúc train? |
| `hp_nogate_old` | bỏ E_old khỏi phép đo (chỉ M_j trên lớp mới, R = I) | Có cần lớp cũ trong phép đo không? (tương đương "Born-rule loss trên lớp mới") |
| `hp_noaffine` | bỏ vế "1" trong v | Đóng góp của thành phần vị trí (khác Wu 2021) |

### 4.5. Dự đoán nếu giả thuyết đúng

- I_orc và pair ratio giảm rõ; task 0–1 tăng về phía oracle (87) và joint (89).
- p(∅|x) nhỏ (< 0.05) sau warmup. Nếu lớn, phần dư đang bị dồn vào phần tử không kết luận.
- `hp_full` hơn `hp_povm`, nếu χ thật sự giúp bộ mã hoá.
- `pgm_stats` trên `hp_full` thu hẹp hoặc đảo khoảng cách với `ridge_stats` (A2: −0.04), vì biểu diễn được huấn
  luyện **cho** phép đo.

## 5. Kiểm tra trước khi chạy (`tests/test_holevo_povm.py`)

| Kiểm tra | Cách |
|---|---|
| `hp_mode: off` bit-identical với RSIAT; `diag` có cùng trọng số với `off` | smoke |
| E_c, E_j, E_∅ nửa xác định dương; tổng = I (sai số < 1e-6) với dữ liệu ngẫu nhiên, mọi δ | đơn vị |
| R = I − Σ E_old = δ(ρ̄_old + δI)^{-1} | so số |
| Σ_c p(c\|x) = 1 với mọi x | đơn vị |
| δ → ∞ thì R → I và E_old → 0; δ → 0 thì R → phép chiếu lên phần bù của support | giới hạn số |
| Gradient không đi qua Ω^{-1/2} (detach) nhưng có đi qua σ_j của batch và ρ_x | kiểm tra `requires_grad` và grad ≠ 0 |
| Không NaN khi σ_j hạng thấp (queue 3 mẫu) | đơn vị |
| `hp_twin` dùng cùng queue và cùng thống kê với `hp_full` | đơn vị |
| Hiệu chỉnh λ cho hai objective: mỗi objective đạt r* = 0.25 trên bước hiệu chỉnh | đơn vị |
| Head `pgm_stats` với `qsd_delta` = 1.33 trùng PGM δ của mục 2.2 khi chỉ có lớp cũ **và dùng cùng bộ mã hoá**. Head QSD dùng `Encoder.from_base` (trọng số theo mẫu), khác m₀, s₀ ở spec A, mục 2.1; test phải ép cùng encoder | so số |
| Chẩn đoán I_orc và χ_oracle không đổi RNG hay trọng số | so log hai smoke |
| Luật R/G/M/O/P/D/H/Q_AB/Q_dec tính đúng trên dữ liệu giả | `SummaryTest` |

## 6. Rủi ro đã biết

- **Trạng thái lớp cũ đã cũ** (như spec A): R dựng từ ρ̄_old ở không gian của task t−1, nên lớp cũ thật có thể bị
  dịch theo adapter. Luật G (χ_oracle và I_orc) bắt trường hợp này.
- **Phần dư quá nhỏ ở task muộn:** khi K lớn, ρ̄_old có thể phủ gần hết không gian, R → 0 ở mọi hướng, và lớp mới
  mất plasticity. Luật P bắt trường hợp này; theo dõi phổ R theo task. Nếu đúng là vậy, phương án sửa
  (**chỉ ghi, chưa dùng**) là cho τ tăng theo K.
- **Gradient mạnh hơn A:** −log p(y|x) có thể lớn khi Σ p(old) gần 1, nên dễ lặp lại kịch bản R1. Hiệu chỉnh λ ở
  epoch 0 và ramp giảm rủi ro này, nhưng không loại bỏ được.
- **Born-rule loss trên trạng thái thuần là dạng toàn phương.** Reviewer sẽ so với FeCAM và Mahalanobis. `hp_twin`
  và các ablation ở mục 4.4 là câu trả lời.
- **Chi phí:** thêm một `eigh` 769 cho Ω mỗi bước, và một `eigh` cho R mỗi task. Dự kiến tổng tăng dưới 25% so với
  RSIAT; phải báo.
- **Trích dẫn trong `research_notes/`** có mục chưa xác minh. Không đưa vào paper trước khi xác minh.

## Nhật ký thay đổi

- 2026-10-30: viết đặc tả (chưa code, chưa chạy). Phụ thuộc spec A đã sửa sau review cùng ngày.
- 2026-10-30 (khi code, trước khi chạy):
  - **`qsd_delta` = 1.3316:** bằng 1024/769, khớp chính xác δ = 1/769.
  - **Bản tương đương Gaussian:** cov của lớp mới được cộng (1e-3 + η)·I, cùng mức 1e-3·I mà cov lớp cũ đã lưu
    vẫn có (`models/base.py::_compute_class_mean`).
  - **Overlay:** `hp_twin_povm` (chỉ softmax Gaussian, λ_A = 0) có sẵn cho trường hợp ở mục 3.3.
  - **Chẩn đoán I_orc:**
    - Overlay `hp_twin` cũng log I_orc: khối lượng softmax mà ảnh lớp cũ thật dồn vào lớp mới.
    - Các overlay có bộ giải mã còn log p(lớp cũ) của ảnh lớp mới thật.
  - **Code:** dùng chung `utils/holevo.py` với spec A. Wave: `kaggle/waves/w03_hp`.
