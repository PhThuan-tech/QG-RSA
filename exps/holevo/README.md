# Holevo-CIL giai đoạn A: huấn luyện adapter tối đa Holevo χ của toàn bộ tập lớp

Đặc tả này được viết **trước khi code và trước khi chạy**, ngày 2026-10-30. Quy tắc bên dưới là cố định. Mọi
thay đổi sau khi thấy kết quả phải ghi vào mục "Nhật ký thay đổi".

- Quy ước seed: `exps/README.md`.
- Hạ tầng dùng lại:
  - mã hoá trạng thái từ thống kê `stats_states` (`utils/qsd.py`, vòng A2, `exps/qsd/README_A2.md`);
  - pass `train_embeddings_old` đầu mỗi task (`models/RSIAT_adapter.py::incremental_train`).
- Kiểm tính mới và các công trình phải đối chiếu: `research_notes/Tính mới Residual POVM cho CIL/` và mục 6.
- Bản ghép với bộ giải mã Residual-POVM (hướng A+B): `exps/holevo_povm/README.md`.

## 1. Vì sao có hướng này

**Dư địa đã đo** (`exps/upper/README.md`):
- Joint − CIL = 3.9 final. Sau khi dùng thống kê oracle vẫn còn 3.4, và phần này nằm ở **biểu diễn**.
- Mất mát dồn vào task 0–1 (+9.0 / +8.3 so với joint). Task 9 ngang joint.
- Tỉ số nhầm khác task / trong task tính trên mỗi cặp lớp: **1.38** ở CIL, **0.90** ở joint.
- Đóng băng feature vẫn quên 70–85% (repo `Research_Idea`, F3). KD có dạng chữ U ngược (F4). Vậy vấn đề là
  **lớp mới lấn vùng lớp cũ (intrusion)**, không phải drift.
- Nguyên nhân gốc (F5): ở task t ≥ 1, loss chỉ là CosFace trên lớp mới
  ([RSIAT_adapter.py:761](../../models/RSIAT_adapter.py#L761)). Lúc train không có tín hiệu nào về lớp cũ.

**Những gì đã cạn, không lặp lại:**
- Mọi vị trí quantum lúc suy luận (PQK, PGM, fusion, SVF).
- Bù drift.
- Phân lập tham số kiểu SPARC.
- CAS hay Q-BRIDGE R1: CosFace trên mọi lớp với hàng classifier cũ detach, cho −3.47 final trên ImageNet-A.

  Khác biệt của hướng này so với R1:
  - Lớp cũ đi vào loss qua **trạng thái bậc hai** (mean + cov) ở mức phân bố, không qua hàng classifier.
  - Objective không ép mẫu mới khỏi từng prototype; nó tối đa một đại lượng của **cả tập lớp**.

**Ý tưởng.** Nhìn CIL như truyền tin lượng tử:
- lớp là thông điệp;
- adapter là bộ mã hoá x → ρ_x;
- classifier là bộ giải mã (POVM).

EFCIL là thêm thông điệp vào kênh khi chỉ còn ρ_c của thông điệp cũ. **Holevo χ** của tập
{p_c, ρ_c} là cận trên của lượng thông tin mà *mọi* bộ giải mã lấy ra được. Huấn luyện adapter để tăng χ trên
**toàn bộ** lớp đã thấy (cũ từ thống kê, mới từ dữ liệu) sẽ:
- buộc lớp mới phân biệt được với lớp cũ ở mức phân bố;
- giữ biểu diễn trải rộng (rank cao). Điều này khớp với bằng chứng của CwD và AdaGauss, và ngược với ý tưởng
  "ngân sách entropy ở task 0" đã loại.

## 2. Công thức

### 2.1. Mã hoá

Bộ mã hoá giống `stats_states`, cố định sau task 0. Với feature thô z ∈ ℝ⁷⁶⁸ (`extract_vector`, chưa chuẩn hoá):

```
v(z) = [ (z − m₀) / s₀ ; 1 ] ∈ ℝ^D,   D = 769      (KHÔNG chuẩn hoá v)
```

- m₀, s₀ được tính **một lần ở cuối task 0**, trực tiếp từ thống kê đã lưu của các lớp task 0. Cách này không
  cần thêm pass dữ liệu và không đụng RNG:
  - m₀ = trung bình các μ_c (c ∈ task 0);
  - s₀² = trung bình_c [ tr Σ_c + ‖μ_c − m₀‖² ].
- Định nghĩa này lấy trọng số đều theo lớp. Nó khác `Encoder.from_base` của QSD, vốn lấy trọng số theo mẫu,
  và có chủ ý: dùng được ngay trong train mà không cần thêm forward. Mọi head QSD vẫn dùng encoder riêng của
  chúng.

### 2.2. Trạng thái lớp

Mọi ρ có vết 1 và cùng một định nghĩa: hỗn hợp các trạng thái thuần ψ(z) = v/‖v‖, trọng số ‖v‖².

- **Lớp cũ c < K** (K = số lớp đã biết). Dùng y nguyên `stats_states(encoder, μ_c, Σ_c)`:
  ```
  ρ_c ∝ [ Σ_c/s₀² + μ̃_c μ̃_cᵀ ,  μ̃_c ;  μ̃_cᵀ , 1 ],   μ̃_c = (μ_c − m₀)/s₀
  ```
  - μ_c là mean **sau SSCA của task trước**; Σ_c là cov đã lưu (không được bù drift, giống A2).
  - Các ρ_c cũ là **hằng số** trong suốt task t: chúng được dựng một lần lúc bắt đầu train task t, rồi detach.
- **Lớp mới c ≥ K.** Dùng **queue feature theo lớp**, cùng kiểu XBM/MoCo, chỉ chứa dữ liệu của task hiện tại
  nên vẫn exemplar-free:
  ```
  ρ_c = Σ_{i∈Q_c} v_i v_iᵀ / Σ_{i∈Q_c} ‖v_i‖²
  ```
  - **Khởi tạo:**
    - Lấy từ `train_embeddings_old`, pass bắt buộc đã chạy ở đầu task t (cùng train_loader, cùng augmentation,
      cùng RNG).
    - Ở bước 0, mạng hiện tại trùng mạng cũ, nên đây đúng là feature hiện tại.
    - Hệ quả: **không có pass dữ liệu mới, luồng RNG giữ nguyên tới khi bắt đầu train**.
  - **Cập nhật FIFO:** các mẫu của lớp c trong batch hiện tại thay các phần tử cũ nhất của Q_c.
    - Phần tử trong batch **mang gradient**.
    - Phần tử còn lại trong queue đã detach.
    - Sức chứa `holevo_queue_cap` (mặc định 256, lớn hơn số ảnh/lớp của ImageNet-R, nên thực tế là toàn bộ lớp).

  Lý do không dùng ước lượng trên batch: batch 48 trên 20 lớp mới chỉ có khoảng 2.4 mẫu mỗi lớp. ρ_c có hạng
  ≤ 3, nên S(ρ_c) vô nghĩa.

### 2.3. Holevo χ

Prior p_c đều trên mọi lớp đã thấy, giống CA và giống cách đánh giá cân bằng. Gọi C = K + n_new.

```
ρ̄  = (1/C) [ Σ_{c<K} ρ_c + Σ_{c≥K} ρ_c ]  =  (K/C)·ρ̄_old  +  (1/C)·Σ_{c≥K} ρ_c
χ  = S(ρ̄) − (1/C) Σ_c S(ρ_c),            S(ρ) = −Tr ρ log ρ
χ̃  = χ / log C                           ∈ [0, 1]   (chuẩn hoá để λ so sánh được giữa các task)
```

- **Vế lớp cũ chỉ đi vào qua ρ̄_old**, một ma trận D×D hằng số tính một lần mỗi task. Các S(ρ_c) của lớp cũ là
  hằng số nên bị bỏ khỏi gradient. Chi phí vì thế không tăng theo số lớp cũ.
- **Diễn giải gradient**, và cũng là cơ chế cần kiểm chứng:
  - ∂S(ρ̄)/∂ρ̄ = −log ρ̄ − I.
  - Một mẫu mới v bị đẩy về các hướng riêng có trị riêng **nhỏ** của ρ̄, tức phần không gian mà lớp cũ chưa
    chiếm. Đây là bản "mềm" của phần dư I − Σ E_old trong hướng B (Residual-POVM).
  - Vế −S(ρ_c) làm từng lớp mới gọn lại.

**Cận lý thuyết và phạm vi của nó** (sửa 2026-10-30 sau review, xem nhật ký):
- Cận Holevo: với mọi POVM tác động lên **một bản sao** của trạng thái, I(Y; Ŷ) ≤ χ. Kết hợp Fano, với p đều:
  ```
  P_err ≥ (log C − χ − log 2) / log(C − 1)
  ```
- **Cận này KHÔNG ràng buộc CA tuyến tính**, hay bất kỳ bộ phân loại nào đọc trực tiếp vectơ cổ điển z:
  - Đọc z tương đương có vô số bản sao của ψ(z), nên I(Y; Z) có thể lớn hơn χ nhiều.
  - Vì vậy, trong hướng A (chỉ có bộ mã hoá, bộ giải mã vẫn là CA), χ chỉ là một **proxy có động cơ** cho độ
    tách giữa các lớp ở mức phân bố. Mối liên hệ giữa χ và accuracy của CA phải được **kiểm bằng thực nghiệm**:
    tương quan giữa Δχ_oracle với pair ratio và final, báo ở mục 4.3.
- Cận chỉ chặt khi bộ giải mã là một phép đo lượng tử một bản sao, ví dụ PGM hay Residual-POVM. Đó là lý do tự
  nhiên để ghép A với B (`exps/holevo_povm/README.md`).
- Paper **không được** viết rằng tăng χ hạ cận lỗi của CA.
- χ bất biến dưới mọi phép quay unitary chung của bộ mã hoá. Điều này nhất quán với Prop. unitary invariance:
  gain, nếu có, phải đến từ việc adapter thay đổi ánh xạ ảnh → trạng thái.

### 2.4. Tính entropy ổn định

- **S(ρ̄):** `eigh` ma trận 769×769 bằng float64 mỗi bước.
  - Gradient dùng surrogate:
    ```
    S_sur(ρ̄) = −Tr[ sg(log(ρ̄ + ε I)) · ρ̄ ]
    ```
    trong đó sg là stop-gradient.
  - Vì Tr ρ̄ = 1 cố định, gradient của surrogate bằng đúng gradient của S. Cách này tránh đạo hàm qua vectơ
    riêng khi trị riêng suy biến.
  - Giá trị log ra dùng S chính xác (không surrogate).
- **S(ρ_c) của lớp mới:** tính qua ma trận Gram k×k thay vì D×D.
  - ρ_c = L Lᵀ, với L = [v_i/√(Σ‖v‖²)] kích thước D×k, k ≤ cap.
  - S(ρ_c) = S(G), G = Lᵀ L (cùng phổ khác 0).
  - Dùng cùng surrogate trên G.
- ε = `holevo_eps` = 1e-8 (cov cũ đã có +1e-3·I nên ρ̄ vốn hạng đủ; ε chỉ để an toàn số học).

### 2.5. Loss

Loss chỉ áp dụng ở task t ≥ 1. **Task 0 giữ nguyên hoàn toàn**, nên accuracy task 0 phải trùng baseline cùng seed.

```
L = CosFace_new  +  β·MSE(f, AE(f_old))  +  γ·orth          (RSIAT, giữ nguyên)
    − λ_H · w(e) · χ̃
w(e) = min(1, e / warmup_epoch)                              (ramp giống lambda_rs)
```

Giữ KD và orth của RSIAT:
- mục tiêu là **thêm** tín hiệu về lớp cũ, không thay thế các điểm mạnh đã đo;
- giữ vậy thì so sánh với RSIAT là sạch.

Không đổi các phần sau (thuộc RSIAT):
- SSCA, CA, optimizer, lr, epoch, AE projector.
- Thống kê lưu sau mỗi task vẫn tính như cũ (`_compute_class_mean`).

### 2.6. Bản classical tương đương và ablation (chạy cùng wave)

| Overlay | `holevo_mode` | Thay đổi so với Holevo | Câu hỏi trả lời |
|---|---|---|---|
| `holevo_vn` | `vn` | phương pháp chính | — |
| `holevo_mcr2` | `mcr2` | thay S(ρ) bằng coding rate R_ε(ρ) = ½ logdet(I + (D/ε²)ρ), ε² = 0.5, trên **đúng** các ρ, queue và thống kê đó (giống MCR², Yu et al. 2020) | Entropy lượng tử có hơn log-det cổ điển không? (**luật Q**) |
| `holevo_spread` | `spread` | chỉ giữ S(ρ̄), bỏ −Σ S(ρ_c) | Gain đến từ trải rộng (kiểu AdaGauss, anti-collapse) hay từ cả cấu trúc Holevo? |

**λ không dùng chung giữa các overlay.** χ và R_ε có thang khác nhau, nên một λ chung sẽ làm bên này mạnh hơn
bên kia một cách giả tạo. Mỗi overlay tự hiệu chỉnh λ của nó theo **cùng một quy tắc tỉ số gradient** (mục 3.3).
Nhờ đó cả ba có cùng ngân sách gradient phụ so với CosFace, và luật Q so sánh công bằng.

## 3. Thiết lập

### 3.1. Config và khoá mới

- **Config:** `exps/rsiat/imagenetr.json` (B0Inc20, 10 task, batch 48, inc_epochs 30, warmup_epoch 4, β 3, γ 1.5),
  cùng dữ liệu với `exps/measurement`.
- **Khoá mới** (mặc định tắt; khi `holevo_mode: off` thì hành vi bit-identical với RSIAT):

| Khoá | Mặc định | Giá trị wave W02 | Ghi chú |
|---|---|---|---|
| `holevo_mode` | `off` | `vn` / `mcr2` / `spread` | |
| `holevo_lambda` | 0 | **`auto`** | hiệu chỉnh theo tỉ số gradient, mục 3.3 |
| `holevo_grad_ratio` | 0.5 | 0.5 | tỉ số mục tiêu r* = ‖∇(λ·obj)‖ / ‖∇ CosFace_new‖ trên adapter |
| `holevo_calib_steps` | 20 | 20 | số bước đầu của task 1 dùng để đo r |
| `holevo_queue_cap` | 256 | 256 | ≥ số ảnh/lớp của ImageNet-R |
| `holevo_eps` | 1e-8 | 1e-8 | |
| `holevo_mcr2_eps2` | 0.5 | 0.5 | chỉ dùng cho `mcr2` |
| `holevo_oracle_diag` | false | **true** | chẩn đoán χ_oracle (mục 4.3); đọc ảnh lớp cũ, chỉ để phân tích |
| `holevo_log_every` | 0 | 25 | log χ, S(ρ̄), mean S(ρ_c), tỉ số gradient |

- **Overlay:** `exps/holevo/holevo_vn.json`, `holevo_mcr2.json`, `holevo_spread.json`. Chỉ chứa các khoá
  `holevo_*` và `prefix`; mọi head QSD tắt.
- **Code dự kiến:**
  - `utils/holevo.py`: encoder từ thống kê, queue, χ / R_ε, surrogate.
  - Một nhánh trong `_compute_rt_loss`.
  - `scripts/summarize_holevo.py`.
  - `tests/test_holevo.py`.

### 3.2. Baseline ghép cặp

- Dùng `noise`, `noise_rep1`, `noise_rep2`, `noise_rep3` (ImageNet-R 1993, `exps/measurement/results/`).
- Final 82.59 ± 0.23, AIA 87.347 ± 0.144.
- Δ được tính so với **trung bình 4 replicate**. SD 0.23 là thang nhiễu cho một run đơn.

### 3.3. Chọn λ (tự động, cùng quy tắc cho mọi overlay)

Hiệu chỉnh được làm **ngay trong run**, trên ImageNet-R. Không dùng smoke CIFAR, vì thang gradient ở đó khác.

1. Ở epoch 0 của task 1, trọng số ramp w(0) = 0, nên objective chưa tác động lên cập nhật. Cập nhật vì vậy
   **giống hệt RSIAT**.
2. Trong `holevo_calib_steps` = 20 bước đầu, tính riêng hai gradient trên tham số adapter (`autograd.grad`,
   không đổi bước cập nhật, không đụng RNG):
   - g_c = ∇ CosFace_new
   - g_o = ∇ obj, với obj = −χ̃ (`vn`), −R̃ (`mcr2`) hoặc −S̃(ρ̄) (`spread`)
3. Đặt λ = r* · median_steps(‖g_c‖ / ‖g_o‖), với r* = `holevo_grad_ratio` = 0.5.
   - Kẹp λ vào [1e-3, 1e3]; nếu bị kẹp thì ghi cảnh báo.
   - λ này cố định cho mọi task sau, được log, và ghi vào `summary.json`.
4. Ở các task sau chỉ log tỉ số r thực tế, không hiệu chỉnh lại.

Quy tắc giống nhau cho `vn`, `mcr2` và `spread`, nên ba overlay có cùng ngân sách gradient phụ ban đầu.

**Không quét λ ở wave sàng lọc.** Lưới r* ∈ {0.25, 1.0} chỉ chạy ở wave sau, và chỉ khi luật M đạt mà luật O
trượt (mục 4).

### 3.5. Bước bắt buộc trước khi chạy wave (ghi kết quả vào nhật ký trước khi chạy)

1. **Thang nhiễu của pair ratio.** Tính pair ratio ở task cuối cho 4 run `noise`, từ thư mục `_stats` ở máy local
   (`utils/qsd.py::head_metrics`, head linear). Ghi mean_PR và SD_PR. Ngưỡng của luật M là:
   ```
   δ_M = max(0.10, 2·SD_PR)
   ```
2. Smoke (`exps/smoke`) với `holevo_mode: vn`: chạy hết, không NaN, có log λ và χ.
3. Smoke với `holevo_mode: off`: bit-identical với RSIAT (test ở mục 5).

### 3.4. Wave

- **Wave `W02_holevo`:** ImageNet-R 1993, 3 run song song: `holevo_vn`, `holevo_mcr2`, `holevo_spread`.
- Nếu GPU chỉ cho 1–2 run, ưu tiên theo thứ tự `holevo_vn` → `holevo_mcr2` → `holevo_spread`.

## 4. Quy tắc quyết định (cố định, `scripts/summarize_holevo.py`)

### 4.1. Sàng lọc (wave W02, ImageNet-R 1993, so với trung bình 4 run `noise`)

| Luật | Điều kiện | Nếu trượt |
|---|---|---|
| **R** (hồi quy) | Accuracy task 0 và mọi số liệu tới hết task 0 trùng `noise` rep0 (task 0 không đổi) | Lỗi code; sửa rồi chạy lại, chưa được kết luận gì |
| **G** (không bị lách) | Trên các task 1–9, median Δχ_oracle(t) **> 0** (định nghĩa ở mục 4.3) | Nếu median Δχ_stale > 0 mà median Δχ_oracle ≤ 0: objective chỉ đẩy lớp mới khỏi vị trí **cũ** của lớp cũ, không tăng độ tách thật → **dừng hướng A**, dù luật O đạt hay không |
| **M** (cơ chế) | Pair ratio (nhầm khác task / trong task, mỗi cặp lớp) ở task cuối **≤ mean_PR − δ_M**, với δ_M = max(0.10, 2·SD_PR) đo trên 4 run `noise` (mục 3.5) | Objective không chạm vào intrusion → **dừng hướng A** |
| **O** (kết quả) | Δ final (`holevo_vn` − noise) **≥ +0.5** (khoảng 2.2 SD của một run) | Nếu M đạt: chạy W03 (λ ∈ {0.3, 3}). Nếu M trượt: dừng |
| **P** (plasticity) | Acc trung bình của 3 task cuối (đo ở task cuối) giảm **≤ 1.0** so với baseline | **Dừng.** Lặp lại kịch bản SPARC hoặc R1 |

- **Đi tiếp sang held-out** khi R, G, M, O, P cùng đạt.
- **Độ nhạy của luật O** (ghi để đọc kết quả đúng): với một run đơn (SD 0.23), nếu không có hiệu ứng thật thì
  xác suất vượt +0.5 chỉ khoảng 1–2%. Nhưng nếu hiệu ứng thật đúng bằng +0.5 thì cũng chỉ có khoảng 50% khả
  năng đạt. Một lần trượt O với Δ trong khoảng [+0.25, +0.5) mà M và G đạt **không** phải bằng chứng là không
  có hiệu ứng; trường hợp đó đi sang wave lưới r* ở mục 3.3.
- Con số trên run sàng lọc chỉ dùng để sàng lọc, không đưa vào bảng chính.

### 4.2. Held-out (chỉ khi sàng lọc đạt)

- **Run:** ImageNet-R 2015, 2026 và ImageNet-A 1993, 2015, 2026, n = 5. Mỗi run gồm `holevo_vn` + `holevo_mcr2`,
  ghép cặp với baseline RSIAT cùng seed (`qsd2_rep0` của A2, đã có sẵn).
- **Luật H (phương pháp):** Δ final (`holevo_vn` − RSIAT) có mean **≥ +0.5**, CI 95% > 0.
- **Luật Q (đóng góp lượng tử):** Δ final (`holevo_vn` − `holevo_mcr2`) có CI 95% > 0.
  - H đạt, Q trượt: phương pháp đứng được, nhưng phải viết là *"objective dẫn ra từ lý thuyết thông tin lượng
    tử, tương đương cổ điển về hiệu năng"*. Không nói là quantum advantage.
  - H trượt: dừng, ghi kết quả âm vào paper.

### 4.3. Đọc kèm (không phải điều kiện dừng, trừ χ_oracle dùng cho luật G)

- **χ_oracle** (`holevo_oracle_diag: true`; đọc ảnh train của lớp cũ, vi phạm exemplar-free nên **chỉ dùng
  phân tích**, giống `_old_statistics_oracle`):
  - Ở **đầu** và **cuối** phần train của mỗi task t ≥ 1, tính χ với ρ_c của **mọi** lớp đã thấy dựng từ feature
    thật của mạng tại thời điểm đó: lớp cũ từ ảnh train cũ, lớp mới từ ảnh train hiện tại. Dùng cùng bộ mã hoá
    m₀, s₀ và cùng định nghĩa trọng số ‖v‖².
  - Δχ_oracle(t) = χ_oracle(cuối) − χ_oracle(đầu).
  - Δχ_stale(t) = χ̂(cuối) − χ̂(đầu), là objective như đã log (lớp cũ từ thống kê).
  - Chạy với `torch.random.fork_rng`, loader tất định, transform test; không đổi trạng thái mạng, thống kê hay
    luồng RNG của huấn luyện. Thời gian chạy được báo riêng, không tính vào chi phí phương pháp.
  - Báo cáo tương quan Spearman giữa Δχ_oracle(t) và mức giảm nhầm khác task ở từng task. Đây là bằng chứng
    thực nghiệm cho việc dùng χ làm proxy khi bộ giải mã là CA (mục 2.3).
- **Drift:** độ lớn trung bình của dịch chuyển SSCA mỗi task so với baseline. EFC++ báo rằng đưa thống kê vào lúc
  train làm drift tăng. Nếu drift tăng quá 50%, ghi rõ trong phân tích.
- **Quỹ đạo χ:** χ, S(ρ̄), mean S(ρ_c) theo epoch.
  - Giả thuyết đúng thì χ tăng mà S(ρ̄_new ‖ ρ̄_old) không sụp.
  - Báo thêm overlap Tr(ρ̄_old ρ̄_new) / √(Tr ρ̄_old² Tr ρ̄_new²).
- Tỉ số gradient r theo task.
- **Chi phí:** thời gian/epoch so với baseline. Phải báo trong paper; dự kiến tăng dưới 15%, chủ yếu do `eigh` 769
  float64.

### 4.4. Báo cáo bắt buộc (dù đạt hay không)

- AIA, final, forgetting.
- Acc task 0 và task 1 ở task cuối.
- Pair ratio (mốc joint 0.90).
- Bảng ba overlay `vn` / `mcr2` / `spread`.
- Kết quả hồi quy.

### 4.5. Dự đoán nếu giả thuyết đúng

- Task 0–1 ở task cuối tăng rõ, về phía oracle (87) và joint (89).
- Task mới giảm không quá 1 điểm.
- Pair ratio giảm từ khoảng 1.38 về phía 0.90.
- `spread` đạt một phần gain. `vn` hơn `spread`, vì vế gọn lớp chống được việc trải rộng phá cấu trúc lớp.

## 5. Kiểm tra trước khi chạy (`tests/test_holevo.py`)

| Kiểm tra | Cách |
|---|---|
| `holevo_mode: off` bit-identical với RSIAT (mọi task) | so log và trọng số trên smoke với overlay rỗng |
| Task 0 không đổi khi bật | so trọng số và acc task 0 |
| ρ lớp cũ = `stats_states` với encoder thống kê | so số |
| ρ có vết 1, nửa xác định dương; 0 ≤ χ ≤ log C | ngẫu nhiên nhỏ |
| Gradient surrogate = autograd của S chính xác | sai phân hữu hạn trên ρ 8×8 không suy biến |
| S qua Gram k×k = S qua D×D | so số |
| χ bất biến dưới phép quay trực giao chung | xoay v rồi tính lại |
| Queue FIFO: đúng lớp, đúng sức chứa, chỉ phần tử của batch mang gradient | đơn vị |
| Khởi tạo queue không tạo thêm pass dữ liệu (RNG giữ nguyên tới khi train) | so trạng thái RNG trước và sau |
| `mcr2` và `spread` chỉ khác đúng vế đã ghi ở bảng 2.6 | đơn vị |
| Hiệu chỉnh λ không đổi cập nhật ở epoch 0 của task 1: trọng số sau epoch 0 trùng `holevo_mode: off` | so trọng số |
| Hiệu chỉnh λ dùng cùng code cho ba overlay; λ được kẹp và log | đơn vị |
| Chẩn đoán χ_oracle không đổi trọng số, thống kê, trạng thái RNG; bật hay tắt cho cùng đường cong accuracy | so log hai run smoke |
| χ_oracle tính từ feature thật = χ̂ khi thống kê lớp cũ trùng thống kê oracle | đơn vị, dữ liệu giả |
| Tóm tắt: luật R/M/O/P/H/Q tính đúng trên dữ liệu giả | `SummaryTest` |

## 6. Rủi ro đã biết

- **Đẩy vào hướng phương sai thấp của ρ̄_old có thể phá đặc trưng ngữ nghĩa** của ViT. KD (β = 3) giới hạn rủi
  ro này; luật P bắt nó.
- **Cov cũ cũ dần** vì không được bù drift. Ảnh hưởng tới ρ̄_old giống A2. Nếu M trượt mà drift lớn, đây là ứng
  viên đầu tiên.
- **Queue chứa feature cũ trong cùng task**, có độ trễ tối đa 1 epoch. Ảnh hưởng nhỏ với lr cosine; đây là cách
  làm chuẩn kiểu XBM.
- **Trùng tiền lệ.** Đây là rủi ro tính mới lớn nhất, xếp từ gần nhất:
  1. **Wu, Baek, You, Ma, *Incremental Learning via Rate Reduction*, arXiv 2011.14593.** Venue nghĩ là CVPR
     2021, **cần xác minh**.
     - Học tăng dần bằng MCR², dùng covariance của lớp cũ đã lưu, không cần mẫu cũ.
     - **Bản `mcr2` của spec này gần như chính là công trình đó**, đặt lên PTM-adapter.
  2. i-CTRL (ICLR 2023): MCR² tăng dần, dùng encoder/decoder vòng kín.
  3. AdaGauss (NeurIPS 2024): loss anti-collapse, gần biến thể `spread`.
  4. PRL (NeurIPS 2024): dành chỗ cho lớp tương lai.
  5. QKD (CVPR 2026): quantum trong PTM-CIL.

- **Khác biệt phải chứng minh** (trả lời trước câu "khác gì Wu 2021 / i-CTRL"):
  - **Trạng thái:** ρ_c từ mean + cov có thêm thành phần affine (vế "1" trong v), nên chứa cả vị trí của lớp
    (μ̃_c μ̃_cᵀ và các khối chéo). Bên kia chỉ dùng covariance đã căn giữa. Cần ablation: bỏ vế "1" (`vn`
    không affine) để biết thành phần này đóng góp bao nhiêu. Ablation này **chỉ chạy ở held-out nếu H đạt**.
  - **Entropy:** von Neumann trên ρ chuẩn hoá vết, so với log-det. Luật Q đo trực tiếp khác biệt này.
  - **Bối cảnh:** adapter trên ViT pretrained, giữ KD và CA của RSIAT. Bên kia train từ đầu.

- **Nếu luật Q trượt**, đóng góp chỉ còn là "đưa rate reduction tăng dần vào PTM-CIL, trình bày bằng ngôn ngữ
  lượng tử". Khi đó phải viết đúng như vậy, và hướng A+B (`exps/holevo_povm/README.md`) trở thành nơi chứa
  phần mới.

## Nhật ký thay đổi

- 2026-10-30: viết đặc tả (chưa code, chưa chạy).
- 2026-10-30 (trước khi code, sau review deep-research chế độ review; đánh giá: Minor Revision, Devil's Advocate:
  REVISE):
  1. **Mục 2.3:** sửa lỗi lý thuyết. Cận Holevo + Fano chỉ ràng buộc phép đo lượng tử một bản sao, không ràng
     buộc CA. χ trong hướng A chỉ là proxy, phải kiểm bằng thực nghiệm.
  2. **Chẩn đoán χ_oracle và luật G** (objective bị lách vì trạng thái lớp cũ đã cũ). Mục 4.1, 4.3.
  3. **Ngưỡng luật M:** δ_M = max(0.10, 2·SD_PR) đo trên 4 run `noise` trước khi chạy (mục 3.5), thay cho 0.10 cố
     định.
  4. **λ:** hiệu chỉnh trong run theo tỉ số gradient, riêng cho từng overlay, cùng quy tắc (mục 3.3). Thay cho λ = 1
     chung và cho bước hiệu chỉnh trên smoke CIFAR.
  5. **Mục 6:** thêm Wu et al. (rate reduction tăng dần với covariance lưu) làm tiền lệ gần nhất, cùng các khác
     biệt phải chứng minh. Sửa tham chiếu `reports/` (thư mục trống).
- 2026-10-30 (khi code, trước khi chạy):
  - **Bước 3.5.1 đã làm.** Pair ratio ở task cuối của 4 run `noise` là 1.3473 / 1.4214 / 1.3794 / 1.3736, nên:
    - mean_PR = 1.3806, SD_PR = 0.0305;
    - δ_M = max(0.10, 0.061) = **0.10**, tức luật M cần pair ratio ≤ **1.281**;
    - trung bình 3 task cuối là 84.732, nên luật P cần ≥ 83.73;
    - task 0 = 96.05 (luật R).

    Các hằng số này nằm trong `scripts/summarize_holevo.py::BASE`, và
    `python scripts/summarize_holevo.py --check-baseline exps/measurement/results/logs/adapter/imagenetr/0/20` tính
    lại chúng.
  - **Queue:** sức chứa mỗi lớp = min(`holevo_queue_cap`, số ảnh của lớp trong pass khởi tạo). Queue FIFO giữ đúng
    kích thước đó, không phình ra bằng các bản augmentation trùng ảnh.
  - **`mcr2`:** chuẩn hoá theo log C giống `vn`, thay cho "giá trị ở bước 0". Vì λ được hiệu chỉnh theo tỉ số
    gradient, thang chuẩn hoá chỉ cần giống nhau giữa các task của cùng một overlay.
  - **Hiệu chỉnh λ:** dùng 20 bước đầu của epoch 0 ở task 1. Nếu epoch 0 có ít hơn 20 bước, λ được chốt ở cuối
    epoch 0 với số bước đã có.
  - **Code:** `utils/holevo.py`, các hook trong `models/RSIAT_adapter.py`, `scripts/summarize_holevo.py`,
    `tests/test_holevo.py` (25 test). Test tích hợp trên vòng lặp RSIAT thật xác nhận chế độ chẩn đoán cho trọng
    số giống hệt RSIAT.
  - **Wave:** `kaggle/waves/w02_holevo`.
