# 🔬 QKSR v3.1 — Projected Local Quantum Metric cho Representation Steering trong RSIAT
## Đặc tả kỹ thuật để AI Agent triển khai (v3.1 — sẵn sàng cho giai đoạn engineering + pilot)

> **Gửi tới AI Agent thực thi code**: Đây là spec, KHÔNG phải code hoàn chỉnh. Giữ đúng tên thuộc tính/khóa config, không sửa gì ngoài mục 12 (Out of scope). Mọi lựa chọn thiết kế là **giả thuyết cần kiểm nghiệm bằng thực nghiệm ở mục 8**, không phải kết luận đã chứng minh. Các quy tắc ở mục 7 (giao thức thống kê) và mục 8.4 (khoá hyperparameter) là **bắt buộc**, không phải khuyến nghị.

---

## 📋 0. Nhật ký thay đổi v3 → v3.1

| # | Vấn đề ở v3 | Đã kiểm | Xử lý ở v3.1 |
|---|---|---|---|
| 1 | **Ablation D không hợp lệ**: `base=false, inc=true, frozen` → module không tham gia loss ở task 0 nên không học gì, sang task 1 bị đóng băng ⇒ thực chất là *random frozen metric*; γ₀ cũng chưa từng được khởi tạo | ✅ Đúng (module có mặt trong `__init__` nhưng không có gradient ở task 0) | Tách **D1 (chẩn đoán, random-frozen)** và **D2 (incremental-trainable, calibrate γ ở task 1)**; đóng góp incremental của metric học từ base đo bằng **B − C**, không phải D (mục 8.1, 5.5) |
| 2 | Optimizer giả định luôn có `encoder`, `gamma_param`; không đúng với `rbf_proj`, `mlp_*`, `median_fixed`, `pqk_random_frozen` | ✅ Đúng | Module **tự cung cấp** `param_groups(...)`; Learner không biết cấu trúc nội bộ (mục 5.6). Quy tắc mới: chỉ đưa vào optimizer những tham số *thực sự nhận gradient* ở nhánh đó |
| 3 | Projector `LayerNorm(768)` mặc định thêm 1.536 tham số affine ⇒ 6.152 sai | ✅ Đúng (`2×768`) | `LayerNorm(768, elementwise_affine=False)`; bảng đếm tham số ở mục 4.2 |
| 4 | Khởi tạo γ mơ hồ (batch đầu nhiễu, không có buffer, không nói thời điểm gọi, thiếu xử lý suy biến) | ✅ Đúng | Quy trình calibration chính xác ở mục 4.5 (subset cố định, `gamma0` là buffer, `u=0`, detach, loại đường chéo, clamp, fallback, đa GPU) |
| 5 | `mlp_metric` "cùng số tham số" không khả thi: phần học được của mạch chỉ ~16 tham số | ✅ Đúng (MLP hữu ích nhỏ nhất đã ~216 tham số) | Hai control: **`mlp_small`** và **`mlp_cap`**; bỏ tên "param-matched"; báo cáo cả tham số lẫn compute (mục 4.4, 8.1) |
| 6 | "Vượt mức std" không phải kiểm định thống kê; sweep lớn trên test ⇒ rò rỉ | ✅ Đúng | Giao thức 4 giai đoạn: tuning trên validation → khoá config → xác nhận trên test; paired difference + bootstrap CI + hiệu chỉnh Holm (mục 7, 8) |
| 7 | H3 nói "quantum-specific" là quá mạnh | ✅ Đúng | H3 đổi thành *circuit-induced feature map*; câu kết luận an toàn ở mục 1 |
| 8 | Unit test thiếu (chuẩn statevector, CNOT, Hermitian/trace/PSD, K(x,x), đối xứng, đối chiếu density matrix đầy đủ, RNG…); `gradcheck` float32 | ✅ Đúng | Bộ test mở rộng ở mục 6.3; module hỗ trợ `dtype` để chạy float64 |
| 9 | *(phát hiện thêm)* Tạo module lượng tử làm dịch RNG toàn cục ⇒ A và B không còn cùng luồng ngẫu nhiên khi bắt đầu, làm hỏng so sánh paired | ✅ Đúng (khởi tạo `nn.Linear`, `randn` đều tiêu thụ RNG) | Khởi tạo module trong `fork_rng` với seed riêng `q_init_seed` (mục 5.2); test RNG ở 6.3 |
| 10 | *(phát hiện thêm)* Chỉ áp `q_lr` nhỏ (×0.1) cho mọi tham số "metric" sẽ thiệt cho control MLP | ✅ Đúng | LR nhân `q_metric_lr_mult` **áp dụng như nhau cho mọi `q_kernel_type`** và được tune cùng ngân sách |
| 11 | *(phát hiện thêm)* Wilcoxon 2 phía với n=5 seed không thể đạt p<0.05 (p tối thiểu 0.0625) | ✅ Đúng | Ưu tiên bootstrap CI + paired t; chỉ dùng Wilcoxon khi n ≥ 6 (mục 7.2) |
| 12 | *(phát hiện thêm)* Nếu chỉ QKSR được tune kỹ còn control tune sơ sài thì so sánh thiên vị | ✅ Đúng | Mọi `q_kernel_type` nhận **cùng ngân sách tuning** (mục 8.3) |

---

## 🎯 1. Định vị đóng góp và giả thuyết

**Contribution (cách viết an toàn):**
> *Projected local quantum metric learning cho representation steering trong exemplar-free, single-adapter class-incremental learning.*

**Định vị:** RSIAT (CVPR 2026) là khung nền, dùng cosine ở `RS_Loss` và `loss_orth`. QKD (CVPR 2026) đã dùng mạch lượng tử biến phân cho quan hệ mẫu–task/routing trong PTM-CIL. QKSR khác ở chỗ dùng **projected kernel như một metric trong loss**, làm việc với **một adapter dùng chung**, và **chỉ tác động lúc huấn luyện** (không thêm chi phí inference).

**KHÔNG được claim:** "lần đầu kết hợp quantum và CIL"; "quantum kernel giảm forgetting" / "quantum advantage"; "bắt tương quan phi cục bộ/bậc cao" cho kernel 1-qubit; "đo semantic shift chính xác hơn" cho nhánh incremental.

**Giả thuyết:**
- **H1 (task 0):** metric phi tuyến học được trên đặc trưng chiếu cho margin lớp tốt hơn cosine ⇒ accuracy task 0 tăng.
- **H2 (incremental):** metric đó, đóng băng sau task 0, làm `loss_orth` cung cấp tín hiệu tốt hơn cho `old_ae` ⇒ forgetting giảm. Đo bằng **B − C** (mục 8.1).
- **H3 (cấu trúc feature map):** lợi ích đến từ **feature map do mạch Ry–CNOT và RDM cục bộ tạo ra** (circuit-induced feature map), vượt các classical control được khảo sát.

**Câu kết luận an toàn nếu H3 được ủng hộ:** *"Kết quả cung cấp bằng chứng thực nghiệm rằng circuit-induced feature map hiệu quả hơn các classical control đã khảo sát."* **Không** viết rằng đã chứng minh lợi thế lượng tử (không thể bao phủ mọi mô hình cổ điển có thể có, và đây là mô phỏng cổ điển).

---

## 🧩 2. Baseline RSIAT (đã đối chiếu với code)

### 2.1 `RS_Loss` (chỉ khi `_cur_task == 0`)
`features = F.normalize(features)`; `dot_prod = features @ features.T` ∈ [-1,1] `[B,B]`; `mask_pos = same_class − eye`, `mask_neg = 1 − same_class`; `loss = mean(relu(1−dot_prod)·mask_pos) + lamda·mean(relu(dot_prod−margin)·mask_neg)` (`rs_margin=0.5` trong `adapter_cub.json`); nhân `lambda_rs·min(1, epoch/warmup_epoch)` trong `_compute_rt_loss`.

### 2.2 `_inc_loss` (chỉ khi `_cur_task > 0`)
`features_old = old_ae(features_old)`; `loss_align = MSE(features, features_old)`; `protos = normalize(old_ae(self._class_means))`; `similarity = protos @ normalize(features_old).T` ∈ [-1,1] `[n_old,B]`; `loss_orth = mean(similarity)`; trả `beta·loss_align + gamma·loss_orth`.

**Hệ quả:** `features_old` từ mạng đóng băng, `_class_means` là hằng ⇒ **`loss_orth` chỉ có gradient lên `old_ae`**; adapter chỉ chịu tác động gián tiếp qua `loss_align`. `loss_orth` không có positive term.

### 2.3 Optimizer trong `_train`
- Task 0: `param_groups` (blocks cuối, blocks còn lại, `fc`) với **`lr` hard-code `0.01`**; `sgd` dùng `param_groups`, `adam` dùng `AdamW(self._network.parameters())`.
- Task > 0: `param_groups` (`convnet`, `fc` lr `init_lr`; `old_ae` lr `ae_init_lr`); `adam` cũng bỏ qua `param_groups`.
- Mọi `exps/*.json` dùng `"optimizer": "sgd"`.

### 2.4 Khác
`BaseLearner.save_checkpoint` chỉ lưu `_network`, không được gọi ở đâu trong repo.

---

## 🏗️ 3. Luồng dữ liệu tổng thể
```
features [B,768] (từ ViT+Adapter; ở nhánh incremental là old_ae(·))
  └► ClassicalProjector ─► h̃ [B,q]
        └► Metric backbone (tuỳ q_kernel_type):
              pqk*      : QuantumFeatureEncoder ─► ψ [B,2^q] ─► LocalRDMExtractor ─► RDM 1-qubit (+ 2-qubit)
              rbf_proj  : h̃ (không biến đổi thêm)
              mlp_*     : MLP(h̃) ─► vector thực
        └► RBF kernel: K = exp(−γ·D) ∈ (0,1]
              ├► RS_Loss   (task 0)     [use_quantum_kernel_base]
              └► _inc_loss (task > 0)   [use_quantum_kernel_inc]
```
Toàn bộ nằm trong `utils/quantum_kernel.py`, đóng gói thành `QuantumKernelModule(nn.Module)`.

---

## 🔧 4. Đặc tả module

### 4.1 Quy ước chung
- **Thứ tự bit:** qubit 0 là bit **có trọng số cao nhất** của chỉ số statevector (`index = Σ_j b_j·2^{q−1−j}`). Mọi test CNOT/RDM dựa trên quy ước này.
- **`q_dtype`** (`float32` mặc định, `float64` cho unit test). Với tập cổng `{Ry, CNOT}` statevector luôn thực ⇒ dùng tensor thực; complex chỉ khi mở rộng tập cổng (ngoài phạm vi).
- Vector hoá theo batch; không dùng PennyLane/Qiskit trong training loop; `q ≤ 12`.

### 4.2 `ClassicalProjector` và bảng đếm tham số
`h̃ = π · tanh( Linear(768→q)( LayerNorm(768, elementwise_affine=False)(features) ) )`, shape `[B,q]`.

| Thành phần (q=8, L=2) | Công thức | Số tham số |
|---|---|---|
| Projector | `q·(768+1)` | **6.152** |
| `pqk*`: θ | `L·q` | 16 |
| `pqk*` (γ học, `u`) | `1` | 1 |
| `mlp_small` (hidden `m=q`; output `2q`) | `(q·m+m)+(m·2q+2q)` | 216 |
| `mlp_cap` (hidden `m=64`; output `2q`) | `(q·64+64)+(64·2q+2q)` | 1.616 |
| `rbf_proj` | không có | 0 |

Ghi chú: output của MLP bằng **số đặc trưng thực độc lập** của biến thể `pqk` tương ứng (bậc 1: `2q`; bậc 2: `2q + 9q = 11q`, vì RDM 2 qubit thực đối xứng có trace 1 có 9 bậc tự do). Số tham số tổng phải được **đọc từ `log_count_parameter`** và báo cáo tách riêng "projector" vs "metric" trong mọi bảng kết quả.

### 4.3 `QuantumFeatureEncoder` (`pqk`, `pqk_no_cnot`, `pqk_random_frozen`)
**Mặc định — mã hoá MỘT lần, không re-uploading:**
$$|\psi\rangle=\Big[\prod_{l=1}^{L}U_{ent}\,U_{var}^{(l)}(\theta_l)\Big]\,U_{enc}(\tilde h)\,|0\rangle^{\otimes q}$$
`U_enc = ⊗_j R_y(h̃_j)`, `U_var^{(l)} = ⊗_j R_y(θ_{l,j})`, `U_ent = CNOT(0→1),…,CNOT(q−1→0)` (vòng), `L=q_num_layers` (mặc định 2, tối đa 3). `θ ∈ ℝ^{L×q}` khởi tạo `randn·0.01`.
**`q_reupload=true` (ablation):** `∏_{l=1}^{L}[U_ent U_var^{(l)} U_enc(h̃)]` — cài đúng MỘT trong hai dạng theo cờ.
- `pqk_no_cnot`: bỏ `U_ent`.
- `pqk_random_frozen`: `θ` là **buffer** ngẫu nhiên (sinh bằng `q_init_seed`), không phải parameter.

### 4.4 `LocalRDMExtractor`, kernel và các control
**RDM bậc 1** (`q_kernel_order=1`): đưa trục qubit `k` về cuối ⇒ `ψ_k` shape `[B,R,2]`, `R=2^{q−1}`;
`ρ_k = torch.einsum('bra,brc->bac', ψ_k, ψ_k.conj())` ⇒ `[B,q,2,2]`. Độ phức tạp `O(q·2^q)`; **cấm** dựng ma trận mật độ đầy đủ `2^q×2^q` (trừ trong unit test, q nhỏ).
> ⚠️ Với Ry+CNOT, mỗi `ρ_k` chỉ có 2 bậc tự do ⇒ kernel bậc 1 ≈ RBF trên vector thực ≤ 2q chiều và mù với các trạng thái cùng RDM 1 qubit (ví dụ các trạng thái Bell). Phải nêu trong báo cáo.

**RDM bậc 2** (`q_kernel_order=2`): thêm các cặp lân cận `(k,k+1 mod q)`; `ψ_{kk'}` shape `[B,R',4]`, `R'=2^{q−2}`; cùng `einsum` ⇒ `[B,q,4,4]`.

$$D(i,j)=\sum_k\|\rho_k^{(i)}-\rho_k^{(j)}\|_F^2+\lambda_2\sum_k\|\rho_{k,k+1}^{(i)}-\rho_{k,k+1}^{(j)}\|_F^2,\quad K=\exp(-\gamma D)$$
(`λ₂=q_order2_weight`, chỉ có khi bậc 2). Tính `[B1,B2]` bằng broadcasting; `clamp` phần mũ.

**Bảng `q_kernel_type`:**
| Kiểu | Mô tả |
|---|---|
| `pqk` (mặc định) | Ry+CNOT, RDM cục bộ theo `q_kernel_order` |
| `pqk_no_cnot` | Bỏ entanglement |
| `pqk_random_frozen` | θ ngẫu nhiên, đóng băng (buffer) |
| `rbf_proj` | RBF trực tiếp trên `h̃` |
| `mlp_small` | `h̃→Linear(q,q)→tanh→Linear(q,d_out)` rồi RBF |
| `mlp_cap` | `h̃→Linear(q,64)→tanh→Linear(64,d_out)` rồi RBF |

`d_out` = số đặc trưng thực độc lập của `pqk` cùng `q_kernel_order` (4.2). Mọi kiểu dùng cùng projector, cùng chính sách γ, cùng loss; chỉ khác "metric backbone".

### 4.5 Chính sách γ và quy trình calibration (chính xác)
**Tham số hoá** (`q_gamma_mode`):
| Chế độ | γ | Thành phần |
|---|---|---|
| `bounded_learned` (**mặc định**) | `γ = γ₀ · 10^{tanh(u)}` ∈ `[γ₀/10, 10γ₀]` | `γ₀`: **registered buffer**; `u`: `nn.Parameter` khởi tạo **0** (⇒ γ=γ₀ ban đầu) |
| `median_fixed` | `γ = γ₀` | chỉ buffer, **không có `u`** |
| `free_learned` (chỉ ablation) | `γ = softplus(u)` | sau calibration đặt `u = log(exp(γ₀) − 1)` |

**Calibration `calibrate_gamma(feature_fn, loader_subset)`** — gọi **đúng một lần**, ở đầu `_train` của **task đầu tiên mà module thực sự được dùng** (task 0 nếu `use_quantum_kernel_base`; ngược lại task 1 nếu chỉ `use_quantum_kernel_inc`), **trước khi bắt đầu tối ưu**, khi `gamma_initialized == False`:
1. Lấy **calibration subset cố định**: `q_calib_samples` (mặc định 512) chỉ số mẫu lấy từ tập train của task hiện tại bằng generator seed `q_init_seed` (không dùng RNG toàn cục), lưu lại danh sách chỉ số vào log.
2. Trích đặc trưng bằng mạng hiện tại ở chế độ `eval()` và `torch.no_grad()`; ở nhánh incremental thì áp `old_ae` đúng như `_inc_loss` sẽ làm (cả `features_old` và `protos` để lấy phân phối khoảng cách thực sự dùng). Khôi phục `train()` sau đó.
3. Tính ma trận `D` **đầy đủ** trên subset (512² là nhỏ), `detach()`, **loại đường chéo**, lấy `median`.
4. `median = max(median, 1e-8)`. Nếu toàn bộ khoảng cách ≈ 0 (`median ≤ 1e-8`) ⇒ **ghi cảnh báo và fallback `γ₀ = 1.0`** (không dừng chương trình).
5. `γ₀ = 1 / median` ghi vào buffer; đặt `gamma_initialized = True` (buffer bool, nằm trong `state_dict`).
6. Đa GPU: `quantum_kernel` **không** bọc DataParallel (được gọi trên feature đã gom); nếu sau này dùng DDP thì broadcast `γ₀` từ rank 0.
- Ở task > 0, γ **không học** trừ khi `q_inc_train_mode="trainable"`.

### 4.6 Miền giá trị và margin
Cosine ∈ [-1,1], K ∈ (0,1] và K ở một khoảng cách cho trước phụ thuộc γ ⇒ không có ngưỡng "tương đương". `rs_margin_q` phải tune độc lập (5.3). Không so giá trị tuyệt đối của `loss_orth` giữa cosine và K (sàn 0 vs −1).

---

## 🔗 5. Tích hợp vào pipeline

### 5.1 Interface `utils/quantum_kernel.py`
```
class QuantumKernelModule(nn.Module):
    def __init__(self, input_dim=768, num_qubits=8, num_layers=2, kernel_type="pqk",
                 kernel_order=1, order2_weight=1.0, reupload=False,
                 gamma_mode="bounded_learned", dtype=torch.float32, init_seed=0): ...
    def forward(self, a, b=None) -> Tensor[Ba,Bb]       # b=None: self-similarity, K ∈ (0,1]
    def encode(self, features) -> Tensor|dict           # metric-side representation (RDM hoặc embedding thực)
    def kernel(self, rep_a, rep_b) -> Tensor[Ba,Bb]
    def calibrate_gamma(self, features_or_fn) -> None   # 4.5
    def set_inc_mode(self, mode: str) -> None           # "frozen" | "trainable": đổi requires_grad
    def param_groups(self, adapter_lr, weight_decay, metric_lr_mult) -> list[dict]   # 5.6
    # buffers: gamma0, gamma_initialized, (theta nếu pqk_random_frozen)
```
Learner chỉ dùng `forward`, `calibrate_gamma`, `set_inc_mode`, `param_groups`; **không** truy cập thuộc tính nội bộ như `encoder` hay `gamma_param`.

### 5.2 `Learner.__init__`
Đặt **ở cuối `__init__`** (sau mọi dòng gốc, để không ảnh hưởng luồng khởi tạo của mạng/`rs_loss_func`), chỉ khi ít nhất một cờ bật:
```
self.use_quantum_kernel_base = args.get("use_quantum_kernel_base", False)
self.use_quantum_kernel_inc  = args.get("use_quantum_kernel_inc", False)
self.quantum_kernel = None
if self.use_quantum_kernel_base or self.use_quantum_kernel_inc:
    with torch.random.fork_rng(devices=[]):        # RNG toàn cục KHÔNG bị tiêu thụ / dịch chuyển
        torch.manual_seed(args["q_init_seed"])
        self.quantum_kernel = QuantumKernelModule(...).to(self._device)
```
Khi cả 2 cờ `false`, không tạo module và không chạm RNG. Khi có module, luồng RNG toàn cục ở đầu huấn luyện phải **giống hệt** baseline (test 6.3).

### 5.3 `RS_Loss.forward` (task 0)
Thêm `quantum_kernel_module=None`, `margin_override=None`. `None` ⇒ giữ nguyên 100%. Có module ⇒ `dot_prod = quantum_kernel_module(features)` (bỏ `F.normalize`); phần còn lại giữ nguyên. Margin dùng `rs_margin_q` với `rs_margin_q_mode ∈ {fixed, quantile}` (`quantile`: margin = quantile `rs_margin_quantile` của K trên các cặp negative trong batch, `detach()`). Warm-up `lambda_rs` giữ nguyên.

### 5.4 `_inc_loss` (task > 0)
Sau khi `features_old`, `protos` đã qua `old_ae` (giữ nguyên), nếu `use_quantum_kernel_inc`: `similarity = self.quantum_kernel(protos, features_old)`. `inc_loss_mode`: `mean` (`similarity.mean()`, **chỉ hợp lệ khi module đóng băng hoặc γ bị chặn**) hoặc `margin` (`relu(similarity − rs_margin_inc).mean()`). `q_inc_pair ∈ {old_proj (mặc định), current}`; `current` so prototype với `features` của adapter hiện tại (gradient vào adapter) — **thiết kế khác**, chỉ khám phá. Diễn giải: mặc định `loss_orth` chỉ huấn luyện `old_ae` (và module nếu `trainable`).

### 5.5 Vòng đời module qua các task (`q_inc_train_mode`)
| Tình huống | Task 0 | Từ task 1 |
|---|---|---|
| `base=true`, `frozen` (mặc định) | calibrate γ₀ ở đầu task 0; module học | `set_inc_mode("frozen")` khi vào task 1; **không** vào optimizer |
| `base=true`, `trainable` | như trên | vẫn học; bắt buộc `bounded_learned` + nên `inc_loss_mode=margin` |
| `base=false`, `inc=true`, `frozen` (**D1, chẩn đoán**) | module không dùng, **không vào optimizer**, không calibrate | calibrate γ₀ ở đầu task 1 rồi `frozen` ⇒ metric ngẫu nhiên đóng băng |
| `base=false`, `inc=true`, `trainable` (**D2**) | như trên | calibrate γ₀ ở đầu task 1 (subset cố định), module học với `bounded_learned` + `margin` |
| `base=true`, `inc=false` (**C**) | học như B | không dùng ở `_inc_loss` |

### 5.6 `_train` — đăng ký tham số (cả 2 nhánh task, cả 2 optimizer)
Trong mỗi nhánh, sau khi dựng `param_groups` gốc:
1. Xác định module có **nhận gradient** ở nhánh này không: task 0 ⇔ `use_quantum_kernel_base`; task > 0 ⇔ `use_quantum_kernel_inc` **và** `q_inc_train_mode=="trainable"`. Nếu không ⇒ **không** đưa vào optimizer.
2. Nếu có: `groups = self.quantum_kernel.param_groups(adapter_lr, weight_decay, q_metric_lr_mult)`, với `adapter_lr` = LR của nhóm adapter trong **đúng nhánh** (task 0: `0.01` như code gốc; task > 0: `init_lr`). Module trả về:
   - nhóm `projector.*`: `lr = adapter_lr`, `weight_decay` như adapter;
   - nhóm còn lại (θ nếu là parameter, MLP, `u`): `lr = adapter_lr · metric_lr_mult`, `weight_decay = 0`;
   - **chỉ gồm tham số có `requires_grad=True`**; nhóm rỗng thì bỏ; buffer không xuất hiện. `metric_lr_mult=None` được coi là `0.1`.
   - Nhờ đó `rbf_proj` (không có metric-side), `median_fixed` (không có `u`), `pqk_random_frozen` (θ là buffer) đều hoạt động mà Learner không cần biết.
3. `param_groups.extend(groups)` (cho `sgd`); `log_count_parameter(param_groups)` tự cộng tham số mới.
4. **Nhánh `adam`:** khi `groups` không rỗng, dựng `AdamW([{'params': self._network.parameters(), 'lr': ..., 'weight_decay': ...}] + groups)` — giữ nguyên hành vi baseline cho mạng, chỉ thêm module. Khi `groups` rỗng, giữ nguyên code gốc.
5. `q_metric_lr_mult` áp dụng **như nhau cho mọi `q_kernel_type`** và được tune cùng ngân sách (8.3).

### 5.7 Checkpoint
Repo chưa gọi `save_checkpoint`. Nếu thêm resume: lưu/nạp `state_dict` của `quantum_kernel` (gồm buffer `gamma0`, `gamma_initialized`, θ nếu là buffer) và `old_ae`, cùng trạng thái đóng băng. Nếu không thêm, báo cáo rõ là không hỗ trợ resume.

### 5.8 Config JSON
```
"use_quantum_kernel_base": true, "use_quantum_kernel_inc": true,
"q_kernel_type": "pqk", "q_kernel_order": 1, "q_order2_weight": 1.0,
"q_num_qubits": 8, "q_num_layers": 2, "q_reupload": false, "q_dtype": "float32",
"q_gamma_mode": "bounded_learned", "q_calib_samples": 512, "q_init_seed": 1234,
"q_metric_lr_mult": 0.1, "q_inc_train_mode": "frozen",
"inc_loss_mode": "mean", "q_inc_pair": "old_proj",
"rs_margin_q": 0.5, "rs_margin_q_mode": "fixed", "rs_margin_quantile": 0.9, "rs_margin_inc": 0.3
```
`rs_margin_q=0.5`, `rs_margin_inc=0.3` chỉ là điểm khởi đầu **chưa được biện minh**. Tạo thêm `exps/adapter_cub_baseline.json` (cả 2 cờ `false`).

---

## 🛠️ 6. Ràng buộc và kiểm thử

### 6.1 Ràng buộc
GPU cùng device; vòng lặp Python chỉ trên hằng số cấu trúc (`q`, `L`); cấm vòng lặp theo sample/cặp `(i,j)`; `eps=1e-8`; clamp mũ; giữ exemplar-free; **khi cả 2 cờ `false` kết quả trùng khớp với repo gốc (cùng seed)**.

### 6.2 Điều kiện hợp lệ hồi quy
Chạy A (baseline) bằng repo đã sửa và bằng repo gốc, cùng seed: kết quả phải trùng (trong dung sai xác định của cuDNN, ghi rõ dung sai).

### 6.3 Unit test bắt buộc
**Gradient / huấn luyện**
1. `gradcheck` ở **float64** (`q_dtype=float64`) cho `forward` trên batch nhỏ, `q` nhỏ.
2. Sau 1 bước `backward()` thật ở task 0: `projector` và (nếu là parameter) θ, `u` có grad khác `None` và khác 0.
3. Task > 0 chế độ `frozen`: tham số module không đổi sau bước optimizer, nhưng `old_ae` có gradient.
4. `q_metric_lr_mult=None` không lỗi; với mỗi `q_kernel_type` (6 kiểu) và mỗi `q_gamma_mode`, `param_groups()` chạy được và không chứa buffer hay tham số `requires_grad=False`.
5. Ở cấu hình D1 (task 0 không dùng module) module **không** có mặt trong optimizer.

**Tính đúng của mạch / RDM** (chạy ở float64)
6. **Chuẩn statevector** `‖ψ‖ = 1` sau từng layer.
7. **CNOT** đúng trên 4 trạng thái cơ sở với quy ước bit ở 4.1: `|00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩` (điều khiển = qubit đầu).
8. **RDM:** đối xứng/Hermitian, `trace = 1`, eigenvalue nhỏ nhất `≥ −1e-8` (float64).
9. **Đối chiếu density matrix đầy đủ:** với `q=3,4`, dựng `|ψ⟩⟨ψ|`, partial trace bằng phương pháp tham chiếu, so với `LocalRDMExtractor` (bậc 1 và bậc 2) — sai số `< 1e-10`.
10. Shape đầu ra đúng cho bậc 1 (`[B,q,2,2]`) và bậc 2 (`[B,q,4,4]`).

**Tính chất kernel** (float64)
11. `K(x,x) = 1`; `K(a,b) = K(b,a)`; `K ∈ (0,1]`.
12. Gram matrix PSD: eigenvalue nhỏ nhất `≥ −1e-8`.

**RNG / tái lập**
13. Khi cả 2 cờ `false`: `torch.get_rng_state()` (và CUDA nếu có) **không đổi** sau `Learner.__init__` so với repo gốc.
14. Khi có module: RNG toàn cục sau `__init__` **bằng** RNG khi không có module (nhờ `fork_rng`), và hai lần tạo với cùng `q_init_seed` cho tham số giống hệt.
15. `calibrate_gamma`: kết quả lặp lại được (cùng subset), `γ₀` là buffer, gọi lần hai không đổi giá trị, fallback hoạt động khi `D ≡ 0`.

---

## 🧪 7. Giao thức thống kê (bắt buộc)

### 7.1 Đơn vị phân tích
So sánh **paired theo seed** (cùng seed, cùng dataset, cùng máy, cùng batch size, cùng số epoch). Với mỗi cặp so sánh, báo cáo `Δ_seed = X_seed − Y_seed` và **trung bình Δ**, **95% bootstrap CI** (percentile, ≥ 10.000 resample) và paired t-test.

### 7.2 Kiểm định
- Dùng **CI bootstrap + paired t** là chính. **Wilcoxon signed-rank chỉ dùng khi n ≥ 6 seed** (với n=5, p hai phía tối thiểu 0,0625 nên không thể đạt 0,05).
- **Effect size tối thiểu δ_min** khai báo *trước* khi xác nhận (gợi ý 0,3 điểm phần trăm cho average/final accuracy); "cải thiện" chỉ được tuyên bố khi CI loại trừ 0 **và** trung bình Δ ≥ δ_min.
- **So sánh chính (confirmatory):** B vs A. **So sánh phụ:** B vs {E1, E2a, E2b, E3, E4}, hiệu chỉnh **Holm** trên nhóm này. Mọi so sánh khác (D1, D2, F, G, H, I…) là **thăm dò**, ghi rõ và không dùng để kết luận.
- Tổng hợp qua dataset: báo cáo theo từng dataset, thêm phân tích gộp (mixed-effects hoặc trung bình theo dataset) và nêu số lần B thắng/thua.

### 7.3 Chỉ số
Task-0 accuracy, average incremental accuracy, final accuracy, forgetting (BWT), old/new/total theo task, số tham số (tách projector vs metric), thời gian/epoch, **peak GPU memory**. Overhead **đo trực tiếp**, không dự báo từ paper khác.

---

## 🧭 8. Kế hoạch thực nghiệm

### 8.1 Ma trận thí nghiệm (mỗi dòng chỉ đổi khoá nêu ra so với B)
| # | Mục đích | Cấu hình | Loại |
|---|---|---|---|
| A | Baseline RSIAT | cả 2 cờ `false` | Confirmatory |
| B | QKSR đầy đủ | mặc định | Confirmatory |
| C | Chỉ task 0 | `inc=false` | Confirmatory (đóng góp task 0 = C − A) |
| **B − C** | **Đóng góp incremental của metric học ở base** | tính từ B và C (cùng task 0) | Confirmatory |
| D1 | Chẩn đoán: incremental với metric **ngẫu nhiên đóng băng** | `base=false, inc=true, q_inc_train_mode=frozen` | Thăm dò |
| D2 | Incremental **trainable** (module học từ task 1) | `base=false, inc=true, trainable, bounded_learned, inc_loss_mode=margin` | Thăm dò |
| E1 | RBF trên `h̃` | `q_kernel_type=rbf_proj` | Confirmatory (control) |
| E2a | MLP nhỏ | `q_kernel_type=mlp_small` | Confirmatory (control) |
| E2b | MLP dung lượng hợp lý | `q_kernel_type=mlp_cap` | Confirmatory (control) |
| E3 | Bỏ CNOT | `q_kernel_type=pqk_no_cnot` | Confirmatory (control) |
| E4 | θ ngẫu nhiên đóng băng | `q_kernel_type=pqk_random_frozen` | Confirmatory (control) |
| E5 | RDM 1 vs 2 qubit | `q_kernel_order=1/2` | Thăm dò |
| E6 | Re-uploading | `q_reupload=true` | Thăm dò |
| F1 | Chính sách γ | `median_fixed / bounded_learned / free_learned` | Thăm dò |
| F2 | Đóng băng vs trainable ở task > 0 | `q_inc_train_mode` | Thăm dò |
| F3 | Loss incremental | `inc_loss_mode` | Thăm dò |
| F4 | Vị trí kernel | `q_inc_pair` | Thăm dò |
| G | Số qubit | `q_num_qubits ∈ {6,8,10,12}` | Thăm dò |
| I | Sức khoẻ gradient | log `theta.grad.abs().mean()`, histogram K mỗi epoch | Chẩn đoán |

**Kiểm tra tính hợp lý bắt buộc:** B và C có **cùng task 0** ⇒ accuracy task 0 phải trùng (trong dung sai xác định ở 6.2). Nếu lệch, có lỗi RNG/tái lập.
**Lưu ý:** phần đóng góp incremental *do metric được học ở base* đo bằng **B − C**; D1/D2 chỉ trả lời các câu hỏi khác (random metric / module học trực tiếp ở incremental) và không được dùng làm bằng chứng cho H2.

### 8.2 Bốn giai đoạn (chống rò rỉ test)
1. **Giai đoạn 0 — Engineering:** cài đặt, toàn bộ unit test 6.3 và hồi quy 6.2 phải pass.
2. **Giai đoạn 1 — Tuning (không đụng test):** dùng **validation split** tách từ tập train (mục 12 cho phép một ngoại lệ hẹp để thêm `val_ratio`, mặc định 0 = không đổi hành vi) hoặc, nếu không thêm được, dùng **dataset phát triển + seed phát triển tách rời** (`1, 2, 3`) hoàn toàn khác tập xác nhận. Mọi lựa chọn `q`, `q_kernel_order`, `q_gamma_mode`, margin, `q_metric_lr_mult` chỉ được dựa trên kết quả ở giai đoạn này.
3. **Giai đoạn 2 — Khoá:** ghi toàn bộ hyperparameter đã chọn cho **từng** `q_kernel_type` vào `locked_config_<kernel_type>.json`, lưu hash, commit **trước** khi chạy giai đoạn 3.
4. **Giai đoạn 3 — Xác nhận trên test:** tối thiểu **3 dataset** (CUB, ImageNet-R, và một trong CIFAR-100/ImageNet-A/Omnibenchmark — repo đã có config) × **≥ 5 seed** (gợi ý `1993, 1996, 1997` + 2 seed nữa; ≥ 6 nếu muốn dùng Wilcoxon), chạy các dòng "Confirmatory". Sau khi khoá không được đổi cấu hình dựa trên kết quả test.

### 8.3 Công bằng ngân sách tuning
**Mọi `q_kernel_type`** (kể cả control) nhận **cùng** ngân sách ở giai đoạn 1: cùng số cấu hình thử (đề xuất ≤ 12/kiểu), cùng lưới cho `q_gamma_mode`, `rs_margin_q(_mode)`, `rs_margin_inc`, `q_metric_lr_mult ∈ {0.1, 1.0}`; ghi lại toàn bộ cấu hình đã thử và kết quả validation. Không cho phép chỉ tune kỹ QKSR.

### 8.4 Quy tắc kết luận
- B không vượt A theo 7.2 ⇒ **không có cải thiện**.
- B vượt A nhưng **không** vượt E1, E2a, E2b (theo 7.2, sau Holm) ⇒ chỉ kết luận *"một learned nonlinear projected metric cải thiện RSIAT"*, **không** quy cho cấu trúc mạch.
- B vượt A, E1, E2a, E2b **và** E3, E4 ⇒ có bằng chứng cho **H3** (circuit-induced feature map); dùng câu kết luận an toàn ở mục 1. Nêu rõ đây là mô phỏng cổ điển với RDM cục bộ và chỉ so với các control đã khảo sát.
- H2 chỉ được ủng hộ nếu **B − C** thoả 7.2.
- Báo cáo **mọi** chạy (kể cả thất bại) và toàn bộ cấu hình đã thử.

---

## 📌 9. Tiêu chí nghiệm thu
- [ ] `utils/quantum_kernel.py` đúng interface 5.1; hỗ trợ đủ 6 `q_kernel_type`; Learner không truy cập thuộc tính nội bộ của module.
- [ ] `LayerNorm(..., elementwise_affine=False)`; bảng tham số ở 4.2 khớp `log_count_parameter`.
- [ ] Module tạo ở **cuối** `__init__` trong `fork_rng`; test RNG 6.3 (13–14) pass.
- [ ] `calibrate_gamma` đúng 4.5, gọi ở task đầu tiên module thực sự được dùng; test 15 pass.
- [ ] `_train` theo 5.6: chỉ đăng ký tham số nhận gradient; xử lý `adam`; `metric_lr_mult=None`.
- [ ] Vòng đời module theo bảng 5.5 (đặc biệt D1/D2/C).
- [ ] Toàn bộ unit test 6.3 (1–15) và hồi quy 6.2 pass.
- [ ] Hai file config (`adapter_cub.json`, `adapter_cub_baseline.json`) chạy được.
- [ ] Giai đoạn 1–3 (8.2) được thực hiện đúng thứ tự, có `locked_config_*.json` trước khi chạy test.
- [ ] Báo cáo theo 7.3, kết luận theo 8.4.

---

## ⚠️ 10. Rủi ro và biện pháp
| Rủi ro | Biện pháp |
|---|---|
| Suy biến `γ → ∞` ở incremental | Đóng băng ở task > 0, γ bị chặn, `margin` loss, F1–F3 |
| Kernel 1-qubit chỉ là RBF ≤ 2q chiều | Nêu trong báo cáo; E5; control E1/E2 |
| Barren plateau / gradient triệt tiêu | `L ≤ 3`, θ khởi tạo nhỏ, `metric_lr_mult`, thí nghiệm I |
| Kernel suy biến về hằng số | Log histogram K; điều chỉnh γ₀, margin |
| Lợi ích chỉ từ learned projection | E1/E2 + quy tắc 8.4 |
| Module không được huấn luyện / bị đưa nhầm vào optimizer | Quy tắc 5.6 + test 3–5 |
| Rò rỉ test / tune thiên vị | 8.2–8.3 |
| Control bị làm yếu chủ ý | Hai control MLP (small/cap), cùng ngân sách tuning, báo cáo tham số + compute |
| Lệch RNG giữa A và B | `fork_rng`, test 13–14, kiểm tra B vs C task 0 |
| Chi phí tính toán | Đo (7.3); giảm `q`/`L` nếu cần |

---

## 🚫 11. Kỳ vọng
Spec này **không đảm bảo** vượt mốc accuracy nào. Số liệu từ paper khác (ví dụ QKD) chỉ cho thấy hướng tương tự từng hiệu quả trong kiến trúc khác. Kết luận chỉ được rút ra theo mục 8.4.

---

## 🚧 12. Ngoài phạm vi (KHÔNG đụng vào)
- Backbone ViT + adapter, `SimpleVitNet`, `network/*`, `utils/inc_net.py`.
- `_stage2_compact_classifier`, `_compute_class_mean`, `displacement()`.
- `AutoencoderSigmoid`, `loss_align`.
- Không thêm exemplar. Ý nghĩa/tên `loss_orth` gốc.
- Việc nhánh `adam` baseline bỏ qua `old_ae` (ghi nhận, không sửa khi cả 2 cờ `false`).
- Data pipeline — **ngoại lệ hẹp duy nhất:** thêm `val_ratio` (tách validation phân tầng từ tập train) để phục vụ giai đoạn 1; mặc định `0` phải giữ nguyên hoàn toàn hành vi gốc (kiểm bằng hồi quy 6.2).

---

## 📎 Phụ lục: Tóm tắt thay đổi theo file
| File | Thay đổi |
|---|---|
| `utils/quantum_kernel.py` (MỚI) | `QuantumKernelModule` 6 kiểu, RDM 1/2 qubit, buffer `gamma0`/`gamma_initialized`, `calibrate_gamma`, `set_inc_mode`, `param_groups`, hỗ trợ `dtype` |
| `models/RSIAT_adapter.py` | Khởi tạo cuối `__init__` trong `fork_rng`; `RS_Loss` (module + margin riêng); `_inc_loss` (2 chế độ, `q_inc_pair`); `_compute_rt_loss`; `_train` (calibrate γ, `param_groups()` của module, xử lý `adam`, chỉ tham số nhận gradient); đóng băng khi vào task 1 theo bảng 5.5 |
| `exps/adapter_cub.json` | Thêm các khoá ở 5.8 |
| `exps/adapter_cub_baseline.json` (MỚI) | Cả 2 cờ `false` |
| Trainer/data (ngoại lệ hẹp) | `val_ratio` mặc định 0 |
| `locked_config_<kernel_type>.json` (MỚI, sinh ở giai đoạn 2) | Hyperparameter đã khoá trước khi chạy test |
