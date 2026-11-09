# QSD-CIL: class-incremental learning như bài toán phân biệt trạng thái lượng tử

> Giai đoạn A đã chạy xong. Kết quả: `results/stage_a/README.md`. Vòng tiếp theo: `README_A2.md`.

Đặc tả này được viết **trước khi chạy**. Các quy tắc quyết định và siêu tham số bên
dưới là cố định; mọi thay đổi sau khi thấy kết quả phải ghi lại ở mục "Nhật ký thay đổi".

Mã nguồn:
- `utils/qsd.py`: trạng thái, kênh, phép đo, CA chạy lại.
- `trainer.py`: hook đánh giá gọi `evaluate_qsd`.
- `scripts/summarize_qsd.py`: tổng hợp kết quả và áp dụng quy tắc.
- `tests/test_qsd.py`: test.

## 1. Căn cứ (đã đo)

| Bằng chứng | Nguồn |
|---|---|
| Toàn bộ mất mát đặc thù của học tăng dần là nhầm lẫn giữa các task: nhầm khác task 16.2% (CIL) so với 12.1% (huấn luyện chung), nhầm trong cùng task 1.24% so với 1.41% | `exps/upper/README.md` |
| Mất mát dồn vào lớp cũ nhất: task 0 −9.0, task 1 −8.3 so với huấn luyện chung | như trên |
| Thống kê đúng (oracle) đưa task 0 từ 80 lên 87, nhưng task 5–9 giảm, ròng chỉ +0.5 final | như trên |
| Covariance của lớp cũ không bao giờ được cập nhật; SSCA chỉ sửa khoảng 15% sai số mean mỗi task | `exps/diagnostics/results/` |
| Head linh hoạt học trên mẫu Gaussian suy biến thì học thuộc (100% trên mẫu giả, khoảng 67% trên test) | `exps/qrfca/results/README.md` |
| Dùng PQK làm "hàm tương đồng thay thế" ở 4 vị trí đều không có lợi | `exps/qksr/results/`, `exps/qrfca/results/` |

## 2. Phương pháp (giai đoạn A: chỉ thay bước suy luận)

**Mã hoá.** Mã hoá biên độ 10 qubit, không mất thông tin:

- ψ(z) = [(z − m₀)/s₀ ; 1] / ‖·‖, đệm 0 lên d = 1024.
- m₀ và s₀ (căn của vết covariance) lấy từ feature task 0 và **cố định** về sau.
- Đệm 0 không làm thay đổi đại lượng nào, nên code tính trong 769 chiều.

**Trạng thái.** ρ_c = (1/n_c) Σ ψψᵀ trên feature **thật** của lớp, tính tại task học lớp đó. Lưu dưới dạng thừa số hạng thấp L_c với vết bằng 1. Không bao giờ lấy mẫu giả từ trạng thái.

**Kênh vận chuyển drift.**

- Ở cuối task t, ảnh của task hiện tại được đưa qua mạng cũ f_{t−1} và mạng mới f_t, theo thứ tự cố định.
- A_t = argmin Σ‖ψ_new − Aψ_old‖² + λ‖A − I‖², giải dạng đóng, với λ = 0.1·N/D.
- Mọi lớp cũ được cập nhật: L_c ← A_t L_c, rồi chuẩn hoá lại vết.

**Phép đo PGM.**

- E_c = π_c W ρ_c W, với W = (ρ̄ + δI)^(−1/2), ρ̄ = Σ π_c ρ_c, π_c đều.
- δ = 0.1/d.

**Bản phân cấp.**

- P(T | z) là tổng của các E_c trong task T.
- P(c | z, T) là PGM trong task T, làm trắng bằng ρ_T.

**Tính chất đã kiểm chứng trong `tests/test_qsd.py`:**

- (i) Bất biến unitary: mạch đặt sau bước mã hoá không làm thay đổi kết quả.
- (ii) Co rút ρ_c về ρ̄ không đổi lớp được chọn khi π đều. Vì vậy δ là bộ điều chuẩn duy nhất có tác dụng.
- (iii) Σ_c E_c bằng phép chiếu lên không gian sinh bởi trạng thái.
- (iv) Kênh khôi phục được phép quay đã biết, và vận chuyển trạng thái khớp với trạng thái tính lại sau phép quay.

## 3. Các head (mọi head chạy trong cùng một run)

| Head | Vai trò |
|---|---|
| `linear` | Head CA của RSIAT, là baseline |
| `linear_rerun` | Chạy lại CA từ head trước CA (`fc.old_state_dict`), có heuristic co mean, dùng generator riêng |
| `linear_noscale` | Như trên, cùng nhiễu và cùng thứ tự xáo, nhưng **bỏ** heuristic co mean 0.9 + 0.1(t+1)/(T+1) |
| `ridge_stats` | Ridge giải tích trên mean và covariance đã lưu, kiểu DPCR. Đây là anh em cổ điển của PGM |
| `pgm_stale` | PGM, trạng thái không được vận chuyển |
| `pgm_channel` | **Phương pháp chính:** PGM cộng kênh vận chuyển |
| `pgm_channel_hier` | Bản phân cấp task → lớp |
| `pgm_fuse` | log p_linear + log p_PGM, trọng số 1 đặt trước; p_linear = softmax(scale · cos) |
| `pgm_nonlinear` | Mã hoá với một qubit cờ: (\|0⟩\|ψ_lin⟩ + \|1⟩\|u⊗u⟩)/√2, trong đó u là phép chiếu ngẫu nhiên trực giao 768 → 32 (hai bản sao 5 qubit, 11 qubit tổng). Giữ nguyên giới hạn tuyến tính, thêm phần phi tuyến |

Vì sao so trong cùng run là hợp lệ: head CA không ảnh hưởng tới việc học adapter ở task sau, và mọi phép ngẫu nhiên dùng generator riêng bên trong `preserve_rng`. Điều này được kiểm tra ở mỗi run bằng hai điều kiện: head `linear` trùng CNN curve, và CNN curve trùng từng chữ số với baseline RSIAT cùng replicate.

## 4. Siêu tham số đặt trước

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `qsd_delta` | 0.1 (× 1/d) | Quét {0.01, 0.1, 1} chỉ để báo cáo độ nhạy (`delta_sensitivity`), **không** dùng để chọn |
| `qsd_channel_ridge` | 0.1 (× N/D) | Kéo về identity, vì drift đo được chỉ khoảng 8% mỗi task |
| `qsd_ridge` | 0.1 (× tr(G)/768) | Cho `ridge_stats`; độ nhạy được báo cáo tương tự |
| `qsd_nl_dim` | 32 | Kích thước khối hai bản sao là 32² = 1024 |
| `qsd_seed` | 20261022 | |
| π_c | đều | Tránh thiên lệch về lớp mới |

## 5. Quy tắc quyết định (cài sẵn trong `scripts/summarize_qsd.py`)

1. **Phương pháp đạt** nếu ít nhất một trong `pgm_channel`, `pgm_channel_hier`, `pgm_fuse`, qua n ≥ 3 run:
   - thắng `linear` với cận dưới CI 95% của Δ AIA > 0 **và** Δ ≥ +0.3, **và**
   - thắng `ridge_stats` với CI 95% > 0.

   Có 3 phép so cho phương pháp, nên đây chỉ là sàng lọc. Kết luận chính thức thuộc bước xác nhận (seed khác).
2. **Kênh có tác dụng:** `pgm_channel − pgm_stale` có CI > 0.
3. **Mã hoá phi tuyến có lợi:** `pgm_nonlinear − pgm_channel` có CI > 0. Chỉ khi điều này đúng mới được nói phần phi tuyến lượng tử có đóng góp.
4. **Heuristic co mean của CA có hại:** `linear_noscale − linear_rerun` có CI > 0.
5. **Chỉ để tham khảo:** `linear_rerun − linear` cho biết nhiễu lấy mẫu của chính CA.
6. **Chỉ số cơ chế**, báo cáo cho mọi head:
   - độ chính xác task 0–1, mục tiêu tiến về mức oracle (87 / 83 trên ImageNet-R);
   - độ chính xác dự đoán task;
   - tỉ số nhầm khác task / trong task trên mỗi cặp lớp (CIL 1.38, huấn luyện chung 0.90).

Nếu không có head PGM nào đạt, kết luận là thống kê và phân xử ở bước suy luận **không phải** đòn bẩy. Khi đó chuyển sang giai đoạn B (steering bằng phép đo trong lúc học adapter), đúng như kế hoạch.

## 6. Kế hoạch chạy (3 tài khoản Kaggle, qua đêm)

Notebook `QGRSA_EXPERIMENTS.ipynb`, cell 1: **chỉ đổi `PLAN`** ở mỗi tài khoản.

| `PLAN` | Dataset | Seed | Run | Thời gian ước tính | Vai trò |
|---|---|---|---|---|---|
| `kaggle1` | ImageNet-R (`zaphat206/imagenet-r`) | 1993 | `qsd_rep0..2` | ~2.5 giờ | Sàng lọc chính |
| `kaggle2` | ImageNet-A (`natural-adversarial-examples-imageneta`) | 1993 | `qsd_rep0..2` | ~1.2 giờ | Sàng lọc phụ |
| `kaggle3` | ImageNet-R + ImageNet-A | 2015, 2026 | `qsd_rep0` | ~3–4 giờ | Xác nhận, chỉ phân tích nếu sàng lọc đạt |

Mỗi run tự chứa baseline `linear`, nên các seed mới không cần chạy thêm baseline.
Kết quả được đóng gói thành `qgrsa_results_<PLAN>.zip`.

## 7. Tổng hợp ở máy local

```bash
python scripts/summarize_qsd.py \
  --root exps/qsd/results/stage_a/kaggle1/logs --root exps/qsd/results/stage_a/kaggle2/logs --root exps/qsd/results/stage_a/kaggle3/logs \
  --reference-root exps/measurement/results/logs \
  --reference-root exps/diagnostics/results/round2_imageneta_s1993/logs \
  --output exps/qsd/results/stage_a/summary.md --json exps/qsd/results/stage_a/summary.json
```

Bảng "gộp mọi seed" xuất hiện khi có nhiều seed của cùng dataset.

## Nhật ký thay đổi

- (trống)
