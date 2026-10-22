# QSD-CIL vòng A2: fusion có hiệu chỉnh và trạng thái từ thống kê đã lưu

Đặc tả này được viết **trước khi chạy**, ngày 2026-10-22. Quy tắc và siêu tham số bên dưới là cố định. Mọi
thay đổi sau khi thấy kết quả phải ghi vào mục "Nhật ký thay đổi".

Giai đoạn A: đặc tả ở `README.md`, kết quả ở `results/stage_a/README.md`.

Mã nguồn:
- `utils/qsd.py`: `stats_states`, `pgm_dense`, `pgm_dense_hierarchical`, `class_statistics`, `fusion_scale`.
- `scripts/summarize_qsd_a2.py`: tổng hợp và áp quy tắc.
- `tests/test_qsd.py`: test.

## 1. Căn cứ (từ giai đoạn A, `results/stage_a/README.md`)

| Phát hiện | Số đo | Hệ quả cho A2 |
|---|---|---|
| Khi chưa có drift, PGM tốt hơn ridge trên cùng thống kê | t=0, R: PGM − ridge = +0.68 [+0.44, +0.92] | Giữ phép đo PGM, sửa phần trạng thái |
| Mức thua của PGM tăng tuyến tính theo task, chủ yếu do nhầm sang task mới hơn | −0.31 (t0) → −2.79 (t9); nhầm sang task mới 11.5% so với 9.9% | Trạng thái lớp cũ bị cũ. Kênh A_t không sửa được |
| `ridge_stats` dùng mean SSCA nên suy giảm chậm hơn | lợi thế +0.68 của PGM mất dần, về khoảng 0 ở t9 | Dựng trạng thái từ **mean SSCA + covariance** đã lưu |
| Fusion trọng số 1 gần như chỉ nghe theo linear | khi bất đồng, fuse chọn theo linear 87% số lần và chỉ đúng 43%, trong khi PGM đúng 3.2% / 14.4% mẫu bất đồng | **Hiệu chỉnh** trọng số fusion |
| Gain của `pgm_fuse` có thể chỉ là "head sinh + head phân biệt" | `ridge_stats` bổ sung cho linear ngang PGM | Đối chứng bắt buộc: **fusion với ridge** |
| `StateBank` lưu từng hàng feature, gần như lưu exemplar dạng feature | | Phương pháp chính phải exemplar-free |

## 2. Phương pháp

**Trạng thái từ thống kê** (`pgm_stats`). Dùng đúng mean μ_c và covariance Σ_c mà RSIAT đã lưu. Không lưu
feature của từng mẫu.

- x = (z − m₀)/s₀, với m₀ và s₀ cố định từ task 0 (như giai đoạn A).
- v = [x; 1].
- ρ_c = E[v vᵀ] / tr, tức ρ_c ∝ [[Σ_x + μ_xμ_xᵀ, μ_x], [μ_xᵀ, 1]].

ρ_c là trạng thái hợp lệ: hỗn hợp các trạng thái mã hoá ψ(z) với trọng số ‖v‖² (đã kiểm chứng trong test).
μ_c là mean đã qua SSCA, nên trạng thái lớp cũ được cập nhật theo đúng cơ chế của RSIAT.

**Các head mới:**

| Head | Vai trò |
|---|---|
| `pgm_stats` | **Phương pháp chính:** PGM trên trạng thái từ (mean SSCA, Σ) |
| `pgm_stats_stale` | Như trên nhưng dùng mean lúc học lớp. Đo tác dụng của việc cập nhật mean |
| `pgm_stats_hier` | Bản phân cấp task → lớp của `pgm_stats` |
| `pgm_stats_oracle` | **Chỉ là đối chứng (không hợp lệ EFCIL):** μ, Σ tính lại từ ảnh train lớp cũ qua mạng hiện tại. Đây là trần của mọi cách vận chuyển trạng thái |

**Fusion có hiệu chỉnh.** Với mỗi nguồn s, head `fuse_<s>_b<w>` là:

  argmax_c [ scale·cos_c(z) + w · k_s · score_s,c(z) ]

- score_s là log xác suất Born với các nguồn PGM, và đầu ra ridge với `ridge_stats`.
- k_s = σ_linear / σ_s, với σ là độ trải của điểm số theo lớp, lấy trung bình theo mẫu. k_s được đo trên ảnh
  train của task hiện tại và **không dùng nhãn**. Nhờ vậy w là trọng số tương đối, so được giữa các nguồn.
- 5 nguồn: `ridge_stats`, `pgm_channel`, `pgm_channel_hier`, `pgm_stats`, `pgm_stats_hier`.
- 5 trọng số: w ∈ {0.25, 0.5, 1, 2, 4}.
- Tổng cộng 25 head fusion.
- Dạng p^w của xác suất Born tương ứng với phép đo trên w bản sao ρ^⊗w. Đây là cách diễn giải, không phải
  giả thuyết cần kiểm định.

**Head giữ lại từ giai đoạn A** để kiểm tra hồi quy: `linear`, `ridge_stats`, `pgm_channel`,
`pgm_channel_hier`, `pgm_fuse`. Phép tính không đổi; đã kiểm tra bằng cách so bản code cũ trong
`dist/QG-RSA_QBRIDGE_Kaggle.zip` với bản mới: 9/9 head trùng hoàn toàn.

**Lưu điểm số:** `qsd_scores_task_<t>.npz` (float16) gồm điểm của linear và của 5 nguồn. Mọi fusion khác có thể
tính offline mà không phải chạy lại.

**Tính chất đã kiểm chứng** (`tests/test_qsd.py`):
- ρ_c từ (μ, Σ) bằng đúng hỗn hợp có trọng số ‖v‖²;
- PGM dạng ma trận đầy đủ trùng PGM dạng thừa số, cả bản phẳng lẫn bản phân cấp;
- `class_statistics` trùng `models/base.py`;
- ở learner giả có thống kê đúng, oracle trùng `pgm_stats`;
- argmax của điểm số đã lưu tái tạo được head đã log.

## 3. Siêu tham số đặt trước

- Giữ nguyên giai đoạn A: δ = 0.1/d, ridge của kênh 0.1, `qsd_ridge` 0.1, seed 20261022.
- Lưới trọng số fusion: {0.25, 0.5, 1, 2, 4}.
- Huấn luyện **không đổi**: overlay `qsd2_rep{k}` chỉ khác `qsd_rep{k}` ở các khoá `qsd_*` và `prefix`. Vì vậy mạng
  giống từng bit với giai đoạn A ở cùng (dataset, seed, replicate).

## 4. Giao thức

- **Nhóm tuning:** ImageNet-R seed 1993, `qsd2_rep0..2` (3 run).
  - Với mỗi nguồn, w* = argmax của AIA trung bình trên nhóm này; nếu hoà, chọn w nhỏ hơn.
  - w* được **cố định** trước khi nhìn held-out.
- **Held-out (8 run):** ImageNet-R seed 2015, 2026, 2042 (3 run) và ImageNet-A seed 1993 ×3, 2015, 2026 (5 run).
  - Riêng ImageNet-A hoàn toàn không dùng để chọn bất cứ thứ gì.

## 5. Quy tắc quyết định (cài trong `scripts/summarize_qsd_a2.py`)

1. **M. Phương pháp đạt** nếu, trên held-out gộp (n=8), fusion `pgm_stats` với w*:
   - Δ AIA so với `linear` ≥ +0.3 và cận dưới CI 95% > 0; **và**
   - mean Δ > 0 trên từng dataset.
2. **Q. Đóng góp của phép đo lượng tử** chỉ được khẳng định nếu, trên held-out:
   - fusion `pgm_stats`(w*) − fusion `ridge_stats`(w*) có CI > 0 khi gộp, **và**
   - mean > 0 trên từng dataset.

   Hai bên dùng cùng thống kê, cùng cách hiệu chỉnh, cùng cách chọn w.
3. **S. Cập nhật mean có ích:** `pgm_stats − pgm_stats_stale` có CI > 0 trên mọi run.
4. **E. Phép đo tốt hơn ridge trên cùng thống kê:** `pgm_stats − ridge_stats` có CI > 0 trên mọi run.
5. **H. Phân cấp có lợi:** `pgm_stats_hier − pgm_stats` có CI > 0.
6. **O. Có đáng đầu tư vào vận chuyển trạng thái không:** `pgm_stats_oracle − pgm_stats` ở **final** có mean ≥ +1.0.
   Nếu dưới mức này thì bỏ hẳn hướng sửa drift.
7. **Hồi quy:**
   - `linear` trùng CNN curve;
   - 5 head giữ lại có curve trùng giai đoạn A ở cùng (dataset, seed, replicate) (`--stage-a-root`).

**Hệ quả:**

| Kết quả | Kết luận | Bước tiếp |
|---|---|---|
| M đạt, Q đạt | Đóng góp lượng tử: "fusion phép đo PGM có hiệu chỉnh, exemplar-free" | Giai đoạn B với loss phép đo, so với loss ridge |
| M đạt, Q không | Đóng góp là fusion sinh–phân biệt; PGM chỉ là cách diễn giải | Giai đoạn B, so loss lượng tử với loss cổ điển ngay từ đầu |
| M không đạt | Chỉnh ở bước suy luận đã chạm trần | Sang thẳng giai đoạn B |
| O ≥ +1 | Còn dư địa ở chất lượng trạng thái | Thiết kế vận chuyển tốt hơn (sau B) |

## 6. Kế hoạch chạy (3 tài khoản Kaggle)

Notebook `QGRSA_EXPERIMENTS.ipynb`, cell 1: chỉ đổi `PLAN`.

| `PLAN` | Run | Vai trò | Thời gian ước tính |
|---|---|---|---|
| `kaggle1` | ImageNet-R seed 1993 × `qsd2_rep0..2` | Tuning | ~3 giờ (giai đoạn A: 2 giờ 48) |
| `kaggle2` | ImageNet-A seed 1993 × `qsd2_rep0..2` + ImageNet-R seed 2042 `qsd2_rep0` | Held-out | ~3 giờ |
| `kaggle3` | ImageNet-R và ImageNet-A, seed 2015 và 2026, `qsd2_rep0` | Held-out | ~2.5 giờ |

Chi phí thêm so với giai đoạn A: trích feature ảnh train lớp cũ cho oracle, và PGM dạng ma trận đầy đủ (khoảng
1.4 TFLOP mỗi lần ở task cuối của R, dưới 1 giây trên GPU).

Điểm số lưu thêm khoảng 50 MB mỗi run ImageNet-R. Phần này nằm trong `_stats/` và không vào git.

## 7. Tổng hợp ở máy local

Giải nén `qgrsa_results_kaggle{1,2,3}.zip` vào `exps/qsd/results/a2/kaggle{1,2,3}/`, rồi chạy:

```bash
python scripts/summarize_qsd_a2.py \
  --root exps/qsd/results/a2/kaggle1/logs --root exps/qsd/results/a2/kaggle2/logs --root exps/qsd/results/a2/kaggle3/logs \
  --stage-a-root exps/qsd/results/stage_a/kaggle1/logs --stage-a-root exps/qsd/results/stage_a/kaggle2/logs \
  --stage-a-root exps/qsd/results/stage_a/kaggle3/logs \
  --output exps/qsd/results/a2/summary.md --json exps/qsd/results/a2/summary.json
```

## Nhật ký thay đổi

- (trống)
