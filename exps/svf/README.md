# SVF giai đoạn A: fusion view plastic với view snapshot task 0 (chỉ suy luận)

Đặc tả này được viết **trước khi chạy**, ngày 2026-10-28. Quy tắc bên dưới là cố định. Mọi thay đổi sau khi
thấy kết quả phải ghi vào mục "Nhật ký thay đổi".

- Quy ước seed: `exps/README.md`.
- Hạ tầng dùng lại: head và fusion của `utils/qsd.py` (vòng A2, `exps/qsd/README_A2.md`).

## 1. Vì sao có hướng này

**Dư địa đã đo:**
- Trên ImageNet-R mọi biến thể đã thử đều dừng ở final ≈ 82.6. Gồm RSIAT, KD các dạng, đơn giản hoá SSCA/cov, quantum, SCF, và SPARC trong repo `Research_Idea`.
- Cận trên (`exps/upper/README.md`): joint − CIL = 3.9 final. Thống kê oracle chỉ sửa được 0.5; phần còn lại nằm ở biểu diễn, tập trung ở task 0–1 (+9.0 / +8.3 so với joint).
- Oracle stats đưa task 0 từ 80.1 lên 87.1, nhưng làm mất ở task mới.

**Bằng chứng cho view snapshot** (repo `Research_Idea`, `exps/sparc/README.md`):
- Khi mạng sau task 0 (A₀) được đóng băng, lớp cũ ổn định hơn hẳn: task 0 cuối 85.6, so với 81.6 của `kd_pure`.
- Đổi lại thiếu plasticity: acc task mới −3.7.

**Ý tưởng:** giữ cả hai view.
- **View plastic:** RSIAT như cũ (adapter hiện tại + CA).
- **View snapshot:** bản sao đóng băng của mạng sau task 0. Thống kê (mean, covariance) của mỗi lớp được tính **khi học lớp đó**, trên ảnh của task hiện tại. Snapshot không bao giờ đổi, nên thống kê này **chính xác mãi mãi**: không có drift, không cần SSCA. Phương pháp vẫn exemplar-free.
- **Fusion:** `argmax(scale·cos + w·k·s_snap)`, với `k = σ_linear / σ_snap` đo không cần nhãn trên ảnh train của task hiện tại, giống hệt vòng A2.

**Hướng CAS bị loại trước khi chạy:**
- Q-BRIDGE R1 chính là CAS với z = 0: CosFace trên mọi lớp đã thấy, hàng classifier cũ detach.
- R1 cho final −3.47 và forgetting +7.1 trên ImageNet-A s1993 (`docs/archive/qbridge/QBRIDGE_VS_RSIAT_IMAGENETA.md`).

## 2. Thiết lập (không chỉnh gì)

**Config:** `exps/rsiat/imagenetr.json` và `exps/rsiat/imageneta.json` (B0Inc20, 10 task), cùng dữ liệu với A2.

**Overlay `svf.json`:**
- Head: `linear`, `ridge_stats`, `snap_ridge`, `qsd_fuse: true`, lưu điểm số.
- So với `exps/scf/scf.json`, overlay chỉ khác các khoá `qsd_*` và `prefix`. Huấn luyện vì vậy **giống hệt** RSIAT gốc cùng seed (`tests/test_svf.py` kiểm tra điều này).

**Head sinh ra:**

| Head | Ý nghĩa |
|---|---|
| `linear` | RSIAT (CA head), phải trùng CNN curve |
| `ridge_stats`, `fuse_ridge_stats_b0p25` | SCF ridge với trọng số đã khoá từ A2 (đối chứng mạnh nhất hiện có) |
| `snap_ridge` | Chỉ view snapshot: ridge closed-form (`qsd_ridge` 0.1) trên thống kê của snapshot. Là ablation kiểu first-session adaptation |
| `fuse_snap_ridge_b<w>` | **SVF**: linear + w·k·snap_ridge, w ∈ {0.25, 0.5, 1, 2, 4} |
| `fuse3_snap_b<w>` | **SVF + SCF**: linear + 0.25·k·ridge_stats + w·k·snap_ridge |

**Seed:** 1993, 2015, 2026, replicate 0. Plan `svf_all` (hoặc `svf_r` + `svf_a`) trong `QGRSA_EXPERIMENTS.ipynb`.

**Chi phí** (phải báo cáo trong paper):
- Suy luận: 2 lượt forward ViT.
- Lưu trữ: thêm một bộ mean/cov 768 chiều cho mỗi lớp, cùng cỡ với thống kê RSIAT đã lưu.
- Một bản sao adapter + fc của task 0.

## 3. Quy tắc quyết định (cố định, `scripts/summarize_svf.py`)

- **Tuning:** chỉ dùng ImageNet-R 1993. Trên run này chọn **một** w cho `fuse_snap_ridge` và **một** w cho `fuse3_snap`, theo final (hoà thì lấy w nhỏ hơn), rồi khoá.
- **Held-out:** ImageNet-R 2015, 2026 và ImageNet-A 1993, 2015, 2026 (n = 5).
- **Rule S (go sang giai đoạn B):** cả hai điều kiện:
  - Δ final (SVF − RSIAT) trên held-out có mean **≥ +1.0** và CI 95% > 0;
  - Δ final (SVF + SCF − SCF ridge) có CI 95% > 0, tức view snapshot thêm được thông tin mà SCF chưa có.
- **Không đạt:** dừng hướng này và ghi kết quả âm vào paper.

**Luật dừng sàng lọc** (chỉ dùng run tuning ImageNet-R 1993, wave `W01_svf`), quyết định có chạy thêm seed hay không:
- **Dừng, không chạy held-out** nếu Δ final (`fuse3_snap` − `fuse_ridge_stats_b0p25`) < **+0.3** tại trọng số đã chọn.
  - Lý do: hai head nằm trong cùng một run, nên nhiễu của hiệu số nhỏ (SD(Δ) khoảng 0.03–0.15, `exps/upper/README.md`).
  - Dưới +0.3, view snapshot không thêm gì đáng kể so với SCF. Rule S gần như chắc chắn trượt.
- **Đi tiếp** (chạy ImageNet-R 2015, 2026 trước, ImageNet-A sau) chỉ khi đạt ngưỡng trên.
- Con số trên run tuning là lạc quan, vì chọn best-of-5 trọng số trên cùng dữ liệu.
  - Chỉ dùng nó để sàng lọc, không đưa vào bảng chính.
  - Báo kèm Δ ở cả 5 trọng số, để thấy gain ổn định hay chỉ nhô lên ở một giá trị.

**Đọc kèm khi sàng lọc** (không phải điều kiện dừng; dùng để chẩn đoán vì sao đạt hay trượt):
- `snap_ridge` một mình: acc theo task, `task_prediction`, `pair_ratio`.
  - Thống kê lớp task 0 được tính trên chính ảnh đã train A₀, nên cụm chặt hơn lớp mới. Nếu task 0 tăng mà task 5–9 giảm, đó là thiên lệch về task 0 (giống kết quả oracle), không phải gain thật.
- Linear probe 200 lớp trên feature cuối (chẩn đoán riêng, không phải phương pháp, chưa cài): cho biết nút thắt là biểu diễn hay bộ phân loại Gaussian.

**Báo cáo bắt buộc**, dù đạt hay không:
- AIA và forgetting;
- `snap_ridge` một mình;
- acc task 0 và task 1 ở task cuối;
- pair ratio, so với mốc joint 0.90;
- kiểm tra hồi quy (head linear = CNN curve).

**Dự đoán nếu giả thuyết đúng:**
- task 0–1 tăng rõ, về gần mức oracle (87);
- task cuối không giảm quá 1 điểm;
- pair ratio giảm từ khoảng 1.38 về phía 0.90.

## 4. Giai đoạn B (chỉ khi Rule S đạt; đặc tả chi tiết viết sau)

- View snapshot đã giữ stability, nên có thể giảm KD để lấy lại 3–4 điểm acc task mới.
- Thêm khoá `beta_scale` (mặc định 1.0, không đổi hành vi) vào `_inc_loss` của `models/RSIAT_adapter.py`. Chạy β·0.33 và β·0 trên ImageNet-R 3 seed.
- So với SVF của giai đoạn A, ghép cặp cùng seed. Mức nhiễu thận trọng: final SD 0.23.
- Luật: Δ final ≥ +0.5, CI > 0.

## 5. Kiểm tra trước khi chạy

| Kiểm tra | Ở đâu |
|---|---|
| Overlay chỉ đổi đánh giá; overlay cũ (A2, SCF) không sinh head mới | `tests/test_svf.py::SettingsTest` |
| Snapshot = mạng sau task 0, đóng băng, khác mạng plastic ở task sau | `tests/test_svf.py::DriverTest` |
| Ở task 0, `snap_ridge` = `ridge_stats` | như trên |
| Thống kê snapshot chỉ được thêm, không bị ghi lại | như trên |
| Trọng số → 0 cho đúng head còn lại (`fuse3` → SCF ridge) | `test_vanishing_weight_recovers_the_other_heads` |
| RNG giữ nguyên, kết quả tất định | `tests/test_svf.py`, `tests/test_qsd.py` |
| Chọn trọng số chỉ trên run tuning; luật S | `tests/test_svf.py::SummaryTest` |
| Hồi quy trên Kaggle: head linear của R 1993 trùng các run `qsd2_rep0` / `noise` cùng seed | `summarize_svf.py` (head linear = CNN curve), so tay với `exps/qsd/results/a2/` |

## Nhật ký thay đổi

- 2026-10-28 (trước khi chạy): vì giới hạn GPU, wave `W01_svf` chỉ chạy run tuning ImageNet-R 1993. Quy tắc giữ nguyên:
  - run này chỉ dùng để chọn trọng số và xem dư địa (sàng lọc);
  - Rule S chỉ được kết luận khi có đủ 5 run held-out, chạy ở các wave sau.
- 2026-10-28 (trước khi chạy): thêm luật dừng sàng lọc ở mục 3 (fuse3 − SCF ridge ≥ +0.3 trên run tuning). Lý do:
  - Recall@1 của A₀ (0.74) gần bằng của mạng cuối (khoảng 0.755), nên dư địa thực tế có thể chỉ khoảng +0.5 đến +1.0;
  - "trần +3.4" là khoảng cách tới joint, tức trần cho mọi phương pháp, không phải trần của SVF.
