# QRF-CA: head phân loại phi tuyến với random feature lượng tử ở bước CA

## Giả thuyết

Ở vòng `UPPER`, khi dùng thống kê lớp cũ chính xác tuyệt đối, final chỉ tăng +0.5.
Nhưng phép đo đó dùng **head cosine tuyến tính** của RSIAT. Giả thuyết ở đây là một head
**phi tuyến**, học trên cùng mô hình Gaussian của từng lớp, có thể giảm phần nhầm lẫn
giữa các task. Hiện tỉ số nhầm khác task / trong task trên mỗi cặp lớp là 1.38 ở CIL,
so với 0.90 ở huấn luyện chung. Cơ sở từ tài liệu:

- RanPAC (NeurIPS 2023): chiếu ngẫu nhiên phi tuyến tăng +4.0 ImageNet-R trên cùng feature.
- KLDA (AAAI 2025): RFF + LDA đạt mức ngang huấn luyện chung.
- FeCAM (NeurIPS 2023): dùng covariance riêng của từng lớp.

Thành phần lượng tử chỉ là **feature map đóng băng**, không có tham số học: các mạch
Ry + CNOT 8 qubit với góc ngẫu nhiên, mã hoá lại dữ liệu ở mỗi lớp, đọc toạ độ Bloch
của RDM 1 và 2 qubit. Head là ridge cổ điển giải dạng đóng. Mã nguồn ở
`utils/ca_heads.py`.

## Thiết kế

Mọi head được dựng và đánh giá **trong cùng một run**, sau `eval_task` của mỗi task.
Trong RSIAT, head của CA không ảnh hưởng tới việc học adapter ở task sau, vì loss
chỉ dùng logit của lớp mới. Thống kê lớp cũng không phụ thuộc head. Vì vậy các head
dùng chung adapter, thống kê và tập test, và hiệu số giữa hai head không chứa nhiễu
huấn luyện. Mọi phép lấy mẫu dùng generator riêng bên trong `preserve_rng`.

Cả 6 head cùng học trên một mô hình lớp như CA: N(s_c·μ_c, Σ_c), lấy 512 mẫu mỗi lớp.

| Head | Vai trò |
|---|---|
| `linear` | Head RSIAT đã huấn luyện (baseline) |
| `qda` | Mahalanobis theo từng lớp, covariance co rút kiểu FeCAM (γ₁ = γ₂ = 1) |
| `rff_ridge` | RBF random Fourier features, M = 8192, bandwidth theo trung vị |
| `rp_ridge` | ReLU random projection, M = 8192 (kiểu RanPAC) |
| `pqk_ridge` | PQK random features, M = 8192 (86 mạch × 96 feature) |
| `pqk_noent_ridge` | Như `pqk_ridge` nhưng bỏ CNOT; dùng **cùng** phép chiếu và góc |

Hệ số ridge được chọn trên 20% mẫu giả để riêng, từ lưới 10⁻³…10³ nhân với tr(G)/M.
Không bao giờ dùng tập test để chọn. Mỗi task còn ghi kernel–target alignment (đã
centered) và geometric difference g(K_classical ‖ K_pqk) (Huang et al. 2021) vào
`cahead_task_<t>.json`.

Chạy 3 replicate (`qrfca_rep0..2`, `rng_replicate` 0, 1, 2), seed 1993, trên ImageNet-A
(input `natural-adversarial-examples-imageneta`, notebook tự chia 80/20 theo lớp với
seed 67, giống vòng 1–2) hoặc ImageNet-R (`zaphat206/imagenet-r`). Overlay không phụ
thuộc dataset; chọn bằng `DATASETS` trong notebook. Với ImageNet-A, baseline để kiểm
tra hồi quy là `diag_legacy`, `_rep1`, `_rep2` trong
`exps/diagnostics/results/round2_imageneta_s1993/`.

## Quy tắc quyết định (đặt trước, cài sẵn trong `scripts/summarize_ca_heads.py`)

1. **QRF-CA đạt** nếu có ít nhất một head phi tuyến mà, qua n ≥ 3 run, cận dưới
   CI 95% của Δ AIA so với `linear` lớn hơn 0 **và** Δ AIA trung bình ≥ +0.3.
2. **Được nói "phần lượng tử có đóng góp"** chỉ khi `pqk_ridge` thắng **mỗi**
   đối chứng `rff_ridge`, `rp_ridge`, `qda`, `pqk_noent_ridge` với CI 95% > 0.
   Không đạt điều này thì chỉ báo cáo QRF-CA như một cải tiến cổ điển, kèm so sánh
   lượng tử trung thực.
3. **Nếu không head phi tuyến nào thắng `linear`**, giả thuyết về bộ phân loại bị
   bác bỏ. Đòn bẩy còn lại là biểu diễn (steering liên tục, D6).
4. **Kiểm tra hồi quy:** CNN curve của mỗi run phải trùng từng chữ số với baseline
   RSIAT cùng replicate (ImageNet-R: `noise` trong `exps/measurement/results/`;
   ImageNet-A: `diag_legacy*` trong `exps/diagnostics/results/round2_*`), tức là
   việc thêm head không làm thay đổi huấn luyện.

Nếu đạt, bước xác nhận là thêm seed 2015 và 2026 cùng ImageNet-A. Mỗi run đã tự chứa
baseline `linear`, nên không cần chạy thêm baseline.

## Tổng hợp ở máy local

```bash
python scripts/summarize_ca_heads.py --root exps/qrfca/results/logs \
  --reference-root exps/measurement/results/logs --output exps/qrfca/results/summary.md
```

ImageNet-A: dùng `--reference-root exps/diagnostics/results/round2_imageneta_s1993/logs`.
