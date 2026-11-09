# Chẩn đoán SSCA và thống kê lớp cũ

Các overlay trong thư mục này trả lời hai câu hỏi trước khi đầu tư vào phương
pháp mới:

1. **SSCA hiện tại ước lượng mean lớp cũ tốt đến đâu, và bản ghép cặp đúng có tốt hơn không?**
2. **Nếu thống kê lớp cũ hoàn hảo thì CA cải thiện được bao nhiêu?** (headroom)

Overlay chỉ là phần ghi đè; luôn ghép với một config nguồn đầy đủ.

## Lỗi ghép cặp trong SSCA legacy

`ssca_mode: "legacy"` giữ nguyên code gốc: `Y1` và `Y2` được trích từ
`train_loader` có `shuffle=True` và augmentation train, rồi lấy `Y2 - Y1`
theo từng hàng. Hàng `i` của hai lượt thường là hai ảnh khác nhau với hai crop
khác nhau, nên drift cục bộ theo lớp bị mất.

`ssca_mode: "paired"` trích cả hai lượt từ cùng một loader không shuffle, dùng
eval transform, nên `Y2[i] - Y1[i]` là drift của đúng một ảnh. Kernel, `sigma`
và công thức giữ nguyên; chỉ việc ghép cặp thay đổi. Lượt legacy vẫn được chạy
để luồng RNG toàn cục giống hệt. Ở task 1, adaptation của hai chế độ trùng
nhau hoàn toàn và khác biệt chỉ đến từ mean ước lượng. Từ task 2, loss
incremental (`loss_orth` của RSIAT, risk/guard của Q-BRIDGE) đọc mean lớp cũ,
nên hai run phân kỳ một cách hợp lệ do chính mean đã thay đổi.

## Các overlay

| Overlay | Thay đổi kết quả? | Mục đích |
|---|---|---|
| `ssca_legacy_observed.json` | Không (chỉ quan sát) | Baseline legacy + báo cáo sai số của stale/legacy/paired so với oracle |
| `ssca_paired.json` | Có | Áp dụng SSCA ghép cặp đúng (phương pháp hợp lệ EFCIL) |
| `oracle_mean.json` | Có — **cận trên, vi phạm EFCIL** | CA dùng mean thật của lớp cũ |
| `oracle_mean_cov.json` | Có — **cận trên, vi phạm EFCIL** | CA dùng mean và covariance thật của lớp cũ |
| `noise_replicate.json` | Có — control | Legacy với `rng_replicate: 1`: cùng base task 0, khác ngẫu nhiên SGD từ task 1 → **sàn nhiễu** |
| `seeds_3.json` | — | Seed 1993, 2015, 2026 |

Mọi overlay bật `save_predictions: true`: dự đoán top-1 từng ảnh test được lưu
vào `predictions_task_<t>.npz` để kiểm định McNemar ghép cặp. Với 1500 ảnh
test, mỗi ảnh = 0.067 pp, nên so mean ± std trên 3 seed không đủ để thấy hiệu
ứng cỡ vài phần mười điểm. Báo cáo chẩn đoán còn ghi accuracy **trước CA**
(cũ/mới, tỉ lệ nhầm cũ→mới) để tách đóng góp của CA.

`stats_diagnostics: true` chỉ đọc: loader chẩn đoán có generator riêng, không
tiêu thụ RNG toàn cục; test `test_diagnostics_do_not_change_training_or_rng`
kiểm tra điều này. Oracle đọc ảnh train của lớp cũ qua encoder hiện tại, nên
chỉ dùng để chẩn đoán.

## Chạy trên ImageNet-A (RSIAT cổ điển)

Đặt `output_root` riêng (ví dụ qua một overlay máy cục bộ) để không lẫn log cũ.

```bash
for v in ssca_legacy_observed noise_replicate ssca_paired oracle_mean oracle_mean_cov; do
  python main.py --config exps/rsiat/imageneta.json \
    --overlay exps/diagnostics/$v.json \
    --overlay exps/diagnostics/seeds_3.json
done

python scripts/summarize_stats_diagnostics.py \
  --root <output_root>/logs/adapter/imageneta \
  --reference diag_legacy \
  --output reports/generated/ssca_diagnostics_imageneta.md
```

Với nhánh Q-BRIDGE R0, thêm overlay chẩn đoán **sau cùng** vào config R0 đã
sinh (để `prefix` của overlay có hiệu lực). Mỗi task ghi
`stats_task_<t>.json` vào `<log>_stats/` (hoặc thư mục `_qbridge` của run).

## Cách đọc kết quả

**Sai số mean** (từ run `diag_legacy`, chỉ quan sát):
`relative = ||ước lượng − oracle|| / ||stale − oracle||`, trung vị theo lớp cũ.

- `legacy ≈ 1` hoặc `> 1`: SSCA hiện tại không tốt hơn việc bỏ bù drift.
- `paired < legacy`: sửa ghép cặp cải thiện bộ ước lượng.
- Cột "task-0 classes": kiểm tra giả thuyết lớp base bị sai số tích lũy.
- `Kernel ESS` gần số mẫu N: kernel gần đều (chỉ còn phép tịnh tiến toàn cục);
  gần 1: chỉ mẫu gần nhất. `Floor rows` cao: `sigma` quá nhỏ so với khoảng cách
  feature, trọng số bị sàn `1e-5` chi phối.
- `Cov rel. error`: sai số Frobenius tương đối của covariance đang lưu (không
  bao giờ được cập nhật) so với covariance thật.

**Headroom** (paired delta theo seed so với `diag_legacy`):

- `oracle_mean − legacy`: phần do mean.
- `oracle_mean_cov − oracle_mean`: phần do covariance.
- `paired − legacy`: phần thu được bằng phương pháp hợp lệ.

**Sàn nhiễu**: `|replicate − legacy|` là mức chênh khi không đổi gì ngoài
ngẫu nhiên SGD. Một hiệu ứng chỉ được tính khi `|Δ AIA| > max(0.3, 2 × sàn nhiễu)`;
McNemar (p < 0.05, gộp các seed) là bằng chứng bổ sung ở mức từng ảnh.

## Cây quyết định (ngưỡng đặt trước khi xem kết quả)

`scripts/summarize_stats_diagnostics.py` tự xếp kịch bản theo đúng các ngưỡng
dưới đây; ngưỡng nằm trong code (`HEADROOM_LARGE`, `HEADROOM_SMALL`,
`MIN_EFFECT`, `ALPHA`) và **không được chỉnh sau khi đã xem kết quả**.

| Kịch bản | Điều kiện | Ý nghĩa | Hành động tiếp |
|---|---|---|---|
| **A** | headroom ≥ 2 pp, paired vượt sàn nhiễu | Thống kê lớp cũ là chỗ nghẽn; sửa ghép cặp đã lấy lại một phần | Giữ paired làm baseline mới; làm covariance transport; thử Q-SSCA (kernel PQK cho trọng số SSCA) với control cosine/RBF/MLP |
| **B** | headroom ≥ 2 pp, paired trong sàn nhiễu | Chỗ nghẽn đúng, nhưng ghép cặp chưa đủ | Đọc `Kernel ESS`/`Floor rows`: nếu kernel gần đều → chọn σ trên validation (`val_ratio`); nếu phần covariance lớn → covariance transport; thử transport bằng T (`old_ae`) kiểu LDC; Q-SSCA là ứng viên đóng góp chính |
| **C** | 0.5 ≤ headroom < 2 pp | Có dư địa nhưng không lớn | Làm phần lớn hơn (mean hay covariance); tính chi phí/lợi ích; cân nhắc kết hợp với hướng representation |
| **D** | headroom < 0.5 pp | Thống kê lớp cũ **không** phải chỗ nghẽn | Đổi hướng, xem mục dưới |

### Nếu kết quả không khả quan (kịch bản D hoặc paired không giúp)

Đây vẫn là kết quả có giá trị, vì nó loại bỏ một giả thuyết bằng bằng chứng
có cận trên. Thứ tự xử lý:

1. **Kiểm tra lại phép đo trước khi kết luận**: run đủ 10 task; cùng
   `target` test giữa các run (script báo lỗi nếu khác); sàn nhiễu đã đo;
   kết quả nhất quán qua seed.
2. **Đọc bảng trước/sau CA** để định vị chỗ mất accuracy:
   - Pre-CA old thấp, nhầm cũ→mới cao, CA kéo lên nhiều → chỗ nghẽn là **ranh giới
     quyết định**: thử cải tiến CA (`logit_norm` có sẵn, scale/ngân sách mẫu),
     hoặc đưa risk/guard của Q-BRIDGE vào chính CA.
   - Pre-CA old thấp mà oracle cũng không cứu → **representation của lớp cũ bị
     hỏng ngay trong adaptation**: hướng giữ representation (ràng buộc adapter,
     sửa đầu ra Sigmoid của `old_ae`, quét liều loss incremental).
   - Accuracy lớp **mới** thấp (ImageNet-A ~70%) → chỗ nghẽn là **plasticity**: base
     RS loss (nhánh đẩy đang tắt vì `rs_margin=1.5` > 1), dung lượng adapter.
3. **Câu chuyện quantum khi đó**: chuyển PQK về nơi đo được trực tiếp: chất
   lượng representation ở base (control rbf_proj/mlp với cùng margin đang bật),
   hoặc PQK học có ý thức quyết định (decision-aware). Không giữ PQK ở vị trí
   đã chứng minh là không có đòn bẩy.
4. **Nếu mọi hướng đều trong sàn nhiễu**: viết thành nghiên cứu phân tích/âm
   tính: lỗi ghép cặp của SSCA trong code upstream, các loss incremental gần
   như trơ (R2 = R0), độ phân giải 1500 ảnh test, và cận trên oracle. Các phát
   hiện này có thể kiểm chứng và hữu ích cho cộng đồng EFCIL.

Lưu ý: trong run oracle, mỗi task bắt đầu từ mean oracle của task trước, nên
báo cáo sai số trong run đó là sai số một bước, không phải tích lũy. Sai số
tích lũy đọc từ `diag_legacy` / `diag_paired`.

## Vòng 2: ứng viên phương pháp (đặt trước khi chạy)

Kết quả vòng 1 (seed 1993): kịch bản C. Ngoài ra, kernel SSCA chết hoàn toàn
(`Floor rows = 1.00`: với σ = 4, khoảng cách bình phương tới mẫu gần nhất
≈ 1400 nên mọi trọng số bằng sàn ⇒ SSCA chỉ là phép tịnh tiến toàn cục và
cách ghép cặp không còn ảnh hưởng). Khoảng 91% lỗi task cuối là nhầm **giữa
các task** (chỉ 3% nhầm trong cùng task).

| Overlay | Một giá trị đặt trước | Giả thuyết cơ chế (kiểm được trong báo cáo) |
|---|---|---|
| `r2_sigma_paired` | `ssca_sigma: "auto_nn"`, paired | `Floor rows` ≈ 0; sai số mean tương đối của paired < legacy ở mọi task |
| `r2_sigma_legacy` | `auto_nn`, legacy | Với kernel cục bộ, ghép sai cặp sẽ gây hại: kỳ vọng ≤ `r2_sigma_paired` |
| `r2_ca_taskmargin` | `ca_task_margin: 0.1` | Tỉ lệ nhầm giữa các task giảm |
| `r2_ca_lognorm` | `ca_logit_norm: 0.1` | Lỗi "to newer task" giảm |
| `r2_combo` | cả ba | Không kém thành phần tốt nhất |

`auto_nn`: ở mỗi task, `2σ² =` trung vị (theo lớp cũ) của khoảng cách bình
phương tới mẫu mới gần nhất, tính trên feature trước adaptation của chính bộ
ước lượng được áp dụng; không dùng test hay oracle.

Quy tắc: một ứng viên **đạt** khi `|Δ AIA| > max(0.3, 2 × sàn nhiễu)` (sàn
nhiễu từ cả hai replicate) **và** McNemar task cuối p < 0.05. Ứng viên đạt
trên seed 1993 mới được xác nhận bằng 3 seed; chỉ kết quả 3 seed mới được
báo cáo như một cải tiến. Ứng viên không đạt vẫn được báo cáo kèm dữ liệu cơ
chế. Baseline `diag_legacy` được chạy lại trong cùng phiên; nếu AIA khác
75.546 thì phải ghi nhận tái lập không trọn vẹn giữa các phiên.

Nếu một ứng viên CA đạt, bước tiếp theo là thay margin đều bằng margin có
trọng số rủi ro PQK theo cặp lớp (Q-CA), so với các control cosine/RBF/MLP và
xáo trộn. Nếu `r2_sigma_paired` đạt, bước tiếp theo là Q-SSCA (kernel PQK
trong trọng số SSCA) với cùng bộ control.
