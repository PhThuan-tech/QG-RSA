# QSD-CIL giai đoạn A: kết quả (2026-10-22)

Bảng đầy đủ: `summary.md`, `summary.json`. Lệnh tạo bảng: README đặc tả, mục 7, với các `--root` trỏ vào
`kaggle{1,2,3}/logs` (thư mục này).

10 run, gồm ImageNet-R và ImageNet-A, mỗi dataset 3 replicate seed 1993 và thêm seed 2015, 2026.
Không run nào lỗi.

## Kiểm tra hồi quy: đạt

- Ở cả 10 run, head `linear` trùng CNN curve.
- 6 run seed 1993 trùng từng chữ số với baseline cùng replicate: `noise` trên R, `diag_legacy` trên A.
- Hook QSD không làm thay đổi quá trình huấn luyện. Các so sánh trong cùng run là hợp lệ.

## Kết quả chính (Δ AIA so với `linear`, CI 95%)

| Head | R seed 1993 (n=3) | R gộp (n=5) | A seed 1993 (n=3) | A gộp (n=5) |
|---|---|---|---|---|
| `pgm_channel` | −1.03 [−1.15, −0.91] | −1.39 [−2.01, −0.77] | −1.19 [−1.83, −0.55] | −0.97 [−1.40, −0.54] |
| `pgm_channel_hier` | −0.53 [−0.83, −0.23] | −0.64 [−0.86, −0.42] | −0.97 [−2.03, +0.08] | −0.62 [−1.42, +0.18] |
| **`pgm_fuse`** | **+0.22 [+0.17, +0.26]** | **+0.25 [+0.19, +0.31]** | **+0.35 [+0.19, +0.50]** | **+0.44 [+0.26, +0.61]** |
| `ridge_stats` | −1.20 | −1.40 | −0.74 | −0.99 |
| `pgm_nonlinear` − `pgm_channel` | −3.75 | −3.63 | −5.70 | −6.50 |

Theo quy tắc đặt trước:

- **ImageNet-A:** `pgm_fuse` **ĐẠT**, cả ở bước sàng lọc lẫn khi gộp mọi seed.
- **ImageNet-R:** CI của `pgm_fuse` > 0, nhưng Δ = 0.22–0.25 **dưới ngưỡng +0.3**, nên "không kết luận".
- `pgm_fuse` thắng ở **cả 10/10 run**, khoảng +0.2 đến +0.6 AIA và +0.3 đến +0.5 final.
- Trên R, mức tăng này lớn hơn headroom của oracle stats (+0.195).

## Các khẳng định thành phần: không khẳng định nào giữ được

1. **Kênh vận chuyển vô tác dụng.**
   - `pgm_channel − pgm_stale` ≈ 0, với R gộp −0.03 [−0.19, +0.14].
   - Kênh đo được lệch khỏi identity khoảng 7–9% mỗi task, đúng như dự báo. Nhưng trạng thái bậc hai không nhạy với drift ở mức này.
   - Đây là bằng chứng thứ ba, sau SSCA và oracle stats, rằng **sửa drift thống kê không phải đòn bẩy**.
2. **Mã hoá phi tuyến "hai bản sao" có hại.** Kém 3.6–6.5 AIA. Thêm u⊗u chỉ thêm chiều nhiễu cho một PGM chỉ có khoảng 30–150 mẫu mỗi lớp.
3. **Heuristic co mean của CA vô hại.** Bỏ heuristic này không giúp gì: R −0.03, A −0.10.
4. **PGM tương đương ridge_stats.**
   - R gộp: `pgm_channel − ridge_stats` = +0.01 [−0.27, +0.29].
   - Đúng như lý thuyết: PGM trên trạng thái bậc hai thực chất là phép đo căn bậc hai, tức một bộ phân loại ridge/LDA trên momen thứ hai.
   - **Không có lợi thế nào từ hình thức lượng tử khi dùng riêng.**
   - Độ nhạy δ phẳng: 0.01 và 0.1 chênh dưới 0.3, còn 1.0 kém hơn khoảng 0.7.
5. **Bản phân cấp cân bằng lại task nhưng không tăng accuracy.**
   - `pgm_channel_hier` nâng task 0 thêm +2.6 (R) và đưa tỉ số nhầm khác/trong task về khoảng 1.0, gần mức huấn luyện chung 0.90.
   - Tuy vậy, độ chính xác trong task giảm nhiều hơn phần thu được.
   - Vậy tỉ số này **không** phải đại lượng nên tối ưu trực tiếp.

## Cơ chế của `pgm_fuse`: là ensemble, chưa chứng minh được là lượng tử

Tính trên dự đoán ở task cuối:

| | R: đúng thêm / sai thêm so với linear | R: oracle(linear ∨ head) | A: đúng thêm / sai thêm | A: oracle |
|---|---|---|---|---|
| `ridge_stats` | +2.9–3.4 / −5.3–6.1 | 85.4–86.0 | +4.6–6.0 / −5.9–7.5 | 71.9–73.5 |
| `pgm_channel` | +2.9–3.4 / −5.8–6.5 | 85.7–86.0 | +4.1–5.5 / −6.9–7.7 | 71.4–73.1 |
| `pgm_fuse` | +0.6–1.0 / −0.3–0.4 | | +1.1–1.7 / −0.7–1.1 | |

Phân tích:

- Một head sinh (generative) trên thống kê đã lưu sửa được khoảng 3% (R) và 5% (A) mẫu mà head CA sai.
- `ridge_stats`, một head **cổ điển**, bổ sung cho linear **ngang bằng** PGM: oracle 85.4–86.0 so với 85.7–86.0.
- Vì vậy mức tăng của `pgm_fuse` rất có thể là lợi ích chung của việc **kết hợp head phân biệt (CA) với head sinh (thống kê)**. Lợi ích này không đến từ phép đo lượng tử.

**Lỗ hổng của đặc tả:** đối chứng đúng cho `pgm_fuse` là `ridge_fuse` (log p_linear + log p_ridge), chứ không phải `ridge_stats` đứng riêng. Head `ridge_fuse` không được đặt trước. File `.npz` chỉ lưu argmax nên không tính bù được. Trước khi đưa `pgm_fuse` vào paper như một đóng góp lượng tử, cần chạy đối chứng này.

## Phân tích sâu (`analyze_qsd.py`, tính trên dự đoán argmax)

Lưu ý: ở t=0 các replicate trùng nhau, vì `rng_replicate` chỉ reseed từ task 1. Vậy ở t=0 chỉ có 3 điểm độc lập, ứng với 3 seed.

1. **Khi chưa có drift, PGM là bộ phân loại tốt.**
   - Ở t=0 trên R: PGM − linear = −0.31 [−0.77, +0.15], và PGM − `ridge_stats` = **+0.68 [+0.44, +0.92]**.
   - Ở t=0 trên A: PGM − linear = +1.57 (cả 3 seed đều dương), SD 0.45 so với 1.62 của linear.
   - Trên cùng thống kê bậc hai, phép đo PGM tốt hơn ridge cổ điển.
2. **Mức thua của PGM tăng tuyến tính theo task.** Trên R, PGM − linear đi từ −0.31 (t0) xuống −2.79 (t9).
   - Phần thua tăng thêm nằm gần như hoàn toàn ở **nhầm sang task mới hơn**: 11.54% so với 9.91% của linear.
   - Đây là thiên lệch về task gần nhất. Trạng thái lớp mới được tính trên feature hiện tại nên "sắc", còn trạng thái lớp cũ thì cũ.
   - Kênh không sửa được điều này (≈ stale).
   - `ridge_stats` dùng mean đã được SSCA cập nhật, nên suy giảm chậm hơn PGM (−0.99 → −2.62). Lợi thế +0.68 của PGM ở t0 mất hết ở t9.
3. **Bản phân cấp cắt được khoảng 2/3 phần suy giảm** (−2.79 → −0.90 ở t9; task 0 +2.6). Cái giá là độ chính xác trong task giảm: within|task 97.9 so với 98.5.
4. **`pgm_fuse` chỉ sửa lỗi phân xử task.**
   - Trên R, nó sửa 0.71% mẫu nhầm khác task nhưng làm hỏng 0.36%. Lỗi trong task gần như không đổi.
   - Gain tăng dần theo t: +0.04 → +0.37 trên R. Gain chia đều cho lớp cũ (+0.29) và lớp mới (+0.26).
5. **Fuse còn bỏ phí phần lớn tính bổ sung.**
   - Trên R, linear và PGM bất đồng ở 14.4% mẫu: linear đúng 5.8%, PGM đúng 3.2%, cả hai sai 5.4%.
   - Khi bất đồng, fuse chọn theo linear 87% số lần và chỉ đúng 43%.
   - Lý do: xác suất Born Tr(E_c ρ) phẳng hơn nhiều so với softmax(scale·cos), nên log p_PGM chỉ đóng vai trò phá hoà.
   - Trên A: bất đồng 22%, PGM đúng 4.6%.
6. **Mã hoá phi tuyến kém ngay từ t=0** (−2.5 trên R). Vấn đề là feature map, không phải drift.

## Kết luận

- QSD-CIL theo đúng đặc tả (PGM, kênh, phân cấp) **không thắng RSIAT**.
- Hướng duy nhất có tín hiệu nhất quán là `pgm_fuse`. Mức tăng nhỏ nhưng cực ổn định: SD(Δ) khoảng 0.05 trên R. Cơ chế là kết hợp head phân biệt với head sinh, và chưa tách được phần nào là lượng tử.

Bước tiếp theo: vòng A2, đặc tả ở `exps/qsd/README_A2.md`. Trong A2:
- ý tưởng `pgm_ssca` được hiện thực bằng `pgm_stats`: trạng thái dựng từ mean SSCA và covariance, exemplar-free;
- `ridge_fuse` trở thành nguồn `ridge_stats` trong lưới fusion.

Đề xuất ban đầu:

- **(a)** Một vòng chỉ suy luận, thêm head `ridge_fuse` và lưu điểm số. Vòng này dùng lại được cùng replicate và chỉ tốn thêm vài phút mỗi run.
  - Nếu `pgm_fuse − ridge_fuse` có CI > 0, phần lượng tử có đóng góp thật.
  - Nếu không, đóng góp được viết thành "fusion sinh–phân biệt", và PGM chỉ là một cách diễn giải.
- **(b)** Chuyển sang giai đoạn B: steering adapter bằng phép đo. Giai đoạn A cho thấy chỉnh ở bước suy luận chỉ đạt trần khoảng +0.2 đến +0.4 AIA. Muốn tăng nhiều hơn phải sửa chính biểu diễn, phù hợp với kết luận của vòng UPPER.
