# QSD-CIL vòng A2: kết quả (2026-10-22)

- Đặc tả: `exps/qsd/README_A2.md`.
- Bảng đầy đủ: `summary.md` và `summary.json`. Lệnh tạo bảng ở mục 7 của đặc tả.
- Phân tích thăm dò (không đặt trước): `a2_explore.py`.

11 run, không run nào lỗi:
- tuning: ImageNet-R seed 1993 × 3;
- held-out: ImageNet-R seed 2015, 2026, 2042 và ImageNet-A seed 1993 × 3, 2015, 2026.

## Kiểm tra hồi quy: đạt

- Ở cả 11 run, `linear` trùng CNN curve.
- Ở 10 run có run giai đoạn A tương ứng, 5 head giữ lại (`linear`, `ridge_stats`, `pgm_channel`, `pgm_channel_hier`, `pgm_fuse`) trùng từng chữ số.
- Vậy huấn luyện giống từng bit, và phép tính của giai đoạn A không đổi.

## Quyết định theo quy tắc đặt trước

| Quy tắc | Kết quả (held-out, Δ AIA, CI 95%) | Kết luận |
|---|---|---|
| **M.** Fusion `pgm_stats` (w*=0.5) thắng linear | gộp **+0.46 [+0.28, +0.64]**; R +0.35, A +0.53 | **ĐẠT** |
| **Q.** Fusion `pgm_stats` − fusion `ridge_stats` (w*=0.25) | **−0.16 [−0.39, +0.07]**; R +0.06, A −0.29 | **Không giữ được** |
| **S.** `pgm_stats − pgm_stats_stale` (mean SSCA so với mean cũ) | +0.13 [+0.08, +0.18] | Có, nhưng nhỏ |
| **E.** `pgm_stats − ridge_stats` (cùng thống kê) | +0.09 [−0.23, +0.40] | Không |
| **H.** `pgm_stats_hier − pgm_stats` | +0.33 [+0.02, +0.64] | Có |
| **O.** `pgm_stats_oracle − pgm_stats`, final | +0.64 [+0.48, +0.80], dưới ngưỡng +1.0 | Không đáng làm transport |

Số tuyệt đối:

| Dataset | linear (RSIAT) | Fusion `pgm_stats` | Fusion `ridge_stats` | Tốt nhất trong lưới |
|---|---|---|---|---|
| ImageNet-R, 6 run | 87.25 | 87.63 | 87.59 | 87.71 (`fuse_pgm_channel_hier_b1`, không exemplar-free) |
| ImageNet-A, 5 run | 74.97 | 75.49 | **75.78** | |

## Diễn giải

1. **Fusion với một head sinh trên thống kê đã lưu là cải tiến thật, nhất quán và tái lập được.**
   - Mọi nguồn đều cho +0.45 đến +0.62 AIA trên held-out, với CI > 0.
   - Đây là mức tăng lớn nhất trong cả chuỗi thí nghiệm của dự án.
   - Mức này lớn hơn 3 lần headroom của oracle stats (+0.195).
2. **Phần lượng tử không mang thêm gì.**
   - Ridge cổ điển trên cùng thống kê ngang PGM trên ImageNet-R và tốt hơn trên ImageNet-A, nơi chỉ có 30 ảnh mỗi lớp.
   - Lưới mịn hơn (thăm dò) xác nhận 0.25 là trọng số tối ưu của ridge, không phải do chạm biên lưới.
   - Lợi thế của PGM ở t=0 (+0.68 trên R, giai đoạn A) không còn sau khi fusion.
3. **Trạng thái tốt hơn chỉ giúp ở các lớp cũ nhất, không giúp tổng thể.**
   - Oracle nâng task 0 thêm +4.4 trên R (81.9 → 86.4), nhưng AIA chỉ +0.08 và final +0.64.
   - Đây là cùng mẫu hình với vòng UPPER: sửa thống kê lớp cũ thì làm hại lớp mới.
   - Nút thắt nằm ở biểu diễn, không ở thống kê.
4. **Kết hợp ba nguồn chỉ thêm một chút** (thăm dò, không đặt trước):
   - linear + ridge + `pgm_stats_hier`: held-out gộp +0.71 [+0.44, +0.97] (R +0.40, A +0.90);
   - linear + ridge + `pgm_stats`: +0.57.
   - Như vậy trần thực tế của bước suy luận là khoảng +0.4 trên R và +0.8 trên A.

## Kết luận

- Đóng góp có bằng chứng là một **fusion sinh–phân biệt có hiệu chỉnh, exemplar-free**:
  - không cần huấn luyện thêm;
  - chi phí dưới 1 phút mỗi task;
  - +0.3 đến +0.4 AIA trên ImageNet-R và +0.5 đến +0.8 trên ImageNet-A.
- Bản cổ điển (ridge) đã đạt toàn bộ mức tăng. PGM/QSD **không được** trình bày như một lợi thế lượng tử. Nó chỉ dùng được như một cách diễn giải thống nhất, ví dụ "ridge trên momen bậc hai ≈ phép đo căn bậc hai".
- Không nên đầu tư thêm vào vận chuyển trạng thái (quy tắc O).
