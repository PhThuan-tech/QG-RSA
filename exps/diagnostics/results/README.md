# Kết quả chẩn đoán đã đo

Mỗi thư mục là một lần chạy notebook `QGRSA_EXPERIMENTS.ipynb` (`configs/`,
`logs/.../*_stats/` gồm `stats_task_*.json` và `predictions_task_*.npz`).
Console đã được bỏ vì lặp lại `.log`.

**Không lưu trùng:** `diag_legacy` và `diag_legacy_rep1` của vòng 2 khớp bit với vòng 1
(predictions và curve giống hệt; stats vòng 2 có thêm `ssca_sigma_used`,
`ca_logit_norm`, `ca_task_margin`). Hai run này chỉ được lưu ở `round2_*`, nên vòng 1
không còn tự đủ dữ liệu. Bảng tổng hợp chuẩn là `summary_imageneta_s1993.md`,
sinh từ cả hai vòng:

```bash
python scripts/summarize_stats_diagnostics.py \
  --root exps/diagnostics/results/round1_imageneta_s1993/logs \
  --root exps/diagnostics/results/round2_imageneta_s1993/logs \
  --output exps/diagnostics/results/summary_imageneta_s1993.md
```

`round*/summary.md` là bảng từng vòng, giữ lại để tham khảo.

| Thư mục | Nội dung | Môi trường |
|---|---|---|
| `round1_imageneta_s1993/` | Vòng 1: legacy, paired, oracle_mean, oracle_mean_cov, replicate 1 | RTX PRO 6000 Blackwell, torch 2.11+cu128, 5 song song |
| `round2_imageneta_s1993/` | Vòng 2: legacy (chạy lại), replicate 1–2, σ auto (legacy/paired), CA logit_norm, CA task margin, combo | như trên, 6 song song |

Tất cả: ImageNet-A, split 80/20 seed 67, B0Inc20 (10 task), seed 1993, RSIAT
cổ điển (`exps/rsiat/imageneta.json`).

## Kết luận vòng 1

- Headroom thống kê lớp cũ (oracle_mean_cov − legacy): +0.66 AIA, +1.73 final.
  Phần do mean +0.41, do covariance +0.25. Headroom tăng theo task (task 9: +1.7).
- Kernel SSCA với σ = 4 bị chết: `Floor rows = 1.00` ⇒ SSCA chỉ là phép tịnh
  tiến toàn cục, cách ghép cặp không ảnh hưởng (paired ≈ legacy).
- Khoảng 91% lỗi task cuối là nhầm **giữa các task** (3% nhầm trong cùng task).
  *Hiệu chỉnh (2026-10-21):* mức nền khi nhầm ngẫu nhiên là 180/199 ≈ 90%, vì có
  180 lớp khác task so với 19 lớp cùng task. Tính trên mỗi cặp lớp, nhầm khác task chỉ
  gấp 1.09 lần nhầm trong task. Vì vậy con số 91% không phải bằng chứng của lỗi đặc
  thù học tăng dần. Xem `exps/measurement/results/README.md`.

## Kết luận vòng 2

1. **Tái lập tuyệt đối giữa các phiên**: `diag_legacy` (75.546) và `diag_legacy_rep1`
   (75.751) trùng từng chữ số với vòng 1.
2. **Nhiễu huấn luyện lớn hơn ước lượng ban đầu**. Ba run chỉ khác ngẫu nhiên
   SGD (legacy, rep1, rep2) có AIA 75.546 / 75.751 / 76.285 (SD 0.38) và final
   67.07 / 67.80 / 68.00 (SD 0.49). Riêng rep2 so với legacy đã là +0.74 AIA,
   McNemar p = 0.087. Như vậy với một seed, chênh lệch AIA dưới khoảng 1 pp
   không diễn giải được.
3. **Sửa phương pháp thống kê**: McNemar chỉ tính phương sai do mẫu test, không
   tính phương sai do huấn luyện. Bằng chứng là rep2 (thuần nhiễu) đạt p = 0.087,
   còn combo p = 0.03 mà vẫn kém rep2 về AIA. Đơn vị kiểm định đúng phải là
   **run huấn luyện** (replicate × seed); McNemar chỉ còn là chỉ số phụ.
4. **Không ứng viên CA nào vượt nhiễu**. So với trung bình 3 run legacy
   (75.861): logit_norm −0.11, task margin +0.01, combo +0.38 AIA. Tỉ lệ nhầm
   "sang task mới" vẫn ≈ 18.3–19.6%, giống các replicate.
5. **Lỗi ghép cặp SSCA được xác nhận là có hại một khi kernel cục bộ**.
   `r2_sigma_legacy` (σ tự động, ghép cặp ngẫu nhiên) cho final 64.07, tức
   −3.6 pp so với trung bình legacy (≈ 7 SD). Sai số mean tăng lên 1.18–1.30
   (tệ hơn không bù), drift ước lượng tích lũy lên 0.36 × norm, nhầm "sang task
   mới" lên 25.3%. RSIAT gốc chỉ an toàn vì σ = 4 làm kernel suy biến thành
   trung bình toàn cục; khi đó tổng các hiệu không phụ thuộc cách ghép cặp.
6. **Locality Euclid trong 768 chiều gần như không đạt được**. Với σ ≈ 26–28,
   `Floor rows` = 0 nhưng ESS vẫn ≈ 85% số mẫu: khoảng cách tới mẫu gần nhất
   (~1400) và trung vị (~2800) chỉ khác nhau một hệ số e (hiện tượng tập trung
   khoảng cách). Vì vậy `r2_sigma_paired` ≈ legacy. Muốn bù drift cục bộ cần một
   metric trong không gian học được, ít chiều (ví dụ projector của PQK).

## Ghi chú vận hành

Với 6 run song song, mỗi run mất khoảng 55 phút; 2 run song song chỉ khoảng 15
phút mỗi run. Tức là chạy 6 song song bị nghẽn CPU/IO, tổng thông lượng không
tốt hơn. Nên đặt `MAX_PARALLEL = 3`.
