# Giai đoạn 1b: cận trên trên ImageNet-R

Một lần chạy `QGRSA_EXPERIMENTS.ipynb` với `VARIANTS = UPPER`. Môi trường: RTX PRO 6000
Blackwell, torch 2.11+cu128, chạy 3 run song song. Dữ liệu là Kaggle `zaphat206/imagenet-r`,
cùng bản với baseline `noise` trong `exps/measurement/results/`, seed 1993.

| Run | AIA | Final | Forgetting |
|---|---:|---:|---:|
| `upper_oracle` (rep 0) | 87.491 | 83.11 | 3.88 |
| `upper_oracle_rep1` | 87.607 | 83.05 | 3.81 |
| `upper_joint` (200 lớp, 1 task) | — | **86.52** | — |
| Baseline `noise` (4 run) | 87.347 ± 0.144 | 82.59 ± 0.23 | 5.25 ± 0.19 |

## Hai con số quyết định

So cặp oracle với baseline (`run_level_stats.py compare`, 2 cặp, rep 0 và rep 1):

| Metric | Δ trung bình | SD(Δ) | Ghi chú |
|---|---:|---:|---|
| AIA | **H = +0.195** | 0.031 | Từng cặp +0.17 / +0.22 |
| Final | +0.515 | 0.148 | |
| Forgetting | −1.46 | 0.16 | |

- **H = +0.20 AIA.** Kể cả khi mean và covariance của lớp cũ chính xác tuyệt đối, AIA
  chỉ tăng 0.2. Trên ImageNet-A con số này là +0.66.
- **G = joint − final = 86.52 − 82.59 = 3.9.** Sau khi trừ phần oracle sửa được, còn
  lại 86.52 − 83.08 = 3.4.

Theo bảng quyết định trong `exps/measurement/README.md`, H < 0.5 và 2 ≤ G < 4 rơi vào
**vùng giữa**, không có hướng mặc định. Lý do quyết định được ghi bên dưới.

## Khoảng cách nằm ở đâu

Độ chính xác cuối theo nhóm lớp (task 0 là các lớp học sớm nhất):

| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Joint | 89.1 | 88.7 | 86.4 | 86.4 | 83.2 | 86.0 | 85.8 | 87.0 | 85.4 | 85.8 |
| CIL (4 run) | 80.1 | 80.4 | 82.7 | 82.3 | 78.8 | 82.7 | 85.3 | 84.2 | 83.9 | 86.2 |
| Oracle (2 run) | 87.1 | 83.1 | 83.5 | 82.2 | 79.1 | 81.6 | 84.7 | 82.9 | 82.1 | 83.6 |
| Joint − CIL | **+9.0** | **+8.3** | +3.7 | +4.1 | +4.4 | +3.3 | +0.5 | +2.9 | +1.6 | −0.3 |

1. **Mất mát nằm ở các lớp cũ nhất, không nằm ở các lớp mới.** Task 9 bằng joint, vậy
   khả năng học lớp mới đủ tốt. Gần một nửa khoảng cách thuộc về task 0 và task 1.
2. **Lỗi đặc thù của học tăng dần là nhầm sang task khác.**
   - 4.5% ảnh test được joint đoán đúng nhưng cả 4 run CIL đều đoán sai; 96% số lỗi đó
     là nhầm sang task khác.
   - Tính trên mỗi cặp lớp, tỉ số nhầm khác task / trong task là 0.90 ở joint và 1.38
     ở CIL. Học tăng dần làm tăng khoảng 50% mức nhầm giữa các task.
   - Điều này hiệu chỉnh cách đọc trong `exps/measurement/results/README.md`: mốc so
     sánh đúng là joint (0.90), không phải 1.0.
3. **Oracle cứu được lớp cũ nhưng bù trừ với lớp mới.**
   - Với thống kê đúng, task 0 tăng từ 80.1 lên 87.1, gần bằng joint.
   - Đổi lại, các task 5–9 giảm 1–2.6 điểm, vì các lớp mới bị đoán sang task 0 nhiều
     hơn (cột task 0 trong ma trận nhầm lẫn tăng từ khoảng 1.2% lên khoảng 3%).
   - Tổng lại chỉ còn +0.5 final.
   - Nghĩa là với **biểu diễn hiện tại**, bộ phân loại CA đã gần tối ưu: thống kê sai
     chỉ dịch chuyển nhầm lẫn từ lớp này sang lớp khác. Phần còn lại, khoảng 3.4 điểm,
     nằm ở **biểu diễn**, vì các lớp của các task khác nhau chồng lấn nhiều hơn so với
     khi huấn luyện chung.
4. **Feature gần như không trôi.** Drift/norm ≈ 0.08 mỗi task, và SSCA chỉ giảm sai số
   mean xuống 0.82–0.92 lần mức không bù. Đây là lý do H nhỏ.

## Quyết định (ghi trước khi chạy ứng viên)

- **Loại** hướng Q-SSCA và các chỉnh sửa CA làm hướng chính, vì H = 0.2 AIA.
- Giả thuyết làm việc: phần mất mát còn lại (3.4 final) là **chồng lấn biểu diễn giữa
  các task**, tập trung ở các lớp cũ nhất.
- Trước khi thiết kế phương pháp, cần **1 run chẩn đoán**: huấn luyện một linear probe
  (cosine head) trên feature thật của toàn bộ 200 lớp với adapter cuối cùng của CIL.
  - Probe ≈ 83 (bằng oracle): nút thắt là biểu diễn → can thiệp vào quá trình học
    adapter, tức tách lớp mới khỏi lớp cũ ở mức feature.
  - Probe ≫ 83, tiến gần 86.5: biểu diễn vẫn đủ tốt, nút thắt là mô hình Gaussian của CA
    → can thiệp vào cách sinh feature cho lớp cũ.

## Ghi chú phương pháp đo

SD(Δ) giữa oracle và baseline chỉ là 0.03 AIA, nhỏ hơn nhiều mức giả định
√2·0.144 ≈ 0.20. Lý do là oracle chỉ thay đổi bước CA, còn quá trình học adapter giống
hệt baseline, nên nhiễu chung triệt tiêu gần hết khi lấy hiệu. Hệ quả:

- Ứng viên **chỉ đổi CA**: 2–3 cặp đủ để phát hiện chênh lệch khoảng 0.1–0.2 AIA.
- Ứng viên **đổi quá trình học adapter**: dùng mức giả định thận trọng ban đầu.

## Lưu trữ

`.log`, config, `summary.md` và README được đưa vào git. `*_stats/` (npz/json) và
`console/` chỉ giữ ở máy (`.gitignore`).
