# Giai đoạn 1: nhiễu huấn luyện trên ImageNet-R

Một lần chạy `QGRSA_EXPERIMENTS.ipynb` (variants `MEASURE`). Môi trường: RTX PRO 6000
Blackwell Server, torch 2.11+cu128, timm 1.0.29, 3 run song song. Dữ liệu: Kaggle
`zaphat206/imagenet-r` (bản chia sẵn 23 918 / 6 082 ảnh), B0Inc20 (10 task), seed 1993.
RSIAT cổ điển (`exps/rsiat/imagenetr.json`) với `noise_rep0..3`.

Bảng chính nằm ở `noise.md`, được sinh bằng `scripts/run_level_stats.py noise`.

| Run | AIA | Final |
|---|---:|---:|
| noise (rep 0) | 87.318 | 82.49 |
| noise_rep1 | 87.390 | 82.64 |
| noise_rep2 | 87.512 | 82.88 |
| noise_rep3 | 87.167 | 82.34 |
| **Trung bình ± SD** | **87.347 ± 0.144** | **82.59 ± 0.23** |

Base task 0 giống nhau ở cả 4 run (96.05), đúng thiết kế của `rng_replicate`.

## Kết luận

1. **Chọn ImageNet-R làm benchmark sàng lọc.** SD của AIA là 0.144, bằng khoảng 1/2.6
   so với ImageNet-A (0.38). Với giả định thận trọng sd(Δ) = √2·SD, cần 3 cặp để phát
   hiện Δ AIA = 0.5, và 4 cặp cho Δ = 0.3.
2. **Có thể dùng lại baseline.** Các run `noise`, `noise_rep1`, `noise_rep2` là baseline
   ghép cặp cho mọi ứng viên, miễn là code của baseline và cùng bản dữ liệu, cùng môi
   trường không đổi. Mỗi ứng viên khi đó chỉ cần 3 run, khoảng 2 giờ nếu chạy 3 song
   song.
3. **Kết quả cũ "QKSR vượt RSIAT trên ImageNet-R" không có giá trị bằng chứng.**
   - Kết quả cũ: QKSR đạt 87.28 (1 run, 2026-10-03), RSIAT đạt 86.924 (1 run,
     2025-11-20, khác phiên bản code, môi trường và có thể khác cách chia dữ liệu).
   - Chính RSIAT, khi chỉ đổi ngẫu nhiên SGD, đã dao động 87.17–87.51.
   - Mức +0.36 vì thế nằm trong nhiễu cộng với chênh lệch môi trường.
   - Muốn khẳng định lại, phải chạy QKSR theo các cặp (seed, replicate) của protocol này.

## Cấu trúc lỗi ở task cuối (trung bình 4 run, % số ảnh test)

| | Lỗi | Trong cùng task | Sang task cũ hơn | Sang task mới hơn |
|---|---:|---:|---:|---:|
| ImageNet-R | 17.4 | 1.24 | 6.29 | 9.88 |
| ImageNet-A (round 2, 3 run) | 32.4 | 2.87 | 11.0 | 18.5 |

**Hiệu chỉnh cách đọc so với vòng 1.** Mỗi lớp có 19 lớp cùng task nhưng có 180 lớp
khác task. Nếu nhầm lẫn là ngẫu nhiên, khoảng 90% lỗi đã rơi vào nhóm "khác task".
Vì vậy chỉ số đúng là **tỉ lệ nhầm trên mỗi cặp lớp**:

| | Nhầm/cặp trong task | Nhầm/cặp khác task | Tỉ số |
|---|---:|---:|---:|
| ImageNet-R | 0.065 | 0.090 | 1.38 |
| ImageNet-A | 0.151 | 0.164 | 1.09 |

Như vậy, nhầm lẫn giữa các task chỉ cao hơn mức nền 1.1–1.4 lần. Phần lớn lỗi là ảnh
khó đối với backbone, không phải hỏng hóc đặc thù của học tăng dần. Câu "91% lỗi là
nhầm giữa task" ở vòng 1 đúng về số liệu, nhưng diễn giải nó như bằng chứng của lỗi
tăng dần là sai.

Các chỉ số khác:

- **Không có thiên lệch về task mới nhất.** Chỉ 14% lỗi khác task trỏ vào task cuối,
  gần mức 1/9 nếu trải đều.
- **Lớp cũ và lớp mới cân bằng.** Ở từng giai đoạn, độ chính xác lớp cũ và lớp mới
  chênh nhau trong vòng ±4 pp, ví dụ task 9: cũ 82.1, mới 86.0. CA đang cân bằng tốt.
- **Quên vừa phải.** Độ chính xác cuối của các lớp task 0 là 80.1, trong khi lúc kết
  thúc task 0 là 96.05. Độ chính xác cuối theo task trải từ 78.8 đến 86.2.
- **Độ ổn định theo ảnh.** 80.4% ảnh luôn đúng và 15.2% ảnh luôn sai ở cả 4 run; chỉ
  4.5% ảnh đổi kết quả. Hai run bất kỳ khác nhau trên 2.45% số ảnh.

## Hệ quả cho bước tiếp theo

Chưa đo được dư địa thực tế trên ImageNet-R. Trước khi chọn phương pháp, cần hai run
cận trên:

1. `oracle_mean_cov`: dư địa của thống kê lớp cũ. Trên ImageNet-A chỉ là +0.66 AIA.
2. Huấn luyện chung 200 lớp trong một task với cùng công thức huấn luyện: khoảng cách
   giữa học tăng dần và cận trên khi không học tăng dần.

Script phân tích: `analyze_noise.py` (đọc `predictions_task_*.npz`).
