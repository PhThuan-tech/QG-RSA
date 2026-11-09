# Protocol đo lường (đặt trước khi có dữ liệu)

## Vì sao cần

Vòng 2 cho thấy trên ImageNet-A, ba run chỉ khác ngẫu nhiên SGD (cùng seed,
cùng base task 0) có SD = 0.38 AIA và 0.49 final. Một replicate thuần nhiễu
đã đạt +0.74 AIA. Vì vậy:

- So sánh một seed, một run cho mỗi bên không phân biệt được hiệu ứng dưới
  khoảng 1 pp.
- McNemar theo ảnh chỉ đo phương sai lấy mẫu test, không đo phương sai huấn
  luyện, nên **không** được dùng để quyết định.

## Đơn vị và thiết kế

**Đơn vị phân tích là một run huấn luyện.** Một run được định danh bởi
(dataset, seed, replicate). `rng_replicate = k` giữ nguyên base task 0 và đổi
luồng ngẫu nhiên từ task 1.

**So sánh ứng viên với baseline theo cặp ghép (seed, replicate)**: hai run cùng
cặp dùng chung base và chung luồng ngẫu nhiên (common random numbers), nên
phần nhiễu chung triệt tiêu khi lấy hiệu.

- Hiệu ứng: Δ = trung bình của d = AIA(ứng viên) − AIA(baseline) qua các cặp.
- Khoảng tin cậy: CI 95% kiểu t trên các d, bậc tự do n − 1. Phương sai lấy
  thực nghiệm từ chính các d, nên tự động phản ánh mức tương quan do common
  random numbers.
- **Đạt**: n ≥ 3 cặp và cận dưới CI 95% của Δ AIA lớn hơn 0.
- **Kém hơn**: cận trên CI nhỏ hơn 0.
- **Không kết luận**: CI chứa 0. Báo cáo Δ, CI và nói rõ hiệu ứng nhỏ hơn mức
  phát hiện được.
- Final, forgetting và McNemar chỉ là chỉ số phụ, không dùng để quyết định.

## Giai đoạn 1: hiệu chỉnh nhiễu (đang làm)

Bốn run baseline legacy cho mỗi benchmark, seed 1993: overlay `noise_rep0`
đến `noise_rep3` (prefix `noise`, `noise_rep1..3`). Tắt chẩn đoán oracle, chỉ
lưu dự đoán.

| Benchmark | Trạng thái | Ghi chú |
|---|---|---|
| ImageNet-A | Đã có 3 run (`diag_legacy`, `_rep1`, `_rep2` trong `exps/diagnostics/results/round2_*`) | 6 000 train / 1 500 test |
| ImageNet-R | **Xong 4 run: AIA 87.347 ± 0.144, final 82.59 ± 0.23** → benchmark sàng lọc (`results/README.md`) | 23 918 train / 6 082 test; ~118 phút/run khi 3 song song |
| CIFAR-100 | Tuỳ chọn | Mỗi run ~8× ImageNet-A; AIA ~95% nên ít dư địa cải thiện |

ImageNet-A được chia 80/20 theo lớp với seed 67. ImageNet-R dùng bản đã chia
sẵn trên Kaggle `zaphat206/imagenet-r` (23 918 train / 6 082 test, 200 lớp
khớp nhau, không trùng ảnh, tỉ lệ train 0.78–0.80 mỗi lớp); notebook tự nhận
bản này và dùng nguyên. Số tuyệt đối có thể khác protocol công bố của RSIAT,
nhưng mọi so sánh đều nội bộ và cùng cách chia, nên mọi run ImageNet-R phải
dùng cùng một bản dữ liệu.

`scripts/run_level_stats.py noise` báo cho mỗi benchmark: SD của AIA/final/
forgetting; số cặp cần để phát hiện Δ AIA ∈ {0.3, 0.5, 1.0} với power 80% và
α = 5%, giả định thận trọng sd(Δ) = √2·SD; và chi phí
= số cặp × 2 run × phút/run.

**Quy tắc chọn benchmark sàng lọc**: benchmark có ≥ 3 replicate hoàn tất và
chi phí thấp nhất để phát hiện Δ AIA = 0.5. Thời gian mỗi run phụ thuộc mức
chạy song song, nên chỉ so chi phí giữa các benchmark đo cùng điều kiện;
ImageNet-A cũ được đo với 6 run song song.

## Giai đoạn 1b: cận trên trên ImageNet-R (variants `UPPER`)

Mục tiêu là đo dư địa trước khi chọn hướng phương pháp:

- (overlay nay nằm ở `exps/upper/`) `upper_oracle_rep0`, `upper_oracle_rep1`: CA dùng mean và covariance thật của lớp
  cũ. Hai run này ghép cặp với `noise` / `noise_rep1`.
- `upper_joint`: một task duy nhất gồm 200 lớp với cùng công thức huấn luyện task 0.
  Đây là cận trên khi không học tăng dần.

Quy tắc đặt trước, với G = joint − final của baseline (82.59) và
H = Δ AIA trung bình của oracle so với baseline:

| Kết quả | Hướng tiếp theo |
|---|---|
| H ≥ 1.0 | Ước lượng thống kê và drift lớp cũ là đòn bẩy chính → Q-SSCA (metric drift trong không gian projector PQK ít chiều) |
| H < 0.5 và G ≥ 4 | Mất mát nằm ở biểu diễn (adapter bị trôi hoặc kém dẻo), không nằm ở thống kê → nhắm vào align/orth của adapter |
| H < 0.5 và G < 2 | RSIAT đã gần trần trên ImageNet-R. Chuyển sang benchmark khó hơn, hoặc nhắm vào giai đoạn đầu (backbone/task 0) |

Các trường hợp nằm giữa (0.5 ≤ H < 1, hoặc 2 ≤ G < 4) không có lựa chọn mặc định: xem cả hai số rồi quyết định, và ghi lý do vào đây trước khi chạy ứng viên.

## Giai đoạn 2: đánh giá ứng viên

1. **Sàng lọc** trên benchmark đã chọn: ứng viên và baseline cùng chạy các cặp
   (seed 1993, replicate 0..n−1). n lấy từ bảng "cặp cần" ở mức Δ = 1.0
   (tối thiểu 3). Ứng viên "kém hơn" bị loại; "đạt" được xác nhận ở bước 2;
   "không kết luận" chỉ được tiếp tục nếu có lý do cơ chế đã ghi trước.
2. **Xác nhận**: thêm seed 2015 và 2026 (base task 0 khác, thứ tự lớp khác),
   mỗi seed ≥ 1 cặp, và chạy thêm một benchmark thứ hai. Chỉ kết quả ở bước
   này được báo cáo như một cải tiến.

Lệnh so sánh:

```bash
python scripts/run_level_stats.py compare --root <logs> --baseline noise --candidate <family>
```

Để ghép cặp, ứng viên cần các overlay riêng cho từng replicate với prefix
`<family>`, `<family>_rep1`, ... và `rng_replicate` tương ứng.
