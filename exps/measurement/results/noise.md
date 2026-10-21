# Nhiễu huấn luyện theo benchmark

SD giữa các replicate (cùng seed, cùng base task 0, khác ngẫu nhiên SGD từ task 1). Số cặp cần để phát hiện Δ AIA với power 80%, α = 5%, giả định thận trọng sd(Δ) = √2·SD (không tính lợi ích của common random numbers).

| Dataset | Family | Seed | n | AIA mean ± SD | Final mean ± SD | Forgetting mean ± SD | Phút/run | Cặp cần (Δ=0.3 / 0.5 / 1.0) | Chi phí Δ=0.5 (giờ GPU tuần tự) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| imagenetr | noise | 1993 | 4 | 87.347 ± 0.144 | 82.59 ± 0.23 | 5.25 ± 0.19 | 118 | 4 / 3 / 3 | 11.8 |

**Benchmark sàng lọc theo quy tắc đặt trước:** `imagenetr` (chi phí thấp nhất để phát hiện Δ AIA = 0.5 với ≥ 3 replicate).

Lưu ý: phút/run đo trong điều kiện song song lúc chạy; chỉ so sánh tương đối giữa các benchmark chạy cùng điều kiện.
