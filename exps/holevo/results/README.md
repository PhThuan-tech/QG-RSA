# Holevo-CIL (A) và Holevo-POVM (A+B): kết quả sàng lọc W02 + W03 (ImageNet-R seed 1993)

Nguồn: `kaggle/waves/w02_holevo` và `kaggle/waves/w03_hp` (base `c7bc4fd` + patch), chạy ngày 2026-10-30. Môi trường: RTX PRO 6000 Blackwell, torch 2.11.0+cu128, timm 1.0.29, 3 run song song.

File gốc:
- `kaggle1/` (A): log, config, `environment.json`, `SUMMARY.txt`, `summarize_holevo.md/json`.
- `../../holevo_povm/results/kaggle1/` (A+B): cùng cấu trúc.
- `console/` và `_stats/` chỉ giữ ở máy.

Đặc tả: `exps/holevo/README.md` (A), `exps/holevo_povm/README.md` (A+B). Bảng luật do `scripts/summarize_holevo.py` sinh ra.

## Kết luận

**Cả A lẫn A+B: DỪNG theo luật sàng lọc, không chạy held-out.**

1. **Không config nào vượt RSIAT.**
   - AIA của mọi config nằm trong dải 4 run RSIAT lặp lại (87.17–87.51).
   - Final cao nhất là `hp_povm` với +0.21, dưới ngưỡng O (+0.5), và vẫn trong dải nhiễu (82.34–82.88).
2. **Cơ chế nhắm tới không dịch chuyển.** Pair ratio, tức tỉ lệ nhầm giữa các task so với trong một task, giữ nguyên ~1.38–1.44. Luật M yêu cầu ≤ 1.281.
3. **Có một tác động thật, nhưng chỉ là đánh đổi.**
   - `spread`, `hp_povm` và `hp_full` giữ task 0–1 tốt hơn +1.4 đến +1.9 điểm (khoảng 5 SD) và giảm forgetting khoảng 0.8.
   - Đổi lại, các task 5–9 mất 0.4–0.7, nên final và AIA gần như đứng yên.
   - Đây là cùng đường đánh đổi ổn định–linh hoạt đã gặp ở oracle (`upper/`) và SVF (`svf/`).
4. **Phần lượng tử không đóng góp.**
   - `holevo_vn` (Holevo χ đầy đủ) cho kết quả giống hệt RSIAT ở mọi nhóm task.
   - Bản cổ điển `mcr2` còn nhỉnh hơn.
   - Thứ có tác dụng là `spread`, tức chỉ tăng S(ρ̄). Số hạng −S(ρ_c), thứ làm χ khác với việc chỉ trải rộng, đã triệt tiêu hiệu ứng.
   - Với A+B, phép so sánh lượng tử–cổ điển **không hợp lệ**, vì `hp_twin` sụp do lỗi hiệu chỉnh (xem mục cuối).
5. **Decoder POVM bị lách (luật G trượt).** Ảnh thật của lớp cũ bị POVM gán sang lớp mới nhiều hơn theo thời gian (I_orc 0.28–0.30, so với 0.22 ở `hp_diag`). Decoder khớp với trạng thái dựng từ thống kê chứ không khớp với feature thật.

## Bảng chính (một run mỗi config)

Baseline là 4 run `noise` (`measurement/results/`): final 82.59 ± 0.23, AIA 87.35 ± 0.14.

`hp_diag` tính mọi thứ nhưng không cộng vào loss, nên chính là RSIAT chạy trên Kaggle. Accuracy theo từng task của nó **trùng tới hai chữ số thập phân** với run RSIAT của W01 (`svf/results/`). Như vậy RSIAT trên Kaggle là tất định, và `hp_diag` là đối chứng khớp nhất.

| Config | Hướng | AIA | Final | Δ final vs RSIAT TB | Δ final vs `hp_diag` | Forgetting ↓ | Pair ratio | λ (hiệu chỉnh) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| RSIAT, TB 4 run (min–max) | — | 87.35 (87.17–87.51) | 82.59 (82.34–82.88) | — | | 5.25 | 1.381 | |
| `hp_diag` (= RSIAT Kaggle) | đối chứng | 87.25 | 82.42 | −0.17 | — | 5.39 | 1.440 | |
| `holevo_vn` **(chính A)** | A, χ lượng tử | 87.25 | 82.41 | −0.18 | −0.01 | 5.31 | 1.381 | enc 722 |
| `holevo_mcr2` | A, bản cổ điển | 87.19 | 82.54 | −0.05 | +0.12 | 5.10 | 1.409 | enc 8.59 |
| `holevo_spread` | A, chỉ S(ρ̄) | 87.28 | 82.70 | +0.11 | +0.28 | 4.56 | 1.395 | enc 273 |
| `hp_full` **(chính A+B)** | χ + POVM | **87.31** | 82.70 | +0.11 | +0.28 | 4.56 | 1.416 | enc 361, dec 0.516 |
| `hp_povm` | chỉ POVM | 87.30 | **82.80** | +0.21 | +0.38 | **4.39** | 1.386 | dec 1.03 |
| `hp_twin` | χ MCR² + Gaussian | 10.45 | 0.46 | sụp | | | | dec 1000 (bị chặn) |

Task 0 sau task 0 đạt 96.05 ở mọi run (luật R ĐẠT), nên phần mới không làm thay đổi task 0.

## Accuracy theo nhóm task (sau task cuối)

| | Task 0–1 | Task 2–4 | Task 5–9 |
|---|---:|---:|---:|
| RSIAT, 4 run (± SD) | 80.23 ± 0.35 | 81.24 ± 0.23 | 84.44 ± 0.21 |
| `hp_diag` | 79.70 | 80.99 | 84.49 |
| `holevo_vn` | 79.63 | 80.98 | 84.46 |
| `holevo_mcr2` | 81.02 | 81.10 | 84.00 |
| `holevo_spread` | 82.03 | 81.20 | 83.76 |
| `hp_full` | 81.58 | 81.14 | 84.04 |
| `hp_povm` | **82.09** | 81.03 | 84.08 |

Mức tăng ở task cũ là thật, nhưng phải trả bằng task mới. Các task giữa (2–4), nơi phần lớn mất mát dồn lại theo `upper/`, không thay đổi.

## Luật sàng lọc

| Config | R | G | M | O | P | D |
|---|---|---|---|---|---|---|
| `holevo_vn` | ĐẠT | ĐẠT (Δχ_oracle +0.006) | TRƯỢT | TRƯỢT (−0.18) | ĐẠT | — |
| `holevo_mcr2` | ĐẠT | ĐẠT | TRƯỢT | TRƯỢT (−0.05) | ĐẠT | — |
| `holevo_spread` | ĐẠT | ĐẠT | TRƯỢT | TRƯỢT (+0.11) | ĐẠT | — |
| `hp_full` | ĐẠT | TRƯỢT (I_orc +0.097 so với diag) | TRƯỢT | TRƯỢT (+0.11) | ĐẠT | TRƯỢT (Δ PGM −0.13) |
| `hp_povm` | ĐẠT | TRƯỢT (I_orc +0.119) | TRƯỢT | TRƯỢT (+0.21) | ĐẠT | TRƯỢT (Δ PGM +0.10) |

## Chẩn đoán cơ chế

**Gradient.** Bước hiệu chỉnh λ hoạt động đúng với A, `hp_full` và `hp_povm`. Ở task 1, gradient phần mới bằng khoảng 0.5–0.66 lần gradient phân loại, đúng mục tiêu r* = 0.5. Tuy nhiên tỉ lệ này **giảm dần** về 0.08–0.15 ở task 9, vì λ bị khoá từ task 1. Ở các task sau, phần mới đóng vai trò yếu dần.

**Objective có tăng.** Δχ_oracle > 0 ở mọi run A, nên χ trên feature thật đúng là tăng. Dù vậy, tăng χ không làm giảm nhầm lẫn giữa các task.

**POVM phân biệt kém.**
- Với ảnh lớp mới ngay trong lúc train, p_y chỉ ~0.18, còn p_old chiếm ~0.66.
- Vùng dư R chiếm ~0.9 số chiều (với τ = 1), trong khi lớp mới vẫn nằm chủ yếu trên các hướng chính của lớp cũ.
- Train làm p_old trên lớp mới giảm (0.73 → 0.66 so với diag), nhưng đồng thời làm I_orc tăng: feature thật của lớp cũ cũng trôi vào vùng dư, và thống kê đã lưu không theo kịp.

**Completeness của POVM đúng.** Sai số < 1e-14 ở mọi epoch, và p_null ~4e-7.

## Phép so sánh lượng tử–cổ điển

| Cặp | Δ final | Δ task 0–1 | Kết luận |
|---|---:|---:|---|
| `holevo_vn` − `holevo_mcr2` | −0.13 | −1.39 | lượng tử **không** hơn cổ điển |
| `holevo_spread` − `holevo_vn` | +0.29 | +2.40 | thành phần có tác dụng là trải rộng S(ρ̄), không phải χ |
| `hp_full` − `hp_twin` | không hợp lệ | | `hp_twin` sụp do lỗi hiệu chỉnh |

## So với các hướng trước

| Run | AIA | Final | Forgetting | Ghi chú |
|---|---:|---:|---:|---|
| QKSR legacy (`archive/legacy_runs/`, ví dụ "+0.36" trong paper) | 87.28 | 82.88 | 4.41 | code cũ, task 0 = 95.62, nên không so trực tiếp từng số được |
| SVF w 0.25 (`svf/results/`) | 87.62 | 82.83 | 4.84 | chỉ suy luận, trọng số chọn trên chính run đó |
| SCF ridge (`svf/results/`) | 87.66 | 82.64 | 5.77 | head SCF của paper |
| `hp_povm` | 87.30 | 82.80 | 4.39 | |

Cả ba hướng lượng tử đều cho cùng một mẫu: forgetting thấp hơn, task cũ tốt hơn, nhưng AIA và final đứng yên. Về AIA, SCF (đóng góp chính hiện tại của paper) vẫn hơn mọi config Holevo khoảng +0.35.

## Lỗi của `hp_twin` (thiết kế, chưa sửa)

- Logit Gaussian 768 chiều rất lớn, nên softmax bão hoà ngay lúc hiệu chỉnh (p_y = 1.0, gradient ≈ 0). λ_dec tính ra 8·10²⁹ và bị chặn ở 1000.
- Khi loss bắt đầu tăng, λ = 1000 làm adapter sụp ở task 1: p_y = 0.05 = 1/20, accuracy ở mức ngẫu nhiên, gradient chết (r = 0).
- Cách sửa:
  1. chia logit cho một nhiệt độ (ví dụ D, hoặc trung vị của khoảng cách Mahalanobis);
  2. cho `_finalize_lambdas` báo lỗi thay vì âm thầm chặn khi λ chạm biên.
- Chỉ đáng sửa nếu cần khép lại câu hỏi "POVM có hơn bản cổ điển không". Vì `hp_povm` cũng không qua ngưỡng, kỳ vọng thấp.

## Việc tiếp theo (đề xuất)

1. **Không chạy held-out** cho A hay A+B.
2. *(Tuỳ chọn, 3–4 giờ)* Chạy thêm 2–3 replicate cho `hp_povm` và `hp_diag` để biết +0.38 final so với `hp_diag` có thật không. Kể cả nếu có thật, mức này chỉ đến từ việc giữ task cũ, và AIA nhiều khả năng vẫn ≈ 0.
3. Bài học cho hướng sau: lần thứ ba (oracle, SVF, Holevo), "đẩy lớp mới tránh vùng lớp cũ" trên một adapter dùng chung chỉ di chuyển dọc đường đánh đổi. Muốn vượt RSIAT có lẽ phải đổi kiến trúc, ví dụ biến đổi isometry theo task hoặc POVM dùng lúc suy luận, thay vì đổi loss.
