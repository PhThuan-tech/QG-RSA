# Q-BOUND: cận phân biệt trạng thái làm chẩn đoán exemplar-free, và đường đánh đổi cũ–mới

Đặc tả này được viết **trước khi code và trước khi chạy phần B**, ngày 2026-10-31. Các quy tắc bên dưới là cố định. Mọi thay đổi sau khi thấy kết quả phải ghi vào mục "Nhật ký thay đổi".

- Quy ước seed: `exps/README.md`.
- Hạ tầng dùng lại:
  - mã hoá trạng thái `stats_states` (`utils/qsd.py`, encoder khoá sau task 0);
  - đặc trưng thật của lớp cũ dưới `preserve_rng` (như `utils/holevo.py::oracle`);
  - prediction lưu theo task (`save_predictions`).
- Bối cảnh:
  - paper (`paper/sections/quantum.tex`) chứng minh rằng với PGM, chỉ encoding là bậc tự do lượng tử;
  - các thí nghiệm `qksr`, `svf`, `holevo`, `holevo_povm` đều chỉ đổi lớp cũ lấy lớp mới (`exps/holevo/results/README.md`).

## 1. Câu hỏi

Hướng này **không** cố làm tăng accuracy. Nó đặt hai câu hỏi đo lường, đúng tinh thần của paper.

**A. Đường đánh đổi (chỉ dùng dữ liệu có sẵn).** Các can thiệp lúc huấn luyện trên adapter dùng chung (QKSR, Holevo, POVM, oracle) có nằm trên cùng một đường đánh đổi giữa lớp cũ và lớp mới của RSIAT không? Các can thiệp lúc suy luận (SCF, SVF) có vượt được đường đó không?

**B. Cận lượng tử làm chẩn đoán.** Trạng thái lớp chỉ dựng từ thống kê đã lưu (không có ảnh cũ). Từ chúng, cận Helstrom và độ trung thực (fidelity) giữa các task có dự đoán được **mức nhầm lẫn thật giữa các task** không? Và có dự đoán tốt hơn các đại lượng cổ điển dùng cùng đầu vào không?

Phần B có giá trị cho paper dù kết quả ra sao:
- **Nếu Q đạt:** đây là vai trò đầu tiên trong paper mà đại lượng lượng tử làm được việc bản cổ điển không làm được.
- **Nếu Q không đạt:** đây là thêm một bằng chứng có kiểm soát cho kết luận "tương đương cổ điển", lần này ở vai trò chẩn đoán, khác với vai trò phân loại.

## 2. Phần A: đường đánh đổi cũ–mới (không cần GPU)

### 2.1. Dữ liệu

Mọi run có `predictions_task_<cuối>.npz` trong `exps/*/results/**/_stats/`, khoảng 61 run. Với run có `qsd_predictions_task_<cuối>.npz`, mỗi head là một điểm riêng trên **cùng một mạng**.

| Nhóm | Run | Loại can thiệp |
|---|---|---|
| RSIAT | `noise*` (4), `hp_diag`, RSIAT của SVF, head `linear` của `qsd2`/`scf` | baseline |
| Huấn luyện | `qksr_*` (6), `holevo_*` (3), `hp_full`, `hp_povm`, oracle `upper/` | đổi adapter |
| Suy luận | head SCF (`fuse_*`), SVF (`fuse_snap_*`) | đổi bộ phân loại, giữ mạng |

Loại ra: `hp_twin` (sụp), các run lỗi, run không đủ task.

### 2.2. Đại lượng

Với run có T task, lấy accuracy theo task ở task cuối, a_0 … a_{T−1}:
- **old** = trung bình a_k với k < T/2; **new** = trung bình a_k với k ≥ T/2;
- **final**, **AIA**, **forgetting** lấy từ log.

Δ của mỗi run được tính so với RSIAT **khớp môi trường**:
- ImageNet-R 1993: trung bình 4 run `noise`, kèm `hp_diag` cho các run Kaggle W01–W03;
- các dataset và seed khác: head `linear` của chính run đó (với can thiệp lúc suy luận) hoặc run RSIAT cùng seed.

Thang nhiễu: SD của old, new và final trên các replicate RSIAT cùng seed. ImageNet-R 1993 có 4 run `noise`. Với nhóm khác, dùng SD gộp từ các bộ replicate có sẵn (`qsd/a2`, `qksr`).

### 2.3. Phân tích

1. Vẽ Δnew theo Δold, có dải ±2 SD. Đường "đổi ngang" là Δfinal = 0, tính theo trọng số số task của old và new.
2. Với mỗi can thiệp có |Δold| > 2 SD_old, ghi **tỉ giá** r = −Δnew / Δold và **Δfinal** tính theo SD_final.
3. So nhóm huấn luyện với nhóm suy luận theo trung bình Δfinal / SD_final.

### 2.4. Cách đọc (khám phá, **không** phải kiểm định)

Phần A dùng dữ liệu đã thấy, nên chỉ là **phân tích hậu kiểm**. Trong paper nó chỉ được trình bày như vậy. Mô tả trước, để không chọn lời kể sau khi xem số:

- **"Đường đánh đổi giữ":** ít nhất 80% can thiệp huấn luyện có |Δold| > 2 SD_old đồng thời có |Δfinal| < 2 SD_final.
- **"Suy luận vượt đường":** trung bình Δfinal / SD_final của nhóm suy luận lớn hơn nhóm huấn luyện ít nhất 1. SCF được kỳ vọng là ví dụ chính.

Kiểm định xác nhận cho câu A dùng các run held-out mới của phần B. Các run đó là RSIAT, không có can thiệp, nên chỉ làm chặt thêm thang nhiễu. Không chạy thêm can thiệp mới chỉ để kiểm định A.

## 3. Phần B: cận lượng tử làm chẩn đoán

### 3.1. Trạng thái

- Encoder: `stats_states` với center và scale khoá sau task 0, giống hệt `utils/holevo.py::StatsEncoder`.
- Sau CA của task t, với mọi lớp đã thấy c ≤ K_t: ρ_c dựng từ (μ_c, Σ_c) **đã lưu**, tức sau SSCA, đúng thứ CA dùng.
- Trạng thái task: ρ_a = (1/|a|) Σ_{c∈a} ρ_c, với a là chỉ số task ≤ t.

**Bản oracle (đọc kèm):** dựng cùng các trạng thái từ μ, Σ tính trên **đặc trưng thật** của ảnh train mọi lớp đã thấy, với mạng hiện tại. Bước này chỉ đọc, chạy dưới `preserve_rng`, không ảnh hưởng huấn luyện.

### 3.2. Đại lượng dự đoán cho mỗi cặp task (a, b), a < b ≤ t

Cặp được xét riêng như bài toán nhị phân, với tiên nghiệm theo số lớp, chuẩn hoá trên hai task.

**Lượng tử (Q):**

| Ký hiệu | Công thức | Ghi chú |
|---|---|---|
| Q1 `helstrom` | P_H = ½(1 − ‖π_a ρ_a − π_b ρ_b‖₁) | lỗi tối ưu của mọi phép đo trên ρ_a, ρ_b |
| Q2 `fidelity` | F = (Tr √(√ρ_a ρ_b √ρ_a))², dùng −log F | |
| Q3 `pgm_pair` | lỗi của PGM trên tập hai trạng thái | δ như head `pgm_stats` (`qsd_delta` 1.3316, tương đối) |

**Cổ điển (C), cùng đầu vào (μ_c, Σ_c):**

| Ký hiệu | Công thức | Ghi chú |
|---|---|---|
| C1 `bhatta` | cận Bhattacharyya √(π_aπ_b)·e^{−D_B} giữa Gaussian của task, khớp moment (mean và cov của hỗn hợp lớp) | ridge 1e-3·tr Σ / D |
| C2 `mc_bayes` | lỗi Bayes Monte Carlo: lấy 200 mẫu mỗi lớp từ N(μ_c, Σ_c), phân loại bằng log-likelihood lớp, đếm lỗi chéo giữa a và b | seed cố định `20261031 + 100t`; đây là "bản CA" của câu hỏi |
| C3 `mean_dist` | khoảng cách chuẩn hoá giữa mean của hai task, ‖m_a − m_b‖ / √(tr Σ̄ / D) | baseline tầm thường |

Mọi đại lượng được đổi dấu nếu cần, để **giá trị lớn nghĩa là dự đoán nhầm lẫn nhiều**.

### 3.3. Mục tiêu đo được

Từ `predictions_task_t.npz` của head `linear` (RSIAT), xét các ảnh test thuộc task a hoặc b:

> c_ab(t) = ½ [ P(dự đoán ∈ b | y ∈ a) + P(dự đoán ∈ a | y ∈ b) ]

Đọc kèm: cùng mục tiêu trên head SCF ridge (`fuse_ridge_stats_b0p25`) và `pgm_stats`.

Mỗi run cho Σ_{t=1}^{T−1} C(t+1, 2) điểm (stage, cặp). ImageNet-R B0Inc20 có 165 điểm, CUB 20 task có 1 330 điểm.

### 3.4. Thước đo

ρ_S(X) là hệ số Spearman giữa đại lượng X và c_ab trên mọi điểm (stage, cặp) của một run.

Đọc kèm: Spearman **trong từng stage**, lấy trung bình theo stage. Thước đo này loại hiệu ứng "stage muộn có nhiều lớp hơn nên nhầm nhiều hơn".

### 3.5. Lựa chọn trên nhóm tuning (ImageNet-R 1993, run W04 đầu tiên), sau đó khoá

- δ của Q3 trong {0.5, 1.3316, 3}, tương đối;
- số mẫu của C2 trong {200, 1000};
- dùng ρ_S toàn cục hay trong từng stage làm thước đo chính. Mặc định: toàn cục, chỉ đổi nếu hai thước đo trái dấu.

Ghi lựa chọn vào nhật ký trước khi xem held-out.

## 4. Quy tắc quyết định (cố định, `scripts/summarize_qbound.py`)

Held-out gồm n = 11 run RSIAT: ImageNet-R 2015 và 2026; ImageNet-A 1993, 2015, 2026; CIFAR-100 1993, 2015, 2026; CUB-200 1993, 2015, 2026. Mọi dataset dùng split chuẩn của `exps/README.md`.

| Luật | Điều kiện | Ý nghĩa |
|---|---|---|
| **S** (dự đoán được) | median ρ_S(Q1) ≥ 0.5 trên held-out, và ≥ 0.3 ở ít nhất 9/11 run | cận Helstrom dự đoán được nhầm lẫn giữa các task |
| **Q** (lượng tử hơn cổ điển) | ρ_S(Q1) − max(ρ_S(C1), ρ_S(C2)), ghép cặp theo run; bootstrap CI 95% (10 000 lần, seed 20261031) có cận dưới > 0 | lượng tử là chẩn đoán tốt hơn |
| **Q≈** (tương đương) | CI của Q nằm trong [−0.05, +0.05] | bản cổ điển làm được như nhau |
| **E** (cảnh báo sớm) | median Spearman giữa Q1 tại stage hình thành cặp (t = b) và c_ab ở stage cuối ≥ 0.4 | biết trước cặp task nào sẽ nhầm |
| **T** (tầm thường) | ρ_S(Q1) − ρ_S(C3) > 0 với CI > 0 | Q1 không chỉ đo khoảng cách mean |

- **Kết quả báo cáo trong paper:** S, Q / Q≈, E, T và bảng đầy đủ ρ_S của Q1–Q3, C1–C3, kể cả khi trượt.
- **Luật dừng:** nếu S trượt, Q1 không phải chẩn đoán hữu ích. Báo cáo một dòng trong paper, không chạy thêm.

**Đọc kèm, không phải luật:**
- ρ_S trên trạng thái **oracle** so với trạng thái **đã lưu**: phần dự đoán bị mất do thống kê cũ (stale);
- ρ_S với mục tiêu SCF và PGM;
- tương quan giữa tổng Q1 qua các cặp và pair ratio của run.

**Dự đoán ghi trước:**
- S đạt (overlap giữa các task là nguồn lỗi chính, `upper/`);
- Q≈ đạt, tức ngang nhau: định lý của paper và quan hệ PGM ≈ ridge gợi ý C2 ≈ Q1;
- Q1 có thể hơn C1, vì C1 khớp moment nên làm mất cấu trúc hỗn hợp.

## 5. Thiết lập

### 5.1. Khoá config mới (mặc định tắt, không đổi RSIAT)

| Khoá | Mặc định | Ý nghĩa |
|---|---|---|
| `qbound_diag` | `false` | bật hook cuối mỗi task |
| `qbound_oracle` | `true` | thêm bản oracle (một lượt trích đặc trưng train) |
| `qbound_mc_samples` | 200 | C2 |
| `qbound_pgm_delta` | 1.3316 | Q3, tương đối như `qsd_delta` |

- **Hook:** `incremental_train`, ngay sau CA của task t (và sau `_compute_class_mean` ở task 0 để khoá encoder).
- **Ghi ra:** `<log>_stats/qbound_task_<t>.json`, gồm ma trận Q1–Q3 và C1–C3 theo cặp task, cho cả trạng thái đã lưu và oracle, kèm thời gian chạy.
- Toàn bộ hook chạy dưới `torch.no_grad()` + `preserve_rng`.

### 5.2. Overlay và wave

- `exps/qbound/qbound_diag.json`: giống `noise_rep0`, cộng `qbound_diag true`, `save_predictions true`, các head SCF (`qsd_heads [linear, ridge_stats, pgm_stats]`, `qsd_fuse true`) để có mục tiêu đọc kèm.
- **W04_qbound:**
  1. ImageNet-R 1993 (tuning);
  2. 11 run held-out, max_parallel 3.

  Ước tính khoảng 6–8 giờ Kaggle, tách thành hai notebook nếu vượt giới hạn phiên.

### 5.3. Chi phí tính

- Task-level: mỗi stage có tối đa C(T, 2) phép eigvalsh / căn ma trận 769×769 (float64, GPU). Không đáng kể.
- C2: 200 × K_t mẫu, chấm điểm theo khối như `HolevoRuntime.gauss`.
- Oracle: một lượt trích đặc trưng train mỗi task. Với CUB 20 task: chạy một lần ở stage t mà không lặp lại cho các stage trước.

## 6. Kiểm tra trước khi chạy (`tests/test_qbound.py`)

1. **Bất biến:** trên bộ dữ liệu đồ chơi `test_stats_diagnostics`, `qbound_diag` bật hay tắt cho trọng số và prediction **giống hệt từng bit**.
2. **Công thức:**
   - Helstrom với hai trạng thái thuần trực giao cho 0, với hai trạng thái giống nhau cho min(π);
   - Fuchs–van de Graaf: 1 − √F ≤ ½‖ρ−σ‖₁ ≤ √(1−F) trên trạng thái ngẫu nhiên;
   - Bhattacharyya với hai Gaussian giống nhau cho √(π_aπ_b).
3. **C2:** hai Gaussian cách xa nhau cho lỗi ≈ 0, hai Gaussian trùng nhau cho lỗi ≈ 0.5 (prior bằng nhau).
4. **Mục tiêu c_ab:** khớp với đếm tay trên một ma trận nhầm lẫn nhỏ.
5. **Summary:** fixture có ρ_S biết trước, kiểm các luật S, Q, Q≈, E, T.

## 7. Rủi ro đã biết

- **Nhiễu c_ab:** cặp task có ít ảnh test (ImageNet-A, khoảng 7 ảnh/lớp) làm c_ab nhiễu. Spearman giảm nhạy với ngoại lai, nhưng ρ_S có thể thấp vì lý do này. Đọc kèm: ρ_S chỉ trên các cặp có ≥ 200 ảnh test.
- **Cận không ràng buộc head tuyến tính:** Helstrom ràng buộc phép đo trên ρ_a, ρ_b, không ràng buộc head tuyến tính trên ảnh test. Ở đây Q1 chỉ dùng **làm đại lượng dự đoán**, không được viết như một "cận đúng" cho accuracy của RSIAT. Giới hạn này giống mục 2.3 của `exps/holevo/README.md`.
- **Đường đánh đổi (phần A) là hậu kiểm:** chỉ báo cáo như phân tích khám phá, kèm số run và cách chọn.
- **Tương quan do stage:** stage muộn có nhiều lớp hơn. Thước đo trong từng stage được đọc kèm vì lý do này.

## 8. Liên quan

- **Phân biệt trạng thái / PGM:** Hausladen & Wootters (1994); Barnum & Knill (2002, PGM ≤ 2 lần lỗi tối ưu, bình phương); Eldar & Forney (2001, PGM ≡ least-squares). Đây là trích dẫn sẵn có của paper.
- **Fuchs & van de Graaf (1999):** liên hệ giữa fidelity và trace distance.
- **Bayes error / Bhattacharyya:** Fukunaga (1990).
- **Ước lượng độ khó / chuyển giao từ thống kê:** cần tra trước khi viết, để kiểm tính mới của vai trò "chẩn đoán CIL exemplar-free". Lượt tìm nhanh ngày 2026-10-31 không thấy bài nào dùng cận phân biệt trạng thái cho mục đích này. Đó **chưa phải** rà soát hệ thống.

## Nhật ký thay đổi

- 2026-10-31: chạy phần A (`scripts/frontier.py`, `results/README.md`).
  - Đường đánh đổi GIỮ (5/5).
  - Suy luận vượt đường KHÔNG ĐẠT (chênh z final +0.83 < 1).
  - Quan sát sau khi xem số, không phải tiêu chí: chỉ SCF tăng cả hai nửa; SVF nằm trên đường của nhóm huấn luyện.
  - Thêm cho phần B (trước khi có run mới): các run held-out W04 dùng để kiểm định xác nhận "SCF tăng cả old lẫn new". Luật: Δold > 0 và Δnew > 0 ở ít nhất 9/11 run, so với head `linear` của cùng mạng.

- 2026-10-31: viết đặc tả. Đã kiểm dữ liệu có sẵn: khoảng 61 run có prediction cuối, 210 run có score theo head, **không** run nào lưu mean/cov lớp. Vì vậy phần B cần hook mới và run mới, còn phần A làm được ngay.
