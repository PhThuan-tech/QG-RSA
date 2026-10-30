# SVF giai đoạn A: kết quả wave W01 (ImageNet-R seed 1993, run tuning)

Nguồn: `kaggle/waves/w01_svf` (base `c7bc4fd` + patch), chạy ngày 2026-10-29. Môi trường: RTX PRO 6000 Blackwell, torch 2.11.0+cu128, timm 1.0.29, 1 run chạy một mình. File gốc nằm trong `w01/`: log, config, `environment.json`, `SUMMARY.txt`; thư mục `_stats/` chỉ giữ ở máy.

## Kết luận

**Luật dừng sàng lọc kích hoạt: DỪNG, không chạy held-out.**
- Δ final (SVF + SCF − SCF ridge) tại trọng số đã chọn là **+0.11**, dưới ngưỡng +0.3 (`exps/svf/README.md` mục 3).
- View snapshot gần như không thêm được gì so với SCF.
- Kết quả đúng như dự đoán ghi trước khi chạy: Recall@1 của A₀ gần bằng của mạng cuối, nên dư địa chỉ ở mức SCF.

## Bảng (final và AIA theo %, cùng một run)

| Head | AIA | Final | Δ final so với RSIAT | Forgetting | Task 0 | Task 1 | Pair ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| RSIAT (`linear`) | 87.25 | 82.42 | 0 | 5.39 | 79.8 | 79.6 | 1.44 |
| SCF ridge (`fuse_ridge_stats_b0p25`) | 87.66 | 82.64 | +0.22 | 5.77 | 80.5 | 79.9 | 1.38 |
| Snapshot một mình (`snap_ridge`) | 83.58 | 75.72 | −6.70 | 5.64 | **87.6** | 77.7 | 1.03 |
| SVF (`fuse_snap_ridge_b0p25`) | 87.62 | **82.83** | +0.41 | **4.84** | 85.0 | 81.0 | 1.25 |
| SVF + SCF (`fuse3_snap_b0p25`) | 87.63 | 82.75 | +0.33 | 5.29 | 85.0 | 80.7 | 1.14 |

**Δ final theo trọng số snapshot:**

| w | 0.25 | 0.5 | 1 | 2 | 4 |
|---|---:|---:|---:|---:|---:|
| `fuse3_snap` − SCF ridge | +0.11 | −0.05 | −0.99 | −2.42 | −4.21 |
| `fuse_snap_ridge` − RSIAT | +0.41 | +0.09 | −0.95 | −2.68 | −4.44 |

- Trọng số tốt nhất nằm ở biên dưới của lưới (0.25), và gain giảm đều khi w tăng.
- Khi w → 0, fusion về đúng RSIAT. Như vậy cực đại thật nằm đâu đó trong (0, 0.25), và theo xu hướng này nhiều khả năng chỉ quanh +0.4.
- Mọi con số đều lạc quan, vì trọng số được chọn trên chính run này.

## Acc theo task ở task cuối

| Head | T0 | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RSIAT | 79.8 | 79.6 | 82.1 | 82.7 | 78.2 | 82.3 | 85.6 | 84.5 | 84.0 | 86.0 |
| Snapshot một mình | 87.6 | 77.7 | 74.0 | 78.1 | 72.3 | 71.1 | 78.1 | 68.7 | 76.3 | 72.9 |
| SVF w 0.25 | 85.0 | 81.0 | 82.4 | 83.4 | 78.0 | 81.6 | 84.1 | 83.7 | 83.4 | 85.3 |
| Joint (`exps/upper/README.md`) | 89.1 | 88.7 | 86.4 | 86.4 | 83.2 | 86.0 | 85.8 | 87.0 | 85.4 | 85.8 |

## Diễn giải (1 seed, mô tả, không phải kiểm định)

1. **Snapshot giữ được task 0, nhưng chỉ task 0.**
   - Task 0 cuối đạt 87.6, ngang mức oracle stats (87.1) và gần joint (89.1).
   - Như vậy thông tin để phân biệt các lớp task 0 không mất khỏi A₀.
   - Nhưng A₀ chưa từng được train trên các lớp sau, nên chúng chỉ đạt 68–78.
   - Đây là bất đối xứng train–chưa thấy đã cảnh báo trước khi chạy.
2. **Fusion đi trên cùng đường đánh đổi với oracle.**
   - Task 0 tăng +5.2, task 1 +1.4; đổi lại task 5–9 giảm 0.6–1.5.
   - Tổng lại final chỉ +0.41, AIA +0.37, ngang SCF (AIA 87.66).
   - Kết quả oracle ở `exps/upper/README.md` có cùng kiểu: task 0 tăng 7, task mới giảm, final +0.5. Như vậy hai cách can thiệp khác nhau cùng chạm một giới hạn: kéo lớp cũ lên thì lớp mới bị lấn.
3. **Pair ratio giảm** từ 1.44 xuống 1.25 với SVF, và 1.14 với SVF + SCF, nhưng final gần như không đổi. Nhầm lẫn giữa task giảm được bù bằng nhầm lẫn trong task tăng, nên chỉ số này không đủ để dự đoán accuracy.
4. **Forgetting của SVF thấp nhất** (4.84 so với 5.39), vì task 0 và 1 tăng. Với 1 seed, không kết luận được gì thêm.

## Kiểm tra hồi quy

- **Head linear = CNN curve:** đạt.
- **So với các run cũ cùng seed:** không trùng bit. Task 0 trùng (96.05), nhưng từ task 1 lệch nhẹ:
  - W01: `[96.05, 91.37, 90.01, 88.54, 86.32, 85.44, 85.06, 83.95, 83.30, 82.42]`;
  - `qsd2_rep0` = `qsd_rep0` = `noise`: `[96.05, 91.44, 90.21, 88.62, 86.52, 85.41, 84.97, 84.13, 83.34, 82.49]`.
- Torch, timm, GPU và config giống nhau. Chỉ khác `prefix` và `offline` (khoá này chỉ dùng khi tải CIFAR).
- **Nguyên nhân khả dĩ, chưa kiểm chứng:**
  - W01 chạy 1 run một mình, nên `OMP_NUM_THREADS` lớn gấp 3 so với lúc chạy 3 song song. Điều này có thể đổi cách làm tròn trong các phép tính CPU của SSCA/CA. Task 0 không có SSCA/CA và vẫn trùng.
  - Bản sao snapshot chiếm thêm bộ nhớ GPU, có thể làm thư viện chọn kernel khác.
- **Ảnh hưởng tới kết luận:** không có. Mọi so sánh ở trên là giữa các head trong cùng một run, cùng một lần train. Nhưng khi so W01 với run cũ thì không được coi là ghép cặp chính xác: lệch 0.07 final, nằm trong nhiễu SD 0.23.

## Việc tiếp theo

- Dừng SVF như một phương pháp; ghi kết quả âm này vào paper (cùng nhóm với SCF và oracle).
- Câu hỏi còn mở: linear probe 200 lớp trên feature cuối (`exps/svf/README.md` mục 3). Phép đo này cho biết nút thắt nằm ở biểu diễn hay ở bộ phân loại Gaussian. Chưa cài đặt.
