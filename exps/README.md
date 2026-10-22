# Thí nghiệm

## Cấu trúc

Mỗi thí nghiệm là một thư mục `exps/<thí nghiệm>/`:

```text
exps/<thí nghiệm>/
  README.md              đặc tả, viết trước khi chạy (quy tắc quyết định cố định)
  <variant>.json         overlay; notebook tìm theo tên: exps/*/<variant>.json
  results/
    README.md            phân tích kết quả và kết luận
    summary.md / .json   bảng do script tổng hợp sinh ra
    *.py                 script phân tích của vòng đó
    <nguồn>/             nguyên trạng file từ Kaggle (vd. kaggle1/)
      configs/  environment.json
      logs/adapter/<dataset>/<init>/<inc>/<run>.log    -> trong git
      logs/.../<run>_stats/   (npz/json theo task)     -> chỉ ở máy local
```

**Giữ gì, bỏ gì:**
- **Trong git:** `.log` của từng run, configs, `environment.json`, summary, README và script phân tích. Đây là bằng chứng của mọi con số trong paper.
- **Chỉ ở máy local (`.gitignore`):** thư mục `_stats/`, khoảng 320 MB. Đây là dữ liệu cho các phân tích sâu và bảng trong paper (`analyze_qsd.py`, `metrics_table.py`, `a2_explore.py`), nên **đừng xoá**.
- **Không giữ:** `console/` từ Kaggle. Nó chỉ dùng để xem traceback khi run lỗi, còn mọi số liệu đã có trong `.log`. Khi giải nén kết quả mới, có thể xoá thư mục này sau khi xác nhận các run đều xong.

## Quy ước seed (áp dụng từ vòng A3)

- **Mọi dataset dùng chung 3 seed: 1993, 2015, 2026**, với `rng_replicate` 0.
  - Seed quyết định thứ tự lớp (và cách chia 80/20 với ImageNet-A).
  - Bảng chính báo cáo **mean ± SD trên 3 seed này**.
- **Nhóm tuning đã khoá:** ImageNet-R seed 1993, replicate 0–2 (vòng A2).
  - Trọng số fusion đã chọn và không được chọn lại: ridge 0.25, PGM 0.5, PGM phân cấp 1.
  - Mọi dataset và seed khác đều là held-out.
- **Run bổ sung** (giữ lại, không đưa vào bảng chính): replicate 1–2 của seed 1993 và ImageNet-R seed 2042. Chúng chỉ dùng để đo nhiễu huấn luyện và kiểm tra protocol.

Tình trạng (✓ = đã có run A2 với đủ head SCF; mọi run đều chứa sẵn baseline RSIAT):

| Dataset | 1993 | 2015 | 2026 | Ghi chú |
|---|:-:|:-:|:-:|---|
| ImageNet-R (B0Inc20) | ✓ | ✓ | ✓ | `qsd/results/a2/` |
| ImageNet-A (B0Inc20) | ✓ | ✓ | ✓ | `qsd/results/a2/` |
| CIFAR-100 (B0Inc10) | ✓ | ✓ | ✓ | `scf/results/kaggle1/`, `scf/results/kaggle2/` |
| CUB-200 (B0Inc10, 20 task) | ✓ | ✓ | ✓ | `scf/results/kaggle2/`, split chính thức 5 994 / 5 794 |

## Danh mục (theo thứ tự thời gian)

| Thí nghiệm | Đặc tả | Kết quả | Kết luận |
|---|---|---|---|
| Run đơn lẻ đời đầu (trước protocol) | — | `archive/legacy_runs/` | RSIAT 1 run trên 6 dataset (11/2025) và QKSR (10/2026). Ví dụ "+0.36 hoá ra là nhiễu" trong paper lấy từ đây |
| Chẩn đoán SSCA/CA, vòng 1–2 (ImageNet-A) | `diagnostics/README.md` | `diagnostics/results/` | Lỗi ghép cặp SSCA có thật nhưng bị che; dư địa của oracle stats chỉ +0.66 AIA; chỉnh CA không có tác dụng |
| Đo nhiễu (ImageNet-R, 4 replicate) | `measurement/README.md` | `measurement/results/` | AIA 87.347 ± 0.144. Các run `noise*` là baseline ghép cặp |
| Cận trên (oracle stats, huấn luyện chung) | `measurement/README.md`, giai đoạn 1b | `upper/README.md`, `upper/results/` | H = +0.195 AIA; G = 3.9 final. Nút thắt là chồng lấn biểu diễn giữa các task |
| QKSR `qksr_cur` so với `qksr_old` | `qksr/README.md` | `qksr/results/README.md` | Không khác biệt: −0.017 [−0.064, +0.030] |
| QRF-CA (head CA phi tuyến, PQK) | `qrfca/README.md` | `qrfca/results/README.md` | Mọi head phi tuyến kém hơn; PQK −17.3 AIA |
| QSD-CIL giai đoạn A | `qsd/README.md` | `qsd/results/stage_a/README.md` | PGM/kênh kém linear; chỉ `pgm_fuse` thắng ở 10/10 run (R +0.25, A +0.44) |
| QSD-CIL vòng A2 (**kết quả chính của paper**) | `qsd/README_A2.md` | `qsd/results/a2/README.md` | SCF ĐẠT (held-out +0.46 đến +0.62 AIA); ridge cổ điển ngang hoặc hơn PGM, nên không có đóng góp lượng tử; forgetting tăng nhẹ |
| SCF vòng A3: CIFAR-100, CUB-200, 3 seed | `scf/README.md` | `scf/results/README.md` | Chuyển giao KHÔNG ĐẠT theo quy tắc: CIFAR +0.20 AIA (CI > 0), CUB +0.02. Final tăng ở cả 4 dataset; PGM − ridge = −0.04 trên 12 run |

`archive/qksr/`: config QKSR cũ (CIFAR, CUB), không còn dùng.

## Config gốc

**RSIAT baseline.** `rsiat/` chứa config RSIAT cổ điển chạy trực tiếp được cho 6 dataset, đã tắt quantum kernel và Q-BRIDGE:

```bash
python main.py --config exps/rsiat/imagenetr.json
```

Mọi vòng trong bảng trên đều chạy bằng `QGRSA_EXPERIMENTS.ipynb`:
- run = `rsiat/<dataset>.json` + overlay + `OVERRIDES`;
- notebook được dựng bằng `python scripts/build_experiments_notebook.py QGRSA_EXPERIMENTS.ipynb`.

**Smoke.** `smoke/rsiat_cifar224.json` chạy một task ngắn, một epoch, chỉ để kiểm tra pipeline.

**Q-BRIDGE (hướng cũ, không còn chạy).** Code và test vẫn còn trong repo:
- `qbridge/sources/`, `qbridge/stages/`, `qbridge/overlays/R0..R10.json`;
- `scripts/generate_qbridge_configs.py`;
- notebook `QBRIDGE_Kaggle.ipynb`.
