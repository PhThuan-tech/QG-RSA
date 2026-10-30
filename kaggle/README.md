# Chạy trên Kaggle theo wave

Code base được khoá **một lần** thành dataset Kaggle. Mỗi lần thử một hướng là một *wave*: một notebook 3 cell
mang theo **patch**, tức các file đã đổi so với commit base. Không cần upload lại repo hay dataset.

## Một lần cho mỗi base

```bash
python scripts/pack_base.py --commit <commit>      # -> dist/qgrsa-base-<sha7>.zip
```

Upload zip làm dataset Kaggle `qgrsa-base-<sha7>`, hoặc làm version mới của cùng một dataset.

- Zip chỉ chứa file **đã commit** của commit đó, không có `results/`, `paper/`, `docs/` và notebook, kèm `BASE_COMMIT`.
- Chỉ đóng gói base mới khi patch đã quá lớn, ví dụ sau khi commit một loạt thay đổi.
- Dataset ảnh (ImageNet-R/A, CIFAR, CUB) và ViT-B/16 vẫn là input Kaggle như cũ.

## Mỗi wave

1. Sửa code trong repo như bình thường. Chưa cần commit.
2. Tạo `kaggle/waves/<wNN_tên>/wave.json`, chép từ wave trước:

   | Khoá | Ý nghĩa |
   |---|---|
   | `base` | commit của dataset base |
   | `jobs` | danh sách `[datasets, seeds, variants]` (variant = overlay `exps/*/<variant>.json`) |
   | `max_parallel` | số run chạy song song |
   | `overrides` | ghi đè config của mọi run |
   | `tests` | test chạy trên Kaggle trước khi train |
   | `summaries` | script tổng hợp chạy sau khi train |
   | `description`, `inputs` | nội dung cell markdown |

3. Build notebook:

   ```bash
   python scripts/build_wave.py kaggle/waves/<wNN_tên>/wave.json
   ```

   Patch gồm mọi file khác commit base (đã sửa hoặc mới, trừ `.md`, notebook, kết quả). Lệnh in danh sách file để kiểm tra.
4. Trên Kaggle: import `<NAME>_kaggle.ipynb`, attach base + dữ liệu, rồi *Save & Run All*.
   - Chỉ đổi job: sửa `PLAN` ngay trong cell 2, không cần build lại.
   - Đổi code: build lại.
5. Tải `results_<NAME>.zip`. Giải nén vào `exps/<thí nghiệm>/results/<nguồn>/` như các vòng trước.

## Runner kiểm tra những gì

`kaggle/runner.py` được nhúng vào cell 2. Nó dừng ngay, kèm thông báo, khi:
- `BASE_COMMIT` của input khác `base` của wave;
- một file bị patch không trùng hash với bản mà patch được build từ đó;
- test của wave lỗi;
- thiếu GPU, dữ liệu hoặc model.

Run đã xong được bỏ qua khi chạy lại. Kết quả nằm ở `runs/SUMMARY.txt`. `environment.json` ghi lại base và danh sách file của patch.

| Wave | Thí nghiệm | Base |
|---|---|---|
| `w01_svf` | SVF giai đoạn A (`exps/svf/README.md`) | `c7bc4fd` |
| `w02_holevo` | Holevo-CIL giai đoạn A, sàng lọc (`exps/holevo/README.md`) | `c7bc4fd` |
| `w03_hp` | Holevo-POVM giai đoạn A, sàng lọc, chạy sau W02 (`exps/holevo_povm/README.md`) | `c7bc4fd` |
