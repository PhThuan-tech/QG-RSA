# Holevo-CIL (A) và Holevo-POVM (A+B)

Baseline sàng lọc (4 run `noise`, ImageNet-R 1993): final 82.588, pair ratio 1.3806 ± 0.0305 (δ_M = 0.100), trung bình 3 task cuối 84.732, task 0 96.05.

## Từng run

| Dataset | Seed | Run | AIA | Final | Forgetting ↓ | Task 0 | Task 1 | 3 task cuối | Pair ratio | PGM | Ridge | λ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ImageNet-R | 1993 | holevo_mcr2 | 87.19 | 82.54 | 5.10 | 81.5 | 80.5 | 84.12 | 1.409 |  |  | enc 8.59 |
| ImageNet-R | 1993 | holevo_spread | 87.28 | 82.70 | 4.56 | 82.9 | 81.1 | 84.04 | 1.395 |  |  | enc 273 |
| ImageNet-R | 1993 | holevo_vn | 87.25 | 82.41 | 5.31 | 79.5 | 79.8 | 84.66 | 1.381 |  |  | enc 722 |

## A (Holevo-CIL)

### Sàng lọc (ImageNet-R 1993, so với baseline `noise`)

- `holevo_vn` **(chính)**: R ĐẠT; G ĐẠT (median Δχ_oracle 0.0064, Δχ_stale 0.0038); M TRƯỢT (pair ratio 1.381 ≤ 1.281); O TRƯỢT (Δ final -0.18); P ĐẠT (3 task cuối 84.66 ≥ 83.73)
  - khoảng cách vết (trạng thái ‖v‖² so với thật), task cuối: trung bình 0.024, max 0.032
- `holevo_mcr2`: R ĐẠT; G ĐẠT (median Δχ_oracle 0.5692, Δχ_stale 0.1347); M TRƯỢT (pair ratio 1.409 ≤ 1.281); O TRƯỢT (Δ final -0.05); P ĐẠT (3 task cuối 84.12 ≥ 83.73)
  - khoảng cách vết (trạng thái ‖v‖² so với thật), task cuối: trung bình 0.032, max 0.043
- `holevo_spread`: R ĐẠT; G ĐẠT (median Δχ_oracle 0.0120, Δχ_stale 0.0020); M TRƯỢT (pair ratio 1.395 ≤ 1.281); O TRƯỢT (Δ final 0.11); P ĐẠT (3 task cuối 84.04 ≥ 83.73)
  - khoảng cách vết (trạng thái ‖v‖² so với thật), task cuối: trung bình 0.026, max 0.032
- **Kết luận sàng lọc của `holevo_vn`: DỪNG hoặc xem luật trượt ở trên**
- Tham khảo (không phải luật): holevo_vn − holevo_mcr2 final = -0.13

### Held-out (n = 5: ImageNet-R 2015, 2026; ImageNet-A 1993, 2015, 2026)

- H: Δ final holevo_vn − RSIAT (≥ 0.5, CI > 0):   (n=0) → **chưa đủ run (0/5)**
- Q: Δ final holevo_vn − holevo_mcr2 (CI > 0):   (n=0) → **chưa đủ run (0/5)**

## A+B (Holevo-POVM)

### Sàng lọc (ImageNet-R 1993, so với baseline `noise`)

- Chưa có run sàng lọc.

### Held-out (n = 5: ImageNet-R 2015, 2026; ImageNet-A 1993, 2015, 2026)

- H: Δ final hp_full − RSIAT (≥ 0.5, CI > 0):   (n=0) → **chưa đủ run (0/5)**
- Q: Δ final hp_full − hp_twin (CI > 0):   (n=0) → **chưa đủ run (0/5)**
- Q_dec: PGM − ridge trên biểu diễn hp_full (CI > 0):   (n=0) → **chưa đủ run (0/5)**

**Kiểm tra hồi quy (task 0 phải = 96.05):** holevo_mcr2 ImageNet-R 1993: task 0 96.05; holevo_spread ImageNet-R 1993: task 0 96.05; holevo_vn ImageNet-R 1993: task 0 96.05

