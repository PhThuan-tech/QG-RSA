# Holevo-POVM (A+B): kết quả sàng lọc W03

Phân tích chung với hướng A nằm trong `exps/holevo/results/README.md`. File gốc nằm trong `kaggle1/`.

**Tóm tắt: DỪNG.**
- `hp_full` (chính): AIA 87.31, final 82.70.
- `hp_povm` tốt nhất: AIA 87.30, final 82.80, forgetting 4.39.
- So với RSIAT (`hp_diag` 87.25 / 82.42; 4 run `noise` 87.35 / 82.59), mức chênh nằm trong dải nhiễu.
- Luật G trượt: I_orc tăng so với diag.
- Luật M và O trượt.
- `hp_twin` sụp do lỗi hiệu chỉnh λ, nên phép so sánh lượng tử–cổ điển của A+B không hợp lệ.
