# QKSR P0: phép đo chỉ đọc (`scripts/summarize_qksr_probe.py`)

## p0_qksr_cur

Log: `/kaggle/working/runs/logs/adapter/imagenetr/0/20/p0_qksr_cur_1993_pretrained_vit_b16_224_in21k_adapter.log`

- Tái lập qksr_cur rep0 (exps/qksr/results): không có run tham chiếu (lệch tối đa — điểm)

| Gradient adapter (task ≥ 1, trung vị) | Giá trị |
|---|---:|
| cos_c_kd | -0.139 |
| frac_neg_c_kd | 1.000 |
| cos_c_sep | -0.002 |
| frac_neg_c_sep | 0.500 |
| cos_kd_sep | 0.001 |
| cos_c_reg | -0.091 |
| pcgrad_keep_sep | 1.000 |
| ratio_sep_c | 0.244 |
| ratio_kd_c | 0.195 |
| task0_cos_c_base | 0.047 |

| Cuối task (trung vị task ≥ 1) | Giá trị |
|---|---:|
| auc_pqk | 0.592 |
| auc_rbf_theta | 0.720 |
| auc_euclid_z | 0.846 |
| auc_cosine_z | 0.949 |
| auc_pqk_cross_task | 0.592 |
| auc_z_cross_task | 0.846 |
| kernel_same | 0.652 |
| kernel_diff | 0.591 |
| w1_between | 0.040 |
| w1_within | 0.019 |
| w1_random | 0.010 |
| saturated | 0.435 |
| cka_rbf_theta | 0.181 |
| cka_rbf_z | 0.114 |
| pairs_all | 0.556 |
| pairs_topk | 0.853 |
| task 0: pqk/auc_class 0.759, twin_euclid_theta/auc_class 0.857, twin_euclid_z/auc_class 0.894, w1/between_share 0.090 | |

**Đọc theo luật P0.3:**

- W6: gần vuông góc (median cos(c,sep) -0.002) → xung đột KHÔNG phải nguyên nhân, bỏ P4
- W7: AUC PQK − RBF(θ) = -0.128 → PQK kém hơn
- W1: AUC PQK − AUC 768-d = -0.254 → phép chiếu làm mất thông tin lớp
- W1: tán xạ giữa lớp trong không gian chiếu = 3.9× ngẫu nhiên → projector bỏ sót hướng phân lớp
- W2: góc bão hoà 43.5% → tanh bão hoà, cần scale
- W7: CKA(PQK, RBF θ) = 0.181 → Gram PQK khác RBF
- W4: kernel top-k − toàn bộ = +0.297 → có cặp khó, loss trung bình đều làm loãng → P3 đáng thử

## p0_rsiat

Log: `/kaggle/working/runs/logs/adapter/imagenetr/0/20/p0_rsiat_1993_pretrained_vit_b16_224_in21k_adapter.log`

- Tái lập hp_diag W03 (same Kaggle environment as W02/W03): không có run tham chiếu (lệch tối đa — điểm)

| Gradient adapter (task ≥ 1, trung vị) | Giá trị |
|---|---:|
| cos_c_kd | -0.150 |
| frac_neg_c_kd | 1.000 |
| cos_c_sep | — |
| frac_neg_c_sep | — |
| cos_kd_sep | — |
| cos_c_reg | -0.150 |
| pcgrad_keep_sep | — |
| ratio_sep_c | 0.000 |
| ratio_kd_c | 0.188 |
| task0_cos_c_base | 0.142 |

| Cuối task (trung vị task ≥ 1) | Giá trị |
|---|---:|
| auc_pqk | — |
| auc_rbf_theta | — |
| auc_euclid_z | 0.870 |
| auc_cosine_z | 0.952 |
| auc_pqk_cross_task | — |
| auc_z_cross_task | 0.869 |
| kernel_same | — |
| kernel_diff | — |
| w1_between | — |
| w1_within | — |
| w1_random | — |
| saturated | — |
| cka_rbf_theta | — |
| cka_rbf_z | — |
| pairs_all | — |
| pairs_topk | — |
| task 0: pqk/auc_class —, twin_euclid_theta/auc_class —, twin_euclid_z/auc_class 0.925, w1/between_share — | |

**Đọc theo luật P0.3:**


