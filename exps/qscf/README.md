# W07: lượng tử lúc train × SCF (spec, viết trước khi chạy)

## Câu hỏi

Khi mạng đã được train với lượng tử ở cả base task (PQK steering) và incremental task (QKSR), SCF có còn tăng
accuracy không, và QKSR có cộng thêm gì lên trên SCF không?

## Thiết kế 2×2

ImageNet-R, seed 1993, 3 replicate (`rng_replicate` 0, 1, 2), cả 6 run trong **một wave** (W07). Lý do: môi
trường Kaggle từng đổi giữa các vòng, task 0 của QKSR khác nhau giữa W04 và W05.

| Overlay | Train | Head được đánh giá |
|---|---|---|
| `qscf_rsiat_rep{0,1,2}` | RSIAT gốc (giống `p1_rsiat_rep*`) | linear, ridge, PGM, SCF mọi w |
| `qscf_qksr_rep{0,1,2}` | QKSR = `p1_qksr_cur_rep*` (`use_quantum_kernel_base` và `_inc` = true, PQK, `q_inc_pair: current`) | như trên |

Các khoá `qsd_*` chép nguyên từ `exps/scf/scf.json`. SCF đánh giá trên cùng mạng với linear head, nên không
ảnh hưởng tới quá trình train.

## Quy tắc chốt trước

- Trọng số SCF giữ nguyên như vòng A2: ridge w = 0.25, PGM w = 0.5. Không chỉnh sau khi thấy kết quả.
- **Kiểm định chính:** Δ Ā = (QKSR + SCF ridge) − (RSIAT + SCF ridge), ghép cặp theo replicate, t-interval 95%.
  - Cận dưới > 0: **ĐẠT** (QKSR cộng thêm được lên SCF).
  - Ngược lại: **KHÔNG ĐẠT**. Báo cáo đúng như vậy.
- **Báo cáo thêm (mô tả, không kiểm định):** gain SCF trên từng mạng; QKSR − RSIAT ở linear head; tương tác
  (gain SCF trên QKSR − gain SCF trên RSIAT); cùng các so sánh đó cho SCF PGM và cho A_B.
- Chỉ dùng run hoàn chỉnh (đủ 10 task). Thiếu cặp nào thì báo n thực tế. Không chạy bù để "đủ" kết quả đẹp.

## Kỳ vọng (ghi trước để đối chiếu)

Pilot Holevo/POVM + SCF (W03, 1 run mỗi config): SCF tăng Ā khoảng +0.4 trên mọi mạng; phần lượng tử lúc
train thêm +0.05 đến +0.10, trong dải nhiễu. Kỳ vọng W07: SCF tăng trên cả hai mạng, kiểm định chính KHÔNG ĐẠT.

## Chạy

```bash
python scripts/build_wave.py kaggle/waves/w07_qscf/wave.json      # -> W07_qscf_kaggle.ipynb
python scripts/summarize_qscf.py --root exps/qscf/results/kaggle1/logs --output exps/qscf/results/summary.md \
    --json exps/qscf/results/summary.json
```

Kết quả Kaggle (`results_W07_qscf.zip`) giải nén vào `exps/qscf/results/kaggle1/`.
