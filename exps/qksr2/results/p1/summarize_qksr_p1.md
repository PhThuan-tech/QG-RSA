# QKSR-v2 P1/P2: so sánh ghép cặp (`scripts/summarize_qksr_p1.py`)

Δ = A − B, ghép theo replicate; CI 95% t với n − 1 bậc tự do. P2 vượt đường đánh đổi khi Δ old > 0, Δ new > 0 và CI Δ AIA > 0 so với QKSR (spec P2.3).

| Run | Replicate | AIA | Final | Forgetting |
|---|---:|---:|---:|---:|
| p1_qksr_cur | 0 | 87.402 | 82.75 | 4.79 |
| p1_qksr_cur | 1 | 87.398 | 82.75 | 4.78 |
| p1_qksr_cur | 2 | 87.482 | 82.97 | 4.43 |
| p1_qksr_nocnot | 0 | 87.362 | 82.64 | 5.15 |
| p1_qksr_nocnot | 1 | 87.353 | 82.57 | 5.13 |
| p1_qksr_nocnot | 2 | 87.438 | 82.72 | 4.97 |
| p1_qksr_rbf | 0 | 87.303 | 82.60 | 5.05 |
| p1_qksr_rbf | 1 | 87.361 | 82.46 | 5.35 |
| p1_qksr_rbf | 2 | 87.397 | 82.75 | 4.81 |

| So sánh | n | Δ AIA | Δ final | Δ forgetting | Δ old | Δ new | Đọc |
|---|---:|---|---|---|---|---|---|
| P1 · PQK − RBF trên cùng góc θ (T2) | 3 | +0.074 [-0.007, +0.154] | +0.220 [+0.046, +0.394] | -0.402 [-0.777, -0.026] | +0.369 [+0.242, +0.496] | +0.010 [-0.265, +0.285] | không kết luận (CI chứa 0, rộng hơn ±0.1) |
| P1 · PQK − PQK bỏ CNOT (T4) | 3 | +0.043 [+0.036, +0.050] | +0.180 [+0.006, +0.354] | -0.412 [-0.674, -0.151] | +0.409 [+0.011, +0.808] | -0.117 [-0.401, +0.166] | A HƠN B (CI Δ AIA > 0) |
