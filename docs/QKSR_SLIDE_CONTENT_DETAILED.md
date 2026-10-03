# Nội dung slide QKSR chi tiết — bắt đầu sau phần giới thiệu RSIAT

*Soạn ngày 02/10/2026. Nội dung để làm slide và tập trình bày; không phải báo cáo thực nghiệm mới.*

## Cách dùng và phạm vi

Bản này nối tiếp phần bài toán và RSIAT đã trình bày. Có 11 slide phương pháp, sau đó là hai slide kết quả và kiểm chứng. **Nội dung trên slide** có thể đưa trực tiếp vào PowerPoint. **Ghi chú thuyết trình** giải thích công thức, ví dụ và giới hạn; **Câu chuyển** nối sang slide tiếp theo.

Nguồn chính:

- [QKSR_PROBLEM_AND_PIPELINE.md](QKSR_PROBLEM_AND_PIPELINE.md), đặc biệt §4–13.
- [QKSR_PRESENTATION_SCRIPT.md](QKSR_PRESENTATION_SCRIPT.md), slide 3–10.
- [QML_FOUNDATIONS_FOR_QKSR.md](QML_FOUNDATIONS_FOR_QKSR.md), §5–13.
- [QKSR_Spec_for_RSIAT_v3.1.md](QKSR_Spec_for_RSIAT_v3.1.md), định vị đóng góp và vòng đời module.

Mô tả chính theo cấu hình tham chiếu [adapter_cifar224.json](../exps/adapter_cifar224.json): QKSR bật ở cả base và incremental, 8 qubit, 2 circuit layers, không reupload, RDM một qubit, incremental metric frozen, cặp old_proj, separation mode mean. Các log pilot lịch sử không được mặc định là dùng mọi giá trị của cấu hình hiện tại.

| Ký hiệu | Cách đọc |
|---|---|
| \(z_t\) | Feature của ảnh hiện tại qua encoder đang học |
| \(z^-\) | Feature của **cùng ảnh hiện tại** qua snapshot encoder cũ |
| \(\mu_c,\Sigma_c\) | Mean và covariance feature của lớp c |
| \(T_\omega\) | Residual projector của RSIAT, 768 → 768 |
| \(P_\eta\) | Classical projector trong QKSR, 768 → q góc |
| \(a,\theta\) | Góc phụ thuộc dữ liệu; góc circuit học được |
| \(D_Q,K_Q\) | Bình phương khoảng cách RDM; kernel similarity |
| \(\gamma_Q\) | Bandwidth kernel, khác trọng số separation \(\lambda_{\rm sep}\) |

Gợi ý bố cục: một bên là sơ đồ, một bên là 4–5 ý giải thích. Dùng xanh cho encoder hiện tại, xám cho frozen snapshot, cam cho QKSR, tím cho memory. Mũi tên liền thể hiện dữ liệu; mũi tên nét đứt dùng khi giải thích gradient. Hai hình vẽ của cùng một projector phải ghi “chung tham số”.

---

## Slide 1 — Đóng góp đề xuất: thay thước đo trong representation steering

### Nội dung trên slide

**RSIAT dùng representation steering để tổ chức feature và nối representation qua các task. QKSR khảo sát một cách tính similarity khác cho các ràng buộc đó.**

- **Thước đo đề xuất:** Feature được chiếu thành góc, đi qua mạch Ry–CNOT mô phỏng, rồi so sánh bằng local RDM và RBF kernel.
- **Ở base task:** Học metric từ quan hệ mẫu–mẫu; tăng similarity trong lớp và hạn chế similarity giữa các lớp.
- **Ở incremental task:** Dùng metric đã học và đóng băng để đo quan hệ giữa prototype lớp cũ và mẫu lớp mới sau residual projector.
- **Phạm vi tác động:** QKSR tạo training loss; đường dự đoán dùng ViT, shared adapter và classifier.
- **Giả thuyết nghiên cứu:** Feature map này có thể cung cấp learning signal phù hợp hơn các metric cổ điển. Hiệu quả giảm forgetting và đóng góp riêng của circuit cần kiểm chứng.

### Sơ đồ

~~~text
                     Representation steering trong RSIAT
                                 ↓
                   Đổi cách tính similarity sang QKSR
                         ↙                     ↘
                Base: mẫu–mẫu       Incremental: prototype–mẫu
                         ↓                     ↓
                   Loss theo nhãn       Separation loss
~~~

### Ghi chú thuyết trình

Classification loss yêu cầu phân loại đúng dữ liệu hiện tại, nhưng nhiều hình học feature khác nhau có thể thỏa mãn yêu cầu đó. Representation steering bổ sung một sở thích về hình học. Thay thước đo similarity sẽ thay đổi cách các ràng buộc được đánh giá và gradient được tạo ra.

Đóng góp đã hiện thực hóa là một module và cách tích hợp vào hai giai đoạn. Kết luận về hiệu quả cần thực nghiệm. Một metric cổ điển phi tuyến cũng có thể tạo hình học khác; sự hiện diện của circuit chưa đủ chứng minh lợi ích riêng.

### Câu chuyển

“Để hiểu metric này tác động như thế nào, trước hết em trình bày luồng từ một feature ảnh đến similarity.”

**Nguồn:** Problem & Pipeline §0, §2.2, §12.1; Presentation Script slide 3; Spec §1.

---

## Slide 2 — Classical projector: từ feature 768 chiều đến góc quay

### Nội dung trên slide

**Đầu vào QKSR là feature từ ViT và adapter. Projector chọn một biểu diễn nhỏ để circuit có thể xử lý.**

- **LayerNorm:** Chuẩn hóa các tọa độ của từng feature để giảm nhạy với scale đầu vào.
- **Linear:** Học q tổ hợp từ 768 chiều, lựa chọn thông tin đưa vào quantum feature map.
- **Tanh và nhân π:** Giới hạn mỗi giá trị thành góc trong \((-\pi,\pi)\).
- **Đầu ra:** \(a(z)\in\mathbb R^q\), mỗi thành phần điều khiển một qubit.
- **Cấu hình hiện tại:** q=8; nén giúp mô phỏng khả thi nhưng có thể làm mất thông tin.

### Công thức và sơ đồ

\[
a(z)=P_\eta(z)=\pi\tanh\!\left(W\,\operatorname{LN}(z)+b\right),
\qquad W\in\mathbb R^{q\times768}.
\]

~~~text
Ảnh → ViT + adapter → Feature z → LayerNorm → Linear → πtanh → q góc
                      768 chiều                            a(z)
~~~

Chú thích nhỏ: **Pη là projector trong QKSR; khác Tω của RSIAT.**

### Ghi chú thuyết trình

Projector thực hiện nén có học. Với batch 64 mẫu, tensor feature 64×768 trở thành tensor góc 64×8. Mỗi mẫu được xử lý riêng; LayerNorm không tính mean theo lớp hay trên toàn dataset.

Statevector của q qubit có \(2^q\) amplitudes, nên chọn q nhỏ là quyết định về chi phí. Tanh giới hạn góc nhưng có thể bão hòa gradient. W,b có 6.152 tham số khi q=8; đây là thành phần cần đối chứng để tránh quy mọi hiệu ứng cho circuit.

### Câu chuyển

“Projector tạo các góc phụ thuộc ảnh. Circuit dùng các góc này để tạo trạng thái của từng mẫu.”

**Nguồn:** Problem & Pipeline §4.1–4.2, §4.6; Foundations §10.1.

---

## Slide 3 — Ry–CNOT: tạo quantum feature map có tham số

### Nội dung trên slide

**Mỗi mẫu được ánh xạ thành statevector bằng circuit mô phỏng. Hai loại góc có vai trò khác nhau.**

- **Khởi tạo:** q qubit ở trạng thái \(|0\rangle^{\otimes q}\).
- **Data encoding:** \(R_y(a_j)\) mã hóa góc từ projector lên qubit j.
- **Variational rotation:** \(R_y(\theta_{\ell,j})\) dùng góc học được để điều chỉnh feature map.
- **CNOT ring:** Ghép các qubit lân cận, tạo tương tác có thể ảnh hưởng các mô tả trạng thái cục bộ.
- **Đầu ra:** \(|\psi(z)\rangle\); q=8 cho 256 amplitudes, hai lớp có 16 góc circuit học được.

### Sơ đồ

~~~text
|0...0⟩ → Ry(a) → Ry(θ₀) → CNOT ring → Ry(θ₁) → CNOT ring → |ψ(z)⟩
          dữ liệu        layer 0                 layer 1
~~~

| Loại góc | Đến từ đâu? | Vai trò |
|---|---|---|
| \(a(z)\) | Projector, thay đổi theo ảnh | Mã hóa dữ liệu |
| \(\theta\) | Tham số circuit dùng chung | Học cách biến đổi dữ liệu |

### Ghi chú thuyết trình

Trên một qubit ban đầu ở trạng thái 0:

\[
R_y(a)|0\rangle=\cos(a/2)|0\rangle+\sin(a/2)|1\rangle.
\]

Góc a thay đổi amplitudes. Khi có nhiều qubit, CNOT tạo sự phụ thuộc giữa các phần của hệ; có thể tạo entanglement tùy đầu vào. Có CNOT không bảo đảm feature map hữu ích cho phân loại.

Circuit cố định là ánh xạ tuyến tính theo statevector đầu vào. Feature map tổng theo z có tính phi tuyến từ projector, sin/cos của encoding và bước tạo RDM. Không gọi CNOT là “cổng phi tuyến”.

Trong cấu hình hiện tại, dữ liệu được encode một lần rồi qua hai lớp variational. Statevector được tính bằng PyTorch trên CPU/GPU; pipeline này chưa có quantum hardware.

### Câu chuyển

“Sau circuit, em trích mô tả cục bộ của từng trạng thái để so sánh hai mẫu.”

**Nguồn:** Problem & Pipeline §4.3; Foundations §5.2–5.3, §6, §10.2.

---

## Slide 4 — Local RDM và RBF: từ trạng thái đến similarity

### Nội dung trên slide

**QKSR đọc thông tin cục bộ, đo khác biệt giữa hai mô tả, rồi chuyển thành similarity.**

- **Local RDM:** Density matrix của một qubit sau khi bỏ qua phần còn lại. Order=1 giữ q ma trận 2×2 cho mỗi mẫu.
- **Khoảng cách Frobenius:** So từng phần tử RDM, bình phương chênh lệch và cộng trên các qubit.
- **RBF kernel:** Khoảng cách nhỏ cho similarity gần 1; khoảng cách lớn cho similarity nhỏ hơn.
- **Bandwidth γQ:** Điều khiển độ nhạy với khoảng cách; calibrate từ train subset khi metric được dùng lần đầu.
- **Đầu ra:** Ma trận similarity cho hai tập feature. K là độ tương đồng, chưa được hiệu chuẩn thành xác suất nhãn.

### Sơ đồ và công thức

~~~text
zi → Projector → Circuit → RDMs của mẫu i ─┐
                                          ├→ DQ → exp(−γQ DQ) → KQ(zi,zj)
zj → Projector → Circuit → RDMs của mẫu j ─┘
~~~

Hai nhánh dùng chung tham số.

\[
D_Q(z_i,z_j)=\sum_{k=1}^{q}
\|\rho_k(z_i)-\rho_k(z_j)\|_F^2,
\qquad
K_Q(z_i,z_j)=\exp[-\gamma_QD_Q(z_i,z_j)].
\]

### Ghi chú thuyết trình

RDM mô tả những gì có thể quan sát trong một phần nhỏ của hệ. Partial trace bỏ một phần thông tin toàn cục, nên order=1 không giữ toàn bộ tương quan nhiều qubit. Module có tùy chọn thêm RDM hai qubit, nhưng cấu hình tham chiếu chỉ dùng một qubit.

Ví dụ γQ=1: D=0 cho K=1, D=1 cho K≈0,368, D=2 cho K≈0,135. Cùng D=1, thay γQ cũng thay K; K nhỏ hơn chưa chứng minh representation phân loại tốt hơn.

Calibration dùng \(\gamma_0=1/\operatorname{median}(D)\) khi median hợp lệ, chạy một lần khi metric được dùng lần đầu. Bounded learned mode dùng \(\gamma_Q=\gamma_0\,10^{\tanh u}\). γQ khác khóa gamma của RSIAT, vốn là trọng số separation.

### Câu chuyển

“Kernel mới biết mức tương đồng. Ở base task, nhãn quyết định quan hệ nào cần tăng hoặc giảm.”

**Nguồn:** Problem & Pipeline §4.4–4.6; Foundations §8, §11–12.

---

## Slide 5 — Base task: học quan hệ mẫu–mẫu bằng nhãn

### Nội dung trên slide

**Task 0 chưa có lớp cũ. QKSR học cùng adapter từ các quan hệ trong minibatch hiện tại.**

- Tính \(K_{ij}=K_Q(z_i,z_j)\), tạo ma trận B×B.
- **Cặp cùng lớp:** Loss \(1-K_{ij}\) khuyến khích similarity tăng về 1.
- **Cặp khác lớp:** Loss \([K_{ij}-m_Q]_+\) chỉ phạt khi similarity vượt margin.
- Nhãn chọn cặp positive/negative; loại đường chéo i=j vì tự so sánh không cung cấp quan hệ giữa hai mẫu.
- Cộng steering với classification loss; warm-up tăng trọng số từ từ để hạn chế xung đột đầu training.

### Công thức

\[
L_{\rm RS}^{Q}
=\operatorname{mean}_{\mathcal P}(1-K_{ij})
+\alpha\,\operatorname{mean}_{\mathcal N}[K_{ij}-m_Q]_+,
\qquad
L_{\rm base}=L_{\rm cls}+\lambda_{\rm RS}(e)L_{\rm RS}^{Q}.
\]

P: cặp cùng lớp khác mẫu; N: cặp khác lớp; [v]+ = max(0,v). Code thêm epsilon để tránh chia 0 khi không có cặp hợp lệ.

### Sơ đồ

~~~text
Ảnh → ViT frozen + adapter → Features ─→ Classifier → Lcls ─┐
                                │                          ├→ Tổng loss → Update
                                └→ QKSR → K[B,B] → LRSQ ───┘
                                             ↑
                                  Nhãn chọn cặp +/−
~~~

Ví dụ, mQ=0,5:

| Cặp | K | Đóng góp loss của cặp |
|---|---:|---:|
| Cùng lớp | 0,8 | 0,2 |
| Khác lớp | 0,8 | 0,3 |
| Khác lớp | 0,2 | 0 |

### Ghi chú thuyết trình

Loss khuyến khích hình học trong không gian metric; không bảo đảm feature cosine của classifier lập tức tốt hơn. Adapter, classifier, projector QKSR, circuit angles và bandwidth học được nhận update; backbone pretrained giữ cố định.

Warm-up: \(\lambda_{\rm RS}(e)=\lambda_{\rm RS}^{\max}\min(1,e/E_{\rm warm})\). Với e=0, trọng số steering bằng 0; từ e=7 đạt 0,2 trong cấu hình tham chiếu. Batch không có cặp cùng lớp khác mẫu sẽ có positive term bằng 0 do thiếu cặp.

### Câu chuyển

“Sau base task, dữ liệu lớp này không được giữ để replay. Ta lưu thông tin cần thiết cho task tiếp theo.”

**Nguồn:** Problem & Pipeline §5–6; Presentation Script slide 5.

---

## Slide 6 — Sau base task: chuẩn bị memory và metric cho incremental learning

### Nội dung trên slide

**Thống kê lớp và snapshot giữ thông tin quá khứ mà không cần lưu ảnh cũ.**

- **Mean μc:** Tâm feature mỗi lớp; dùng làm prototype trong incremental separation.
- **Covariance Σc:** Mô tả độ phân tán và quan hệ giữa các chiều; dùng cùng mean để sinh Gaussian features cho classifier alignment.
- **Snapshot encoder:** Bản sao encoder sau task, đóng băng khi làm nhánh cũ ở task tiếp theo.
- **Metric QKSR:** Đã học ở base; toàn bộ metric đóng băng khi sang incremental trong cấu hình mặc định.
- Memory chứa thống kê trong không gian feature 768 chiều, không phải density matrix trung bình của lớp.

### Công thức và sơ đồ

\[
\mu_c=\frac1{n_c}\sum_{i:y_i=c}z_i,
\qquad
\Sigma_c=\frac1{n_c-1}\sum_{i:y_i=c}
(z_i-\mu_c)(z_i-\mu_c)^T+\epsilon_{\rm cov}I.
\]

~~~text
Encoder sau base ─→ Trích feature theo lớp ─→ μc, Σc ─→ Memory
       │
       └→ Tạo snapshot frozen cho task tiếp theo

QKSR đã học ở base ─→ Giữ tham số cố định khi sang incremental
~~~

### Ghi chú thuyết trình

Mean là trung bình feature; covariance giữ thông tin phân tán mà mean không có. Ridge εI giúp dùng covariance cho Gaussian sampling. Statistics được tính sau training bằng evaluation transforms.

Prototype μc khác classifier weight wc: μc tính từ dữ liệu, wc được optimizer học. Biến đổi quantum của μc thường khác trung bình các quantum representations, vì feature map phi tuyến.

Không lưu ảnh cũ vẫn có memory: head và statistics tăng theo số lớp. Base tính statistics nhưng chưa chạy incremental classifier alignment trong call hiện tại.

### Câu chuyển

“Với memory và snapshot này, mỗi ảnh lớp mới được nhìn qua cả encoder hiện tại và encoder cũ.”

**Nguồn:** Problem & Pipeline §7, §10.1, §11.2.

---

## Slide 7 — Incremental task: classification và representation alignment

### Nội dung trên slide

**Mô hình vừa học nhãn lớp mới, vừa nối representation trước và sau adaptation.**

- Mở rộng classifier cho lớp mới; main classification loss dùng scores của các lớp thuộc task hiện tại.
- Cùng ảnh mới x tạo \(z_t=f_{\phi_t}(x)\) và \(z^-=f_{\phi_{t-1}}(x)\).
- Encoder cũ là snapshot frozen. “Old feature” là feature của ảnh mới qua encoder cũ.
- Residual projector Tω biến đổi z⁻ thành \(\hat z^-=T_\omega(z^-)\), vẫn 768 chiều.
- Alignment MSE khuyến khích zt phù hợp với \(\hat z^-\), cập nhật adapter hiện tại và Tω.

### Sơ đồ

~~~mermaid
flowchart LR
    X["Cùng ảnh lớp mới x"] --> CUR["Encoder hiện tại<br/>ViT frozen + adapter train"]
    X --> OLD["Snapshot encoder cũ<br/>frozen"]
    CUR --> Z["Feature z_t"]
    Z --> HEAD["Classifier mở rộng"]
    HEAD --> LC["Classification loss<br/>trên lớp mới"]
    OLD --> O["Feature z⁻"]
    O --> T["Residual projector Tω<br/>train"]
    T --> P["Feature đã chiếu<br/>Tω(z⁻)"]
    Z --> LA["Alignment loss"]
    P --> LA
~~~

\[
L_{\rm align}=\operatorname{MSE}(z_t,\hat z^-)
=\frac1{Bd}\sum_{i=1}^{B}\|z_{t,i}-\hat z_i^-\|_2^2,
\qquad d=768.
\]

### Ghi chú thuyết trình

MSE lấy trung bình cả batch và tọa độ feature. Projector tạo cầu nối để tránh buộc representation hiện tại bằng trực tiếp feature cũ. T vẫn học nên alignment target cũng thay đổi.

Tω của RSIAT khác Pη bên trong QKSR. Code có final Sigmoid ở nhánh residual, nên T không phải phép vận chuyển hoàn toàn tự do. T được tạo ở incremental task đầu tiên và giữ qua các task sau.

### Câu chuyển

“Alignment nối feature của cùng ảnh qua thời gian. QKSR bổ sung quan hệ với prototype lớp cũ trong memory.”

**Nguồn:** Problem & Pipeline §5.2, §8.1–8.3, §10.2.

---

## Slide 8 — Incremental task: QKSR separation giữa prototype cũ và mẫu mới

### Nội dung trên slide

**Trong old_proj, QKSR so prototype lớp cũ và feature ảnh mới sau cùng residual projector.**

- Lấy mean μc⁻ của mỗi lớp cũ từ memory.
- Chiếu hai đầu vào bằng cùng Tω: \(\hat\mu_c=T_\omega(\mu_c^-)\), \(\hat z_i^-=T_\omega(z_i^-)\).
- QKSR frozen tạo cross-kernel kích thước Cold×B.
- Kci cho biết prototype cũ c giống feature mới i đến mức nào theo metric.
- Mean separation giảm similarity trung bình; không có positive-pair term như ở base.

### Sơ đồ và công thức

~~~text
Memory μc⁻ ───────────────→ Tω → Tω(μc⁻) ─┐
                                          ├→ QKSR frozen → K[Cold,B] → LsepQ
Ảnh mới → Encoder cũ → z⁻ → Tω → Tω(z⁻) ──┘
                           chung tham số
~~~

\[
K_{ci}^{\rm inc}
=K_Q(T_\omega(\mu_c^-),T_\omega(z_i^-)),
\qquad
L_{\rm sep}^{Q}=\frac1{C_{\rm old}B}\sum_{c,i}K_{ci}^{\rm inc}.
\]

Chú thích: **LsepQ tương ứng biến loss_orth; cấu hình hiện tại inc_loss_mode=mean.**

### Ghi chú thuyết trình

Kernel nhận transformed old prototype và transformed feature từ encoder cũ. zt không trực tiếp đi vào kernel ở old_proj. Giảm RDM-kernel similarity không đồng nghĩa với ép Euclidean dot product bằng 0.

Margin mode lấy mean của [Kci−minc]+, ngừng phạt khi similarity dưới margin; đây không phải cấu hình tham chiếu. Biến thể current thay đầu vào thứ hai bằng zt, đổi đường gradient và là ablation riêng.

Prototype sai hoặc lớp cũ–mới chia sẻ cấu trúc hữu ích có thể làm việc giảm đồng loạt similarity không có lợi. Cần đo old/new accuracy cùng classifier cuối cùng.

### Câu chuyển

“Để hiểu tác động thật của QKSR, cần ghép các loss và nhìn đường gradient.”

**Nguồn:** Problem & Pipeline §8.4–8.7; Presentation Script slide 6.

---

## Slide 9 — Tổng loss và đường cập nhật trong incremental task

### Nội dung trên slide

\[
L_{\rm inc}
=L_{\rm cls}+\beta L_{\rm align}+\lambda_{\rm sep}L_{\rm sep}^{Q}.
\]

| Thành phần | Mục đích | Nhận gradient trực tiếp ở old_proj, frozen |
|---|---|---|
| Lcls | Phân biệt lớp mới | Adapter hiện tại, classifier |
| Lalign | Nối feature hiện tại với feature cũ đã chiếu | Adapter hiện tại, Tω |
| LsepQ | Giảm similarity prototype cũ–mẫu mới | Tω |

- **Frozen metric:** Pη, θ và bandwidth giữ cố định; kernel vẫn khả vi theo đầu vào.
- **Tác động gián tiếp:** Separation đổi Tω → alignment target đổi → ảnh hưởng adapter qua alignment.
- **Cân bằng loss:** Trọng số mạnh có thể cản thích nghi lớp mới; yếu có thể không tạo tác động hữu ích.

### Sơ đồ gradient

~~~text
LsepQ → Frozen kernel → Tω(prototype) ─┐
                      → Tω(z⁻) ──────┴→ Cập nhật Tω

Lalign → zt ───────────────────────────→ Cập nhật adapter
        → Tω(z⁻) ─────────────────────→ Cập nhật Tω

Lcls → Classifier và zt ───────────────→ Cập nhật head và adapter
~~~

### Ghi chú thuyết trình

Frozen parameter không ngắt gradient theo input. Hàm h(x)=2x có hệ số 2 cố định nhưng đạo hàm theo x vẫn bằng 2. QKSR frozen tương tự về nguyên tắc.

Metric cố định không làm toàn geometry cố định: T và đầu vào metric vẫn đổi. Không mô tả separation trực tiếp kéo current feature ở old_proj.

λsep là trọng số loss; γQ là bandwidth. Trong cấu hình tham chiếu, beta=1 và khóa gamma=0,4 tương ứng λsep.

### Câu chuyển

“Khi train xong, representation đã đổi. Memory và classifier cần điều chỉnh trước khi đánh giá task.”

**Nguồn:** Problem & Pipeline §8.6–8.8, §11.2; Foundations §13.3.

---

## Slide 10 — Sau task: bù drift và classifier alignment

### Nội dung trên slide

**Main training chỉ là bước đầu. Pipeline còn cập nhật memory và cân chỉnh classifier trên các lớp đã gặp.**

- **Ước lượng drift:** Thu feature trước/sau adaptation trên dữ liệu hiện tại; dùng displacement để ước lượng dịch chuyển mean lớp cũ.
- **Bù mean cũ:** Mẫu gần prototype cũ trong old feature space được gán trọng số cao hơn; cộng weighted displacement vào mean.
- **Statistics:** Tính mean/covariance lớp mới; giữ covariance lớp cũ trong implementation hiện tại.
- **Gaussian replay:** Sinh feature tổng hợp theo thống kê các lớp đã học.
- **Classifier alignment:** Chỉ tối ưu classifier bằng cross-entropy trên toàn bộ lớp đã gặp; chuẩn bị snapshot/checkpoint cho task tiếp theo.

### Sơ đồ

~~~text
Features trước/sau train → Ước lượng drift → Bù mean lớp cũ ───┐
                                                             ↓
Features lớp mới → Mean/covariance lớp mới ───────────────→ Memory
                                                             ↓
                                                   Gaussian sampling
                                                             ↓
                                                Chỉ cập nhật classifier
                                                             ↓
                                         Evaluate và lưu cho task tiếp theo
~~~

\[
\mu_c\leftarrow\mu_c+\Delta_c,
\qquad
\tilde z_c\sim\mathcal N(\tilde\mu_c,\Sigma_c).
\]

μ̃c là mean sampling; code còn scale mean theo tuổi lớp.

### Ghi chú thuyết trình

Theo cơ chế mô tả, displacement lấy difference trước/sau adaptation của cùng ảnh, cùng view. Weighted displacement giả định mẫu gần nhau có drift tương tự. Đây là estimator, không phải truy cập feature lớp cũ sau adaptation.

§13.1 lưu ý hai lượt extract trong code dùng shuffled loader và bỏ sample ID, nên correspondence theo ảnh chưa được bảo đảm. Không khẳng định estimator đã ghép đúng mọi ảnh; cần kiểm chứng trước khi quy kết kết quả cho metric.

Bù mean dùng Gaussian distance weighting của RSIAT, không dùng QKSR. Old covariance được copy, chưa transport. Mỗi CA epoch lấy 256 synthetic features/lớp, encoder ở eval mode, chỉ train head. Mean sampling scale theo tuổi lớp. QKSR không có trong CA loss.

### Câu chuyển

“Các module phụ trợ phục vụ học. Dự đoán sử dụng encoder và classifier cuối cùng.”

**Nguồn:** Problem & Pipeline §9–10, §13.1.

---

## Slide 11 — Inference và toàn bộ vòng đời phương pháp

### Nội dung trên slide

**Ảnh test được phân loại trên tất cả lớp đã gặp, không cần biết task của ảnh.**

\[
x\ \xrightarrow{\rm ViT+adapter}\ z
\ \xrightarrow{\rm cosine\ classifier}\ s
\ \xrightarrow{\arg\max}\ \hat y.
\]

- QKSR tạo learning signal trong training; prediction dùng encoder và classifier cuối cùng.
- Suy luận không cần quantum projector, circuit, RDM, snapshot cũ hay Tω.
- Chi phí bổ sung của QKSR nằm ở training; số lớp vẫn làm classifier và memory tăng như baseline.

### Dòng thời gian

~~~text
BASE: Train adapter + classifier + QKSR
             ↓
Tính statistics và lưu snapshot
             ↓
INCREMENTAL: Classification + alignment + QKSR separation
             ↓
Bù drift → statistics lớp mới → classifier alignment
             ↓
Lưu snapshot → task tiếp theo ... → INFERENCE
~~~

### Ghi chú thuyết trình

Nếu QKSR có ích, lợi ích thể hiện qua tham số được học, không cần gọi kernel lúc test. “Không thêm inference computation từ QKSR” đúng với architecture hiện tại; không suy ra toàn chi phí mô hình giữ nguyên khi số lớp tăng.

### Câu chuyển

“Sau khi thấy toàn bộ pipeline, em trình bày pilot và mức kết luận dữ liệu hiện hỗ trợ.”

**Nguồn:** Problem & Pipeline §10.3, §11.2–11.4; Presentation Script slide 7.

---

## Slide 12 — Pilot trong tài liệu: chưa thấy cải thiện

### Nội dung trên slide

**Các mốc task 0–4 sau classifier alignment trong log đã cung cấp:**

| Task | RSIAT repo (%) | QKSR (%) | QKSR − RSIAT, điểm % |
|---:|---:|---:|---:|
| 0 | 99,10 | 99,00 | −0,10 |
| 1 | 97,70 | 97,50 | −0,20 |
| 2 | 97,03 | 96,70 | −0,33 |
| 3 | 96,30 | 96,02 | −0,28 |
| 4 | 95,18 | 94,96 | −0,22 |

- Trung bình chênh lệch năm mốc: **−0,226 điểm phần trăm**.
- Đoạn chuỗi quan sát chưa cho thấy improvement.
- Mới có một run và resume; chưa có nhiều paired seeds.
- Chưa đủ bằng chứng quy nguyên nhân cho metric, base checkpoint, tích hợp hay điều kiện thực nghiệm.

### Ghi chú thuyết trình

Đây là log lịch sử được ghi trong tài liệu, không phải kết quả mới tạo khi soạn slide. Task 5 QKSR ở log được đọc mới có pre-CA, không đưa vào bảng post-CA.

Mean năm chênh lệch không phải forgetting. Cần accuracy từng nhóm lớp/task qua thời gian để đo forgetting. Kernel histogram minibatch cuối cũng không đại diện toàn task.

Không quy kết mặc định rằng chỉ cần tune thêm sẽ tốt hơn.

### Câu chuyển

“Để tìm nguyên nhân, em tách đóng góp base, incremental và cấu trúc feature map.”

**Nguồn:** Problem & Pipeline §13.2–13.3; Presentation Script slide 8.

---

## Slide 13 — Đối chứng và đóng góp cần kiểm chứng

### Nội dung trên slide

| Câu hỏi | Cách kiểm tra |
|---|---|
| H1: QKSR base tạo điểm khởi đầu hữu ích hơn không? | Baseline A và base-only C; các task sau cùng cơ chế classical |
| H2: Separation QKSR incremental có lợi không? | Full B và base-only C từ **cùng base checkpoint đã học QKSR** |
| H3: Circuit đóng góp gì? | Projected RBF/MLP với budget phù hợp; no-CNOT, random frozen circuit |

- **Cơ chế:** Old/new accuracy, forgetting, pre/post CA, prototype drift, kernel distribution và gradient.
- **Công bằng:** Nhiều paired seeds; cùng class order, protocol và ngân sách chọn hyperparameter.
- **Tái lập:** Chạy liên tục hoặc kiểm chứng phục hồi đầy đủ RNG; kiểm tra correspondence feature trước/sau.
- **Validation:** Held-out samples không tham gia statistics/CA fitting; tuning dùng validation.
- **Đóng góp kỳ vọng:** Xác định điều kiện metric hữu ích, vai trò base/incremental, phần hiệu ứng liên quan circuit và giới hạn.

### Ghi chú thuyết trình

Dùng tên run theo §12.2: A=classical/classical, B=quantum/quantum, C=quantum/classical. B−C cô lập incremental trong bối cảnh quantum base nếu dùng chung checkpoint và protocol.

Classical base + frozen quantum incremental cần mô tả metric học ở đâu. Chỉ tắt base quantum và bật frozen incremental sẽ dùng metric chưa có supervision base; đó là control random/frozen.

§13.2 lưu ý held-out validation có thể vẫn nằm trong nguồn tính statistics nếu không xử lý đúng. Đây là yêu cầu cho thí nghiệm tiếp theo, không mặc định gate đã hoàn thành.

### Lời kết

“QKSR đã được hiện thực hóa như metric khả vi trong loss của RSIAT. Base học quan hệ mẫu–mẫu; incremental mặc định điều khiển residual projector qua prototype–mẫu. Pilot chưa cho thấy cải thiện, nên bước tiếp theo là đối chứng công bằng trước khi kết luận hiệu quả.”

**Nguồn:** Problem & Pipeline §12, §13.2–13.4; Presentation Script slide 9–10.

---

## Khi chuyển sang PowerPoint

- Slide 2–4: góc → trạng thái → similarity. Giữ dải pipeline ở đầu, tô nổi bước đang nói.
- Slide 7–9: dùng cùng hình incremental. Hiện current/old và alignment, rồi thêm memory/QKSR, cuối cùng nhấn gradient và tổng loss.
- Đặt công thức cạnh mục tiêu; đọc ý nghĩa trước ký hiệu.
- Chi tiết sigmoid projector, calibration, margin/current ablation và giới hạn protocol có thể vào speaker notes hoặc slide dự phòng.
- Phân biệt “đã triển khai”, “giả thuyết”, “kết quả quan sát” và “kế hoạch kiểm chứng” trong cách đặt câu.
