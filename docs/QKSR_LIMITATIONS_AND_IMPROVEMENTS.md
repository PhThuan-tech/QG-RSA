# Hạn chế QKSR và các cải tiến đã triển khai

Đánh giá, mở rộng ngày 03/10/2026 từ source hiện tại và log ImageNet-A/R trong repo.
Đây là phân tích cơ chế và sửa triển khai, **chưa phải kết quả huấn luyện mới**.
Không sửa log cũ, không tự bật objective mới cho cấu hình legacy.
Các cải tiến thay đổi thuật toán được bật riêng qua profile `retention` và chuỗi
ablation `G0`–`G8`; sửa ổn định số/kiểm tra checkpoint áp dụng chung.

## 0. Phạm vi đọc và bằng chứng từ run hiện tại

Đã đối chiếu luồng của `models/RSIAT_adapter.py`, `models/base.py`,
`utils/quantum_kernel.py`, `utils/loss.py`, `utils/toolkit.py`, `utils/inc_net.py`,
`network/classifier.py`, `network/vision_transformer_adapter.py`,
`data/data.py`, `data/data_manager.py`, `trainer.py`, `main.py`, các config A/R,
notebook, tests và tài liệu nghiên cứu trong repo. Audit theo toàn bộ chuỗi:

```text
Ảnh train -> adapter + loss phân loại/regularization
          -> đặc trưng trước/sau trên cùng ảnh train
          -> cập nhật bộ nhớ mean/covariance
          -> lấy mẫu Gaussian + classifier alignment (CA)
          -> accuracy từng task, BWT, forgetting
```

Các kết luận được chia thành ba mức: lỗi/giới hạn nhìn thấy trực tiếp trong
code; cơ chế có nguy cơ gây hại nhưng cần ablation; hiệu quả accuracy chỉ được
khẳng định khi có run dữ liệu thật, cùng protocol và nhiều seed.

| Run QKSR đã có, seed 1993, B0I20 | Final accuracy | Old / new accuracy cuối | Forgetting cuối | BWT cuối |
| --- | ---: | ---: | ---: | ---: |
| ImageNet-A | 64.70% | 63.71% / 75.18% | 10.3178 điểm % | -10.3178 điểm % |
| ImageNet-R | 82.88% | 82.65% / 85.15% | 4.4122 điểm % | -4.3467 điểm % |

Nguồn: hai file `logs/adapter/imageneta/0/20/imageneta_qksr_1993_pretrained_vit_b16_224_in21k_adapter.log`
và `logs/adapter/imagenetr/0/20/01_imagenetr_qksr_1993_pretrained_vit_b16_224_in21k_adapter.log`.
Log RSIAT A gốc có final 66.23%, old 66.02%, new 68.55%, nhưng chưa có ma trận
accuracy theo task và chưa chứng minh manifest/split giống hệt run QKSR.
Do đó **chưa được kết luận QKSR làm forgetting tăng bao nhiêu so với RSIAT**.
Chênh lệch total task đầu/cuối không phải công thức forgetting: số lớp và
độ khó của tập đánh giá đã thay đổi. Dữ liệu A gợi ý ưu tiên kiểm tra khả năng
giữ lớp cũ, không tự chứng minh nguyên nhân là quantum kernel.

## 1. Những hạn chế có căn cứ trong source

### 1.1. Bù dịch chuyển đang có thể ghép sai ảnh

Trong `Learner.incremental_train`, bản cũ lấy embedding trước và sau huấn luyện
bằng hai lần duyệt `self.train_loader`. Loader này có `shuffle=True` và transform
huấn luyện ngẫu nhiên. Nhưng `BaseLearner.displacement` tính `DY = Y2 - Y1` theo
từng hàng, đòi hỏi hai hàng tương ứng là **cùng ảnh, cùng cách tiền xử lý**.
Như vậy độ dịch chuyển có thể chứa cả khác biệt giữa hai ảnh và augmentation.

Đây là lỗi tích hợp kế thừa RSIAT, không riêng quantum kernel. Không thể dùng nó
để kết luận QKSR là nguyên nhân của chênh lệch accuracy trên ImageNet-A.

Đã thêm `ssca_feature_mode='paired_eval'`: lấy đúng tập con train, transform
đánh giá xác định, không shuffle, generator riêng và `num_workers=0`.
Chế độ `legacy` vẫn giữ đường lấy đặc trưng cũ để làm đối chứng.
Khi thử sửa này, phải chạy lại **cả RSIAT và QKSR** cùng chế độ.

### 1.2. Lực đẩy incremental không tác động trực tiếp vào adapter ở `old_proj`

Với cấu hình đã chạy:

```text
old_proto -> old_ae -> quantum kernel <- old_ae <- frozen old_encoder(image)
current_adapter(image) -------- MSE alignment ---------^
```

`old_proj` so sánh prototype đã biến đổi với đặc trưng encoder cũ đã biến đổi.
Gradient của repulsion đi vào `old_ae`; adapter hiện tại nhận nó gián tiếp qua
MSE. Metric frozen vẫn truyền gradient đến đầu vào, không phải tắt toàn bộ
gradient. Test tách riêng repulsion (`beta=0`) xác nhận gradient lên đặc trưng
hiện tại bằng 0 ở `old_proj`, khác 0 ở `current`.

`q_inc_pair='current'` đã tồn tại trong source trước lần sửa này; đây là ablation
đổi đường gradient, không được trình bày như thuật toán mới. Đã sửa calibration
của run **inc-only** để dùng đúng nhánh được chọn. Run dùng cả base/inc vẫn giữ
quy tắc gamma chuẩn hóa một lần tại base, không âm thầm chuẩn hóa lại mỗi task.

### 1.3. Giảm similarity trung bình không có điểm dừng theo độ tách

Kernel có dạng `K=exp(-gamma_Q*d)`, nằm trong (0,1]. `inc_loss_mode='mean'`
tiếp tục giảm similarity của mọi cặp prototype cũ/ảnh mới, kể cả cặp đã đủ xa.
Điều này có thể cạnh tranh với loss phân loại và alignment; cần ablation để
biết có thực sự làm giảm accuracy hay không.

`margin` đã có sẵn: `mean(relu(K-m))`, ngừng repulsion với cặp có `K<=m`.
Lần sửa này thêm hệ số riêng và warmup để không phải thay cả loss alignment:

```text
L_inc = beta * MSE + gamma * w(e) * L_repulsion
w(e)  = q_inc_weight * min(1, (e+1)/q_inc_warmup_epochs)
```

`e` bắt đầu từ 0. Warmup bằng 0 nghĩa là dùng hệ số đầy đủ ngay lập tức.
Mặc định `q_inc_weight=1`, warmup=0 giữ loss cũ; hai lựa chọn này không thay đổi
loss cosine RSIAT khi quantum incremental bị tắt.
`gamma` là hệ số loss, **khác** bandwidth `gamma_Q` của kernel.

### 1.4. ImageNet-A đang thay đổi nhiều hơn chỉ loại kernel

`exps/adapter_imageneta.json` có `rs_margin=1.5`. Cosine không vượt 1, nên
`relu(cosine-1.5)` bằng 0: nhánh đẩy cặp khác lớp trong base RSIAT không hoạt động.
Notebook QKSR lại dùng `rs_margin_q=0.5`, tức nhánh negative-pair hoạt động trở lại.
ImageNet-R dùng margin cosine 0.4. Ngoài ra hệ số incremental `gamma` là 6 ở A,
1.5 ở R; giữ nguyên hệ số không bảo đảm hai metric có cùng cường độ gradient.

Vì vậy không thể suy ra hơn/kém chỉ do thành phần quantum từ hai log này.
Không tự sửa margin RSIAT đã công bố: baseline nguyên bản cần được giữ riêng.
Đã thêm đối chứng RBF cổ điển với **cùng cấu hình current/margin/weight/warmup**.
Đối chứng này giúp kiểm tra metric, không loại bỏ hết khác biệt kiến trúc hay
chứng minh ưu thế lượng tử.

### 1.5. Bộ nhớ lớp cũ chưa theo kịp đầy đủ biến đổi biểu diễn

Source lưu mỗi lớp bằng mean và covariance cho classifier alignment. SSCA cập
nhật mean cũ; `_compute_class_mean` sao chép covariance cũ, chỉ tính covariance
mới cho lớp mới. Kernel incremental chủ yếu so với mean, không mô hình hóa
đa mode hay độ bất định của từng lớp.

Đã bổ sung **transport mean và covariance** dạng residual ridge có kiểm tra
độ tin cậy, bật riêng bằng `statistics_transport='guarded_ridge'`; chi tiết ở
1.9. Chế độ `legacy` vẫn giữ mean-only SSCA để đối chứng. Chưa triển khai
nhiều prototype; bật metric trainable không tự sửa bộ nhớ bị lệch.

### 1.6. Giới hạn biểu diễn và bằng chứng nghiên cứu

- Projector nén đặc trưng ViT 768 chiều xuống 8 góc ở cấu hình hiện tại.
  Đây là nút thắt biểu diễn có thể quan trọng, nhưng chưa có ablation chứng minh
  nó là nguyên nhân thua trên ImageNet-A.
- Kernel bậc 1 chỉ quan sát RDM từng qubit, không biểu diễn mọi tương quan nhiều
  qubit. `order=2` và re-upload đã có, cần thử riêng với đối chứng phù hợp.
- Đây là mô phỏng statevector bằng PyTorch, chi phí tăng theo `2**q`.
  Không có bằng chứng về quantum advantage từ việc vượt baseline một seed.
- Khi incremental frozen, projector/metric/bandwidth không thích nghi với drift
  ở task sau. Đổi trainable là ablation riêng vì metric cũng có thể học cách
  giảm similarity mà không tạo biểu diễn hữu ích cho classifier.

### 1.7. Residual Sigmoid không mô tả được drift có dấu

Trong `utils/toolkit.py`, projector legacy trả về:

```text
g(z) = z + sigmoid(decoder(encoder(z)))
```

Mọi tọa độ residual đều thuộc `(0,1)`. Ví dụ, muốn ánh xạ tọa độ `2.0 -> 1.7`
thì cần residual `-0.3`, nhưng kiến trúc này không thể làm được. Khởi tạo
decoder gần 0 còn tạo residual quanh `+0.5`, không phải identity. Vì MSE
alignment tối ưu cả adapter và projector, adapter có thể phải dịch theo một
hướng không phù hợp chỉ để khớp mục tiêu bị ràng buộc này. Giới hạn biểu diễn
là chắc chắn; mức ảnh hưởng đến accuracy A/R vẫn là giả thuyết.

Đã thêm `SignedResidualAutoencoder` với `ae_type='signed_residual'`:

```text
g(z) = z + decoder(encoder(z))
```

Bỏ Sigmoid cuối, giữ nguyên các tầng/kích thước và số tham số. Linear cuối
được khởi tạo weight/bias bằng 0, nên bắt đầu đúng `g(z)=z`; có thể học cả
drift âm và dương. Khởi tạo dùng RNG CPU riêng qua `fork_rng`, không dịch
chuỗi ngẫu nhiên dùng cho adapter/augmentation. Lưu ý: gradient encoder ở
bước đầu có thể bằng 0 do weight decoder cuối bằng 0; sau khi tầng cuối cập
nhật, đường gradient mở lại. Đây không phải bằng chứng barren plateau.
Residual có dấu không bị chặn biên độ: cần theo dõi MSE/accuracy, không bảo
đảm projector không overfit. Giới hạn operator norm ở 1.9 chỉ áp dụng cho
ánh xạ hậu xử lý ridge, **không** áp dụng cho `old_ae` trong huấn luyện.

### 1.8. Tái sử dụng projector giữa các phép ánh xạ khác nhau

Source legacy chỉ tạo `old_ae` khi `_cur_task==1`, rồi giữ tham số cho mọi
task sau. Nhưng task `t` yêu cầu ánh xạ từ không gian `t-1` sang `t`, không
phải tiếp tục dùng phép ánh xạ `t-2 -> t-1`. Warm-start này có thể có ích,
nhưng cũng có thể mang bias của drift cũ; không nên mặc định coi là đúng.

Đã thêm `ae_reset_each_task=True`, mỗi task incremental tạo một projector
mới, khởi tạo bằng seed riêng `ae_init_seed + task`. Profile retention dùng
projector có dấu nên mỗi lần reset bắt đầu từ identity. Checkpoint khôi phục
đúng loại projector và trạng thái; không được resume checkpoint Sigmoid sang
signed. Các bước G1/G2 tách lợi ích của bỏ Sigmoid khỏi lợi ích của reset.

### 1.9. SSCA mean-only bỏ qua rotation/scaling và ngoại suy thiếu hỗ trợ

Ngoài covariance cũ không đổi, `BaseLearner.displacement` còn dùng
`exp(-distance/(2*sigma**2))+1e-5`. Khi prototype cũ rất xa tất cả ảnh lớp
mới, mọi trọng số có thể gần `1e-5`: thuật toán vẫn cộng gần như drift trung
bình của lớp mới vào một lớp cũ không có dữ liệu hỗ trợ. Thêm epsilon giúp
tránh chia 0 nhưng không tạo ra bằng chứng rằng phép ngoại suy là đúng.

LDC phân tích vì sao bù bằng translation có thể sai khi đặc trưng bị xoay
hoặc co giãn, rồi học ánh xạ giữa đặc trưng trước/sau trên ảnh task hiện tại.
Đây là cơ sở để thử một ánh xạ thay vì chỉ cộng displacement.
[LDC, mục 3.2–3.3](https://arxiv.org/html/2407.08536v1).

Triển khai mới trong `utils/statistics_transport.py` là **ablation riêng do
repo thiết kế, không phải tái hiện nguyên bản LDC**. Đặt `X/Y` là đặc trưng
của cùng ảnh train trước/sau stage 1, dùng quy ước vector hàng:

```text
c = mean(X), b = mean(Y-X)
V = các hướng PCA/SVD của X-c, kích thước D x r
H = (X-c)V
R = argmin_R ||H R - ((Y-X)-b)||² + lambda ||R||²
T(z) = z + b + (z-c)V R
```

`lambda = transport_ridge * trace(H.T H)/r`, có sàn số học. Rank thực tế
không vượt rank dữ liệu, `N-1` hay D. Phần residual được giới hạn
`||R||₂ <= transport_max_change < 1`. Với V trực chuẩn, đây cũng là chặn
`||V R||₂`. Rank bằng 0 thì chỉ còn translation.

Không áp dụng T một cách mù quáng:

1. Tách 20% **các cặp thuộc train của task hiện tại** bằng generator riêng;
   fit trên phần còn lại. Nếu MSE trên cặp giữ lại không tốt hơn identity,
   giữ nguyên bộ nhớ cũ. Đây là kiểm tra fit drift, không phải validation
   accuracy và không dùng test/held-out của lựa chọn siêu tham số.
2. Tính `gain = clip(1 - MSE_map/MSE_identity, 0, 1)`, rồi refit trên toàn bộ
   cặp train. Cho mỗi lớp cũ, tính khoảng cách đến ảnh train gần nhất và
   `support = exp(-distance²/(2*support_scale²*spread))`, trong đó spread là
   median khoảng cách bình phương của X tới c.
3. `a = gain * support`; nếu a dưới `transport_support_floor` thì giữ nguyên
   mean/covariance của lớp đó. Tập dưới 4 cặp hoặc không có drift cũng là no-op.
4. Cập nhật nhất quán cả hai moment bằng cùng ánh xạ, với trust riêng từng lớp:

```text
mu' = mu + a * (b + (mu-c)V R)
M   = I + a V R
Sigma' = M.T Sigma M + jitter I        # Chỉ cộng jitter khi a > 0.
```

Nếu Sigma đầu vào xác định dương, chặn operator norm dưới 1 làm M khả nghịch;
biến đổi covariance giữ tính xác định dương trong số học chính xác. Thực tế
dùng float64 cho fit/transport, đối xứng hóa và jitter; lưu covariance theo
dtype ban đầu. Khai triển hạng thấp tránh nhân covariance D x D với ma trận
đặc đầy đủ nhiều lần: chi phí covariance `O(C*D²*r)`, nhưng SVD vẫn có chi
phí riêng; đây không phải cải tiến miễn phí về thời gian/bộ nhớ. Bộ nhớ full
covariance 200 lớp, D=768, float32 đã khoảng 450 MiB, chưa tính bản sao/tạm.

Giới hạn còn lại: holdout trên **lớp mới** không kiểm chứng được drift thật
của **lớp cũ**; support là heuristic trong đặc trưng nhiều chiều, không phải
xác suất được hiệu chuẩn. Không có exemplar cũ, nên covariance transport
vẫn là xấp xỉ affine, không bổ sung covariance của residual phi tuyến.
Khi bị từ chối, memory cũ có thể vẫn stale; safety gate chỉ giảm nguy cơ
cập nhật tùy tiện, không đảm bảo giảm forgetting. Không dùng ảnh lớp cũ để
fit ánh xạ nhằm giữ protocol huấn luyện exemplar-free.

MACIL lưu ý covariance cũng drift và đưa ra loss Mahalanobis để giữ cấu trúc
phân bố. Repo chỉ lấy đây làm động cơ kiểm tra covariance; công thức
`M.T Sigma M` ở trên **không phải loss covariance của MACIL**.
[MACIL, mục 3.4.2](https://arxiv.org/html/2502.07560v2).

### 1.10. Prototype cũng có thể di chuyển để né repulsion; thiếu neo cosine

Ngay cả `q_inc_pair='current'`, `protos=old_ae(old_means)` vẫn có gradient.
Loss repulsion có thể giảm bằng cách di chuyển prototype thông qua projector,
thay vì chỉ cải thiện adapter. Đồng thời MSE so với một projector đang học
không phải một teacher cố định. Đây là một đường tối ưu có thể làm loss đẹp
hơn nhưng không tự chứng minh giữ được vùng quyết định của lớp cũ.

Đã thêm `q_detach_prototypes=True` cho **nhánh current**: chỉ detach prototype
ở quantum repulsion. Projector vẫn học qua alignment; metric frozen vẫn
truyền gradient đến current features. Không dùng tùy chọn này với old_proj
vì sẽ làm thay đổi sang một đường gradient có ý nghĩa khác.

Classifier trong `network/classifier.py` dùng cosine, còn QKSR tách lớp trong
metric đã chiếu xuống mạch. Tách tốt theo Q-distance chưa chắc là tách tốt
theo cosine của feature gốc. Thêm `prototype_relation_kl` để neo quan hệ với
prototype cũ theo đúng hình học cosine mà classifier sử dụng:

```text
p_old = softmax(cos(f_old(x), mu_old) / tau)
p_new = softmax(cos(f_current(x), detach(g(mu_old))) / tau)
L_relation = tau² * KL(detach(p_old) || p_new)
L_incremental = beta*MSE + gamma*w(epoch)*L_repulsion
                + relation_distill_weight*L_relation
```

Teacher, prototype gốc và prototype đã map đều detach trong relation loss;
chỉ current adapter nhận gradient. Mặc định weight=0 không thêm loss.
Profile retention thử weight=1, tau=0.2; đây **không phải kết quả tuning**.
Một prototype thì softmax chỉ có một phần tử, không cung cấp quan hệ nên trả 0.

Ý tưởng giữ phân phối quan hệ prototype–instance được đối chiếu PRD của
CCLIS. CCLIS dùng replay buffer; repo này chỉ dùng ảnh lớp mới và prototype
đã lưu, nên **không** tái hiện CCLIS hay kế thừa bảo đảm thực nghiệm của nó.
[CCLIS, mục PRD](https://arxiv.org/html/2403.04599v1).
Relation trên lớp mới có thể bảo toàn cả thiên lệch của teacher và hạn chế
plasticity nếu quá mạnh; không có nhãn/samples lớp cũ để kiểm chứng trực tiếp.

### 1.11. Ước lượng covariance thiếu mẫu và checkpoint làm mất tương quan

`torch.cov` cần nhiều hơn một mẫu; một ảnh dẫn tới mẫu số N-1 bằng 0 và NaN.
Nếu N nhỏ hơn D=768, sample covariance hạng thấp và nhiễu: jitter giúp lấy
mẫu được nhưng không chứng minh phân bố ước lượng đáng tin cậy.

Đã thêm `estimate_gaussian_statistics`: kiểm tra finite, một mẫu dùng
covariance 0 trước regularization và cảnh báo; nhiều mẫu vẫn dùng covariance
unbiased như legacy. Mean/covariance hiện tích lũy float64 trước khi lưu,
thay vì mean NumPy float32 như trước; không tuyên bố bitwise parity với
mọi run legacy cũ. Shrinkage tùy chọn:

```text
Sigma_reg = (1-s) Sigma + s diag(Sigma) + 1e-3 I
```

Mặc định s=0; retention thử s=0.05. Không tự bỏ mọi tương quan. Shrinkage
giảm off-diagonal nhiễu nhưng có thể bỏ cả tương quan hữu ích, cần ablation
G5/G6. FeCAM nghiên cứu sự khác nhau giữa covariance từng lớp; đây là nguồn
động cơ, không phải triển khai đầy đủ classifier Mahalanobis của FeCAM.
[FeCAM](https://arxiv.org/abs/2309.14062).

`compact_diagonal_checkpoint=True` chỉ lưu diagonal; load lại dựng diagonal
matrix, không khôi phục covariance ban đầu. Vì thế resume có thể đổi replay
so với chạy liên tục. Đã thêm trường này vào kiểm tra protocol, nhận diện cả
checkpoint diagonal cũ, và không cho dùng compaction cùng guarded transport.
Exact-resume cần giữ full covariance. Chưa thay single Gaussian bằng mixture.

### 1.12. LR thực tế, overflow loss và partial pretrained load

- Stage base SGD trước đây hardcode LR 0.01 cho adapter/head, dù config có
  `init_lr` khác. Đã lộ ra tùy chọn `base_adapter_lr`, mặc định vẫn 0.01 để
  không âm thầm đổi baseline. Tùy chọn này điều khiển nhóm SGD base; nhánh
  AdamW mạng vẫn dùng `init_lr`. Không suy ra LR base từ `init_lr` trong log.
- Nhánh AdamW baseline legacy bỏ projector khỏi optimizer. Các control bật
  signed/reset projector hoặc relation retention nay có cùng nhóm projector
  cần học, kể cả tắt quantum. Baseline legacy chưa bật các option vẫn giữ
  đường optimizer cũ. Config ImageNet-A/R hiện tại dùng SGD nên lỗi AdamW
  không giải thích trực tiếp chênh lệch của hai log này.
- `AngularPenaltySMLoss` từng tính exp rồi log; scale cao dễ overflow. Đã đổi
  sang logsumexp sau biến đổi target margin: tương đương công thức, ổn định
  số hơn. Không tuyên bố overflow đã xảy ra trong log scale=30.
- Bộ loader pretrained dùng `strict=False`, rồi cho train mọi missing key;
  một checkpoint không đúng có thể khiến backbone thiếu weight được train
  ngẫu nhiên. Nay chỉ cho thiếu adapter mới/head; missing backbone hoặc key
  bất ngờ sẽ lỗi rõ. Notebook cũng kiểm tra load safetensors local vào timm;
  checkpoint task phải khớp đầy đủ weight sau dựng lại các head. Không tải
  weight mới và không tự thay model đã upload bằng model khác.

### 1.13. Validation/đánh giá có thể che nguồn forgetting hoặc rò dữ liệu

Giữ old/new task accuracy mới biết liệu cải thiện new-class đang đánh đổi
old-class. Total cuối không cho biết adapter đã quên trước CA hay CA làm
classifier tệ đi. Đã thêm `record_stage_metrics=True`: log và lưu checkpoint
accuracy theo task, old/new/total ngay trước/sau CA cùng `CA accuracy delta`.
Các probe dùng transform đánh giá và generator riêng, không thay RNG replay.
Vì cập nhật thống kê chưa ảnh hưởng head trước CA, đây là probe tác dụng CA,
**không** tách riêng tác dụng mean khỏi covariance; cần G4/G5 để tách transport.

Khi `val_ratio>0`, source trước đây vẫn đánh giá CA/final trên test; validation
mới chỉ thuộc các lớp task hiện tại. Nay tái tạo đúng held-out đã tách ở từng
task để đánh giá **toàn bộ lớp đã thấy**, không đưa ảnh held-out vào stage1,
calibration, fit transport hoặc class statistics. Log ghi rõ validation hay
test. `val_ratio=0` vẫn đánh giá test như protocol cũ. Không chọn siêu tham số
bằng curve test rồi dùng chính curve đó làm xác nhận tăng chất lượng.

Lưu ý protocol: validation ở các task cũ được giữ để **đánh giá** mà không
huấn luyện lại bằng ảnh cũ. Đây là chế độ nghiên cứu/offline tuning trong
repo, không nên gọi là một hệ thống hoàn toàn không giữ dữ liệu quá khứ ở
tầng đánh giá. Final paper run cần đóng băng lựa chọn rồi chạy lại từ đầu
với `val_ratio=0`; split của tuning không phải split final để so trực tiếp.

### 1.14. Những hạn chế chưa giải quyết bằng bản sửa này

1. Một Gaussian/lớp chưa mô tả phân bố nhiều mode, outlier hoặc đặc trưng
   thiên lệch trên ImageNet-A. Hướng tiếp theo: nhiều centroid với ngân sách
   cố định và covariance shrinkage theo số mẫu; cần xử lý transport từng
   mode, quy định bộ nhớ và đối chứng trước khi thay protocol.
2. Quantum projector D->8, RDM bậc 1 và frozen bandwidth có thể không phù
   hợp task sau. Cần qubits/order2/reupload và matched RBF ablation; tăng
   qubits làm statevector tăng theo 2^q, không tự là tăng chất lượng.
3. Với `pqk_no_cnot`, chỉ Ry và không reupload, các phép quay chung cùng
   trục không đổi khoảng cách RDM giữa các mẫu. Gradient góc circuit có thể
   0 vì bất biến kiến trúc; không được mặc định gọi là barren plateau.
4. `data/data.py` đang dùng ToTensor mà chưa Normalize cho A/R. Đây là điểm
   phải kiểm tra với **đúng checkpoint pretrained đã upload**; timm hướng
   dẫn lấy preprocessing từ data config của model, không dùng một bộ mean/std
   tùy ý cho mọi weights. Chưa tự sửa để tránh đổi đồng thời dữ liệu và loss.
   Thử normalization như ablation chung cho RSIAT/QKSR, sau khi xác minh
   provenance của weights. [timm Quickstart](https://huggingface.co/docs/timm/quickstart).
5. CA dùng 256 Gaussian samples/lớp, cùng ngân sách ngay cả lớp hiếm/không
   chắc chắn, và nhân mean bằng hệ số tuổi task mà không nhân covariance
   tương ứng. Đó là heuristic inherited, không phải một affine transport
   nhất quán. Chưa tự bỏ heuristic; dùng stage deltas và ablation mean-decay/
   số samples/CA LR để kiểm tra. CA chỉ sửa head, không phục hồi feature
   phân biệt lớp cũ đã thực sự bị mất trong adapter.
6. Chưa đo gradient riêng CE/alignment/repulsion/relation lên adapter để
   định lượng xung đột; chưa có positive/negative kernel histogram riêng.
   Không chọn weight tự động chỉ bằng độ lớn scalar loss: gradient mới là
   đường tác động lên biểu diễn. Weight relation quá lớn cũng có thể hại.
7. Checkpoint task-boundary và process RNG chưa fingerprint toàn bộ dữ liệu,
   preprocessing/weights/runtime, cũng không lưu worker persistent đang sống.
   Exact-resume test hiện tại workers=0, một runtime CPU; không được suy
   rộng thành bitwise-reproducible giữa mọi Kaggle GPU/runtime.

## 2. Các sửa về đo lường và tái lập

- Log kernel nay gộp **toàn epoch**, trung bình theo số cặp, loại đường chéo của
  self-kernel, nhưng giữ đầy đủ cross-kernel kể cả khi ma trận vuông. Min/max,
  độ lệch chuẩn và histogram được cộng dồn với bộ nhớ hữu hạn. Batch một ảnh
  không còn làm cả epoch trông như kernel toàn 1.
- Gradient quantum được lấy trung bình qua các minibatch có gradient, không
  chỉ minibatch cuối. Metric frozen có dictionary gradient rỗng là bình thường.
  Log chưa đo gradient của từng loss lên adapter, cũng chưa tách histogram
  positive/negative; không được suy ra toàn bộ chất lượng biểu diễn từ mean K.
- Checkpoint mới lưu Python, NumPy, Torch CPU và CUDA RNG; khôi phục sau khi
  tái tạo classifier/autoencoder. Checkpoint cũ không có RNG sẽ có cảnh báo.
  Thay topology GPU cũng được cảnh báo; không có cam kết bitwise giữa runtime,
  phiên bản thư viện hoặc trạng thái worker còn sống khác nhau.
- Metadata chặn đổi weight, warmup, chế độ SSCA, validation ratio, margin,
  loại/reset projector, relation loss, covariance transport/shrinkage và
  checkpoint diagonal khi resume. Đây chưa phải fingerprint toàn bộ
  dataset/backbone/runtime hoặc toàn bộ siêu tham số optimizer.
- Khi `val_ratio>0`, class mean/covariance nay chỉ lấy tập con train thực tế.
  Trước đây chúng lấy lại toàn bộ train gốc, khiến dữ liệu held-out đi vào
  classifier alignment. `val_ratio=0` giữ tập ảnh thống kê cũ.

## 3. Cách dùng trên Kaggle

Upload lại ZIP **source mới**, thay notebook bằng bản hiện tại và khởi động
session mới để tránh cache module từ source cũ. Dataset và model vẫn đọc Input,
không cần tải lại. Ở cell cấu hình:

```python
QKSR_PROFILE = 'current_margin'
SSCA_FEATURE_MODE = 'legacy'  # Tách kiểm tra loss khỏi thay đổi SSCA trước.
RUN_PROFILE = 'smoke'
SMOKE_TASKS = 2
```

Profile thử nghiệm dùng `current`, margin 0.3, weight 0.25 và warmup 3 epoch.
Các giá trị này là điểm bắt đầu kiểm thử, **không phải siêu tham số tối ưu đã
được chọn trên ImageNet-A/R**. Smoke chạy hai task, mỗi task hai epoch, CA tắt;
chỉ kiểm tra code/gradient, không so accuracy với paper. Nếu qua smoke, chuyển
`RUN_PROFILE='full'`. Prefix riêng giúp tránh lẫn checkpoint/log với legacy.
Không resume từ checkpoint legacy khi đổi objective.

Muốn thử **toàn bộ ứng viên retention** đã thêm trong lần audit sâu này:

```python
QKSR_PROFILE = 'retention'
USE_QKSR = True
RUN_PROFILE = 'smoke'
SMOKE_TASKS = 2
```

Profile tự ép `paired_eval`, thêm signed/reset projector, detach prototype
ở repulsion, relation loss, guarded mean/covariance transport và shrinkage;
đặt `keep_last_checkpoint=False` để không tự bỏ checkpoint task trước.
Full ViT/covariance checkpoint tốn dung lượng đáng kể; cần kiểm tra quota
Kaggle khi giữ nhiều task/profile/seed và chủ động lưu kết quả trước khi đầy.
Đổi `USE_QKSR=False` giữ các sửa chung về drift/retention nhưng tắt quantum;
đây là RSIAT control đã cải tiến, không phải RSIAT nguyên bản. Trong smoke,
CA vẫn tắt: muốn chạy qua khâu lấy mẫu/head update trên Kaggle, đặt
`config['ca_epochs']=1` **trước** dòng lưu CONFIG_PATH ở cell cấu hình.
Test pipeline local bên dưới đã bật CA=1 thật.

Để chọn siêu tham số, đặt `config['val_ratio']=0.1` trước khi lưu config và
dùng prefix/session mới, không resume run final. Tất cả số của retention
(rank32, ridge0.01, max_change0.25, support_floor0.05, shrinkage0.05,
relation_weight1/tau0.2) là điểm khởi đầu có kiểm tra code, chưa được chọn
theo validation ImageNet. Các sửa shared đều có thể thử độc lập qua G configs.

Muốn kiểm tra SSCA, dùng `SSCA_FEATURE_MODE='paired_eval'` cho cả baseline và
QKSR; không so bản đã sửa SSCA với log baseline cũ rồi quy phần tăng cho QKSR.

## 4. Ablation đề xuất và kiểm chứng local

Tạo cấu hình từ JSON legacy của từng dataset, không từ một config đã sửa nhiều
yếu tố, để các đối chứng một-yếu-tố giữ đúng ý nghĩa:

```bash
python scripts/generate_qksr_ablation_configs.py --source exps/adapter_imageneta.json --output-dir exps/qksr_imageneta_ablation
python scripts/generate_qksr_ablation_configs.py --source exps/adapter_imagenetr.json --output-dir exps/qksr_imagenetr_ablation
python -B -m unittest discover -s tests -v
```

Generator bổ sung mặc định quantum nếu source chỉ là config RSIAT, tránh tạo
`B_qksr` nhưng thực tế quantum vẫn tắt. Dataset, lịch lớp và tham số gốc được giữ.

| Đối chứng | Câu hỏi kiểm tra |
| --- | --- |
| B / F1 | Đưa repulsion trực tiếp vào adapter có ích không? |
| B / F2 | Dừng repulsion bằng margin có ích không? |
| F1 / F3 | Margin có ích khi dùng nhánh current không? |
| F3 / F4 | Giảm weight riêng có ích không? |
| F4 / F5 | Warmup có ích không? |
| A / F6 và B / F7 | Sửa ghép ảnh SSCA ảnh hưởng từng phương pháp ra sao? |
| F5 / F8 | PQK so với RBF cổ điển trong cùng objective cải tiến? |
| G0 / G1 | Residual có dấu + identity-init thay Sigmoid có ích không? |
| G1 / G2 | Reset projector mỗi task có ích không? |
| G2 / G3 | Bỏ đường né repulsion qua prototype có ích không? |
| G3 / G4 | Neo quan hệ cosine có ích không? |
| G4 / G5 | Guarded transport cả mean/cov có ích hơn mean-only SSCA không? |
| G5 / G6 | Shrinkage covariance có ích không? |
| G6 / G7 | PQK có ích hơn RBF khi cùng toàn bộ sửa retention không? |
| A / G8 | Các sửa chung có ích cho RSIAT, không cần quantum không? |

G0 bắt đầu từ current/margin/weight/warmup cùng paired_eval. G0–G6 mỗi bước
thêm một cơ chế; G1 gộp signed residual với identity-init, muốn tách hai yếu
tố này cần thêm ablation riêng. G5 cũng gộp affine mean transport, covariance
transport và safety gate; **chưa** tách riêng lợi ích covariance khỏi mean.
G6/G8 không phải đối chứng thuần kernel: RSIAT vẫn dùng cosine old_proj và
loss base gốc; matched-kernel control chính là G6/G7. Không gán mọi chênh
lệch G6/G8 cho quantum. A–F giữ đối chứng cũ; G configs giữ mọi checkpoint.

Để tách các yếu tố với G configs có ý nghĩa, source phải là JSON RSIAT gốc
như lệnh trên, không phải config retention đã chỉnh tay. Giữ seed, lịch lớp,
weights, split, số epoch, batch size và preprocessing giống nhau. Sau smoke,
chạy G0–G6 trên validation, ưu tiên ImageNet-A; đóng băng ứng viên rồi báo
cáo final/average accuracy, forgetting, BWT, per-task accuracy và thời gian/
peak-memory trên tối thiểu 3 paired seeds. Bật retention không sửa được run
legacy đã hoàn thành; phải chạy lại từ đầu với prefix riêng.

Test local gồm loss legacy parity, đường gradient current/old_proj, margin,
weight/warmup, calibration đúng nhánh, thống kê epoch, loader ghép đúng ảnh,
validation không rò vào thống kê, checkpoint RNG và optimizer smoke CPU cho
cả hai nhánh. Không chạy huấn luyện ImageNet đầy đủ; máy local không có CUDA.

### Kết quả kiểm chứng bản audit sâu

Lệnh thực tế trên máy Windows hiện tại, dùng Python đã có torch/timm:

```powershell
py -3.12 -X utf8 -B -m unittest discover -s tests -q
```

Đã chạy 55 tests: **54 đạt, 1 bỏ qua do không có CUDA**. Kiểm chứng thêm:

- Projector identity và số tham số bằng legacy; học được shift âm; init không
  thay training RNG.
- Gradient relation chỉ vào current features; detach loại đường repulsion
  tới projector mà vẫn có gradient adapter.
- Singleton covariance finite/SPD; mặc định nhiều mẫu khớp covariance legacy.
- Transport covariance khớp công thức affine đặc đầy đủ; biết từ chối
  holdout xấu/lớp không được hỗ trợ, no-op khi không drift/thiếu cặp.
  Kiểm tra bổ sung D=768, 64 cặp, 2 lớp, rank32 cho covariance finite và
  Cholesky thành công trên CPU; không phải benchmark ImageNet toàn bộ.
- Trên bài toán **synthetic affine đã biết**, error mean/cov giảm trên 99%
  so với giữ nguyên; chỉ là test toán học, không phải tăng accuracy ImageNet.
- CosFace/logsumexp khớp cross-entropy margin, scale=1000 vẫn finite.
- Checkpoint không cho đổi protocol/partial weights; AdamW control có projector;
  notebook compile và tạo đúng profile trên cả A/R, QKSR/RSIAT, smoke/full.
- Pipeline offline thật qua 3 task, stage1, signed/reset projector, SSCA/transport,
  Gaussian CA=1 và stage probes. Resume ở task boundary khớp **từng tensor
  weight, mean, covariance** với chạy liên tục (rtol=atol=0) trên CPU.
  Toy data chỉ là fixture kiểm tra code, không đại diện chất lượng nghiên cứu.

Không cài thư viện mới, không tải dataset/model, không chạy ImageNet full,
không sửa/xóa log hay checkpoint người dùng. File tạm do test tự tạo nằm
trong checkout và được test thu hồi; không ghi ra ngoài thư mục làm việc.

Notebook [`QKSR_Kaggle_Ablations.ipynb`](../QKSR_Kaggle_Ablations.ipynb)
tách `A`, `B`, `G0`–`G8` thành từng cell run độc lập, hỗ trợ ImageNet-A,
ImageNet-R và CIFAR224 đã upload trong Kaggle Input. Smoke và full dùng prefix
khác nhau nên không resume hoặc ghi đè lẫn nhau.

## 5. Đối chiếu tài liệu nghiên cứu và mức độ tái hiện

Các paper được đọc qua trang web, không tải file vào máy. LDC 3.2–3.3,
MACIL 3.4.2, CCLIS phần PRD được dùng để kiểm tra động cơ drift/geometry,
không lấy số accuracy của paper làm cam kết cho bản sửa. FeCAM được đối
chiếu abstract về phân bố khác nhau từng lớp; timm dùng hướng dẫn preprocessing.
Các link nguồn đặt ngay tại mục liên quan ở trên.

Với RSIAT, đã xem [trang CVPR chính thức](https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_Representation-Steered_Incremental_Adapter-Tuning_for_Class-Incremental_Learning_with_Pre-Trained_Models_CVPR_2026_paper.html)
và phần Algorithm 1 được search index trích từ
[supplemental chính thức](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Zhao_Representation-Steered_Incremental_Adapter-Tuning_CVPR_2026_supplemental.pdf).
Index mô tả khởi tạo projector theo task; mở trực tiếp PDF/main page bị 403
trong lượt audit nên **không tuyên bố đã đọc toàn bộ PDF RSIAT**. Source trong
repo mới là bằng chứng trực tiếp cho hành vi reuse/Sigmoid đang được sửa.

Kết luận: đã khắc phục các lỗi ổn định và bổ sung cơ chế thử nghiệm có đối
chứng cho những giới hạn sát với forgetting. Chưa có bằng chứng bản sửa làm
ImageNet-A/R tốt hơn RSIAT; đặc biệt no-exemplar drift estimation, covariance
phi tuyến và phân bố nhiều mode vẫn là các hướng cần nghiên cứu tiếp.

Đánh giá accuracy cần cùng pretrained weights, manifest train/test, lịch lớp,
epoch và nhiều paired seeds. Chọn margin/weight trên validation thuộc train,
không chọn bằng test rồi dùng chính test đó làm bằng chứng tăng chất lượng.
