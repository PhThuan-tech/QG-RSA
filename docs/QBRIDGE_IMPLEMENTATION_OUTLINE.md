# Q-BRIDGE v1.1 — outline triển khai

Triển khai theo `QBRIDGE_Implementation_Spec_v1.1.md`. Đây là trạng thái phần mềm, không phải báo cáo cải thiện độ chính xác. Xem `QBRIDGE_RUNBOOK.md` để chuẩn bị checkpoint/artifact và chạy ablation.

## 1. Contribution đã đưa vào pipeline

- Base dùng lại checkpoint QKSR task 0 theo từng dataset/seed, gồm encoder, adapter, classifier, PQK, gamma và thống kê lớp. Không huấn luyện lại base trong runner Q-BRIDGE.
- Khi `qbridge.enabled=true`, incremental thay separation cũ bằng cross-task margin và old-anchor guard, giữ classification và alignment. R1 có nhánh riêng thay classification bằng all-seen CosFace.
- Cross loss dùng raw cosine và mẫu số cố định `B*C_old`; risk chỉ điều chỉnh trọng số, đã detach. Không chuẩn hóa theo số pair đang vi phạm.
- Guard dùng `T(mu_old).detach()`, tính lại mỗi bước. Gradient của guard chỉ vào classifier mới; gradient cross vẫn đi qua feature hiện tại dù trọng số classifier cũ đã detach.
- PQK/calibrator/classical selector được freeze. Cache biểu diễn prototype theo task; không cache anchor qua nhiều bước khi T đang học. Bỏ tính selector khi không cần.
- Mapping class ID gốc → vị trí head là tường minh, không giả định ID gốc liên tiếp.
- SSCA, Gaussian CA, covariance và inference giữ thuật toán hiện tại. Guard chỉ tác động trong adaptation; báo cáo riêng trước/sau CA.

## 2. Outline theo file

| File | Thay đổi |
|---|---|
| `utils/qbridge.py` | Validate cấu hình, cross/guard, mapping ID, frozen runtime, R1–R10 selectors, kiểm tra artifact/hash, diagnostics |
| `utils/qbridge_calibration.py` | Cache contract, split P/F/V, fit calibrator đơn điệu, gate, Brier/ECE/NLL/rank metrics, fit R7/R8 |
| `utils/qbridge_experiment.py` | Preflight checkpoint, seed/path templates, bảo toàn RNG khi đánh giá thêm, pre/post-CA và manifest |
| `models/RSIAT_adapter.py` | Gắn nhánh loss, lifecycle prototype/snapshot, optimizer T, ghi epoch/provenance; nhánh disabled giữ loss cũ |
| `network/classifier.py` | Thêm `qbridge_scores` với old rows detached; không đổi forward inference |
| `models/base.py` | Strict full-state restore khi branch từ base và lưu provenance ở checkpoint cuối |
| `utils/inc_net.py` | Khởi tạo ViT từ kiến trúc local trước strict restore, không tải timm weights; tuning config dùng `SimpleNamespace` |
| `trainer.py` | Branch theo base/seed, output độc lập, báo lỗi, lưu checkpoint chỉ ở task cuối, metric JSON |
| `main.py` | `--overlay` merge đệ quy, có thể dùng nhiều overlay |
| `data/data.py`, `data/data_manager.py` | Root dữ liệu local và chế độ offline cho runner; giữ split/transform cũ |
| `scripts/qbridge_prepare.py` | CLI tạo feature cache, calibrator, selector RBF/MLP và fit manifest |
| `scripts/generate_qbridge_configs.py` | Sinh R0–R10 từ manifest/spec người dùng, gắn đường dẫn/seed mà không tự chọn hyperparameter dataset |
| `scripts/summarize_qbridge.py` | Tổng hợp run thực đã hoàn tất, paired deltas; từ chối khác base hash hoặc training/data budget |
| `tests/test_qbridge.py` | Kiểm tra loss, gradients, protocol, offline restore và vòng incremental tổng hợp qua SSCA/CA |
| `exps/qbridge/sources/` | Cấu hình dataset dùng chung để train paired base và sinh nhánh |
| `exps/qbridge/stages/` | Overlay cho từng giai đoạn, gồm `train_base.json` |
| `exps/qbridge/overlays/` | Overlay R0–R10 và manifest sinh từ spec |

## 3. Config nào là full contribution?

**R6 = full calibrated Q-BRIDGE v1.1**: new-class CosFace + alignment + cross margin có calibrated PQK risk + old-anchor guard. File `exps/qbridge/overlays/R6.json` là overlay; phải ghép source config và cung cấp calibrator đã qua gate cho đúng base/seed.

**R5 = full raw Q-BRIDGE**, nên dùng làm pilot trước R6. Khác R6 ở chỗ dùng PQK risk trực tiếp.

| Variant | Vai trò |
|---|---|
| R0 | Legacy QKSR incremental `old_proj` separation |
| R1 | All-seen CosFace + alignment, không cross/guard/separation |
| R2 | Uniform cross, không guard |
| R3 | Uniform cross + guard |
| R4 | Cosine risk + cross + guard |
| R5 | Raw PQK risk + cross + guard |
| R6 | Calibrated PQK risk + cross + guard |
| R7 | RBF trên đúng projector base, bandwidth fit base-only |
| R8 | MLP selector fit trên feature base cố định, rồi freeze |
| R9 | Hoán vị risk PQK trong các pair active của batch bằng RNG riêng |
| R10 | Thay risk active bằng trung bình active |

`enabled=false` khôi phục loss cũ. `enabled=true` và hai lambda bằng 0 chỉ còn classification + alignment, **không** tương đương legacy separation.

## 4. Source audit và quyết định tích hợp

1. Head thực tế là `SimpleContinualLinear` gồm nhiều Linear, đầu ra cosine chuẩn hóa. API mới ghép weight để detach theo class ID; không thay cấu trúc head hay nhân thêm `scale` trong margin.
2. `DataManager` đã remap nhãn sang vị trí class order. Nhánh mới đổi lại ID gốc trước khi ánh xạ sang hàng classifier; cache calibration dùng nhãn vị trí được kiểm tra khớp base head.
3. Optimizer Adam cũ chỉ đưa `old_ae` vào nhóm tham số khi bật quantum incremental. Nhánh Q-BRIDGE sử dụng nhóm như QKSR/R0 để T nhận alignment gradient dù `use_quantum_kernel_inc=false`. SGD và ngân sách CA không thay đổi.
4. Detach old rows không có nghĩa bảo đảm old weights bất biến sau optimizer: decay/momentum theo baseline vẫn có thể tác động. Có test riêng cho hành vi này.
5. Loader checkpoint cũ kiểm tra cả incremental flags. Branch task 0 cho phép khác riêng các flag incremental, vẫn kiểm tra base metadata/class order và strict model state. Full covariance 768D là bắt buộc.
6. Factory pretrained cũ có thể tải timm weights trước khi load checkpoint. Restore-only branch dựng cùng kiến trúc local; không thay đường train base mặc định.
7. SSCA hiện trích feature trước/sau bằng hai lượt loader; việc ghép mẫu và chất lượng mean/covariance vẫn là đặc tính baseline. Không sửa trong contribution này. Snapshot ID ghi nhận phiên bản ước lượng, không chứng minh mean là phân phối thật.
8. Calibrator dùng base TRAIN theo P/F/V=60/20/20, không dùng eval/test hoặc task mới. Encoder đã thấy toàn bộ base train trước đó: đây là **calibrator-held-out**, không phải encoder-held-out.
9. R8 là conditional selector control trên feature base có sẵn, không phải classical base được học từ đầu với ngân sách bằng PQK. Artifact ghi rõ objective, optimizer, epochs, parameters, thời gian và phạm vi kết luận.
10. Nhánh runner reseed khi bắt đầu từ cùng base. Không hỗ trợ exact incremental resume; chỉ lưu full checkpoint ở task dataset cuối. Pilot dừng sớm cần chạy lại từ base nếu muốn full run.

## 5. Báo cáo và trạng thái kiểm chứng

- Các fixture CPU kiểm tra baseline loss/gradient, base-task flag parity, fixed denominator, empty/active pairs, global IDs, gradient ownership, optimizer decay, R1, R6 fail-fast, gamma/cache task 1/5/9, inference isolation, artifact fitting và offline model factory.
- Một test chạy `Learner.incremental_train` thực qua adaptation → SSCA → thống kê → Gaussian CA → pre/post report bằng encoder/dữ liệu 4D tổng hợp. Đây không phải ViT dataset benchmark.
- Chạy `python -B -m unittest discover -s tests -q` để kiểm tra source hiện tại; test CUDA có thể skip nếu môi trường không có GPU.
- Run thực sẽ ghi manifest, task JSON, epoch JSONL, stage metrics, old→new/new→old errors, guard trước/sau CA, norm gradient, risk quantiles, conflict proxy và cosine agreement. Diagnostics risk hiện lấy batch đầu mỗi epoch; không phải ước lượng future risk trên toàn dataset.
- Thời gian epoch/ms mỗi batch và GPU peak cho phép so chi phí giữa paired runs; chúng gồm instrumentation, không phải phép đo kernel thuần. RDM encoding count ghi theo task.
- Chưa chạy CIFAR/ImageNet-A, chưa fit artifact thật, chưa có AIA/delta/VRAM GPU thực. Full ViT strict restore trên checkpoint người dùng và Kaggle execution cần được xác nhận khi có assets.
- Không kết luận tăng accuracy, quantum advantage hay novelty từ unit tests. R6–R5 kiểm tra weight shaping; sigmoid đơn điệu không cải thiện thứ hạng pair.

Các chỉnh sửa/xóa notebook và tài liệu pipeline đã có sẵn trong workspace không thuộc patch này. Spec và hai JSON thiết kế người dùng được giữ nguyên.
