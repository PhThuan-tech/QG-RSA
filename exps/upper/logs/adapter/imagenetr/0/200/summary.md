# SSCA / oracle diagnostics

Oracle prefixes read old-class training images: upper bounds only.

| Run | Seed | Tasks | AIA | Final | Final old | Final new |
|---|---:|---:|---:|---:|---:|---:|
| upper_joint | 1993 | 1 | 86.520 | 86.52 | 0.00 | 86.52 |

## Kịch bản theo quy tắc đặt trước

- Sàn nhiễu: **chưa đo** (thiếu `diag_legacy_rep*`); dùng ngưỡng tối thiểu 0.3 pp, kết luận yếu hơn.
- **Chưa đủ run oracle** để xếp kịch bản.

## Phân rã lỗi task cuối (% ảnh test)

Inter-task = to-older + to-newer. Ứng viên CA nhắm vào inter-task.

| Run | Seed | Error | Within-task | To older task | To newer task |
|---|---:|---:|---:|---:|---:|
| upper_joint | 1993 | 13.48 | 13.48 | 0.00 | 0.00 |

## Before vs after CA (test accuracy)

| Run | Seed | Task | Pre-CA total | Pre-CA old | Pre-CA new | Pre-CA old→new err | Post-CA total | Post-CA old | Post-CA new |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Old-mean estimation error per task

Relative error = ||estimate - oracle|| / ||stale - oracle|| (median over old classes); 1.0 means no better than skipping drift compensation.

| Run | Seed | Task | Stale | Legacy | Paired | Applied | Legacy (task-0 classes) | Paired (task-0 classes) | Cov rel. error | Drift / norm | Kernel ESS | Floor rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
