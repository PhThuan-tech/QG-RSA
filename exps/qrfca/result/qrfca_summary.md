# QRF-CA: so sánh head CA trong cùng run

## imageneta | qrfca | seed 1993 | 3 replicate

| Head | AIA (mean) | Final (mean) | Task 0 | Task 1 |
|---|---:|---:|---:|---:|
| linear | 75.861 | 67.623 | 66.667 | 65.080 |
| qda | 68.534 | 60.133 | 60.520 | 57.317 |
| rff_ridge | 74.447 | 67.133 | 65.363 | 63.493 |
| rp_ridge | 74.623 | 66.337 | 64.060 | 62.783 |
| pqk_ridge | 58.532 | 48.200 | 52.890 | 41.623 |
| pqk_noent_ridge | 65.457 | 54.777 | 56.053 | 51.147 |

### Head phi tuyến so với head tuyến tính RSIAT (ghép trong cùng run)

| Head | n | Δ AIA | CI 95% | Δ final | Δ task 0 | Kết luận |
|---|---:|---:|---|---:|---:|---|
| qda | 3 | -7.327 | [-8.606, -6.048] | -7.490 | -6.147 | KÉM HƠN head tuyến tính |
| rff_ridge | 3 | -1.414 | [-2.522, -0.305] | -0.490 | -1.303 | KÉM HƠN head tuyến tính |
| rp_ridge | 3 | -1.238 | [-2.290, -0.186] | -1.287 | -2.607 | KÉM HƠN head tuyến tính |
| pqk_ridge | 3 | -17.329 | [-19.073, -15.585] | -19.423 | -13.777 | KÉM HƠN head tuyến tính |
| pqk_noent_ridge | 3 | -10.403 | [-11.098, -9.708] | -12.847 | -10.613 | KÉM HƠN head tuyến tính |

### pqk_ridge so với các đối chứng (Δ AIA)

| Đối chứng | n | Δ AIA | CI 95% |
|---|---:|---:|---|
| rff_ridge | 3 | -15.915 | [-16.613, -15.217] |
| rp_ridge | 3 | -16.091 | [-16.803, -15.379] |
| qda | 3 | -10.002 | [-10.640, -9.364] |
| pqk_noent_ridge | 3 | -6.925 | [-8.351, -5.500] |

**Kết luận quantum:** chưa chứng minh được đóng góp quantum

**Kiểm tra hồi quy:** rep 0: head linear = CNN curve; trùng từng chữ số với baseline: không có; rep 1: head linear = CNN curve; trùng từng chữ số với baseline: không có; rep 2: head linear = CNN curve; trùng từng chữ số với baseline: không có

