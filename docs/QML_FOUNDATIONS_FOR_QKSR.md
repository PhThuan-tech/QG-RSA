# QML nền tảng cho QKSR — học từ trực giác đến công thức

*Cập nhật 02/10/2026. Dành cho người biết machine learning cơ bản, mới tiếp cận quantum computing.*

## 0. Học tài liệu này như thế nào?

**Mục tiêu:** sau khi đọc, bạn giải thích được vì sao QKSR đi qua chuỗi “feature → góc → statevector → RDM → similarity”, thay vì chỉ đọc lại công thức.

Không cần học toàn bộ cơ học lượng tử để hiểu repository này. Cần nắm đại số tuyến tính của một mạch nhỏ và cách đưa nó vào loss của neural network. Mỗi phần dưới đây đi theo thứ tự: **vấn đề cần giải quyết → trực giác → công thức và nguồn gốc → ví dụ → liên hệ QKSR**.

Lộ trình đọc:

| Lượt đọc | Phần | Câu hỏi cần trả lời |
|---|---|---|
| 1 — hiểu trạng thái | §1–6 | Qubit chứa gì? Cổng thay đổi nó thế nào? |
| 2 — hiểu cách lấy đặc trưng | §7–10 | Vì sao statevector chưa phải feature cuối? RDM giữ và bỏ gì? |
| 3 — hiểu học máy | §11–14 | RDM tạo kernel/loss và gradient như thế nào? |
| 4 — chuẩn bị giải thích | §15–18 | Có thể tính tay một ví dụ và trả lời câu hỏi phản biện không? |

Khi thấy công thức, tự hỏi ba câu: **đầu vào là gì, đầu ra là gì, thao tác này phục vụ mục tiêu nào?** Các đoạn “Đọc sâu” có thể để lượt sau.

Tài liệu thứ hai, [QKSR_PROBLEM_AND_PIPELINE.md](QKSR_PROBLEM_AND_PIPELINE.md), giải thích bài toán incremental learning và toàn bộ training loop. Tài liệu này tập trung vào nền tảng quantum/kernel.

## 1. QKSR sử dụng quantum theo nghĩa nào?

Một neural network có thể biến ảnh thành vector đặc trưng $z$. Thông thường, ta đo độ giống giữa hai vector bằng cosine. QKSR thử học một phép biến đổi khác trước khi đo độ giống:

```text
feature từ ViT/Adapter
    → một vector góc ngắn
    → mạch Ry–CNOT được mô phỏng
    → thông tin cục bộ của trạng thái
    → khoảng cách giữa hai mẫu
    → độ tương đồng dùng trong loss
```

**Phần “quantum” là cấu trúc toán học của feature map.** Implementation hiện dùng tensor PyTorch trên CPU/GPU để mô phỏng mạch; không gửi ảnh hay circuit lên máy tính lượng tử.

QKSR cũng không thay classifier bằng quantum classifier. Nó tạo thêm tín hiệu huấn luyện; dự đoán nhãn vẫn dùng ViT, adapter và classifier classical. Đây là điểm nên nói ngay đầu buổi báo cáo.

**Vì sao thử cấu trúc này?** Các phép quay và tương tác qubit tạo một họ hàm đặc trưng có cấu trúc. Giả thuyết là cấu trúc ấy có thể giúp metric phù hợp với representation steering. Đây là lý do thử nghiệm, chưa phải bảo đảm tốt hơn MLP hay RBF thông thường.

## 2. Bộ ký hiệu tối thiểu — đọc được trước khi tính

### 2.1 Ket và bra chỉ là cách viết vector

Ta chọn hai vector cơ sở:

$$
|0\rangle=\begin{bmatrix}1\\0\end{bmatrix},
\qquad
|1\rangle=\begin{bmatrix}0\\1\end{bmatrix}.
$$

Ký hiệu $|\,\rangle$ gọi là **ket**. Số 0 và 1 ở đây là tên của hai trạng thái cơ sở; ket không phải một phép toán bí ẩn.

Một vector bất kỳ có thể viết:

$$
|\psi\rangle=a|0\rangle+b|1\rangle
=\begin{bmatrix}a\\b\end{bmatrix}.
$$

Ký hiệu $\langle\psi|$ gọi là **bra**, bằng chuyển vị liên hợp của ket:

$$
\langle\psi|=\begin{bmatrix}a^*&b^*\end{bmatrix}.
$$

Dấu $*$ đổi $u+iv$ thành $u-iv$. Dấu $\dagger$ gộp “chuyển vị + liên hợp”. Trong QKSR hiện tại, amplitudes là số thực, nên phép này trở thành chuyển vị thông thường.

### 2.2 Đừng nhầm hai phép nhân

| Biểu thức | Kết quả | Ý nghĩa |
|---|---|---|
| $\langle\psi\vert \phi\rangle$ | Một số | Inner product: mức chồng lấp giữa hai vector |
| $\vert \psi\rangle\langle\psi\vert $ | Một ma trận | Outer product: cách biểu diễn trạng thái dùng ở density matrix |

Ví dụ với $|\psi\rangle=[3/5,\;4/5]^T$:

$$
\langle\psi|\psi\rangle=\frac{9}{25}+\frac{16}{25}=1,
\qquad
|\psi\rangle\langle\psi|
=\frac1{25}\begin{bmatrix}9&12\\12&16\end{bmatrix}.
$$

Cùng hai vector nhưng đổi thứ tự nhân cho hai kiểu đối tượng khác nhau. Nắm điều này sẽ giúp đọc density matrix ở §7.

### 2.3 Các ký hiệu khác

| Ký hiệu | Cách đọc / thao tác |
|---|---|
| $\mathbb R^d$, $\mathbb C^d$ | Vector d chiều, phần tử thực hoặc phức |
| $\Vert v\Vert _2^2$ | Tổng bình phương độ lớn các phần tử |
| $I$ | Ma trận đơn vị; nhân với vector không thay đổi vector |
| $\operatorname{Tr}(A)$ | Trace: tổng phần tử trên đường chéo |
| $A\succeq0$ | Ma trận positive semidefinite, giải thích ở §7 và §11 |
| $\otimes$ | Tensor product: ghép các hệ con, xem §4 |
| $E[X]$ | Kỳ vọng: trung bình theo phân phối của X |
| $[u]_+$ | $\max(0,u)$, cũng chính là ReLU |

## 3. Qubit: amplitude, probability và phase

### 3.1 Amplitude là gì?

Một pure state của qubit có dạng $|\psi\rangle=a|0\rangle+b|1\rangle$, với:

$$
|a|^2+|b|^2=1.
$$

$a,b$ là **biên độ xác suất — amplitudes**. Khi đo trong cơ sở $\{|0\rangle,|1\rangle\}$, quy tắc Born cho:

$$
P(0)=|a|^2,\qquad P(1)=|b|^2.
$$

Đây là **quy tắc nền tảng của mô hình lượng tử**, không phải một công thức suy ra từ loss ML. Điều kiện chuẩn hóa ở trên làm tổng xác suất bằng 1.

Ví dụ $a=3/5,b=4/5$: đo ra 0 với xác suất 0,36 và ra 1 với xác suất 0,64. Một lần đo chỉ cho một kết quả. Bạn không đọc được đồng thời cả hai amplitudes từ một shot.

**Superposition — chồng chập** nghĩa là trạng thái là tổ hợp tuyến tính của các vector cơ sở. Nói “qubit là 0 và 1 cùng lúc” chỉ là cách nói tắt, không đủ để tính toán hoặc phân biệt với một bit ngẫu nhiên.

### 3.2 Tại sao cần amplitude thay vì chỉ lưu xác suất?

Hai trạng thái sau cho cùng xác suất 0/1 khi đo trong cơ sở trên:

$$
|+\rangle=\frac1{\sqrt2}\begin{bmatrix}1\\1\end{bmatrix},
\qquad
|-\rangle=\frac1{\sqrt2}\begin{bmatrix}1\\-1\end{bmatrix}.
$$

Nhưng chúng khác nhau ở **dấu tương đối**, một trường hợp của relative phase. Nếu áp dụng $R_y(-\pi/2)$ trước khi đo:

$$
R_y(-\pi/2)|+\rangle=|0\rangle,\qquad
R_y(-\pi/2)|-\rangle=-|1\rangle.
$$

Lúc này kết quả đo phân biệt hoàn toàn hai trạng thái. Phép nhân ma trận ở §5 cho phép kiểm tra hai đẳng thức này.

Vậy chỉ giữ xác suất trong một cơ sở sẽ làm mất thông tin ảnh hưởng đến các phép biến đổi tiếp theo. Những amplitude có thể cộng hoặc triệt tiêu nhau: đó là cơ sở của **interference — giao thoa**.

Ngược lại, nhân *toàn bộ* statevector với cùng phase $e^{i\chi}$ không thay đổi trạng thái vật lý quan sát được. Ví dụ $|1\rangle$ và $-|1\rangle$ chỉ khác global phase. Đừng nhầm với việc chỉ đổi dấu một thành phần như ở $|+\rangle$ và $|-\rangle$.

### 3.3 Liên hệ implementation

QKSR bắt đầu bằng trạng thái thực và chỉ dùng các ma trận thực $R_y$, CNOT. Vì thế nó không cần lưu số phức. Tuy vậy amplitude vẫn có thể âm; không được coi statevector là probability vector.

“Exact statevector simulation” nghĩa là tính trực tiếp amplitudes, không lấy mẫu shots. Vẫn có sai số float32/float64; từ “exact” không có nghĩa số học chính xác vô hạn.

Nguồn nền tảng để đọc tiếp về trạng thái, measurement và Dirac notation: [IBM Quantum Learning — Quantum information](https://quantum.cloud.ibm.com/learning/en/courses/basics-of-quantum-information/single-systems/quantum-information).

**Tự kiểm tra:** $[0.6,0.8]^T$ là statevector hợp lệ; $[0.6,0.4]^T$ chưa được chuẩn hóa, dù hai phần tử cộng thành 1. Vì sao?

## 4. Nhiều qubit: tensor product từ đâu ra?

### 4.1 Ghép hai hệ cần bốn khả năng cơ sở

Một qubit có hai basis states. Hai qubit có bốn tổ hợp: $00,01,10,11$. Ta dùng thứ tự này trong ví dụ và code của repo, với qubit 0 ở phía trái.

Nếu qubit A có amplitudes $[a,b]^T$ và B có $[c,d]^T$, trạng thái product là:

$$
\begin{bmatrix}a\\b\end{bmatrix}
\otimes
\begin{bmatrix}c\\d\end{bmatrix}
=
\begin{bmatrix}ac\\ad\\bc\\bd\end{bmatrix}.
$$

Đây là tensor/Kronecker product: nhân mỗi amplitude của A với mỗi amplitude của B. Bình phương độ lớn cho xác suất chung bằng tích xác suất của hai phần khi state là product.

Ví dụ:

$$
|+\rangle\otimes|0\rangle
=\frac1{\sqrt2}\begin{bmatrix}1\\0\\1\\0\end{bmatrix}
=\frac{|00\rangle+|10\rangle}{\sqrt2}.
$$

Mỗi hệ thêm vào nhân đôi số tổ hợp cơ sở. Vì vậy q qubit cần $2^q$ amplitudes cho một statevector tổng quát.

| q | Số amplitudes | Điều cần hiểu |
|---:|---:|---|
| 2 | 4 | Có thể tính tay |
| 8 | 256 | Cấu hình QKSR đang khảo sát |
| 12 | 4.096 | Giới hạn q hiện được code cho phép |
| 30 | 1.073.741.824 | Statevector simulation không còn nhẹ |

### 4.2 Nhiều chiều không có nghĩa nhiều thông tin đầu vào hơn

Nén feature 768 chiều thành 8 góc rồi tạo 256 amplitudes không khôi phục thông tin đã mất ở bước nén. Các amplitudes bị ràng buộc bởi 8 góc đầu vào và circuit đã chọn; chúng không phải 256 biến tự do độc lập.

Tương tự, từ một số $a$ tạo vector $[a,a^2,\sin a,\cos a]$ cho nhiều tọa độ hơn, nhưng không sinh thêm thông tin về dữ liệu gốc. Feature map mới vẫn có thể giúp phân tách bằng một quyết định đơn giản hơn. Đây là lý do ML để quan tâm, thay vì chỉ đếm chiều.

## 5. Cổng lượng tử: phép biến đổi bảo toàn chuẩn

### 5.1 Vì sao unitary?

Statevector được chuẩn hóa để mô tả tổng xác suất 1. Một closed-system quantum gate được mô hình bằng ma trận unitary $U$, thỏa $U^\dagger U=I$. Khi đó:

$$
\|U|\psi\rangle\|_2^2
=\langle\psi|U^\dagger U|\psi\rangle
=\langle\psi|\psi\rangle=1.
$$

Ta vừa suy ra chuẩn được giữ nguyên từ điều kiện unitary. Với ma trận thực của QKSR, unitary tương ứng với orthogonal matrix.

Một linear layer ML bất kỳ không có ràng buộc này. Circuit là một họ biến đổi có cấu trúc chặt hơn.

### 5.2 $R_y$: mã hóa một số thực bằng phép quay

QKSR dùng:

$$
R_y(a)=
\begin{bmatrix}
\cos(a/2)&-\sin(a/2)\\
\sin(a/2)&\cos(a/2)
\end{bmatrix}.
$$

Nhân với $|0\rangle=[1,0]^T$ lấy cột đầu:

$$
R_y(a)|0\rangle=
\begin{bmatrix}\cos(a/2)\\\sin(a/2)\end{bmatrix}.
$$

Chuẩn bằng 1 vì $\cos^2(a/2)+\sin^2(a/2)=1$. Input $a$ trở thành amplitudes qua hàm lượng giác. Đây là **angle encoding**.

| Góc a | Trạng thái sau quay | Xác suất đo 0/1 |
|---|---|---|
| 0 | $\vert 0\rangle$ | 1 / 0 |
| $\pi/2$ | $\vert +\rangle$ | 1/2 / 1/2 |
| $\pi$ | $\vert 1\rangle$ | 0 / 1 |

**Tại sao là a/2, không phải a?** Trên Bloch sphere, vị trí state được mô tả bởi vector $(\sin a,0,\cos a)$; góc a là góc của vector đó. Amplitudes cần half-angle để khi lập density matrix thu được $\sin a$ và $\cos a$ qua công thức góc đôi. §9 sẽ tính ra cụ thể.

**Đọc sâu — nguồn gốc toán tử:** với Pauli matrix $Y=\begin{bmatrix}0&-i\\i&0\end{bmatrix}$ và $Y^2=I$:

$$
R_y(a)=e^{-iaY/2}
=\cos(a/2)I-i\sin(a/2)Y.
$$

Tách chuỗi mũ ma trận thành các lũy thừa chẵn/lẻ sẽ cho cosine/sine. Thay ma trận Y vào thu được ma trận thực ở trên. Đây là quy ước phép quay spin/qubit; không phải lựa chọn half-angle để tune accuracy.

### 5.3 Hai loại góc trong repo

- **Góc dữ liệu** $a_j=P_\eta(z)_j$: thay đổi theo ảnh; được tính từ một projector classical.
- **Góc tham số** $\theta_{l,j}$: tham số model, dùng chung cho mọi ảnh; optimizer có thể cập nhật khi không frozen.

Cả hai đều đi vào $R_y$, nhưng một bên là input và một bên là trọng số. Nhầm chúng sẽ dẫn đến hiểu sai điều gì còn thay đổi khi “freeze quantum module”.

## 6. CNOT và entanglement: tính một ví dụ thật

### 6.1 CNOT là phép hoán vị amplitudes

Với qubit thứ nhất làm control, thứ hai làm target:

| Basis input | Basis output |
|---|---|
| $\vert 00\rangle$ | $\vert 00\rangle$ |
| $\vert 01\rangle$ | $\vert 01\rangle$ |
| $\vert 10\rangle$ | $\vert 11\rangle$ |
| $\vert 11\rangle$ | $\vert 10\rangle$ |

Trong thứ tự $00,01,10,11$:

$$
U_{\mathrm{CNOT}}=
\begin{bmatrix}
1&0&0&0\\
0&1&0&0\\
0&0&0&1\\
0&0&1&0
\end{bmatrix}.
$$

Nó đổi chỗ hai amplitudes cuối. Vì thế code có thể thực hiện CNOT bằng indexing permutation mà không nhân ma trận $2^q\times2^q$.

Mô tả “nếu control=1 thì đảo target” là bảng tác động trên basis states. Với superposition, cổng tác động **tuyến tính lên toàn bộ statevector**, không đo control rồi chạy một câu lệnh if cổ điển.

### 6.2 Entanglement xuất hiện thế nào?

Bắt đầu từ $|00\rangle$, quay qubit đầu góc $\pi/2$, rồi CNOT:

$$
|00\rangle
\longrightarrow\frac{|00\rangle+|10\rangle}{\sqrt2}
\longrightarrow\frac{|00\rangle+|11\rangle}{\sqrt2}=|\Phi^+\rangle.
$$

Trạng thái cuối là Bell state. Nó không thể viết thành một ket riêng của A nhân tensor một ket riêng của B.

Chứng minh ngắn: nếu tách được, amplitudes phải là $ac,ad,bc,bd$. Bell state đòi $ac$ và $bd$ khác 0, nên cả $a,b,c,d$ khác 0. Nhưng lại cần $ad=bc=0$: mâu thuẫn.

Đây là tiêu chuẩn entanglement cho **pure states**. Với mixed states cần định nghĩa separability tổng quát hơn; QKSR không cần giải bài toán ấy để tính kernel.

### 6.3 Có CNOT không bảo đảm có lợi

CNOT tác động lên $|00\rangle$ vẫn cho $|00\rangle$, hoàn toàn không entangled. Một dãy cổng cũng có thể tạo rồi gỡ entanglement.

Trong QKSR, CNOT cho các góc tương tác. Việc tương tác ấy có giúp phân loại hay không là câu hỏi cần kiểm tra bằng ablation. Ví dụ Bell ở trên chỉ dùng **một CNOT**; một layer q=2 của repo dùng cả $0\to1$ và $1\to0$, nên không được đồng nhất hai circuit.

## 7. Density matrix: vì sao cần một ma trận nữa?

### 7.1 Statevector chưa mô tả được mọi tình huống cục bộ

Statevector đủ cho pure state của toàn circuit. Nhưng khi chỉ quan sát một qubit trong một hệ entangled, qubit đó thường không có một pure statevector riêng. Ta cần biểu diễn được cả **mixed state**.

Density matrix giải quyết điều này và cho một công thức thống nhất để tính thống kê phép đo.

Với pure state:

$$
\rho=|\psi\rangle\langle\psi|.
$$

Với một ensemble chuẩn bị state $|\psi_j\rangle$ với xác suất $p_j$:

$$
\rho=\sum_j p_j|\psi_j\rangle\langle\psi_j|,
\qquad p_j\ge0,\quad \sum_jp_j=1.
$$

Công thức thứ hai đến từ lấy trung bình các thống kê của từng trường hợp chuẩn bị. Density matrix của subsystem entangled cũng có thể là mixed, ngay cả khi toàn hệ không có nhiễu.

### 7.2 Diagonal và off-diagonal có ý nghĩa gì?

Với $|\psi\rangle=[a,b]^T$:

$$
\rho=
\begin{bmatrix}
|a|^2&ab^*\\
ba^*&|b|^2
\end{bmatrix}.
$$

Diagonal cho xác suất trong cơ sở 0/1. Off-diagonal giữ coherence, liên quan phase tương đối; cần nó để dự đoán đo trong cơ sở khác.

So sánh:

$$
\rho_+=\frac12\begin{bmatrix}1&1\\1&1\end{bmatrix},
\qquad
\rho_{\mathrm{mix}}=\frac12|0\rangle\langle0|
+\frac12|1\rangle\langle1|
=\frac12\begin{bmatrix}1&0\\0&1\end{bmatrix}.
$$

Cả hai có diagonal $1/2,1/2$, nhưng một bên là pure superposition, bên kia là mixture. Nếu chỉ lưu diagonal, bạn đã xóa khác biệt này.

**Vì sao phép outer product đúng?** Với vector đo $|v\rangle$:

$$
\langle v|\rho|v\rangle
=\langle v|\psi\rangle\langle\psi|v\rangle
=|\langle v|\psi\rangle|^2.
$$

Nó trả đúng quy tắc xác suất Born. Density matrix không được tạo tùy ý; nó đóng gói chính các xác suất đo này.

### 7.3 Ba tính chất và nguồn gốc

1. **Hermitian:** $\rho^\dagger=\rho$. Pure-state outer product tự có tính chất này; trung bình của chúng cũng vậy.
2. **Trace 1:** $\operatorname{Tr}(\rho)=1$, vì tổng xác suất đo bằng 1.
3. **PSD:** $v^\dagger\rho v\ge0$, vì ở pure state biểu thức bằng $|\langle\psi|v\rangle|^2$; trung bình có trọng số không âm vẫn không âm.

**Purity** là $\operatorname{Tr}(\rho^2)$. Pure state có $\rho^2=\rho$, nên purity=1; với $I/2$, purity=$1/2$. Nó phân biệt mức “mixed” của trạng thái, không phải accuracy hay độ tinh khiết của dữ liệu.

## 8. Partial trace: tự suy ra từ hai qubit

### 8.1 Ta muốn giữ cái gì khi bỏ một phần hệ?

Với hai qubit A, B, viết:

$$
|\psi\rangle
=a_{00}|00\rangle+a_{01}|01\rangle
+a_{10}|10\rangle+a_{11}|11\rangle.
$$

Nếu chỉ đo A, xác suất A=0 là $|a_{00}|^2+|a_{01}|^2$: cộng trên hai khả năng của B. Đó là marginalization quen thuộc trong xác suất.

Nhưng density matrix còn có off-diagonal, nên cần tổng quát phép cộng ấy để giữ cả coherence của A:

$$
(\rho_A)_{a,a'}
=\sum_{b=0}^{1} a_{ab}a^*_{a'b}.
$$

Viết ra ma trận:

$$
\rho_A=
\begin{bmatrix}
|a_{00}|^2+|a_{01}|^2&
a_{00}a_{10}^*+a_{01}a_{11}^*\\
a_{10}a_{00}^*+a_{11}a_{01}^*&
|a_{10}|^2+|a_{11}|^2
\end{bmatrix}.
$$

Đây chính là $\rho_A=\operatorname{Tr}_B(\rho_{AB})$, gọi là **partial trace**. Nó không phải lấy một góc $2\times2$ của ma trận toàn phần và cũng không phải lấy trung bình amplitudes.

### 8.2 Ví dụ có số: Bell state

Với $a_{00}=a_{11}=1/\sqrt2$, $a_{01}=a_{10}=0$:

- ô (0,0): $1/2+0=1/2$;
- ô (1,1): $0+1/2=1/2$;
- ô (0,1): $(1/\sqrt2)\cdot0+0\cdot(1/\sqrt2)=0$.

Do đó $\rho_A=I/2$; tương tự $\rho_B=I/2$.

Toàn hệ là pure Bell state, nhưng mỗi phần riêng lẻ mixed. Đây là do bỏ phần tương quan với hệ còn lại, không chứng minh hardware noise.

### 8.3 RDM giữ và bỏ gì?

**Reduced density matrix — RDM** giữ mọi thống kê phép đo chỉ trên subsystem đang xét. Nó bỏ thông tin chỉ tồn tại trong quan hệ với các qubit không được giữ.

Ví dụ:

$$
|\Phi^\pm\rangle=\frac{|00\rangle\pm|11\rangle}{\sqrt2}
$$

là hai trạng thái toàn hệ khác nhau, trực giao, nhưng đều có $\rho_A=\rho_B=I/2$. Kernel chỉ đọc one-body RDM sẽ cho chúng similarity 1. Hai-qubit density matrix vẫn phân biệt được chúng.

Từ đó rút ra đúng mức: **one-body RDM có thể bỏ mất thông tin phân biệt**. Không thể kết luận two-body luôn tốt hơn trên dữ liệu thực, vì còn regularization, noise, compute và số mẫu.

### 8.4 Code tính RDM mà không dựng full density matrix

Sau khi sắp lại trục tensor, code xem amplitudes là $M_{r,a}$, với r chỉ phần bị bỏ, a chỉ subsystem cần giữ. Nó tính:

$$
\rho_{a,c}=\sum_r M_{r,a}M^*_{r,c}.
$$

Đây chính là công thức partial trace ở trên, được thực hiện bằng `einsum("bra,brc->bac", ...)`; b đầu tiên là batch index.

Nhờ vậy, q=8 vẫn lưu statevector 256 phần tử nhưng không cần dựng full density matrix 256×256 cho mỗi mẫu. Xem [LocalRDMExtractor](../utils/quantum_kernel.py).

## 9. Bloch vector: một cách hiểu RDM gọn hơn

Phần này giúp trả lời câu hỏi “mỗi qubit cuối cùng cho bao nhiêu thông tin?”.

Ma trận Hermitian $2\times2$ trace 1 có thể viết:

$$
\rho=\frac12(I+r_xX+r_yY+r_zZ),
$$

với:

$$
X=\begin{bmatrix}0&1\\1&0\end{bmatrix},
\quad
Y=\begin{bmatrix}0&-i\\i&0\end{bmatrix},
\quad
Z=\begin{bmatrix}1&0\\0&-1\end{bmatrix}.
$$

$I,X,Y,Z$ tạo một cơ sở cho các ma trận loại này. Vector $(r_x,r_y,r_z)$ gọi là Bloch vector; $r_j=\operatorname{Tr}(\rho\,\sigma_j)$ là expectation của phép đo Pauli tương ứng.

Với trạng thái $R_y(a)|0\rangle$, đặt $c=\cos(a/2),s=\sin(a/2)$:

$$
\rho(a)=
\begin{bmatrix}c^2&cs\\cs&s^2\end{bmatrix}
=\frac12
\begin{bmatrix}
1+\cos a&\sin a\\
\sin a&1-\cos a
\end{bmatrix}.
$$

Công thức góc đôi cho $r_x=\sin a,r_y=0,r_z=\cos a$. Đây là lời giải thích half-angle đã hẹn ở §5.

Trong circuit thực Ry–CNOT, one-qubit RDM là ma trận thực nên $r_y=0$. Mỗi RDM có tối đa hai tọa độ tự do, dù lưu bốn ô.

Two-qubit RDM thực đối xứng 4×4 có 10 ô độc lập trước ràng buộc trace, còn tối đa 9 sau trace. **Không được cộng 2q+9q rồi coi là đúng số chiều thông tin của toàn biểu diễn:** các RDM chồng lắp phải nhất quán với nhau, và toàn map còn phụ thuộc q góc input. `11q` trong MLP control là quy ước kích thước so sánh, không phải chứng minh hai model có capacity bằng nhau.

## 10. Một mẫu đi qua circuit QKSR như thế nào?

### 10.1 Projector chọn góc, circuit biến đổi góc

Feature ViT có d=768 chiều; không đưa trực tiếp 768 số vào 8 qubit. Projector classical học:

$$
a=P_\eta(z)=\pi\tanh(W\operatorname{LN}(z)+b).
$$

Đọc từ trong ra:

1. LayerNorm chuẩn hóa các tọa độ của từng feature để scale đầu vào ổn định hơn.
2. $Wz+b$ chọn q tổ hợp học được từ feature 768 chiều.
3. tanh giới hạn mỗi số vào khoảng (-1,1).
4. Nhân $\pi$ biến chúng thành góc trong (-π,π).

Đây là **lựa chọn thiết kế ML**, không phải định luật quantum. Có thể dùng encoding khác. Nó có trade-off: giảm compute nhưng có thể mất thông tin; tanh bão hòa còn làm gradient nhỏ.

### 10.2 Mạch mặc định theo đúng thứ tự code

```text
Khởi tạo |00...0>
    → Ry(a_j) trên từng qubit j: encoding một lần
    → layer 0: Ry(theta_0,j), rồi vòng CNOT
    → layer 1: Ry(theta_1,j), rồi vòng CNOT
    → statevector
```

Vòng CNOT chạy control $j=0,1,\ldots,q-1$, target $(j+1)\bmod q$. Cổng sau có thể tác động lên kết quả cổng trước; không được tự ý đổi thứ tự.

**Variational quantum circuit — VQC** chỉ có nghĩa circuit có tham số có thể tối ưu. Trong config q=8, L=2, có 16 góc $\theta$, không phải 256 trọng số tùy ý tương ứng 256 amplitudes.

Nếu `q_reupload=true`, mỗi layer encode lại cùng a trước khi áp variational gates. Reupload có thể thay đổi độ phức tạp hàm theo input; không mặc định tốt hơn. Việc encoding quyết định expressive structure được phân tích trong [Schuld, Sweke, Meyer — Effect of data encoding…](https://arxiv.org/abs/2008.08605).

**Một điểm tinh tế:** với $\theta$ cố định, toàn circuit là ánh xạ tuyến tính theo statevector vào. Nhưng từ z đến a có tanh, từ a đến amplitudes có sin/cos, từ amplitudes đến RDM có tích. Do đó feature map tổng theo **dữ liệu z** vẫn phi tuyến. Không cần nói sai rằng “CNOT là một nonlinear gate”.

## 11. Từ RDM đến kernel: tại sao dùng hai bước?

### 11.1 Khoảng cách trước: đo hai biểu diễn khác nhau bao nhiêu

Frobenius norm của ma trận là Euclidean norm sau khi trải phẳng:

$$
\|A-B\|_F^2=\sum_{i,j}|A_{ij}-B_{ij}|^2.
$$

Với q one-body RDMs:

$$
D_1(z,z')=\sum_{k=0}^{q-1}\|\rho_k(z)-\rho_k(z')\|_F^2.
$$

Công thức đến từ yêu cầu đơn giản: so từng ô, bình phương để không triệt tiêu dấu, rồi cộng. “Squared distance” không cùng đơn vị với distance chưa bình phương.

Nếu thêm two-body RDM trên q cặp lân cận:

$$
D_2(z,z')=D_1(z,z')
+\lambda_2\sum_{k=0}^{q-1}
\|\rho_{k,(k+1)\bmod q}(z)-\rho_{k,(k+1)\bmod q}(z')\|_F^2.
$$

$\lambda_2\ge0$ quyết định ảnh hưởng phần cặp. Nó không phải một số đo entanglement.

**Tại sao vẫn là Euclidean distance?** Đặt r là vector nối one-body matrices đã trải phẳng và two-body matrices đã nhân $\sqrt{\lambda_2}$. Khi bình phương norm, hệ số trở thành $\lambda_2$:

$$
D_2(z,z')=\|r(z)-r(z')\|_2^2.
$$

Thiếu căn bậc hai trong bước nối vector sẽ làm hệ số thành $\lambda_2^2$, khác implementation.

### 11.2 Vì sao chuyển distance thành similarity?

Loss đang cần “hai mẫu giống nhau bao nhiêu”, với giá trị lớn là giống. Distance có chiều ngược lại: 0 là giống. Ta chọn Gaussian RBF:

$$
K(z,z')=\exp[-\gamma_QD(z,z')],\qquad \gamma_Q>0.
$$

Đây là một **lựa chọn kernel classical** trên RDM features. Không có định luật vật lý buộc phải dùng hàm mũ này.

Tại D=0, K=1. D tăng thì K giảm trơn, luôn dương trong toán học số thực chính xác. Ví dụ $\gamma_Q=1$:

| D | K | Cách đọc |
|---:|---:|---|
| 0 | 1 | Hai biểu diễn RDM bằng nhau |
| 0,5 | 0,6065 | Tương đồng trung gian |
| 1 | 0,3679 | Khác hơn |
| 2 | 0,1353 | Tương đồng thấp hơn nữa |

**K=0,8 không có nghĩa xác suất cùng lớp là 80%.** Kernel chưa được calibrate thành xác suất nhãn. Nó cũng không phải quantum measurement probability.

### 11.3 Kernel PSD là gì và từ đâu ra?

Một similarity tùy ý chưa chắc là kernel PSD. Với cùng một tập n mẫu, Gram matrix $G_{ij}=K(z_i,z_j)$ phải thỏa $v^TGv\ge0$ cho mọi v thực.

Nếu $G_{ij}=\langle\Phi_i,\Phi_j\rangle$, thì:

$$
v^TGv=\left\|\sum_i v_i\Phi_i\right\|^2\ge0.
$$

Đây là ý nghĩa đại số của PSD: tính nhất quán với một inner-product representation. PSD không bảo đảm phân loại tốt; kernel luôn bằng 1 cũng PSD nhưng không phân biệt mẫu.

**Đọc sâu — vì sao Gaussian RBF có tính chất đó?**

$$
e^{-\gamma_Q\|r-s\|^2}
=e^{-\gamma_Q\|r\|^2}e^{-\gamma_Q\|s\|^2}
\sum_{n=0}^{\infty}\frac{(2\gamma_Q)^n}{n!}(r^Ts)^n.
$$

Khai triển squared distance rồi dùng chuỗi mũ cho ra biểu thức trên. Mỗi $(r^Ts)^n$ là inner product của tensor features bậc n; các hệ số đều không âm. Vì vậy có thể ghép chúng thành một feature map chung.

Phân biệt: **RDM feature r hữu hạn chiều**, nhưng feature space tương ứng với RBF kernel có thể vô hạn chiều. Không nên gọi hai thứ này là cùng một không gian.

Cross-kernel giữa C prototypes và B samples là ma trận C×B, có thể không vuông. Không áp điều kiện PSD trực tiếp lên ma trận chữ nhật ấy; PSD là tính chất của kernel khi tạo Gram matrix trên cùng một tập.

## 12. Bandwidth: gamma thay đổi “giống” thành “khác” thế nào?

### 12.1 Gamma là thước đo độ nhạy

Cùng D=1:

- $\gamma_Q=0,1$ cho K≈0,905;
- $\gamma_Q=1$ cho K≈0,368;
- $\gamma_Q=10$ cho K≈0,0000454.

Vậy kernel thấp chưa chắc feature tốt hơn: có thể gamma đơn giản đã tăng.

Ký hiệu $\gamma_Q$ ở đây dành riêng cho bandwidth. Trong config RSIAT còn có khóa `gamma` là **trọng số loss incremental**, hai đại lượng khác nhau.

### 12.2 Median heuristic xuất phát từ yêu cầu scale

Ta muốn một khoảng cách “điển hình” không làm K gần như luôn 0 hoặc 1. Lấy median các khoảng cách calibration là $m_D>0$, đặt:

$$
\gamma_0=\frac1{m_D}.
$$

Khi D=$m_D$, K=$e^{-1}$≈0,368. Ta vừa chọn được một mốc similarity cụ thể cho typical distance. Đây là heuristic về scale, không phải ước lượng tối ưu từ xác suất hay chứng minh generalization.

Code loại diagonal khi tính self-distance vì $D(z_i,z_i)=0$ không phản ánh quan hệ hai mẫu khác nhau. Cross-distance prototype–sample không có “self diagonal” để tự động loại bỏ.

Nếu median không hữu hạn hoặc ≤$10^{-8}$, implementation cảnh báo và dùng $\gamma_0=1$. Không nên mô tả code này thành “nghịch đảo epsilon”: hai fallback khác nhau.

### 12.3 Vì sao dùng bounded learned gamma?

Đặt u là tham số thực không bị chặn:

$$
\gamma_Q=\gamma_0\,10^{\tanh u}.
$$

Vì tanh u nằm trong (-1,1), gamma dương và nằm giữa $\gamma_0/10$ với $10\gamma_0$ (cận giới hạn). Tại u=0, gamma=$\gamma_0$.

Mục đích là cho model điều chỉnh scale nhưng hạn chế độ lớn thay đổi. **Không bảo đảm kernel không collapse:** representation vẫn có thể co lại hoặc khoảng cách tập trung. Để kết luận phải đo phân phối K và gradients trên probe set cố định.

Chi tiết số học: code dùng $K_\epsilon=(1-\epsilon)K+\epsilon$, với epsilon là `torch.finfo(dtype).tiny`. Đây là mixture với constant kernel để chống underflow về 0; giữ PSD trong mô hình toán học. Nó không phải một cải tiến semantic hay nguồn tăng accuracy.

## 13. Loss học được gì qua kernel?

### 13.1 Từ mục tiêu bằng lời đến công thức

Nếu hai mẫu cùng lớp, muốn K lớn: chọn $L_+=1-K$. Nếu khác lớp, chỉ muốn chúng không quá giống: chọn $L_-=[K-m]_+$, với m là margin.

Ví dụ m=0,5:

- positive pair K=0,8 đóng góp 0,2;
- negative pair K=0,8 đóng góp 0,3;
- negative pair K=0,2 đóng góp 0.

Margin tạo vùng “đã đủ khác thì ngừng phạt”. Kernel chưa biết nhãn; nhãn xuất hiện khi quyết định pair thuộc positive hay negative.

### 13.2 Gradient giải thích “kéo gần/đẩy xa”

Đặt $\Delta=r_i-r_j$, $K=e^{-\gamma_Q\|\Delta\|^2}$. Quy tắc đạo hàm hàm hợp cho:

$$
\frac{\partial K}{\partial r_i}=-2\gamma_QK(r_i-r_j).
$$

Với $L_+=1-K$, gradient đổi dấu; gradient descent kéo $r_i$ về $r_j$. Với negative loss đang active, gradient của loss bằng gradient K; gradient descent đẩy hai điểm xa nhau.

Đây là giải thích **cục bộ trong không gian r**. Update thật tác động qua mạng dùng chung cho nhiều mẫu, nên không bảo đảm mọi pair đều di chuyển đúng mong muốn sau một step.

Cũng không nên kết luận “K cao thì gradient mạnh”: nếu $r_i=r_j$, hệ số $(r_i-r_j)$ bằng 0. Nếu K quá nhỏ, gradient cũng có thể rất nhỏ. Scalar loss lớn không đồng nghĩa gradient hữu ích lớn.

### 13.3 Backpropagation trong simulator

Mọi bước hiện tại là phép toán khả vi:

```text
loss → kernel → khoảng cách RDM → RDM → amplitudes
     → góc circuit và góc input → projector → feature đầu vào
```

PyTorch dùng chain rule để tính derivatives. Không cần parameter shift trong implementation hiện tại.

**Frozen không đồng nghĩa detach.** Với một hàm có tham số cố định $h(x)=2x$, trọng số 2 không học nhưng đạo hàm theo x vẫn bằng 2. Tương tự, QKSR frozen vẫn truyền gradient về input, miễn code không đặt toàn đường đó trong `no_grad` hoặc detach input.

Với `q_inc_pair=old_proj`, input kernel là output của residual projector RSIAT. Do vậy steering cập nhật projector đó trực tiếp; ảnh hưởng lên adapter diễn ra thông qua alignment. Đây là nội dung quan trọng trong tài liệu pipeline.

### 13.4 Nếu chạy hardware thì khác gì?

Không đọc trực tiếp statevector/RDM chính xác. Cần chuẩn bị cùng state nhiều lần, đo observables rồi ước lượng statistics. “Shot” là một lần chạy/đo.

Với các rotation thích hợp, expectation $F(\theta)$ có parameter-shift rule:

$$
\frac{dF}{d\theta}
=\frac{F(\theta+\pi/2)-F(\theta-\pi/2)}2.
$$

Công thức này áp dụng khi giả định generator/gate phù hợp và xét sự xuất hiện của parameter tương ứng. Với loss phi tuyến của nhiều expectations, phải kết hợp chain rule; không tùy tiện shift toàn scalar loss rồi khẳng định luôn đúng.

Finite shots tạo sampling error; thiết bị còn có gate/readout noise. Kết quả simulator lý tưởng chưa kiểm chứng các điều kiện ấy.

## 14. Phân biệt fidelity, PQK và quantum advantage

Fidelity kernel giữa pure states thường viết:

$$
K_{\mathrm{fid}}(z,z')=|\langle\psi(z)|\psi(z')\rangle|^2.
$$

QKSR dùng **RBF trên local RDM features**, không dùng công thức này. Hai loại có thể đánh giá một cặp state rất khác nhau: cặp Bell $\Phi^+,\Phi^-$ có fidelity 0 nhưng one-body PQK bằng 1.

Một điểm sâu hơn: nếu cùng một unitary không phụ thuộc dữ liệu U được đặt sau encoding, thì:

$$
\langle U\psi(z)|U\psi(z')\rangle
=\langle\psi(z)|U^\dagger U|\psi(z')\rangle
=\langle\psi(z)|\psi(z')\rangle.
$$

Vậy học riêng unitary chung đó không thay đổi full fidelity. Nhưng **local** RDM sau một entangling unitary có thể đổi theo cách khác nhau giữa input, nên phép triệt tiêu trên không áp nguyên cho PQK. Đây là lý do cần hiểu readout, không chỉ tên circuit.

Projected quantum kernels được nghiên cứu như một cách xây inductive bias qua local features; bằng chứng trong các setting của literature không tự chuyển thành accuracy gain trên RSIAT. Xem [Huang và cộng sự, Power of data in quantum machine learning](https://www.nature.com/articles/s41467-021-22539-9).

**Ba mức kết luận khác nhau:**

| Kết quả | Có thể nói | Chưa thể nói |
|---|---|---|
| QKSR hơn cosine trong simulator | Phương pháp hybrid này tốt hơn control đã thử | Gain chắc chắn do quantum resource |
| Hơn MLP/RBF/Fourier với budget tương đương | Circuit-induced features có bằng chứng hữu ích trong setting đó | Mọi classical model đều thua |
| Chạy được hardware | Có bằng chứng feasibility ở thiết bị đó | Có speedup hoặc quantum advantage |

Kernel concentration cũng có thể xảy ra với projected kernels dưới các điều kiện nhất định; tăng qubit/depth không tự giải quyết. [Thanasilp và cộng sự](https://arxiv.org/abs/2208.11060) phân tích vấn đề này. Hướng benchmark của [Bowles, Ahmed, Schuld](https://arxiv.org/abs/2403.07059) nhấn mạnh tầm quan trọng của classical controls.

## 15. Ví dụ tính tay nối toàn bộ đường đi

### 15.1 Một qubit: từ góc tới loss

Ví dụ sư phạm này dùng q=1 để tính tay; module hiện yêu cầu q≥2. Giả sử projector cho hai mẫu các góc a=0 và b=$\pi/2$, không có variational rotation:

$$
|\psi_a\rangle=|0\rangle,\qquad|\psi_b\rangle=|+\rangle.
$$

RDM của hệ một qubit chính là density matrix toàn phần:

$$
\rho_a=\begin{bmatrix}1&0\\0&0\end{bmatrix},
\quad
\rho_b=\frac12\begin{bmatrix}1&1\\1&1\end{bmatrix}.
$$

Lấy hiệu:

$$
\rho_a-\rho_b=
\begin{bmatrix}1/2&-1/2\\-1/2&-1/2\end{bmatrix},
\quad D=4(1/2)^2=1.
$$

Với $\gamma_Q=1$, K=$e^{-1}$≈0,3679. Nếu cùng lớp, loss≈0,6321. Nếu khác lớp và margin=0,5, loss=0: chúng đã khác đủ theo mục tiêu này.

**Câu chuyện đầy đủ:** ảnh tạo góc khác nhau → state khác nhau → RDM khác nhau → distance=1 → similarity≈0,368 → nhãn quyết định có cần kéo gần hay không.

### 15.2 Hai qubit: dùng đúng thứ tự vòng CNOT của repo

Chọn q=2, L=1, $\theta=0$, order 1. Mẫu A có góc (0,0); mẫu B có góc $(\pi/2,0)$.

Mẫu A luôn ở $|00\rangle$. Mẫu B đi qua:

$$
\frac{|00\rangle+|10\rangle}{\sqrt2}
\xrightarrow{\mathrm{CNOT}_{0\to1}}
\frac{|00\rangle+|11\rangle}{\sqrt2}
\xrightarrow{\mathrm{CNOT}_{1\to0}}
\frac{|00\rangle+|01\rangle}{\sqrt2}=|0\rangle|+\rangle.
$$

Qubit 0 của cả hai mẫu đều là $|0\rangle$: đóng góp distance 0. Qubit 1 là $|0\rangle$ so với $|+\rangle$: đóng góp 1. Tổng D=1, K=$e^{-1}$.

Ví dụ còn chỉ ra rằng circuit có thể tạo Bell state ở giữa rồi kết thúc bằng product state. Không lấy số CNOT làm số đo “entanglement hữu ích”.

## 16. Tự kiểm tra và đáp án

1. **Statevector $[3/5,-4/5]^T$ có hợp lệ không?** Có: tổng bình phương bằng 1. Dấu âm là hợp lệ, probability vẫn không âm.
2. **$\rho_+$ và $I/2$ cùng diagonal, có phải cùng state không?** Không; off-diagonal khác, đo ở cơ sở khác có thể phân biệt.
3. **Vì sao RDM Bell là mixed dù simulator không có noise?** Vì bỏ thông tin của subsystem còn lại bằng partial trace.
4. **q=8, one-body RDM được lưu bao nhiêu số?** 8×2×2=32 số/mẫu, tối đa 16 tọa độ local độc lập; còn có ràng buộc từ toàn map. Không phải 256 independent measured features.
5. **K=1 có chứng minh hai ảnh cùng lớp không?** Không; chỉ nói RDM features bằng nhau trong kernel này, và feature map có thể mất thông tin.
6. **Gamma tăng gấp đôi, K tại D cố định thay đổi thế nào?** Thành $K^2$, vì $e^{-2\gamma_QD}=(e^{-\gamma_QD})^2$.
7. **Freeze circuit có làm output không đổi giữa các ảnh?** Không; angles input vẫn phụ thuộc ảnh. Freeze là cố định trọng số.
8. **Freeze toàn metric có ngăn gradient vào projector RSIAT không?** Không, nếu graph theo input còn nguyên.
9. **Vì sao cần đối chứng projected RBF?** Để biết projection+RBF đã đủ tạo gain hay phải cần circuit.
10. **QKSR hiện có chạy trên quantum hardware không?** Không; đây là differentiable classical simulation của quantum feature map.

## 17. Mẫu giải thích bằng lời trong khoảng một phút

> Em dùng ViT và adapter để tạo feature cho ảnh. QKSR học một phép chiếu feature thành một số góc, rồi đưa các góc này qua mạch Ry–CNOT được mô phỏng bằng PyTorch. Mạch tạo statevector, nhưng em chỉ lấy thông tin cục bộ qua reduced density matrices. Khoảng cách giữa các RDM được chuyển thành similarity bằng Gaussian RBF. Similarity này đi vào loss để khuyến khích cấu trúc representation phù hợp với phân loại liên tục. Classifier lúc suy luận vẫn là classical. Giả thuyết cần kiểm chứng là feature map do circuit tạo ra có ích hơn cosine và các nonlinear classical controls; hiện chưa có kết luận quantum advantage.

Sau đoạn này, giảng viên hỏi khâu nào thì mở phần tương ứng: góc ở §5/10, RDM ở §7–9, kernel ở §11–12, gradient ở §13, evidence ở §14.

## 18. Từ điển và bước đọc tiếp

| Thuật ngữ | Cách hiểu để dùng trong QKSR |
|---|---|
| Pure state | Có thể mô tả bằng một normalized statevector |
| Mixed state | Cần density matrix tổng quát; có thể do bỏ subsystem |
| Statevector | Danh sách amplitudes theo basis đã chọn |
| Entanglement | Với pure state, không tách thành product của các subsystem |
| Coherence | Thông tin phase biểu hiện qua off-diagonal trong basis đang xét |
| Observable | Toán tử Hermitian mô tả đại lượng đo; expectation là Tr(ρO) |
| RDM | Density matrix chỉ của subsystem được giữ |
| VQC/PQC | Circuit có tham số điều chỉnh được |
| Angle encoding | Dùng dữ liệu làm góc quay |
| Data reuploading | Đưa lại dữ liệu vào nhiều layer circuit |
| Kernel bandwidth | Tham số quy định similarity giảm nhanh/chậm theo distance |
| Inductive bias | Cấu trúc khiến model ưu tiên một số họ hàm/lời giải |
| Ablation | Thay/bỏ một thành phần có kiểm soát để kiểm tra vai trò |
| Shot | Một lần thực thi và đo circuit |

Đọc tiếp theo thứ tự:

1. [Bài toán và pipeline QKSR](QKSR_PROBLEM_AND_PIPELINE.md): vì sao cần giữ lớp cũ, loss nào dùng kernel, gradient đi đâu.
2. [Implementation notes](QKSR_IMPLEMENTATION_NOTES.md) và [source quantum module](../utils/quantum_kernel.py): kiểm tra thiết kế đã được hiện thực ra sao.
3. [Research map](RSIAT_QML_RESEARCH_MAP.md): các hướng tương lai và prior art; không đồng nhất đề xuất trong đó với QKSR đã triển khai.
4. [Spec v3.1](QKSR_Spec_for_RSIAT_v3.1.md): quy ước kỹ thuật/thí nghiệm; đọc sau khi đã hiểu ý nghĩa.

Để học thêm density matrix trong một giáo trình quantum có hệ thống, xem [IBM Quantum Learning — General formulation](https://quantum.cloud.ibm.com/learning/en/courses/general-formulation-of-quantum-information/density-matrices/introduction). Các ví dụ tính tay ở tài liệu này được dựng để nối trực tiếp với code QKSR.
