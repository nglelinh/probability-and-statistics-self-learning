---
layout: post
title: "Regularization: Ridge Regression và Lasso"
date: 2021-01-01 00:00:04 +0700
categories:
- chapter07
tags: [linear-regression, regularization, ridge, lasso, bias-variance]
chapter: "07"
lang: vi
order: 6
lesson_type: required
---

Trong nghiên cứu y sinh hiện đại, một bộ dữ liệu có thể chỉ gồm $$N = 80$$ bệnh nhân nhưng đo tới $$p = 20{,}000$$ mức độ biểu hiện gene. Nếu bạn đưa thẳng dữ liệu này vào hồi quy OLS, ma trận $$\mathbf{X}^T\mathbf{X}$$ có kích thước $$20{,}000 \times 20{,}000$$ nhưng hạng tối đa chỉ là 80 — nó bị suy biến hoàn toàn, tạo ra vô số bộ hệ số khớp 100% tập huấn luyện nhưng dự báo thảm họa trên bệnh nhân mới!

Bài học mở rộng này đào sâu vào cơ chế toán học đứng sau **Điều chuẩn (Regularization)**: vì sao cộng thêm $$\lambda I$$ trong Ridge lập tức chữa khỏi bệnh suy biến ma trận, hình học của chuẩn $$L_1$$ so với $$L_2$$, vai trò cầu nối của **Elastic Net**, và cách đánh giá trực quan trên dữ liệu thưa (sparse) bằng Python.

---

## Kiến thức cần nhớ: Cơ chế ổn định hóa và chọn lọc trong không gian nhiều chiều

Để làm chủ điều chuẩn khi số biến lớn ($$p \gg n$$) hoặc đa cộng tuyến nặng, hãy nắm ba trụ cột giải tích – đại số sau:

1. **Dịch chuyển phổ giá trị riêng (Spectral Shift) trong Ridge:** Phân rã giá trị kỳ dị cho thấy OLS chia cho các giá trị riêng $$d_j^2$$ của $$\mathbf{X}^T\mathbf{X}$$ — khi $$d_j \approx 0$$, nhiễu bị khuếch đại ra vô cực. Ridge thay mẫu số thành $$d_j^2 + \lambda$$, chặn đứng mọi sự bùng nổ chỉ bằng một hằng số dương $$\lambda$$.
2. **Đạo hàm dưới (Subgradient) tại gốc tọa độ trong Lasso:** Hàm $$\lvert\beta_j\rvert$$ có góc gãy (không khả vi) tại $$\beta_j = 0$$. Chính "điểm gãy" này tạo ra một vùng chết (dead zone) $$[-\lambda, +\lambda]$$: bất kỳ đặc trưng nào có tương quan với phần dư yếu hơn $$\lambda$$ đều bị giữ chặt ở đúng 0.
3. **Hiệu ứng nhóm (Grouping Effect) của Elastic Net:** Khi hai đặc trưng mạnh tương quan sát nhau ($$r \approx 0{,}99$$), L1 thuần có xu hướng giữ ngẫu nhiên 1 biến và xóa biến kia, còn thành phần L2 trong Elastic Net buộc cả hai biến cùng chia sẻ trọng số ổn định.

---

## 1. Hạn Chế Của OLS

Giả sử bạn có $p = 100$ biến độc lập nhưng chỉ có $N = 50$ quan sát (hoặc các biến độc lập tương quan rất mạnh với nhau). Khi đó ma trận $\mathbf{X}^T\mathbf{X}$ bị suy biến hoặc gần suy biến (singular), khiến các hệ số $\hat{\beta}$ của OLS có độ lớn rất cao và dao động mạnh. Chỉ một thay đổi nhỏ trong tập huấn luyện cũng làm $\hat{\beta}$ biến động đáng kể, dẫn đến quá khớp.

Giải pháp là kiểm soát độ lớn của các hệ số $\beta$ thông qua kỹ thuật co hệ số (shrinkage).

---

## 2. Hồi Quy Ridge (Điều Chuẩn $L_2$)

Hồi quy Ridge bổ sung tổng bình phương các hệ số vào hàm mục tiêu:

$$ \hat{\boldsymbol{\beta}}^{ridge} = \arg\min_{\beta} \left\{ \sum_{i=1}^N (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^p \beta_j^2 \right\} $$

Trong đó:
*   $\lambda \ge 0$ là siêu tham số điều chuẩn (tuning parameter / hyperparameter):
    *   Khi $\lambda = 0$, mô hình quay về OLS thông thường.
    *   Khi $\lambda \to \infty$, tất cả các hệ số $\beta_j \to 0$ (trừ hệ số chặn $\beta_0$).
*   Số hạng phạt có dạng $\lambda ||\beta||_2^2$.

### Cơ Sở Toán Học Của Tính Ổn Định Trong Ridge

Với dữ liệu đã căn giữa (không phạt hệ số chặn $\beta_0$), nghiệm dạng đóng của Ridge là:

$$ \hat{\boldsymbol{\beta}}^{ridge} = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{Y} $$

Việc cộng thêm $\lambda \mathbf{I}$ vào đường chéo chính giúp ma trận luôn khả nghịch và ổn định hơn về mặt số học (numerically stable).

---

## 3. Hồi Quy Lasso (Điều Chuẩn $L_1$)

Lasso (Least Absolute Shrinkage and Selection Operator) thay tổng bình phương bằng tổng giá trị tuyệt đối của các hệ số:

$$ \hat{\boldsymbol{\beta}}^{lasso} = \arg\min_{\beta} \left\{ \sum_{i=1}^N (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^p |\beta_j| \right\} $$

### Khả Năng Chọn Lọc Biến Của Lasso (Feature Selection)

Khác với Ridge (chỉ co $\beta_j$ tiến gần về 0), Lasso có thể ép một số hệ số $\beta_j$ về **đúng bằng 0**. Nhờ đó, Lasso tự động thực hiện **lựa chọn biến (feature selection)** và tạo ra các mô hình thưa (sparse models), đặc biệt hữu ích khi số biến $p$ rất lớn.

> **Trực giác hình học:** Trong không gian tham số 2 chiều, miền ràng buộc của Ridge là hình tròn (tổng quát là hình cầu), còn của Lasso là hình thoi (tổng quát là đa diện). Hình thoi có các đỉnh nhọn nằm trên các trục tọa độ; khi các đường elip đồng mức của RSS tiếp xúc với miền ràng buộc, điểm tiếp xúc dễ rơi vào các đỉnh này hơn, làm cho một số hệ số $\beta_j = 0$.

---

## 4. Elastic Net

Elastic Net kết hợp đồng thời cả hai số hạng phạt của Lasso và Ridge:

$$ \lambda_1 \sum |\beta_j| + \lambda_2 \sum \beta_j^2 $$

Phương pháp này phù hợp khi ta muốn kết hợp ưu điểm của cả hai: vừa chọn lọc biến (như Lasso), vừa xử lý tốt các nhóm biến có đa cộng tuyến cao (như Ridge).

---

## 5. Cách Chọn Tham Số $\lambda$

Trong thực hành, ta không chọn $\lambda$ một cách tùy ý mà sử dụng **kiểm chứng chéo (Cross-Validation - CV)**. Với thư viện `scikit-learn`, các lớp `RidgeCV` và `LassoCV` tự động thử nghiệm một dải giá trị $\lambda$ (tham số `alpha`) và chọn giá trị cho sai số kiểm định thấp nhất.

### Ví Dụ Minh Họa: So Sánh Ridge và Lasso

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

# Giả lập dữ liệu thưa
np.random.seed(42)
N, p = 50, 100
X = np.random.randn(N, p)
true_beta = np.zeros(p)
true_beta[:5] = 10 # Chỉ 5 biến đầu tiên là quan trọng
y = X @ true_beta + np.random.randn(N)

# Chuẩn hóa dữ liệu (QUAN TRỌNG cho Regularization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y)

# Train Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_scaled, y)

# So sánh hệ số
plt.figure(figsize=(12, 5))
plt.plot(true_beta, 'k.', label='True Beta')
plt.plot(ridge.coef_, 'b^', alpha=0.5, label='Ridge Coef')
plt.plot(lasso.coef_, 'rx', label='Lasso Coef')
plt.legend()
plt.title("Comparison: Lasso ep he so ve 0 (Sparsity) tot hon Ridge")
plt.show()

print(f"Số biến Lasso giữ lại: {np.sum(lasso.coef_ != 0)}")
```

---

### Ví dụ tính tay làm từng bước: Chữa bệnh suy biến ma trận $$2\times 2$$ bằng Ridge

Giả sử hai đặc trưng $$X_1$$ và $$X_2$$ giống hệt nhau (đa cộng tuyến 100%), tạo ra ma trận thiết kế suy biến:

$$\mathbf{X}^T\mathbf{X} = \begin{pmatrix} 2 & 2 \\ 2 & 2 \end{pmatrix}, \qquad \mathbf{X}^T\mathbf{y} = \begin{pmatrix} 6 \\ 6 \end{pmatrix}.$$

1. **Thử giải OLS:** Định thức $$\det(\mathbf{X}^T\mathbf{X}) = 2\times 2 - 2\times 2 = 0$$ (hai giá trị riêng là $$\mu_1 = 4$$ và $$\mu_2 = 0$$). Ma trận **không khả nghịch** — mọi cặp $$(\beta_1, \beta_2)$$ thỏa $$\beta_1 + \beta_2 = 3$$ (kể cả $$(+1000, -997)$$) đều cho cùng sai số OLS!
2. **Bật Ridge với $$\lambda = 1$$:** Cộng ma trận đơn vị $$I$$ vào đường chéo chính:
   $$\mathbf{X}^T\mathbf{X} + I = \begin{pmatrix} 3 & 2 \\ 2 & 3 \end{pmatrix}.$$
   Định thức lập tức nhảy từ $$0$$ lên $$3\times 3 - 2\times 2 = 5 \neq 0$$ (hai giá trị riêng tăng đều lên $$5$$ và $$1$$)!
3. **Tính nghiệm duy nhất của Ridge:**
   $$(\mathbf{X}^T\mathbf{X} + I)^{-1} = \frac{1}{5}\begin{pmatrix} 3 & -2 \\ -2 & 3 \end{pmatrix} \implies \hat{\boldsymbol{\beta}}_{\text{Ridge}} = \frac{1}{5}\begin{pmatrix} 3 & -2 \\ -2 & 3 \end{pmatrix}\begin{pmatrix} 6 \\ 6 \end{pmatrix} = \frac{1}{5}\begin{pmatrix} 6 \\ 6 \end{pmatrix} = \begin{pmatrix} 1{,}2 \\ 1{,}2 \end{pmatrix}.$$

Nhìn nghiệm $$(1{,}2;\; 1{,}2)$$: Ridge không chỉ cứu bài toán khỏi suy biến mà còn **chia đều trọng số** cho hai biến tương quan hoàn toàn (tổng bằng $$2{,}4$$, hơi co nhẹ từ $$3{,}0$$ về 0 để đánh đổi lấy sự ổn định tuyệt đối!).

### Giải thích: Bốn bước triển khai điều chuẩn chuẩn xác trong thực tế

1. **Kiểm tra độ suy biến của $$\mathbf{X}^T\mathbf{X}$$:** Khi số điều kiện lớn hoặc $$p > n$$, OLS không còn đáng tin — chuyển ngay sang Ridge/Lasso/Elastic Net.
2. **Chuẩn hóa bắt buộc bên trong Cross-Validation:** Tính mean/std trên *tập train của từng fold* rồi áp cho tập validation để tránh rò rỉ dữ liệu (data leakage).
3. **Vẽ đường dẫn điều chuẩn (Regularization Path):** Quan sát các hệ số $$\hat{\beta}_j(\lambda)$$ co về 0 theo $$\log\lambda$$ giúp bạn nhận diện ngay những đặc trưng "sống sót" bền bỉ nhất.
4. **Dùng quy tắc "1 sai số chuẩn" (1-SE rule):** Thay vì chọn đúng $$\lambda$$ có lỗi CV thấp nhất tuyệt đối, nhiều nhà thống kê chọn $$\lambda$$ *lớn hơn* (mô hình đơn giản, gọn nhẹ hơn) có lỗi nằm trong vòng 1 sai số chuẩn quanh mức tối thiểu.

**Kiểm tra nhanh.** Trong ví dụ ma trận $$\mathbf{X}^T\mathbf{X} = \begin{pmatrix} 2 & 2 \\ 2 & 2 \end{pmatrix}$$ ở trên, nếu bạn tăng tham số phạt Ridge từ $$\lambda = 1$$ lên $$\lambda = 4$$, nghiệm $$\hat{\boldsymbol{\beta}}_{\text{Ridge}}$$ mới bằng bao nhiêu? *(Gợi ý: $$(\mathbf{X}^T\mathbf{X} + 4I)\hat{\boldsymbol{\beta}} = (6,6)^T$$ với đối xứng $$\hat{\beta}_1 = \hat{\beta}_2$$.)*

---

## Video tham khảo

- [Ridge vs Lasso Regression, Visualized!!!](https://www.youtube.com/watch?v=Xm2C_gTAl8c) — StatQuest with Josh Starmer, tiếng Anh. **Xem trước.** Minh họa trực quan sự khác biệt giữa phạt L2 (co hệ số) và phạt L1 (đưa hệ số về đúng 0).
- [Regularization Part 3: Elastic Net Regression](https://www.youtube.com/watch?v=1dKRdX9bfIo) — StatQuest with Josh Starmer, tiếng Anh. Giới thiệu Elastic Net khi cần kết hợp đồng thời tính ổn định của Ridge và tính thưa của Lasso.
- [Elastic Net Regularization : Data Science Concepts](https://www.youtube.com/watch?v=t6E0Dt5rg9U) — ritvikmath, tiếng Anh. Phân tích toán học bổ sung về Elastic Net và vai trò của tham số điều chuẩn.

### Bài Tập

1. Sinh một tập dữ liệu có mức độ đa cộng tuyến cao (các cột của ma trận $X$ tương quan mạnh với nhau).
2. So sánh MSE của OLS, Ridge và Lasso trên tập kiểm tra (test set).
3. Quan sát và nhận xét mức độ ổn định của các hệ số Ridge so với OLS.

---
**Tiếp theo:** Các bài vừa qua tập trung vào bài toán hồi quy (dự đoán giá trị liên tục). Để xử lý bài toán **phân loại** (chẳng hạn phân loại email là Spam hay Không phải Spam), chúng ta sẽ chuyển sang **Hồi quy Logistic (Logistic Regression)** ở bài tiếp theo.
