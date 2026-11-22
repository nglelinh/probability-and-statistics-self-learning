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

Ở các bài trước, chúng ta đã thấy nhiều tính chất tối ưu của OLS. Tuy nhiên, OLS vẫn có một hạn chế đáng kể: việc đòi hỏi ước lượng phải hoàn toàn **không chệch (unbiased)** đôi khi khiến **phương sai (variance)** của ước lượng tăng quá cao, dẫn đến quá khớp (overfitting).

**Điều chuẩn (Regularization)** giải quyết vấn đề này bằng cách thêm một số hạng phạt (penalty term) vào hàm mất mát. Mô hình chấp nhận tăng nhẹ **độ chệch (bias)** để giảm mạnh **phương sai (variance)**, nhờ đó giảm tổng sai số dự báo (MSE).

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
