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

Trong bài trước, chúng ta đã thần thánh hóa OLS. Tuy nhiên, OLS có điểm yếu chí tử: nó cố gắng **không chệch (unbiased)**, nhưng đôi khi cái giá phải trả là **phương sai (variance) quá lớn**, dẫn đến Overfitting.

**Regularization** là kỹ thuật thêm một thành phần "phạt" (penalty term) vào hàm mất mát, chấp nhận một chút **bias** để giảm mạnh **variance**, từ đó giảm tổng lỗi dự báo (MSE).

---

## 1. Vấn Đề Của OLS
Hãy tưởng tượng bạn có $p=100$ biến nhưng chỉ có $N=50$ dòng dữ liệu (hoặc các biến tương quan rất mạnh).
Khi đó ma trận $\mathbf{X}^T\mathbf{X}$ gần như suy biến (singular).
Hệ số $\hat{\beta}$ của OLS sẽ cực kỳ lớn và dao động mạnh. Một thay đổi nhỏ trong dữ liệu train sẽ làm $\hat{\beta}$ thay đổi hoàn toàn -> Overfitting.

Giải pháp: Đừng cho phép $\beta$ quá lớn. Hãy co nó lại (Shrinkage).

---

## 2. Ridge Regression ($L_2$ Regularization)

Ridge thay đổi hàm mục tiêu bằng cách thêm tổng bình phương các hệ số:

$$ \hat{\boldsymbol{\beta}}^{ridge} = \arg\min_{\beta} \left\{ \sum_{i=1}^N (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^p \beta_j^2 \right\} $$

Trong đó:
*   $\lambda \ge 0$ là tham số điều chỉnh (tuning parameter / hyperparameter).
    *   $\lambda = 0$: Trở về OLS.
    *   $\lambda \to \infty$: Tất cả $\beta_j \to 0$ (trừ $\beta_0$).
*   Thành phần phạt: $\lambda ||\beta||_2^2$.

### Tại sao Ridge tốt hơn?
Về mặt toán học:
$$ \hat{\boldsymbol{\beta}}^{ridge} = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{Y} $$
Việc cộng thêm $\lambda \mathbf{I}$ vào đường chéo chính giúp ma trận luôn khả nghịch và ổn định số học (numerically stable).

---

## 3. Lasso Regression ($L_1$ Regularization)

Lasso (Least Absolute Shrinkage and Selection Operator) sử dụng trị tuyệt đối thay vì bình phương:

$$ \hat{\boldsymbol{\beta}}^{lasso} = \arg\min_{\beta} \left\{ \sum_{i=1}^N (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^p |\beta_j| \right\} $$

### Điều kỳ diệu của Lasso: Feature Selection
Khác với Ridge (chỉ làm $\beta$ nhỏ đi tiệm cận 0), Lasso có khả năng ép một số hệ số $\beta_j$ về **đúng bằng 0**.
-> Lasso thực hiện **lựa chọn biến (Feature Selection)** tự động. Nó tạo ra các mô hình thưa (sparse models), rất hữu ích khi $p$ rất lớn.

> **Trực giác hình học:** Miền ràng buộc của Ridge là hình tròn (cầu), của Lasso là hình thoi (đa diện). Hình thoi có các "đỉnh" nhọn nằm trên các trục tọa độ. Khi elip của RSS tiếp xúc với miền ràng buộc, nó dễ chạm vào các đỉnh này hơn -> $\beta_j = 0$.

---

## 4. Elastic Net
Là sự kết hợp của cả hai:
$$ \lambda_1 \sum |\beta_j| + \lambda_2 \sum \beta_j^2 $$
Dùng khi muốn tận dụng ưu điểm của cả hai: vừa chọn biến (Lasso), vừa xử lý đa cộng tuyến tốt (Ridge).

---

## 5. Chọn $\lambda$ như thế nào?
Không thể đoán mò. Chúng ta dùng **Cross-Validation (CV)**.
Thường dùng `RidgeCV` hoặc `LassoCV` trong Scikit-Learn để tự động thử một dải các giá trị $\lambda$ (alpha) và chọn giá trị có sai số kiểm định thấp nhất.

### Code Example: Ridge vs Lasso

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

### Bài Tập
1. Sinh một tập dữ liệu có đa cộng tuyến cao (các cột X tương quan mạnh).
2. So sánh MSE của OLS, Ridge và Lasso trên tập test.
3. Quan sát hệ số của Ridge xem chúng có ổn định hơn OLS không?

---
**Tiếp theo:** Chúng ta đã xử lý bài toán hồi quy (dự đoán số). Nhưng làm sao để **phân loại** (ví dụ: Email là Spam hay Not Spam)? Bài tới: **Logistic Regression**.
