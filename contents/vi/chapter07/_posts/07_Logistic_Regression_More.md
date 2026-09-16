---
layout: post
title: "Logistic Regression và Phân Loại Tuyến Tính"
date: 2021-01-01 00:00:05 +0700
categories:
- chapter07
tags: [classification, logistic-regression, sigmoid, mle, cross-entropy]
chapter: "07"
lang: vi
order: 7
lesson_type: required
---

Đừng để cái tên đánh lừa: **Logistic Regression** là một thuật toán **Phân loại (Classification)**, không phải Hồi quy (Regression). Nó dùng để dự đoán xác suất thuộc về một lớp nào đó (ví dụ: Có bệnh/Không bệnh).

Tại sao không dùng Linear Regression cho bài toán này?
Vì Linear Regression trả về giá trị liên tục $(-\infty, +\infty)$, trong khi xác suất $P$ phải nằm trong $[0, 1]$.

---

## 1. Hàm Sigmoid và Log-Odds

Để ép đầu ra của mô hình tuyến tính $z = \beta_0 + \beta_1 X$ vào khoảng $(0, 1)$, ta dùng **hàm Sigmoid** (hay Logistic function):

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

Khi đó, mô hình Logistic Regression dự đoán xác suất $P(Y=1|X)$:

$$ P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X)}} $$

### Log-Odds (Logit)
Nếu biến đổi ngược lại:
$$ \ln \left( \frac{P}{1-P} \right) = \beta_0 + \beta_1 X $$
Đại lượng $\frac{P}{1-P}$ gọi là **Odds** (tỷ lệ cược).
Đại lượng $\ln(\text{Odds})$ gọi là **Log-odds** hay **Logit**.

> **Ý nghĩa hệ số $\beta$:** Nếu $\beta_1 = 0.5$, nghĩa là khi $X$ tăng 1 đơn vị, log-odds của việc $Y=1$ tăng 0.5 đơn vị (hay Odds tăng $e^{0.5}$ lần).

---

## 2. Ước Lượng Tham Số: MLE (Maximum Likelihood)

Không thể dùng OLS (Bình phương tối thiểu) cho Logistic Regression vì hàm lỗi không lồi (non-convex), dễ bị kẹt ở cực trị địa phương.
Thay vào đó, ta dùng **Maximum Likelihood Estimation (MLE)**.

Ta muốn tìm $\beta$ sao cho xác suất quan sát được bộ dữ liệu là lớn nhất (Likelihood max).
Trong thực tế, ta tối thiểu hóa **Negative Log-Likelihood**, hay còn gọi là **Log Loss** (Cross-Entropy Loss):

$$ J(\beta) = - \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right] $$

*   Nếu $y=1$: Loss muốn $\hat{y}$ càng gần 1 càng tốt (để $\log(\hat{y}) \to 0$).
*   Nếu $y=0$: Loss muốn $\hat{y}$ càng gần 0 càng tốt (để $\log(1-\hat{y}) \to 0$).

Hàm này lồi (convex), đảm bảo Gradient Descent sẽ tìm được cực trị toàn cục.

---

## 3. Đường Ranh Giới Quyết Định (Decision Boundary)

Mặc dù hàm dự đoán là phi tuyến (đường cong Sigmoid), nhưng Logistic Regression là một **bộ phân loại tuyến tính (Linear Classifier)**.

Tại sao? Vì ranh giới quyết định (nơi $P=0.5$) xảy ra khi:
$$ \sigma(z) = 0.5 \iff z = 0 $$
$$ \iff \beta_0 + \beta_1 X_1 + ... + \beta_p X_p = 0 $$
Đây là phương trình của một siêu phẳng (hyperplane) - một đường thẳng trong 2D hoặc mặt phẳng trong 3D.

---

## 4. Code Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# 1. Tạo dữ liệu giả lập
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, 
                           n_redundant=0, n_clusters_per_class=1, random_state=42)

# 2. Train mô hình
model = LogisticRegression()
model.fit(X, y)

# 3. Vẽ Decision Boundary
# Tạo lưới điểm
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Dự đoán trên lưới
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
plt.title("Linear Decision Boundary of Logistic Regression")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Vẽ đường thẳng phân cách thực sự: beta0 + beta1*x1 + beta2*x2 = 0
# => x2 = -(beta0 + beta1*x1) / beta2
w = model.coef_[0]
b = model.intercept_[0]
x_line = np.linspace(x_min, x_max, 100)
y_line = -(b + w[0] * x_line) / w[1]
plt.plot(x_line, y_line, 'k--', linewidth=2, label='Decision Boundary Equation')
plt.legend()
plt.show()
```

### Bài Tập
1. Thử dùng Logistic Regression cho dữ liệu không phân tách tuyến tính (ví dụ dạng hình tròn `make_circles`). Quan sát kết quả.
2. Làm thế nào để Logistic Regression phân loại được biên giới cong? (Gợi ý: Feature Engineering, thêm $x_1^2, x_2^2$).

---
**Tổng kết Chương 03:**
Chúng ta đã nắm vững các mô hình tuyến tính cơ bản:
- **Linear Regression**: Dự báo số thực.
- **Regularization**: Kiểm soát overfitting.
- **Logistic Regression**: Phân loại.

Ở chương sau, chúng ta sẽ học cách đánh giá xem các mô hình này tốt xấu ra sao bằng các phương pháp Resampling.
