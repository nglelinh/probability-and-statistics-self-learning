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

Mặc dù mang tên gọi có từ "Regression", **Hồi quy Logistic (Logistic Regression)** là một thuật toán **phân loại (classification)**, dùng để dự đoán xác suất một quan sát thuộc về một lớp cụ thể (chẳng hạn có bệnh hay không có bệnh). Ta không thể áp dụng trực tiếp hồi quy tuyến tính cho bài toán này vì hồi quy tuyến tính trả về giá trị thực trên toàn bộ khoảng $(-\infty, +\infty)$, trong khi xác suất $P$ phải nằm trong đoạn $[0, 1]$.

---

## 1. Hàm Sigmoid và Log-Odds

Để ánh xạ đầu ra của mô hình tuyến tính $z = \beta_0 + \beta_1 X$ vào khoảng $(0, 1)$, ta sử dụng **hàm Sigmoid** (hay hàm Logistic):

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

Khi đó, mô hình hồi quy logistic dự đoán xác suất $P(Y=1|X)$ theo công thức:

$$ P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X)}} $$

### Log-Odds (Logit)

Biến đổi ngược phương trình trên, ta thu được:

$$ \ln \left( \frac{P}{1-P} \right) = \beta_0 + \beta_1 X $$

Trong đó, tỷ số $\frac{P}{1-P}$ gọi là **Odds** (tỷ lệ chênh), còn $\ln(\text{Odds})$ gọi là **Log-odds** hay **Logit**.

> **Diễn giải hệ số $\beta$:** Nếu $\beta_1 = 0.5$, thì khi $X$ tăng thêm 1 đơn vị, log-odds của biến cố $Y=1$ tăng thêm 0.5 đơn vị (tương đương với Odds tăng $e^{0.5}$ lần).

---

## 2. Ước Lượng Tham Số Bằng Hợp Lý Cực Đại (MLE)

Ta không dùng hàm mất mát bình phương tối thiểu (OLS) kết hợp với đầu ra sigmoid vì hàm mục tiêu thu được không lồi (non-convex), dễ gặp khó khăn bởi cực trị địa phương khi tối ưu. Thay vào đó, hồi quy logistic sử dụng phương pháp **ước lượng hợp lý cực đại (Maximum Likelihood Estimation - MLE)**.

Mục tiêu của MLE là tìm $\beta$ để cực đại hóa xác suất quan sát được tập dữ liệu huấn luyện. Trong thực hành, điều này tương đương với việc cực tiểu hóa hàm **Negative Log-Likelihood**, còn gọi là **Log Loss** (hay **Cross-Entropy Loss**):

$$ J(\beta) = - \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right] $$

*   Khi $y_i = 1$: Hàm mất mát nhỏ khi $\hat{y}_i$ tiến gần 1 (lúc đó $\log(\hat{y}_i) \to 0$).
*   Khi $y_i = 0$: Hàm mất mát nhỏ khi $\hat{y}_i$ tiến gần 0 (lúc đó $\log(1-\hat{y}_i) \to 0$).

Hàm mục tiêu này là hàm lồi (convex), đảm bảo các thuật toán tối ưu như Gradient Descent hội tụ về cực trị toàn cục.

---

## 3. Đường Ranh Giới Quyết Định (Decision Boundary)

Dù hàm dự đoán xác suất là đường cong Sigmoid phi tuyến, hồi quy logistic vẫn là một **bộ phân loại tuyến tính (linear classifier)**. Lý do là với ngưỡng phân loại mặc định $P = 0.5$, ranh giới quyết định thỏa mãn:

$$ \sigma(z) = 0.5 \iff z = 0 $$
$$ \iff \beta_0 + \beta_1 X_1 + ... + \beta_p X_p = 0 $$

Đây là phương trình của một siêu phẳng (hyperplane) — tương ứng với đường thẳng trong không gian 2 chiều hoặc mặt phẳng trong không gian 3 chiều.

---

## 4. Ví Dụ Minh Họa Bằng Python

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

## Video tham khảo

- [Odds and Log(Odds), Clearly Explained!!!](https://www.youtube.com/watch?v=ARfXDSkQf1Y) — StatQuest with Josh Starmer, tiếng Anh. **Xem trước.** Giải thích khái niệm odds và log-odds, bước biến đổi cốt lõi dẫn tới hàm sigmoid.
- [Logistic Regression (and why it's different from Linear Regression)](https://www.youtube.com/watch?v=3bvM3NyMiE0) — Visually Explained, tiếng Anh. Làm rõ sự khác biệt giữa hồi quy logistic (đầu ra là xác suất qua hàm sigmoid) và hồi quy tuyến tính.
- [What is a Decision Boundary?](https://www.youtube.com/watch?v=KJJFfRXFbuc) — Nicholas Leonhard, tiếng Anh. Minh họa ranh giới quyết định tương ứng với tập điểm tại đó giá trị logit bằng 0.

### Bài Tập

1. Áp dụng hồi quy logistic cho dữ liệu không phân tách tuyến tính (chẳng hạn dữ liệu dạng vòng tròn đồng tâm từ `make_circles`) và quan sát kết quả.
2. Làm thế nào để hồi quy logistic tạo ra ranh giới quyết định dạng đường cong? (Gợi ý: tạo thêm các đặc trưng phi tuyến như $x_1^2, x_2^2$).

---
**Tổng kết các mô hình tuyến tính cơ bản:**
Qua các bài học trong chương này, chúng ta đã tìm hiểu ba nhóm mô hình tuyến tính nền tảng:
- **Hồi quy tuyến tính (Linear Regression):** Dự báo biến mục tiêu liên tục.
- **Điều chuẩn (Regularization):** Kiểm soát quá khớp và đa cộng tuyến.
- **Hồi quy Logistic (Logistic Regression):** Phân loại dựa trên xác suất.
