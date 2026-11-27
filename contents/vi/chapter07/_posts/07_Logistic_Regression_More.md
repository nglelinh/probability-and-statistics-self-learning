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

Hãy tưởng tượng bạn đang vận hành hệ thống phát hiện giao dịch thẻ tín dụng gian lận. Mô hình hồi quy logistic trả về một con số xác suất mượt mà $$p \in (0, 1)$$ cho mỗi giao dịch. Câu hỏi thực chiến không chỉ là *"làm sao vẽ đường thẳng ngăn cách hai lớp?"*, mà còn là: **nếu ta hạ ngưỡng cảnh báo từ $$p = 0{,}50$$ xuống $$p = 0{,}20$$ để không bỏ sót kẻ gian, hình học của ranh giới quyết định sẽ thay đổi như thế nào?**

Bài học củng cố này tập trung vào góc nhìn hình học và trực quan hóa của **Hồi quy Logistic**: cách hàm sigmoid biến khoảng cách tới siêu phẳng thành xác suất, cách đọc ý nghĩa của **log-odds (logit)**, và cách vẽ trực tiếp đường ranh giới quyết định trên mặt phẳng đặc trưng 2D bằng Python.

---

## Kiến thức cần nhớ: Ba trụ cột hình học – xác suất của Phân loại Tuyến tính

![Mở rộng Logistic: Phân loại Đa lớp Softmax & ROC-AUC]({{ site.baseurl }}/img/course/ch07_07_softmax.svg)

Để hiểu thật chắc cơ chế phân loại của Hồi quy Logistic trên mặt phẳng 2D cũng như không gian $$p$$ chiều:

1. **Siêu phẳng phân cách & Độ tin cậy theo khoảng cách:** Biểu thức $$z = \beta_0 + \boldsymbol{\beta}^T\mathbf{x}$$ tỷ lệ thuận với khoảng cách có dấu từ điểm $$\mathbf{x}$$ tới đường ranh giới $$z = 0$$. Điểm nằm ngay trên đường ranh giới có $$z = 0 \implies p = 0{,}5$$ (bất định tối đa); điểm càng xa đường ranh giới về hai phía, $$\lvert z\rvert$$ càng lớn và xác suất càng tiến sát về $$0$$ hoặc $$1$$.
2. **Dịch chuyển ngưỡng xác suất = Dịch song song đường ranh giới:** Chọn ngưỡng quyết định $$p \ge \tau$$ tương đương với $$z \ge \ln\frac{\tau}{1-\tau}$$. Nói cách khác, đổi ngưỡng $$\tau$$ **không hề xoay hướng** đường thẳng phân cách — nó chỉ tịnh tiến song song đường thẳng đó sang phía lớp được ưu tiên!
3. **Tối ưu lồi qua Log-Loss:** Cực đại hóa hàm hợp lý Bernoulli tương đương cực tiểu hóa hàm log-loss lồi nghiêm ngặt (khi có điều chuẩn L2), đảm bảo nghiệm hình học tìm được là duy nhất và ổn định.

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

---

### Ví dụ tính tay làm từng bước: Vẽ đường ranh giới quyết định và dịch ngưỡng từ $$p=0{,}5$$ sang $$p=0{,}2$$

Giả sử sau khi huấn luyện hồi quy logistic với 2 đặc trưng $$(x_1, x_2)$$, ta thu được hàm tuyến tính log-odds:

$$z(x_1, x_2) = -4 + 2x_1 + x_2.$$

1. **Đường ranh giới quyết định ở ngưỡng mặc định $$p = 0{,}5$$ (tức $$z = 0$$):**
   $$-4 + 2x_1 + x_2 = 0 \iff x_2 = 4 - 2x_1.$$
   Đường thẳng này cắt trục hoành tại $$(2, 0)$$ và trục tung tại $$(0, 4)$$.
2. **Tính xác suất dự báo cho hai điểm nằm đối xứng qua đường ranh giới:**
   - Điểm $$A(3, 0)$$: $$z = -4 + 2(3) + 0 = +2 \implies p(A) = \frac{1}{1 + e^{-2}} \approx 0{,}881$$ (phân loại lớp 1 với độ tự tin 88,1%).
   - Điểm $$B(1, 0)$$: $$z = -4 + 2(1) + 0 = -2 \implies p(B) = \frac{1}{1 + e^{2}} \approx 0{,}119$$ (phân loại lớp 0; để ý $$p(A) + p(B) = 1$$!).
3. **Hạ ngưỡng cảnh báo xuống $$p = 0{,}20$$ (để tăng Recall bắt gian lận):**
   Đổi ngưỡng xác suất sang thang log-odds:
   $$z_{\text{ngưỡng}} = \ln\frac{0{,}20}{1 - 0{,}20} = \ln(0{,}25) = -\ln 4 \approx -1{,}386.$$
   Đường ranh giới mới là:
   $$-4 + 2x_1 + x_2 = -1{,}386 \iff x_2 = 2{,}614 - 2x_1.$$
   Đường phân cách giữ nguyên độ dốc $$-2$$ nhưng **tịnh tiến xuống dưới**, mở rộng vùng dự báo lớp 1 để bắt thêm các trường hợp nghi ngờ!

### Giải thích: Bốn bước đọc vị mô hình phân loại tuyến tính

1. **Kiểm tra khả năng tách tuyến tính trước khi đổ lỗi cho thuật toán:** Nếu hai lớp phân bố kiểu vòng tròn đồng tâm (lớp 1 ở giữa, lớp 0 bao quanh), không một đường thẳng $$\beta_0 + \beta_1 x_1 + \beta_2 x_2 = 0$$ nào tách được — bạn cần thêm đặc trưng bậc hai (như $$x_1^2 + x_2^2$$) hoặc chuyển sang mô hình phi tuyến.
2. **Độ lớn vector $$\|\boldsymbol{\beta}\|$$ quyết định độ dốc vùng chuyển tiếp:** Nhân đôi cả $$\beta_0, \beta_1, \beta_2$$ không làm thay đổi vị trí đường $$z = 0$$, nhưng làm hàm sigmoid dựng đứng hơn (xác suất chuyển từ 0 sang 1 đột ngột hơn).
3. **Tách biệt bước *học xác suất* với bước *chọn ngưỡng ra quyết định*:** Huấn luyện mô hình để tối ưu Log-Loss (xác suất hiệu chuẩn tốt), sau đó mới chọn ngưỡng $$\tau$$ dựa trên ma trận chi phí thực tế của bài toán.
4. **Nhớ kiểm tra độ cân bằng lớp và hiệu chuẩn xác suất:** Khi lớp dương hiếm, hãy kiểm tra thêm biểu đồ hiệu chuẩn (calibration plot) để chắc chắn xác suất dự báo 0,20 thực sự tương ứng khoảng 20% ca dương.

**Kiểm tra nhanh.** Với mô hình $$z(x_1, x_2) = -4 + 2x_1 + x_2$$ ở trên, điểm $$C(1{,}5;\; 0)$$ sẽ bị xếp vào lớp nào nếu dùng ngưỡng $$p = 0{,}50$$, và xếp vào lớp nào nếu hạ ngưỡng xuống $$p = 0{,}20$$?

---

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
