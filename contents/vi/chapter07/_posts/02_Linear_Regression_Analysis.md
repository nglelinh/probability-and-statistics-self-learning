---
layout: post
title: "Hồi Quy Tuyến Tính và Phương Pháp Bình Phương Tối Thiểu (OLS)"
date: 2021-01-01 00:00:03 +0700
categories: [probability-and-statistics]
tags: [linear-regression, ols, gauss-markov, residuals]
chapter: "07"
lang: vi
order: 2
lesson_type: required
---

**Mô hình tuyến tính (Linear Model)** là "công cụ lao động" (workhorse) của thống kê và machine learning. Mặc dù đơn giản, nó là nền tảng để hiểu các phương pháp phức tạp hơn như Neural Networks (có thể coi là nhiều lớp các mô hình tuyến tính ghép lại với hàm kích hoạt phi tuyến).

Trong bài này, chúng ta sẽ đi sâu vào toán học của Hồi quy tuyến tính, không chỉ dừng ở việc `model.fit()`.

---

## 1. Từ Đơn Biến đến Đa Biến (Simple to Multiple Regression)

Trong chương trước, ta đã làm quen với hồi quy đơn biến: $Y = \beta_0 + \beta_1 X + \varepsilon$.
Thực tế, $Y$ thường phụ thuộc vào nhiều biến $X_1, X_2, ..., X_p$.

**Mô hình Hồi quy Tuyến tính Đa biến:**

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p + \varepsilon$$

Trong đó:
*   $Y$: Biến phụ thuộc (Target/Response).
*   $X_j$: Các biến độc lập (Features/Predictors).
*   $\beta_j$: Các hệ số hồi quy (Coefficients/Weights).
*   $\varepsilon$: Sai số ngẫu nhiên (Error term), thường giả định $\varepsilon \sim N(0, \sigma^2)$.

### Dạng Ma Trận (Matrix Notation)
Để tính toán hiệu quả, ta chuyển sang dạng ma trận.
Với $N$ quan sát và $p$ biến (cộng thêm 1 cột bias $x_0=1$):

$$ \mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon} $$

Trong đó:
*   $\mathbf{Y}$ là vector cột $N \times 1$.
*   $\mathbf{X}$ là ma trận thiết kế $N \times (p+1)$.
*   $\boldsymbol{\beta}$ là vector tham số $(p+1) \times 1$.
*   $\boldsymbol{\varepsilon}$ là vector sai số $N \times 1$.

```python
import numpy as np

# Ví dụ minh họa Matrix Notation python
X = np.array([[1, 2], [1, 3], [1, 4]]) # N=3, p=1 (cột 1 là bias)
beta = np.array([2, 5]) # beta_0=2, beta_1=5
y_true = X @ beta # [12, 17, 22]
print(f"Y true: {y_true}")
```

---

## 2. Phương Pháp Bình Phương Tối Thiểu (Ordinary Least Squares - OLS)

Mục tiêu: Tìm $\hat{\boldsymbol{\beta}}$ sao cho tổng bình phương sai số (Residual Sum of Squares - RSS) là nhỏ nhất.

$$ RSS(\boldsymbol{\beta}) = \sum_{i=1}^N (y_i - \hat{y}_i)^2 = (\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})^T (\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}) $$

### Giải pháp dạng đóng (Closed-form Solution)
Bằng cách lấy đạo hàm RSS theo $\boldsymbol{\beta}$ và gán bằng 0, ta thu được phương trình pháp tuyến (Normal Equation):

$$ \mathbf{X}^T (\mathbf{Y} - \mathbf{X}\hat{\boldsymbol{\beta}}) = 0 $$

Giải phương trình này ta được công thức OLS nổi tiếng:

$$ \hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y} $$

> **Lưu ý:** Để tính được, ma trận $\mathbf{X}^T \mathbf{X}$ phải khả nghịch (không có đa cộng tuyến hoàn hảo - multicollinearity).

```python
# Tự implement OLS với numpy
def fit_ols(X, y):
    # Thêm cột bias 1 vào đầu X nếu chưa có (thường sklearn làm tự động, ở đây ta làm thủ công)
    N = len(y)
    X_b = np.c_[np.ones((N, 1)), X] 
    
    # Công thức: (X^T * X)^-1 * X^T * Y
    beta_hat = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return beta_hat

# Test
X_data = np.array([[2], [3], [4]])
y_data = 2 + 5 * X_data.flatten() + np.random.normal(0, 0.5, 3) # True beta=[2, 5]
beta_est = fit_ols(X_data, y_data)
print(f"Estimated beta: {beta_est}")
```

---

## 3. Ý Nghĩa Hình Học (Geometric Interpretation)

Đây là một góc nhìn rất đẹp của thống kê.
*   Chúng ta có không gian $\mathbb{R}^N$ (không gian các quan sát).
*   Vector $\mathbf{Y}$ nằm trong không gian này.
*   Các cột của ma trận $\mathbf{X}$ căng ra một không gian con (Subspace) có số chiều là $p+1$.
*   Mọi vector dự đoán $\hat{\mathbf{Y}} = \mathbf{X}\boldsymbol{\beta}$ đều phải nằm trong subspace này.

Để sai số $\mathbf{Y} - \hat{\mathbf{Y}}$ là nhỏ nhất (khoảng cách Euclidean ngắn nhất), thì vector sai số $\mathbf{e}$ phải **vuông góc** với subspace đó.
$$\rightarrow$$ **$\hat{\mathbf{Y}}$ chính là hình chiếu vuông góc (Orthogonal Projection) của $\mathbf{Y}$ lên không gian cột của $\mathbf{X}$.**

Ma trận chiếu (Hat Matrix) $\mathbf{H}$ biến $\mathbf{Y}$ thành $\hat{\mathbf{Y}}$:
$$ \hat{\mathbf{Y}} = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y} = \mathbf{H}\mathbf{Y} $$

---

## 4. Định Lý Gauss-Markov

Tại sao lại dùng OLS mà không phải phương pháp khác (như giảm tổng trị tuyệt đối sai số)?

**Định lý Gauss-Markov:**
Nếu các sai số $\varepsilon_i$ không tương quan (uncorrelated) và có phương sai đồng nhất (homoscedasticity) $\sigma^2$, và kỳ vọng bằng 0, thì:

**Ước lượng OLS là ước lượng tuyến tính không chệch tốt nhất (BLUE - Best Linear Unbiased Estimator).**

*   **Linear:** Là hàm tuyến tính của biến phụ thuộc $Y$.
*   **Unbiased:** $E[\hat{\boldsymbol{\beta}}] = \boldsymbol{\beta}$.
*   **Best:** Có phương sai (variance) nhỏ nhất trong tất cả các ước lượng không chệch tuyến tính.

---

## 5. Phân Tích Phần Dư (Residual Analysis)

Sau khi mô hình chạy xong, làm sao biết nó có tốt hay sai giả định? Ta nhìn vào **Residuals** ($e = y - \hat{y}$).

Các giả định quan trọng cần kiểm tra:
1.  **Linearity:** Quan hệ giữa X và Y là tuyến tính. Nếu Plot Residuals vs Predicted Value thấy có hình cong (parabol) -> Sai giả định (cần thêm feature bậc cao).
2.  **Homoscedasticity (Phương sai đồng nhất):** Variance của residuals không đổi theo $\hat{y}$. Nếu plot có hình cái phễu (to dần hoặc nhỏ dần) -> Heteroscedasticity (Phương sai thay đổi).
3.  **Normality:** Residuals phân phối chuẩn (quan trọng cho kiểm định giả thuyết, không quan trọng cho dự báo thuần túy). Dùng Q-Q Plot để check.
4.  **No Multicollinearity:** Các biến $X$ không được tương quan quá mạnh với nhau.

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Residuals vs Fitted
    sns.scatterplot(x=y_pred, y=residuals, ax=axes[0])
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_xlabel('Fitted values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Fitted (Check Linearity & Homoscedasticity)')
    
    # Histogram of Residuals
    sns.histplot(residuals, kde=True, ax=axes[1])
    axes[1].set_title('Distribution of Residuals (Check Normality)')
    
    plt.tight_layout()
    plt.show()
```

### Bài Tập Thực Hành
Sử dụng bộ dữ liệu `Advertising.csv` (hoặc Boston Housing) để:
1. Thực hiện Hồi quy tuyến tính đa biến.
2. Vẽ đồ thị Residuals vs Fitted values.
3. Nhận xét về tính phù hợp của mô hình.

---
**Tiếp theo:** Nếu mô hình quá phức tạp hoặc các biến độc lập tương quan mạnh thì sao? Chúng ta sẽ học về **Regularization (Ridge & Lasso)** ở bài sau.
