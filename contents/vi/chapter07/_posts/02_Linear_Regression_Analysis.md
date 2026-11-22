---
layout: post
title: "Hồi Quy Tuyến Tính và Phương Pháp Bình Phương Tối Thiểu (OLS)"
date: 2021-01-01 00:00:03 +0700
categories:
- chapter07
tags: [linear-regression, ols, gauss-markov, residuals]
chapter: "07"
lang: vi
order: 2
lesson_type: required
---

**Mô hình tuyến tính (Linear Model)** là một trong những công cụ nền tảng của thống kê và học máy. Dù có cấu trúc đơn giản, nó là cơ sở để hiểu nhiều phương pháp phức tạp hơn, chẳng hạn mạng nơ-ron (có thể xem như nhiều lớp mô hình tuyến tính ghép nối qua các hàm kích hoạt phi tuyến). Trong bài này, chúng ta sẽ đi sâu vào cơ sở toán học của hồi quy tuyến tính thay vì chỉ dừng lại ở việc gọi hàm `model.fit()`.

---

## 1. Từ Đơn Biến đến Đa Biến (Simple to Multiple Regression)

Trong bài trước, ta đã làm quen với hồi quy đơn biến: $Y = \beta_0 + \beta_1 X + \varepsilon$. Trong thực tế, biến mục tiêu $Y$ thường phụ thuộc đồng thời vào nhiều biến giải thích $X_1, X_2, ..., X_p$.

**Mô hình hồi quy tuyến tính đa biến:**

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p + \varepsilon$$

Trong đó:
*   $Y$: Biến phụ thuộc (target/response).
*   $X_j$: Các biến độc lập (features/predictors).
*   $\beta_j$: Các hệ số hồi quy (coefficients/weights).
*   $\varepsilon$: Sai số ngẫu nhiên (error term), thường giả định $\varepsilon \sim N(0, \sigma^2)$.

### Dạng Ma Trận (Matrix Notation)

Để biểu diễn và tính toán gọn hơn với $N$ quan sát và $p$ biến độc lập (bổ sung thêm một cột hằng số $x_0 = 1$ cho hệ số chặn), ta viết mô hình dưới dạng ma trận:

$$ \mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon} $$

Trong đó:
*   $\mathbf{Y}$ là vector cột kích thước $N \times 1$.
*   $\mathbf{X}$ là ma trận thiết kế kích thước $N \times (p+1)$.
*   $\boldsymbol{\beta}$ là vector tham số kích thước $(p+1) \times 1$.
*   $\boldsymbol{\varepsilon}$ là vector sai số kích thước $N \times 1$.

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

Mục tiêu của OLS là tìm ước lượng $\hat{\boldsymbol{\beta}}$ sao cho tổng bình phương phần dư (Residual Sum of Squares - RSS) đạt giá trị nhỏ nhất:

$$ RSS(\boldsymbol{\beta}) = \sum_{i=1}^N (y_i - \hat{y}_i)^2 = (\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})^T (\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}) $$

### Nghiệm Dạng Đóng (Closed-form Solution)

Lấy đạo hàm của $RSS(\boldsymbol{\beta})$ theo $\boldsymbol{\beta}$ và đặt bằng 0, ta thu được phương trình pháp tuyến (Normal Equation):

$$ \mathbf{X}^T (\mathbf{Y} - \mathbf{X}\hat{\boldsymbol{\beta}}) = 0 $$

Giải phương trình này cho ta công thức ước lượng OLS:

$$ \hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y} $$

> **Lưu ý:** Để công thức trên xác định, ma trận $\mathbf{X}^T \mathbf{X}$ phải khả nghịch, tức là các cột của $\mathbf{X}$ độc lập tuyến tính (không có đa cộng tuyến hoàn hảo — multicollinearity).

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

Góc nhìn hình học giúp làm rõ bản chất của phương pháp bình phương tối thiểu:
*   Ta xét không gian $\mathbb{R}^N$ (không gian của các quan sát), trong đó chứa vector $\mathbf{Y}$.
*   Các cột của ma trận $\mathbf{X}$ sinh ra một không gian con (column space) có số chiều tối đa là $p+1$.
*   Mọi vector giá trị dự đoán $\hat{\mathbf{Y}} = \mathbf{X}\boldsymbol{\beta}$ đều nằm trong không gian con này.

Để độ dài vector phần dư $\mathbf{e} = \mathbf{Y} - \hat{\mathbf{Y}}$ đạt giá trị nhỏ nhất theo khoảng cách Euclid, vector $\mathbf{e}$ phải **vuông góc** với không gian cột của $\mathbf{X}$. Do đó, **$\hat{\mathbf{Y}}$ chính là hình chiếu vuông góc (orthogonal projection) của $\mathbf{Y}$ lên không gian cột của $\mathbf{X}$.**

Ma trận chiếu (Hat Matrix) $\mathbf{H}$ ánh xạ $\mathbf{Y}$ thành $\hat{\mathbf{Y}}$:

$$ \hat{\mathbf{Y}} = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y} = \mathbf{H}\mathbf{Y} $$

---

## 4. Định Lý Gauss-Markov

Vì sao ta thường chọn OLS thay vì các tiêu chí khác (chẳng hạn cực tiểu hóa tổng trị tuyệt đối sai số)? Định lý Gauss-Markov cung cấp cơ sở lý thuyết cho lựa chọn này.

**Định lý Gauss-Markov:** Nếu các sai số $\varepsilon_i$ có kỳ vọng bằng 0, không tương quan với nhau (uncorrelated) và có phương sai đồng nhất $\sigma^2$ (homoscedasticity), thì **ước lượng OLS là ước lượng tuyến tính không chệch tốt nhất (BLUE — Best Linear Unbiased Estimator)**:

*   **Linear (Tuyến tính):** $\hat{\boldsymbol{\beta}}$ là hàm tuyến tính của biến phụ thuộc $\mathbf{Y}$.
*   **Unbiased (Không chệch):** $E[\hat{\boldsymbol{\beta}}] = \boldsymbol{\beta}$.
*   **Best (Tốt nhất):** Có phương sai nhỏ nhất trong lớp tất cả các ước lượng tuyến tính không chệch.

---

## 5. Phân Tích Phần Dư (Residual Analysis)

Sau khi huấn luyện mô hình, ta cần kiểm tra xem các giả định của hồi quy tuyến tính có được thỏa mãn hay không thông qua **phần dư (residuals)** $e = y - \hat{y}$. Các giả định chính gồm:

1.  **Tính tuyến tính (Linearity):** Mối quan hệ giữa $X$ và $Y$ là tuyến tính. Nếu đồ thị phần dư theo giá trị dự đoán (Residuals vs. Fitted) xuất hiện dạng cong (như parabol), giả định tuyến tính bị vi phạm và cần cân nhắc bổ sung đặc trưng phi tuyến hoặc bậc cao.
2.  **Phương sai đồng nhất (Homoscedasticity):** Phương sai của phần dư không đổi theo $\hat{y}$. Nếu đồ thị có dạng hình phễu (độ phân tán phình to hoặc thu hẹp dần), mô hình gặp hiện tượng phương sai thay đổi (heteroscedasticity).
3.  **Tính chuẩn (Normality):** Phần dư tuân theo phân phối chuẩn. Giả định này quan trọng cho suy diễn và kiểm định giả thuyết (dù ít ảnh hưởng hơn đến dự báo điểm thuần túy), thường được kiểm tra bằng Q-Q plot.
4.  **Không có đa cộng tuyến mạnh (No Multicollinearity):** Các biến độc lập $X_j$ không tương quan tuyến tính quá chặt với nhau.

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

## Video tham khảo

- [Multiple Regression, Clearly Explained!!!](https://www.youtube.com/watch?v=zITIFTsivN8) — StatQuest with Josh Starmer, tiếng Anh. **Xem trước.** Mở rộng từ đường thẳng sang mặt phẳng: hồi quy đa biến vẫn dựa trên bình phương tối thiểu khi thêm biến giải thích, tương ứng với mục 1 của bài.
- [16. Projection Matrices and Least Squares](https://www.youtube.com/watch?v=osh80YCg_GM) — MIT OpenCourseWare, tiếng Anh. Bài giảng về hình chiếu vuông góc và ma trận chiếu, làm rõ ý nghĩa hình học của OLS.
- [Gauss-Markov assumptions part 1](https://www.youtube.com/watch?v=NjTpHS5xLP8) — Ben Lambert, tiếng Anh. Trình bày các giả định Gauss–Markov giúp OLS trở thành ước lượng BLUE.

### Bài Tập Thực Hành

Sử dụng bộ dữ liệu `Advertising.csv` (hoặc Boston Housing) để:
1. Xây dựng mô hình hồi quy tuyến tính đa biến.
2. Vẽ đồ thị phần dư theo giá trị dự đoán (Residuals vs. Fitted values).
3. Nhận xét về mức độ phù hợp của mô hình và các giả định hồi quy.

---
**Tiếp theo:** Khi mô hình quá phức tạp hoặc các biến độc lập tương quan mạnh với nhau, ta xử lý ra sao? Chúng ta sẽ tìm hiểu kỹ thuật **điều chuẩn (Regularization: Ridge & Lasso)** trong các bài tiếp theo.
