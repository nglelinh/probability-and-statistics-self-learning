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

Khi bạn gõ một dòng lệnh ngắn gọn `LinearRegression().fit(X, y)` trong Python, điều gì thực sự diễn ra trong một phần nghìn giây bên dưới lớp vỏ tiện lợi đó? Không có phép màu nào cả — chỉ có **hình học Euclid trong không gian $$n$$ chiều**! Bài toán tìm bộ hệ số tối ưu $$\hat{\boldsymbol{\beta}}$$ thực chất là bài toán tìm **hình chiếu vuông góc duy nhất** của vector dữ liệu $$\mathbf{y} \in \mathbb{R}^n$$ lên siêu phẳng (không gian cột) sinh bởi các cột của ma trận đặc trưng $$\mathbf{X}$$.

Bài học này mở nắp ca-pô của **Hồi quy Tuyến tính Đa biến và Phương pháp Bình phương Tối thiểu (OLS)**: từ ký hiệu ma trận, nghiệm dạng đóng $$(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$, ý nghĩa hình học trực giao, **Định lý Gauss–Markov** (vì sao OLS là bộ ước lượng tuyến tính không chệch có phương sai nhỏ nhất — BLUE), đến kỹ thuật phân tích phần dư.

---

## Kiến thức cần nhớ: Ba góc nhìn thống nhất về Hồi quy Tuyến tính Đa biến

Một người tốt nghiệp đại học nắm vững OLS khi nhìn thấy **ba góc nhìn hoàn toàn tương đương** của cùng một phương trình chuẩn $$\mathbf{X}^T\mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}^T\mathbf{y}$$:

1. **Góc nhìn Giải tích (Tối ưu hóa):** Cực tiểu hóa hàm mất mát bình phương $$\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2$$ bằng cách lấy gradient theo vector $$\boldsymbol{\beta}$$ và đặt bằng $$\mathbf{0}$$. Vì hàm lồi bậc hai, điểm dừng chính là cực tiểu toàn cục duy nhất khi $$\mathbf{X}$$ đủ hạng cột.
2. **Góc nhìn Đại số Tuyến tính & Hình học:** Đặt $$\hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}}$$ nằm trong không gian cột $$\mathrm{Col}(\mathbf{X})$$. Vector khoảng cách ngắn nhất từ $$\mathbf{y}$$ tới $$\mathrm{Col}(\mathbf{X})$$ buộc phải **vuông góc** với mọi cột của $$\mathbf{X}$$, tức $$\mathbf{X}^T(\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}) = \mathbf{0}$$ — ra ngay phương trình chuẩn! Ma trận mũ $$\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$$ chính là toán tử chiếu vuông góc.
3. **Góc nhìn Thống kê (Định lý Gauss–Markov):** Chỉ cần nhiễu có kỳ vọng 0, không tương quan và phương sai đồng nhất (chưa cần giả định phân phối chuẩn!), OLS đã là **BLUE** (*Best Linear Unbiased Estimator*) — không có bộ ước lượng tuyến tính không chệch nào khác có phương sai nhỏ hơn OLS.

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

---

### Ví dụ tính tay làm từng bước: Giải phương trình chuẩn OLS và kiểm chứng tính trực giao trên ma trận $$3\times 2$$

Xét 3 điểm dữ liệu $$(x_i, y_i)$$: $$(0, 1),\; (1, 2),\; (2, 4)$$. Ta muốn khớp mô hình $$y = \beta_0 + \beta_1 x + \varepsilon$$ bằng công thức ma trận OLS:

1. **Lập ma trận thiết kế $$\mathbf{X}$$ (thêm cột toàn số 1 cho hệ số chặn) và vector $$\mathbf{y}$$:**
   $$\mathbf{X} = \begin{pmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{pmatrix}, \qquad \mathbf{y} = \begin{pmatrix} 1 \\ 2 \\ 4 \end{pmatrix}.$$
2. **Tính $$\mathbf{X}^T\mathbf{X}$$ và $$\mathbf{X}^T\mathbf{y}$$:**
   $$\mathbf{X}^T\mathbf{X} = \begin{pmatrix} 1 & 1 & 1 \\ 0 & 1 & 2 \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{pmatrix} = \begin{pmatrix} 3 & 3 \\ 3 & 5 \end{pmatrix}, \qquad \mathbf{X}^T\mathbf{y} = \begin{pmatrix} 1+2+4 \\ 0+2+8 \end{pmatrix} = \begin{pmatrix} 7 \\ 10 \end{pmatrix}.$$
3. **Nghịch đảo ma trận $$2\times 2$$ và tìm $$\hat{\boldsymbol{\beta}}$$:**
   Định thức $$\det(\mathbf{X}^T\mathbf{X}) = 3\times 5 - 3\times 3 = 6 \neq 0$$, nên
   $$(\mathbf{X}^T\mathbf{X})^{-1} = \frac{1}{6}\begin{pmatrix} 5 & -3 \\ -3 & 3 \end{pmatrix} \implies \hat{\boldsymbol{\beta}} = \frac{1}{6}\begin{pmatrix} 5 & -3 \\ -3 & 3 \end{pmatrix}\begin{pmatrix} 7 \\ 10 \end{pmatrix} = \frac{1}{6}\begin{pmatrix} 5 \\ 9 \end{pmatrix} = \begin{pmatrix} 5/6 \\ 3/2 \end{pmatrix}.$$
   Vậy đường hồi quy là $$\hat{y} = \frac{5}{6} + \frac{3}{2}x \approx 0{,}833 + 1{,}5x$$.
4. **Kiểm chứng trực giao hình học $$\mathbf{X}^T\mathbf{e} = \mathbf{0}$$:**
   Dự báo $$\hat{\mathbf{y}} = (5/6,\; 14/6,\; 23/6)^T$$, suy ra vector phần dư $$\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}} = (1/6,\; -2/6,\; 1/6)^T$$. Kiểm tra tích vô hướng với cả hai cột của $$\mathbf{X}$$:
   - Với cột hệ số chặn $$(1,1,1)^T$$: $$\frac{1}{6} - \frac{2}{6} + \frac{1}{6} = 0$$ (tổng phần dư luôn bằng 0!).
   - Với cột đặc trưng $$(0,1,2)^T$$: $$0\cdot\frac{1}{6} + 1\cdot\left(-\frac{2}{6}\right) + 2\cdot\frac{1}{6} = 0$$ (phần dư trực giao hoàn toàn với $$\mathbf{X}$$!).

### Giải thích: Bốn điều cần ghi nhớ từ cấu trúc ma trận của OLS

1. **Điều kiện tồn tại nghiệm duy nhất nằm ở $$\det(\mathbf{X}^T\mathbf{X}) \neq 0$$:** Nếu hai cột đặc trưng tỷ lệ tuyến tính với nhau (đa cộng tuyến hoàn toàn) hoặc số biến lớn hơn số mẫu ($$p > n$$), ma trận $$\mathbf{X}^T\mathbf{X}$$ suy biến — không thể nghịch đảo được.
2. **Trong thực tế tính toán, không bao giờ đảo $$\mathbf{X}^T\mathbf{X}$$ trực tiếp:** Tính $$\mathbf{X}^T\mathbf{X}$$ bình phương số điều kiện (condition number) của ma trận, dễ gây sai số dấu phẩy động. Các thư viện như NumPy/scikit-learn dùng phân rã **QR** hoặc **SVD** để giải trực tiếp phép chiếu vuông góc ổn định hơn hẳn.
3. **Gauss–Markov chỉ nói về lớp bộ ước lượng *không chệch*:** Nếu ta chấp nhận hy sinh một chút độ chệch (như Ridge/Lasso ở bài sau), ta hoàn toàn có thể đạt MSE nhỏ hơn cả OLS khi đặc trưng nhiều hoặc đa cộng tuyến mạnh!
4. **Đừng quên đọc phần dư:** Ngay cả khi nghiệm ma trận chạy mượt mà, phần dư có cấu trúc phi tuyến hoặc phương sai không đều sẽ báo hiệu mô hình tuyến tính đang thiếu biến hoặc cần biến đổi.

**Kiểm tra nhanh.** Trong ví dụ 3 điểm trên, tổng bình phương phần dư bằng $$\|\mathbf{e}\|_2^2 = (1/6)^2 + (-2/6)^2 + (1/6)^2 = 6/36 = 1/6$$. Hãy kiểm chứng định lý Pythagoras trong $$\mathbb{R}^3$$: liệu $$\|\hat{\mathbf{y}}\|_2^2 + \|\mathbf{e}\|_2^2$$ có bằng đúng $$\|\mathbf{y}\|_2^2 = 1^2 + 2^2 + 4^2 = 21$$ không?

---

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
