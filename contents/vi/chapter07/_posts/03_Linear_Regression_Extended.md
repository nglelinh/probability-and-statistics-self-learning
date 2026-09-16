---
layout: post
title: 01-04-00 Hồi Quy Tuyến Tính và Phương Pháp Bình Phương Tối Thiểu
chapter: "07"
order: 3
owner: nglelinh
lang: vi
categories:
- chapter07
lesson_type: required
---

Hồi quy tuyến tính là nền tảng của statistical learning. Mặc dù đơn giản, nó chứa đựng hầu hết các ý tưởng quan trọng mà chúng ta sẽ gặp trong các mô hình phức tạp hơn: loss function, optimization, inference, và model diagnostics. Trong bài học này, chúng ta sẽ không chỉ học công thức mà còn hiểu sâu về ý nghĩa hình học, các giả định, và cách đánh giá chất lượng mô hình thông qua phân tích residuals.

---

## Mô Hình Hồi Quy Tuyến Tính

Giả sử chúng ta có $$n$$ quan sát với $$p$$ features. Mô hình tuyến tính giả định:

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p + \epsilon$$

hoặc dạng ma trận:

$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$

trong đó:
- $$\mathbf{y}$$: vector $$n \times 1$$ của target values
- $$\mathbf{X}$$: ma trận $$n \times (p+1)$$ của features (cột đầu tiên là 1 cho intercept)
- $$\boldsymbol{\beta}$$: vector $$(p+1) \times 1$$ của coefficients
- $$\boldsymbol{\epsilon}$$: vector $$n \times 1$$ của errors, giả định $$\epsilon_i \sim \mathcal{N}(0, \sigma^2)$$ độc lập

## Ordinary Least Squares (OLS)

**Ý tưởng:** Tìm $$\boldsymbol{\beta}$$ để minimize tổng bình phương residuals (Residual Sum of Squares - RSS):

$$\text{RSS}(\boldsymbol{\beta}) = \sum_{i=1}^n (y_i - \mathbf{x}_i^T\boldsymbol{\beta})^2 = ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2$$

**Lời giải giải tích:**

Đạo hàm theo $$\boldsymbol{\beta}$$ và cho bằng 0:

$$\frac{\partial \text{RSS}}{\partial \boldsymbol{\beta}} = -2\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) = 0$$

Giải ra:

$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

Đây là **normal equation**, công thức closed-form cho OLS.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def implement_ols_from_scratch():
    """
    Lập trình OLS từ đầu và so sánh với sklearn
    """
    # Sinh dữ liệu
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=1, noise=10)
    
    # Thêm cột intercept
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    
    # OLS từ đầu: β = (X'X)^(-1) X'y
    beta_ols = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
    y_pred_ols = X_with_intercept @ beta_ols
    
    # So sánh với sklearn
    model_sklearn = LinearRegression()
    model_sklearn.fit(X, y)
    y_pred_sklearn = model_sklearn.predict(X)
    
    # Vẽ biểu đồ
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.6, s=50, label='Data')
    plt.plot(X, y_pred_ols, 'r-', linewidth=2, label='OLS từ đầu')
    plt.plot(X, y_pred_sklearn, 'g--', linewidth=2, label='sklearn')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('So Sánh OLS Implementation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Residuals
    residuals = y - y_pred_ols
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred_ols, residuals, alpha=0.6, s=50)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()
    
    print("Coefficients từ OLS:")
    print(f"  Intercept: {beta_ols[0]:.4f}")
    print(f"  Slope:     {beta_ols[1]:.4f}")
    print("\nCoefficients từ sklearn:")
    print(f"  Intercept: {model_sklearn.intercept_:.4f}")
    print(f"  Slope:     {model_sklearn.coef_[0]:.4f}")
    print(f"\nR²: {r2_score(y, y_pred_ols):.4f}")
    print(f"MSE: {mean_squared_error(y, y_pred_ols):.4f}")

# implement_ols_from_scratch()
```

## Ý Nghĩa Hình Học

OLS có thể hiểu theo góc nhìn hình học đẹp đẽ:

**Projection Interpretation:** 
- $$\mathbf{y}$$ là một vector trong không gian $$\mathbb{R}^n$$
- $$\mathbf{X}$$ định nghĩa một không gian con (column space) của $$\mathbb{R}^n$$
- $$\hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}}$$ là **hình chiếu vuông góc** của $$\mathbf{y}$$ lên column space của $$\mathbf{X}$$
- Residual vector $$\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$$ vuông góc với column space

**Hat Matrix:**

$$\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$$

$$\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$$

Ma trận $$\mathbf{H}$$ được gọi là "hat matrix" vì nó "đội mũ" lên $$\mathbf{y}$$ để tạo ra $$\hat{\mathbf{y}}$$.

## Định Lý Gauss-Markov

Dưới các giả định:
1. $$E[\boldsymbol{\epsilon}] = 0$$
2. $$\text{Var}(\boldsymbol{\epsilon}) = \sigma^2 \mathbf{I}$$ (homoscedasticity)
3. Errors không tương quan

Thì **OLS estimator là Best Linear Unbiased Estimator (BLUE)**:
- **Unbiased:** $$E[\hat{\boldsymbol{\beta}}] = \boldsymbol{\beta}$$
- **Best:** Có variance nhỏ nhất trong tất cả linear unbiased estimators

```python
def verify_gauss_markov(n_simulations=1000):
    """
    Kiểm chứng tính unbiased của OLS estimator
    """
    true_beta = np.array([2.0, 3.5])  # [intercept, slope]
    n_samples = 50
    
    beta_estimates = []
    
    for _ in range(n_simulations):
        # Sinh dữ liệu
        X = np.random.uniform(0, 10, n_samples)
        epsilon = np.random.normal(0, 2, n_samples)
        y = true_beta[0] + true_beta[1] * X + epsilon
        
        # Ước lượng OLS
        X_with_intercept = np.column_stack([np.ones(n_samples), X])
        beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
        beta_estimates.append(beta_hat)
    
    beta_estimates = np.array(beta_estimates)
    
    # Vẽ biểu đồ
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for i, param_name in enumerate(['Intercept', 'Slope']):
        axes[i].hist(beta_estimates[:, i], bins=50, density=True, alpha=0.7)
        axes[i].axvline(true_beta[i], color='red', linestyle='--', linewidth=2, 
                       label=f'True {param_name} = {true_beta[i]}')
        axes[i].axvline(beta_estimates[:, i].mean(), color='green', linestyle='--', linewidth=2,
                       label=f'Mean estimate = {beta_estimates[:, i].mean():.4f}')
        axes[i].set_xlabel(param_name)
        axes[i].set_ylabel('Density')
        axes[i].set_title(f'Phân Phối của {param_name} Estimator')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()
    
    print("Kiểm chứng Unbiased Property:")
    print(f"True β₀: {true_beta[0]:.4f}, Mean estimate: {beta_estimates[:, 0].mean():.4f}")
    print(f"True β₁: {true_beta[1]:.4f}, Mean estimate: {beta_estimates[:, 1].mean():.4f}")

# verify_gauss_markov()
```

## Đánh Giá Mô Hình

### R² (Coefficient of Determination)

$$R^2 = 1 - \frac{\text{RSS}}{\text{TSS}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

- $$R^2 \in [0, 1]$$
- $$R^2 = 1$$: Mô hình fit hoàn hảo
- $$R^2 = 0$$: Mô hình không tốt hơn việc dự đoán bằng mean

**Vấn đề:** $$R^2$$ luôn tăng khi thêm features, ngay cả khi features đó vô dụng!

### Adjusted R²

$$R^2_{\text{adj}} = 1 - \frac{\text{RSS}/(n-p-1)}{\text{TSS}/(n-1)}$$

Điều chỉnh cho số lượng parameters, phạt mô hình phức tạp.

### Residual Standard Error (RSE)

$$\text{RSE} = \sqrt{\frac{\text{RSS}}{n-p-1}}$$

Ước lượng cho $$\sigma$$ (standard deviation của errors).

```python
from sklearn.datasets import load_diabetes

def comprehensive_model_evaluation():
    """
    Đánh giá toàn diện mô hình linear regression
    """
    # Load dataset
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Tính các metrics
    n, p = X.shape
    rss = np.sum((y - y_pred)**2)
    tss = np.sum((y - y.mean())**2)
    r2 = 1 - rss/tss
    adj_r2 = 1 - (rss/(n-p-1))/(tss/(n-1))
    rse = np.sqrt(rss/(n-p-1))
    
    print("Model Evaluation Metrics:")
    print(f"  R²:          {r2:.4f}")
    print(f"  Adjusted R²: {adj_r2:.4f}")
    print(f"  RSE:         {rse:.4f}")
    print(f"  MSE:         {mean_squared_error(y, y_pred):.4f}")
    
    # Diagnostic plots
    residuals = y - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Residuals vs Fitted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6, s=30)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Fitted values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Q-Q plot
    from scipy import stats as sp_stats
    sp_stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Scale-Location
    standardized_residuals = residuals / residuals.std()
    axes[1, 0].scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), alpha=0.6, s=30)
    axes[1, 0].set_xlabel('Fitted values')
    axes[1, 0].set_ylabel('√|Standardized residuals|')
    axes[1, 0].set_title('Scale-Location')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Residuals histogram
    axes[1, 1].hist(residuals, bins=30, density=True, alpha=0.7)
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Histogram of Residuals')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()

# comprehensive_model_evaluation()
```

## Các Giả Định của Linear Regression

1. **Linearity:** Mối quan hệ giữa $$X$$ và $$Y$$ là tuyến tính
2. **Independence:** Các quan sát độc lập
3. **Homoscedasticity:** Variance của errors không đổi
4. **Normality:** Errors tuân theo phân phối Chuẩn
5. **No multicollinearity:** Các features không tương quan cao với nhau

**Kiểm tra giả định:**
- Linearity: Residual plot
- Homoscedasticity: Scale-location plot
- Normality: Q-Q plot, Shapiro-Wilk test
- Multicollinearity: VIF (Variance Inflation Factor)

## Bài Tập Thực Hành

**Bài 1: Implement OLS với QR Decomposition**
Thay vì dùng $$(X^TX)^{-1}$$, sử dụng QR decomposition để giải OLS (numerically stable hơn).

**Bài 2: Multicollinearity Detection**
Tải dataset với features tương quan cao. Tính VIF và quan sát ảnh hưởng lên coefficients.

**Bài 3: Polynomial Regression**
Implement polynomial regression bằng cách tạo polynomial features. So sánh với sklearn's PolynomialFeatures.
