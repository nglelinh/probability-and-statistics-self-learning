---
layout: post
title: 01-05-00 Regularization - Ridge Regression và Lasso
chapter: "07"
order: 4
owner: nglelinh
lang: vi
categories:
- chapter07
lesson_type: required
---

Regularization là một trong những ý tưởng quan trọng nhất trong machine learning hiện đại. Khi đối mặt với overfitting hoặc multicollinearity, thay vì loại bỏ features, chúng ta "co" (shrink) các coefficients về phía zero. Bài học này giới thiệu hai phương pháp regularization phổ biến nhất - Ridge (L2) và Lasso (L1) - và giải thích tại sao Lasso có khả năng "kỳ diệu" là tự động chọn features quan trọng.

---

## Vấn Đề với OLS

Ordinary Least Squares có hai vấn đề chính:

**1. Overfitting khi $$p$$ lớn:**
- Khi số features gần bằng số samples ($$p \approx n$$), mô hình có thể fit noise
- Khi $$p > n$$, OLS không có nghiệm duy nhất

**2. Multicollinearity:**
- Khi các features tương quan cao, $$\mathbf{X}^T\mathbf{X}$$ gần singular
- Coefficients có variance rất lớn, không ổn định

**Giải pháp:** Thêm penalty term vào loss function để "co" coefficients.

## Ridge Regression (L2 Regularization)

Ridge regression minimize:

$$\text{Loss}_{\text{Ridge}} = \sum_{i=1}^n (y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij})^2 + \lambda \sum_{j=1}^p \beta_j^2$$

hoặc dạng ma trận:

$$\text{Loss}_{\text{Ridge}} = ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2 + \lambda ||\boldsymbol{\beta}||^2$$

**Lời giải:**

$$\hat{\boldsymbol{\beta}}_{\text{Ridge}} = (\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$$

**Tham số $$\lambda$$:**
- $$\lambda = 0$$: OLS thông thường
- $$\lambda \to \infty$$: Tất cả coefficients $$\to 0$$
- $$\lambda$$ càng lớn, regularization càng mạnh

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

def implement_ridge_from_scratch():
    """
    Lập trình Ridge regression từ đầu
    """
    # Sinh dữ liệu
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=20, n_informative=10, noise=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize (quan trọng cho regularization!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Ridge từ đầu
    def ridge_regression(X, y, lambda_param):
        n, p = X.shape
        I = np.eye(p)
        beta = np.linalg.inv(X.T @ X + lambda_param * I) @ X.T @ y
        return beta
    
    # Thử các giá trị lambda khác nhau
    lambdas = np.logspace(-2, 4, 50)
    coefs = []
    train_errors = []
    test_errors = []
    
    for lam in lambdas:
        beta = ridge_regression(X_train_scaled, y_train, lam)
        coefs.append(beta)
        
        # Tính errors
        train_pred = X_train_scaled @ beta
        test_pred = X_test_scaled @ beta
        train_errors.append(np.mean((y_train - train_pred)**2))
        test_errors.append(np.mean((y_test - test_pred)**2))
    
    coefs = np.array(coefs)
    
    # Vẽ biểu đồ
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Regularization path
    axes[0].plot(np.log10(lambdas), coefs, linewidth=1.5)
    axes[0].set_xlabel('log₁₀(λ)')
    axes[0].set_ylabel('Coefficients')
    axes[0].set_title('Ridge Regularization Path')
    axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[0].grid(True, alpha=0.3)
    
    # Train/Test error
    axes[1].plot(np.log10(lambdas), train_errors, label='Training Error', linewidth=2)
    axes[1].plot(np.log10(lambdas), test_errors, label='Test Error', linewidth=2)
    axes[1].set_xlabel('log₁₀(λ)')
    axes[1].set_ylabel('MSE')
    axes[1].set_title('Training vs Test Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()
    
    # Tìm lambda tối ưu
    optimal_idx = np.argmin(test_errors)
    optimal_lambda = lambdas[optimal_idx]
    print(f"Optimal λ: {optimal_lambda:.4f}")
    print(f"Test MSE at optimal λ: {test_errors[optimal_idx]:.4f}")

# implement_ridge_from_scratch()
```

## Lasso Regression (L1 Regularization)

Lasso minimize:

$$\text{Loss}_{\text{Lasso}} = \sum_{i=1}^n (y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij})^2 + \lambda \sum_{j=1}^p |\beta_j|$$

**Sự khác biệt quan trọng:** L1 penalty ($$|\beta|$$) thay vì L2 ($$\beta^2$$)

**Hệ quả kỳ diệu:** Lasso có thể đưa một số coefficients về **chính xác bằng 0**, tức là tự động loại bỏ features không quan trọng (feature selection)!

**Lý do hình học:**
- L1 ball có các góc nhọn tại các trục
- Khi contour của RSS chạm vào L1 ball, thường chạm tại góc (nơi một số coefficients = 0)
- L2 ball tròn, ít khi chạm tại trục

```python
from sklearn.linear_model import Lasso, LassoCV

def compare_ridge_lasso():
    """
    So sánh Ridge và Lasso
    """
    # Sinh dữ liệu với nhiều features vô dụng
    np.random.seed(42)
    n_samples, n_features = 100, 50
    n_informative = 10
    
    X = np.random.randn(n_samples, n_features)
    # Chỉ 10 features đầu tiên có ích
    true_coef = np.zeros(n_features)
    true_coef[:n_informative] = np.random.randn(n_informative) * 5
    y = X @ true_coef + np.random.randn(n_samples) * 2
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train models với cùng lambda
    lambda_param = 1.0
    ridge = Ridge(alpha=lambda_param)
    lasso = Lasso(alpha=lambda_param)
    
    ridge.fit(X_scaled, y)
    lasso.fit(X_scaled, y)
    
    # Vẽ biểu đồ so sánh coefficients
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    x_pos = np.arange(n_features)
    
    axes[0].bar(x_pos, true_coef, alpha=0.7)
    axes[0].set_xlabel('Feature index')
    axes[0].set_ylabel('Coefficient')
    axes[0].set_title('True Coefficients')
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(x_pos, ridge.coef_, alpha=0.7, color='orange')
    axes[1].set_xlabel('Feature index')
    axes[1].set_ylabel('Coefficient')
    axes[1].set_title(f'Ridge (λ={lambda_param})')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    axes[2].bar(x_pos, lasso.coef_, alpha=0.7, color='green')
    axes[2].set_xlabel('Feature index')
    axes[2].set_ylabel('Coefficient')
    axes[2].set_title(f'Lasso (λ={lambda_param})')
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    # plt.show()
    
    # Đếm số coefficients = 0
    ridge_zeros = np.sum(np.abs(ridge.coef_) < 1e-10)
    lasso_zeros = np.sum(np.abs(lasso.coef_) < 1e-10)
    
    print(f"Số coefficients ≈ 0:")
    print(f"  Ridge: {ridge_zeros}/{n_features}")
    print(f"  Lasso: {lasso_zeros}/{n_features}")
    print(f"\nLasso đã loại bỏ {lasso_zeros} features!")

# compare_ridge_lasso()
```

## Elastic Net

Kết hợp cả L1 và L2:

$$\text{Loss}_{\text{ElasticNet}} = ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2 + \lambda_1 ||\boldsymbol{\beta}||_1 + \lambda_2 ||\boldsymbol{\beta}||^2$$

hoặc dạng sklearn:

$$\text{Loss} = ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2 + \alpha \rho ||\boldsymbol{\beta}||_1 + \frac{\alpha(1-\rho)}{2} ||\boldsymbol{\beta}||^2$$

**Ưu điểm:**
- Kết hợp feature selection của Lasso với stability của Ridge
- Tốt khi có nhóm features tương quan cao (Lasso chỉ chọn 1, Elastic Net chọn cả nhóm)

## Chọn Hyperparameter λ: Cross-Validation

```python
def cross_validation_for_lambda():
    """
    Sử dụng CV để chọn lambda tối ưu
    """
    # Sinh dữ liệu
    np.random.seed(42)
    X, y = make_regression(n_samples=200, n_features=30, n_informative=15, noise=10)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Ridge với CV
    ridge_cv = RidgeCV(alphas=np.logspace(-2, 4, 50), cv=5)
    ridge_cv.fit(X_scaled, y)
    
    # Lasso với CV
    lasso_cv = LassoCV(alphas=np.logspace(-2, 2, 50), cv=5, random_state=42)
    lasso_cv.fit(X_scaled, y)
    
    print("Optimal hyperparameters:")
    print(f"  Ridge λ: {ridge_cv.alpha_:.4f}")
    print(f"  Lasso λ: {lasso_cv.alpha_:.4f}")
    
    # Vẽ CV path cho Lasso
    plt.figure(figsize=(10, 6))
    plt.semilogx(lasso_cv.alphas_, lasso_cv.mse_path_.mean(axis=1), linewidth=2)
    plt.axvline(lasso_cv.alpha_, color='r', linestyle='--', linewidth=2, 
               label=f'Optimal λ = {lasso_cv.alpha_:.4f}')
    plt.xlabel('λ')
    plt.ylabel('Mean Squared Error (CV)')
    plt.title('Lasso Cross-Validation Path')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()

# cross_validation_for_lambda()
```

## Ứng Dụng Thực Tế: Feature Selection

```python
from sklearn.datasets import load_diabetes

def practical_feature_selection():
    """
    Sử dụng Lasso để feature selection trên dataset thực
    """
    # Load data
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    feature_names = diabetes.feature_names
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Lasso với các lambda khác nhau
    alphas = np.logspace(-1, 1, 20)
    coefs = []
    
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X_scaled, y)
        coefs.append(lasso.coef_)
    
    coefs = np.array(coefs)
    
    # Vẽ regularization path
    plt.figure(figsize=(12, 6))
    for i in range(coefs.shape[1]):
        plt.plot(np.log10(alphas), coefs[:, i], label=feature_names[i], linewidth=2)
    
    plt.xlabel('log₁₀(λ)')
    plt.ylabel('Coefficients')
    plt.title('Lasso Regularization Path - Diabetes Dataset')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.show()
    
    # Chọn lambda tối ưu bằng CV
    lasso_cv = LassoCV(cv=5, random_state=42)
    lasso_cv.fit(X_scaled, y)
    
    # Features được chọn
    selected_features = np.abs(lasso_cv.coef_) > 1e-5
    print(f"\nOptimal λ: {lasso_cv.alpha_:.4f}")
    print(f"\nFeatures được chọn ({selected_features.sum()}/{len(feature_names)}):")
    for i, (name, coef) in enumerate(zip(feature_names, lasso_cv.coef_)):
        if abs(coef) > 1e-5:
            print(f"  {name:10s}: {coef:8.4f}")

# practical_feature_selection()
```

## So Sánh Ridge vs Lasso vs Elastic Net

| Đặc điểm | Ridge (L2) | Lasso (L1) | Elastic Net |
|----------|------------|------------|-------------|
| Penalty | $$\sum \beta_j^2$$ | $$\sum \|\beta_j\|$$ | L1 + L2 |
| Coefficients = 0? | Không | Có | Có |
| Feature selection? | Không | Có | Có |
| Grouped features? | Tất cả nhỏ | Chọn 1 | Chọn nhóm |
| Computational | Closed-form | Iterative | Iterative |

**Khi nào dùng gì?**
- **Ridge:** Khi tất cả features đều có ích, chỉ cần giảm overfitting
- **Lasso:** Khi nghi ngờ nhiều features vô dụng, cần feature selection
- **Elastic Net:** Khi có nhóm features tương quan cao, hoặc $$p > n$$

## Bài Tập Thực Hành

**Bài 1: Implement Lasso với Coordinate Descent**
Lasso không có closed-form solution. Implement coordinate descent algorithm để giải Lasso.

**Bài 2: Regularization Path**
Vẽ regularization path cho Ridge và Lasso trên cùng dataset. Quan sát sự khác biệt.

**Bài 3: High-Dimensional Regression**
Tạo dataset với $$p = 100, n = 50$$. So sánh hiệu suất của OLS (sẽ fail), Ridge, Lasso, và Elastic Net.
