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

Điều chuẩn (regularization) là một trong những kỹ thuật quan trọng trong học máy hiện đại. Khi mô hình đối mặt với quá khớp (overfitting) hoặc đa cộng tuyến (multicollinearity), thay vì loại bỏ thủ công các đặc trưng, ta "co" (shrink) các hệ số hồi quy về gần 0. Bài học này giới thiệu hai phương pháp điều chuẩn phổ biến là Ridge (L2) và Lasso (L1), đồng thời giải thích cơ chế giúp Lasso tự động chọn lọc đặc trưng.

---

## Hạn Chế Của OLS

Phương pháp bình phương tối thiểu thông thường (OLS) gặp hai vấn đề chính:

**1. Quá khớp khi số đặc trưng $$p$$ lớn:**
- Khi số đặc trưng xấp xỉ số mẫu ($$p \approx n$$), mô hình dễ khớp cả nhiễu trong dữ liệu huấn luyện.
- Khi $$p > n$$, bài toán OLS không còn nghiệm duy nhất.

**2. Đa cộng tuyến (Multicollinearity):**
- Khi các đặc trưng tương quan mạnh với nhau, ma trận $$\mathbf{X}^T\mathbf{X}$$ gần suy biến (near-singular).
- Các hệ số ước lượng có phương sai rất lớn và kém ổn định.

**Giải pháp:** Bổ sung một số hạng phạt (penalty term) vào hàm mất mát để thu nhỏ độ lớn của các hệ số.

## Hồi Quy Ridge (Điều Chuẩn L2)

Hồi quy Ridge tối thiểu hóa hàm mất mát:

$$\text{Loss}_{\text{Ridge}} = \sum_{i=1}^n (y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij})^2 + \lambda \sum_{j=1}^p \beta_j^2$$

Khi dữ liệu đã được căn giữa (không phạt hệ số chặn $$\beta_0$$), hàm mất mát viết dưới dạng ma trận là:

$$\text{Loss}_{\text{Ridge}} = ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2 + \lambda ||\boldsymbol{\beta}||^2$$

**Nghiệm dạng đóng:**

$$\hat{\boldsymbol{\beta}}_{\text{Ridge}} = (\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$$

**Vai trò của tham số $$\lambda$$:**
- $$\lambda = 0$$: Quay về hồi quy OLS thông thường.
- $$\lambda \to \infty$$: Tất cả các hệ số hồi quy tiến về 0.
- $$\lambda$$ càng lớn, mức độ điều chuẩn càng mạnh.

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

## Hồi Quy Lasso (Điều Chuẩn L1)

Hồi quy Lasso tối thiểu hóa hàm mất mát:

$$\text{Loss}_{\text{Lasso}} = \sum_{i=1}^n (y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij})^2 + \lambda \sum_{j=1}^p |\beta_j|$$

**Điểm khác biệt chính:** Lasso sử dụng chuẩn L1 ($$|\beta_j|$$) thay cho chuẩn L2 ($$\beta_j^2$$).

**Đặc điểm quan trọng:** Lasso có thể ép một số hệ số về **chính xác bằng 0**, tức là tự động loại bỏ các đặc trưng ít ảnh hưởng (feature selection).

**Giải thích hình học:**
- Miền ràng buộc L1 có các đỉnh nhọn nằm trên các trục tọa độ.
- Khi các đường đồng mức của RSS mở rộng và tiếp xúc với miền ràng buộc L1, điểm tiếp xúc thường rơi vào các đỉnh này — nơi một số hệ số bằng 0.
- Ngược lại, miền ràng buộc L2 có dạng hình tròn (hoặc hình cầu) trơn nên hiếm khi tiếp xúc đúng tại các trục tọa độ.

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

Elastic Net kết hợp cả hai thành phần phạt L1 và L2:

$$\text{Loss}_{\text{ElasticNet}} = ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2 + \lambda_1 ||\boldsymbol{\beta}||_1 + \lambda_2 ||\boldsymbol{\beta}||^2$$

hoặc theo quy ước cài đặt trong `scikit-learn`:

$$\text{Loss} = \frac{1}{2n}||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2 + \alpha \rho ||\boldsymbol{\beta}||_1 + \frac{\alpha(1-\rho)}{2} ||\boldsymbol{\beta}||^2$$

**Ưu điểm:**
- Kết hợp khả năng chọn lọc đặc trưng của Lasso với tính ổn định của Ridge.
- Hoạt động hiệu quả khi có các nhóm đặc trưng tương quan cao (Lasso thường chỉ chọn một biến đại diện, trong khi Elastic Net có thể giữ lại cả nhóm).

## Chọn Siêu Tham Số $$\lambda$$ Bằng Kiểm Chứng Chéo (Cross-Validation)

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

## Ứng Dụng Thực Tế: Chọn Lọc Đặc Trưng (Feature Selection)

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

## So Sánh Ridge, Lasso và Elastic Net

| Đặc điểm | Ridge (L2) | Lasso (L1) | Elastic Net |
|----------|------------|------------|-------------|
| Số hạng phạt | $$\sum \beta_j^2$$ | $$\sum \|\beta_j\|$$ | L1 + L2 |
| Đưa hệ số về đúng 0 | Không | Có | Có |
| Chọn lọc đặc trưng | Không | Có | Có |
| Nhóm đặc trưng tương quan | Co đều về gần 0 | Chọn 1 biến | Giữ cả nhóm |
| Cách tìm nghiệm | Dạng đóng | Lặp (iterative) | Lặp (iterative) |

**Gợi ý lựa chọn:**
- **Ridge:** Phù hợp khi hầu hết các đặc trưng đều đóng góp thông tin và mục tiêu chính là giảm quá khớp.
- **Lasso:** Phù hợp khi nhiều đặc trưng có thể không hữu ích và cần mô hình thưa để chọn lọc biến.
- **Elastic Net:** Phù hợp khi có các nhóm đặc trưng tương quan mạnh hoặc khi $$p > n$$.

## Video tham khảo

- [Regularization Part 1: Ridge (L2) Regression](https://www.youtube.com/watch?v=Q81RR3yKn30) — StatQuest with Josh Starmer, tiếng Anh. **Xem trước.** Giải thích hồi quy Ridge (cộng $$\lambda$$ lần tổng bình phương hệ số vào tổng bình phương phần dư) và cách chọn $$\lambda$$ bằng kiểm chứng chéo.
- [Regularization Part 2: Lasso (L1) Regression](https://www.youtube.com/watch?v=NGf0voTMlcs) — StatQuest with Josh Starmer, tiếng Anh. Trình bày cách phạt trị tuyệt đối đưa một số hệ số về đúng 0 để chọn lọc đặc trưng.
- [Regularization Part 3: Elastic Net Regression](https://www.youtube.com/watch?v=1dKRdX9bfIo) — StatQuest with Josh Starmer, tiếng Anh. Giới thiệu Elastic Net kết hợp đồng thời hai số hạng phạt L1 và L2.

## Bài Tập Thực Hành

**Bài 1: Cài đặt Lasso bằng Coordinate Descent**
Vì Lasso không có nghiệm dạng đóng, hãy cài đặt thuật toán hạ tọa độ (coordinate descent) để giải bài toán tối ưu Lasso.

**Bài 2: Đường điều chuẩn (Regularization Path)**
Vẽ đường biến thiên hệ số theo $$\lambda$$ của Ridge và Lasso trên cùng một tập dữ liệu, sau đó so sánh sự khác biệt.

**Bài 3: Hồi quy dữ liệu chiều cao (High-Dimensional Regression)**
Tạo tập dữ liệu với $$p = 100, n = 50$$ rồi so sánh hiệu suất giữa OLS (không có nghiệm duy nhất), Ridge, Lasso và Elastic Net.
