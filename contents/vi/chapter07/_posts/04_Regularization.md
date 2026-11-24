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

Hãy tưởng tượng bạn xây mô hình dự đoán giá nhà bằng OLS và vô tình đưa vào đồng thời hai biến đo gần như cùng một thứ: diện tích sàn tính bằng $$\text{m}^2$$ ($$X_1$$) và diện tích sàn tính bằng $$\text{ft}^2$$ cộng thêm chút sai số đo ($$X_2$$). Khi giải phương trình OLS, ma trận $$\mathbf{X}^T\mathbf{X}$$ trở nên gần suy biến và các hệ số bỗng nổ tung theo hai hướng ngược nhau: chẳng hạn $$\hat{\beta}_1 = +5{,}000$$ và $$\hat{\beta}_2 = -464$$! Trên tập huấn luyện chúng triệt tiêu nhau rất khéo, nhưng chỉ cần một sai số đo nhỏ trên căn nhà mới, dự báo sẽ văng lệch hàng tỷ đồng.

**Điều chuẩn (Regularization) — Ridge (L2), Lasso (L1) và Elastic Net** giải quyết triệt để hiện tượng quá khớp và đa cộng tuyến bằng một ý tưởng kinh tế cực kỳ sâu sắc: **đánh thuế lên độ lớn của các hệ số**. Chấp nhận hy sinh một chút độ chệch (bias) để kéo tụt phương sai (variance) xuống — đó là minh họa sống động nhất cho nguyên lý *Bias–Variance Trade-off* trong học máy hiện đại.

---

## Bản đồ kiến thức nền: Bốn góc nhìn giúp hiểu thấu Ridge, Lasso và Elastic Net

Tại sao chỉ đổi bình phương $$\beta_j^2$$ (L2) thành trị tuyệt đối $$\lvert\beta_j\rvert$$ (L1) lại tạo ra khác biệt một trời một vực về khả năng chọn lọc đặc trưng? Hãy nhìn qua bốn lăng kính:

1. **Tối ưu có ràng buộc & Hình học hình cầu L2 vs. Hình thoi L1:** Thêm hình phạt $$\lambda\|\boldsymbol{\beta}\|_2^2$$ tương đương ràng buộc $$\boldsymbol{\beta}$$ nằm trong một **hình cầu tròn trơn**; đường đồng mức hình elip của hàm mất mát OLS thường tiếp xúc tại điểm có các tọa độ *khác 0*. Ngược lại, hình phạt $$\lambda\|\boldsymbol{\beta}\|_1$$ tương đương ràng buộc trong một **hình thoi có góc nhọn nằm đúng trên các trục tọa độ** — nơi các hệ số bị ép về **đúng bằng 0**!
2. **Nghịch đảo ổn định ma trận trong Ridge:** Nghiệm dạng đóng của Ridge là $$\hat{\boldsymbol{\beta}}_{\text{Ridge}} = (\mathbf{X}^T\mathbf{X} + \lambda I)^{-1}\mathbf{X}^T\mathbf{y}$$. Cộng thêm $$\lambda I$$ đẩy mọi giá trị riêng của $$\mathbf{X}^T\mathbf{X}$$ tăng thêm $$\lambda > 0$$, giúp ma trận luôn khả nghịch kể cả khi $$p > n$$ hay đa cộng tuyến hoàn toàn.
3. **Diễn giải Bayes (MAP Estimation):** Ridge hoàn toàn tương đương với ước lượng hậu nghiệm cực đại (MAP) khi đặt tiên nghiệm **Gauss** $$\beta_j \sim \mathcal{N}(0, \tau^2)$$; còn Lasso tương đương với tiên nghiệm **Laplace** (đỉnh nhọn tại 0, đuôi dày hơn Gauss).
4. **Chuẩn hóa đặc trưng (Scaling) & Cross-Validation:** Vì hình phạt đánh trực tiếp lên độ lớn số học của $$\beta_j$$, bạn **bắt buộc phải chuẩn hóa** mọi biến về cùng thang đo (trung bình 0, độ lệch chuẩn 1) trước khi chạy Ridge/Lasso (và không phạt hệ số chặn $$\beta_0$$); siêu tham số $$\lambda$$ được chọn bằng kiểm chứng chéo.

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
| Số hạng phạt | $$\sum \beta_j^2$$ | $$\sum \vert\beta_j\vert$$ | L1 + L2 |
| Đưa hệ số về đúng 0 | Không | Có | Có |
| Chọn lọc đặc trưng | Không | Có | Có |
| Nhóm đặc trưng tương quan | Co đều về gần 0 | Chọn 1 biến | Giữ cả nhóm |
| Cách tìm nghiệm | Dạng đóng | Lặp (iterative) | Lặp (iterative) |

**Gợi ý lựa chọn:**
- **Ridge:** Phù hợp khi hầu hết các đặc trưng đều đóng góp thông tin và mục tiêu chính là giảm quá khớp.
- **Lasso:** Phù hợp khi nhiều đặc trưng có thể không hữu ích và cần mô hình thưa để chọn lọc biến.
- **Elastic Net:** Phù hợp khi có các nhóm đặc trưng tương quan mạnh hoặc khi $$p > n$$.

---

### Ví dụ tính tay cầm tay chỉ việc: Tận mắt thấy Lasso ép nhiễu về 0 còn Ridge thì không

Xét trường hợp trực giao chuẩn hóa ($$\mathbf{X}^T\mathbf{X} = I$$) với 2 đặc trưng độc lập:
- Đặc trưng 1 là **tín hiệu mạnh** có ước lượng OLS $$\hat{\beta}_1^{\text{OLS}} = 3{,}0$$.
- Đặc trưng 2 là **nhiễu yếu** có ước lượng OLS $$\hat{\beta}_2^{\text{OLS}} = 0{,}4$$.

Với hàm mục tiêu chuẩn hóa $$\frac{1}{2}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \text{phạt}$$ và chọn $$\lambda = 1{,}0$$:

1. **Hồi quy Ridge** (phạt $$\frac{\lambda}{2}\sum \beta_j^2$$, đạo hàm bằng 0 cho nghiệm đóng $$\hat{\beta}_j^{\text{Ridge}} = \frac{\hat{\beta}_j^{\text{OLS}}}{1 + \lambda}$$):
   - $$\hat{\beta}_1^{\text{Ridge}} = \frac{3{,}0}{1 + 1} = 1{,}50.$$
   - $$\hat{\beta}_2^{\text{Ridge}} = \frac{0{,}4}{1 + 1} = 0{,}20 \neq 0.$$
   *(Ridge co tỷ lệ tất cả hệ số xuống 50%, nhưng không bao giờ loại bỏ hẳn biến nhiễu!)*

2. **Hồi quy Lasso** (phạt $$\lambda\sum \lvert\beta_j\rvert$$, nghiệm ngưỡng mềm *soft-thresholding* $$\hat{\beta}_j^{\text{Lasso}} = \mathrm{sign}(z_j)\max(\lvert z_j\rvert - \lambda,\, 0)$$ với $$z_j = \hat{\beta}_j^{\text{OLS}}$$):
   - $$\hat{\beta}_1^{\text{Lasso}} = \max(3{,}0 - 1{,}0,\, 0) = 2{,}00.$$
   - $$\hat{\beta}_2^{\text{Lasso}} = \max(0{,}4 - 1{,}0,\, 0) = \mathbf{0}!$$

Chỉ một dòng phép trừ đã lột tả trọn vẹn sự khác biệt: **Ridge co theo tỷ lệ** (chia cho $$1+\lambda$$), còn **Lasso trừ đi một lượng cố định $$\lambda$$ và cắt phẳng về 0** mọi hệ số yếu hơn ngưỡng $$\lambda$$!

### Khung giải thích: Bốn câu hỏi giúp chọn đúng giữa OLS, Ridge, Lasso và Elastic Net (nên đọc chậm)

1. **Đã chuẩn hóa đặc trưng chưa?** Nếu biến $$X_1$$ đo bằng km còn $$X_2$$ đo bằng mét, hệ số $$\beta_1$$ sẽ lớn gấp 1.000 lần $$\beta_2$$ và bị Ridge/Lasso phạt oan nặng gấp triệu lần. Luôn đưa bước `StandardScaler` vào bên trong *Pipeline* (chỉ fit trên tập train!).
2. **Bài toán của bạn là *thưa* (sparse) hay *dày* (dense)?** Nếu tin rằng chỉ có một nhóm nhỏ đặc trưng thực sự quan trọng trong hàng trăm biến (ví dụ chọn gene), dùng **Lasso**. Nếu hầu hết các biến đều đóng góp một chút và có tương quan chặt với nhau, **Ridge** thường dự báo tốt hơn.
3. **Khi nhiều biến quan trọng tương quan mạnh với nhau:** Lasso có xu hướng chọn ngẫu nhiên 1 biến và đẩy các biến anh em về 0. **Elastic Net** (trộn cả L1 và L2) vừa giữ được tính thưa, vừa giữ trọn nhóm biến tương quan cùng nhau.
4. **Cẩn thận khi suy diễn p-value sau Lasso:** Chọn biến bằng Lasso rồi chạy lại OLS lấy p-value trên cùng dữ liệu sẽ tạo ra p-value quá lạc quan (lỗi *post-selection inference*). Hãy dùng tập kiểm tra tách biệt để đánh giá công bằng.

**Kiểm tra nhanh.** Trong ví dụ trực giao trên ($$\hat{\beta}_1^{\text{OLS}} = 3{,}0$$, $$\hat{\beta}_2^{\text{OLS}} = 0{,}4$$), nếu ta tăng hệ số phạt lên $$\lambda = 4{,}0$$, hai hệ số của **Ridge** và **Lasso** sẽ lần lượt bằng bao nhiêu?

---

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
