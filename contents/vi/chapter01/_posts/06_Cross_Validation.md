---
layout: post
title: 01-07-00 Cross-Validation và Resampling Methods
chapter: "01"
order: 7
owner: nglelinh
lang: vi
categories:
- chapter01
lesson_type: required
---

Model selection là một trong những thách thức lớn nhất trong machine learning: Làm thế nào để chọn độ phức tạp mô hình phù hợp? Làm sao biết mô hình sẽ hoạt động tốt trên dữ liệu mới? Cross-validation và các phương pháp resampling cung cấp câu trả lời thực tế cho những câu hỏi này. Thay vì dựa vào các công thức lý thuyết, chúng ta sử dụng chính dữ liệu để ước lượng test error một cách đáng tin cậy.

---

## Vấn Đề: Training Error vs Test Error

Chúng ta đã biết training error không phải là thước đo tốt cho hiệu suất thực tế:
- Training error luôn giảm khi mô hình phức tạp hơn
- Test error (trên dữ liệu mới) mới là thứ chúng ta quan tâm

**Vấn đề:** Chúng ta không có test set vô hạn để đánh giá mọi mô hình!

**Giải pháp:** Sử dụng dữ liệu huấn luyện một cách thông minh để **ước lượng** test error.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression

def demonstrate_train_test_split():
    """
    Minh họa cách chia train/test set cơ bản
    """
    # Sinh dữ liệu
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=1, noise=10)
    
    # Chia 70-30
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Thử các độ phức tạp khác nhau
    degrees = range(1, 16)
    train_errors = []
    test_errors = []
    
    for degree in degrees:
        model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=0.1))
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_errors.append(np.mean((y_train - train_pred)**2))
        test_errors.append(np.mean((y_test - test_pred)**2))
    
    # Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, train_errors, 'o-', label='Training Error', linewidth=2)
    plt.plot(degrees, test_errors, 's-', label='Test Error', linewidth=2)
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error')
    plt.title('Training vs Test Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()
    
    optimal_degree = degrees[np.argmin(test_errors)]
    print(f"Optimal degree: {optimal_degree}")
    print(f"Test error at optimal degree: {min(test_errors):.4f}")

# demonstrate_train_test_split()
```

## K-Fold Cross-Validation

**Ý tưởng:** Thay vì chia một lần, chia dữ liệu thành $$K$$ folds (thường $$K=5$$ hoặc $$K=10$$):

1. Chia dữ liệu thành $$K$$ phần bằng nhau
2. Lặp $$K$$ lần:
   - Dùng fold thứ $$k$$ làm validation set
   - Dùng $$K-1$$ folds còn lại làm training set
   - Tính validation error
3. CV error = trung bình của $$K$$ validation errors

$$\text{CV}_{(K)} = \frac{1}{K}\sum_{k=1}^K \text{MSE}_k$$

**Ưu điểm:**
- Sử dụng hiệu quả dữ liệu (mỗi điểm được dùng cả train và validation)
- Ước lượng ổn định hơn single train/test split
- Giảm variance của ước lượng

```python
def k_fold_cross_validation_demo():
    """
    Implement và minh họa K-Fold CV
    """
    # Sinh dữ liệu
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=5, noise=10)
    
    # Thử các giá trị lambda khác nhau cho Ridge
    alphas = np.logspace(-2, 3, 20)
    
    # K-Fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        # cross_val_score tự động thực hiện K-fold CV
        scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
        cv_scores.append(-scores.mean())  # Chuyển về MSE (dương)
    
    # So sánh với single train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    single_split_scores = []
    
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        single_split_scores.append(np.mean((y_test - pred)**2))
    
    # Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    plt.semilogx(alphas, cv_scores, 'o-', label='5-Fold CV', linewidth=2, markersize=6)
    plt.semilogx(alphas, single_split_scores, 's--', label='Single Train/Test Split', 
                linewidth=2, markersize=6, alpha=0.7)
    plt.xlabel('Regularization parameter (α)')
    plt.ylabel('Mean Squared Error')
    plt.title('K-Fold CV vs Single Split')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()
    
    optimal_alpha_cv = alphas[np.argmin(cv_scores)]
    optimal_alpha_single = alphas[np.argmin(single_split_scores)]
    
    print(f"Optimal α (CV): {optimal_alpha_cv:.4f}")
    print(f"Optimal α (Single split): {optimal_alpha_single:.4f}")

# k_fold_cross_validation_demo()
```

## Leave-One-Out Cross-Validation (LOOCV)

Trường hợp đặc biệt của K-Fold khi $$K = n$$:
- Mỗi lần, dùng 1 điểm làm validation, $$n-1$$ điểm làm training
- Lặp $$n$$ lần

$$\text{CV}_{(n)} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$

**Ưu điểm:**
- Sử dụng tối đa dữ liệu (training set có $$n-1$$ điểm)
- Ước lượng gần như unbiased cho test error

**Nhược điểm:**
- Tốn kém tính toán (phải train $$n$$ lần)
- Variance cao (các training sets rất giống nhau)

**Khi nào dùng LOOCV?**
- Khi $$n$$ nhỏ (< 100)
- Khi training nhanh (linear models)

```python
from sklearn.model_selection import LeaveOneOut

def compare_cv_methods():
    """
    So sánh K-Fold và LOOCV
    """
    # Sinh dữ liệu nhỏ
    np.random.seed(42)
    X, y = make_regression(n_samples=50, n_features=3, noise=5)
    
    alphas = np.logspace(-1, 2, 15)
    
    # 5-Fold CV
    kfold_scores = []
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        kfold_scores.append(-scores.mean())
    
    # LOOCV
    loo = LeaveOneOut()
    loocv_scores = []
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        scores = cross_val_score(model, X, y, cv=loo, scoring='neg_mean_squared_error')
        loocv_scores.append(-scores.mean())
    
    # Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    plt.semilogx(alphas, kfold_scores, 'o-', label='5-Fold CV', linewidth=2)
    plt.semilogx(alphas, loocv_scores, 's-', label='LOOCV', linewidth=2)
    plt.xlabel('α')
    plt.ylabel('CV Error')
    plt.title('5-Fold CV vs LOOCV')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()
    
    print(f"Optimal α (5-Fold): {alphas[np.argmin(kfold_scores)]:.4f}")
    print(f"Optimal α (LOOCV):  {alphas[np.argmin(loocv_scores)]:.4f}")

# compare_cv_methods()
```

## Stratified K-Fold (Cho Classification)

Khi làm classification với classes imbalanced, cần đảm bảo mỗi fold có tỷ lệ classes giống nhau.

**Stratified K-Fold:** Chia dữ liệu sao cho tỷ lệ classes trong mỗi fold giống với tổng thể.

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

def stratified_cv_demo():
    """
    Minh họa tầm quan trọng của Stratified CV
    """
    # Sinh dữ liệu imbalanced (20% class 1, 80% class 0)
    X, y = make_classification(n_samples=100, n_features=5, n_informative=3,
                               n_redundant=0, weights=[0.8, 0.2], random_state=42)
    
    print(f"Tỷ lệ class 1 trong toàn bộ dữ liệu: {y.mean():.2%}")
    print()
    
    # Regular K-Fold
    print("Regular K-Fold:")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"  Fold {i+1}: {y[val_idx].mean():.2%} class 1")
    
    print("\nStratified K-Fold:")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"  Fold {i+1}: {y[val_idx].mean():.2%} class 1")
    
    # So sánh hiệu suất
    model = LogisticRegression()
    
    regular_scores = cross_val_score(model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42))
    stratified_scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
    
    print(f"\nRegular CV accuracy: {regular_scores.mean():.4f} ± {regular_scores.std():.4f}")
    print(f"Stratified CV accuracy: {stratified_scores.mean():.4f} ± {stratified_scores.std():.4f}")

# stratified_cv_demo()
```

## Bootstrap

Bootstrap là phương pháp resampling mạnh mẽ khác, đặc biệt hữu ích để ước lượng uncertainty.

**Thuật toán:**
1. Từ dataset gốc có $$n$$ điểm, lấy mẫu **có hoàn lại** $$n$$ điểm → bootstrap sample
2. Train mô hình trên bootstrap sample
3. Test trên các điểm **không** xuất hiện trong bootstrap sample (out-of-bag)
4. Lặp lại $$B$$ lần (thường $$B = 100$$ hoặc $$1000$$)

**Bootstrap estimate of test error:**

$$\text{Err}_{\text{boot}} = \frac{1}{B}\sum_{b=1}^B \frac{1}{|C^{-b}|}\sum_{i \in C^{-b}} L(y_i, \hat{f}^{*b}(x_i))$$

trong đó $$C^{-b}$$ là tập các điểm không có trong bootstrap sample thứ $$b$$.

```python
def bootstrap_error_estimation(n_bootstrap=100):
    """
    Ước lượng test error bằng Bootstrap
    """
    # Sinh dữ liệu
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=5, noise=10)
    
    n_samples = len(X)
    oob_errors = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample (lấy mẫu có hoàn lại)
        bootstrap_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        oob_idx = np.array([i for i in range(n_samples) if i not in bootstrap_idx])
        
        if len(oob_idx) == 0:
            continue
        
        X_boot, y_boot = X[bootstrap_idx], y[bootstrap_idx]
        X_oob, y_oob = X[oob_idx], y[oob_idx]
        
        # Train trên bootstrap sample
        model = Ridge(alpha=1.0)
        model.fit(X_boot, y_boot)
        
        # Test trên out-of-bag
        pred_oob = model.predict(X_oob)
        oob_error = np.mean((y_oob - pred_oob)**2)
        oob_errors.append(oob_error)
    
    bootstrap_estimate = np.mean(oob_errors)
    
    # So sánh với CV
    cv_scores = cross_val_score(Ridge(alpha=1.0), X, y, cv=5, scoring='neg_mean_squared_error')
    cv_estimate = -cv_scores.mean()
    
    print(f"Bootstrap estimate of test error: {bootstrap_estimate:.4f}")
    print(f"5-Fold CV estimate: {cv_estimate:.4f}")
    
    # Vẽ histogram của bootstrap errors
    plt.figure(figsize=(10, 6))
    plt.hist(oob_errors, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(bootstrap_estimate, color='red', linestyle='--', linewidth=2, 
               label=f'Mean = {bootstrap_estimate:.4f}')
    plt.xlabel('Out-of-Bag Error')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Bootstrap Errors ({n_bootstrap} iterations)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()

# bootstrap_error_estimation()
```

## Lựa Chọn Phương Pháp CV

| Phương pháp | Khi nào dùng | Ưu điểm | Nhược điểm |
|-------------|--------------|---------|------------|
| **Train/Test Split** | Quick baseline | Nhanh, đơn giản | Variance cao |
| **5-Fold CV** | Mặc định cho hầu hết | Cân bằng bias-variance | - |
| **10-Fold CV** | Dataset trung bình | Bias thấp hơn 5-fold | Chậm hơn |
| **LOOCV** | Dataset nhỏ (n < 100) | Bias thấp nhất | Variance cao, chậm |
| **Stratified CV** | Classification imbalanced | Đảm bảo tỷ lệ classes | - |
| **Bootstrap** | Ước lượng uncertainty | Linh hoạt | Phức tạp hơn |

## Bài Tập Thực Hành

**Bài 1: Implement K-Fold từ Đầu**
Viết hàm `my_kfold_cv(X, y, model, k=5)` không dùng sklearn.

**Bài 2: Nested CV**
Implement nested cross-validation để chọn hyperparameter và đánh giá mô hình không biased.

**Bài 3: Time Series CV**
Tìm hiểu về TimeSeriesSplit và áp dụng cho dữ liệu chuỗi thời gian (không thể shuffle!).
