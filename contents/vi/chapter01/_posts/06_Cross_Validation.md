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

Lựa chọn mô hình là một bài toán cốt lõi trong học máy: làm thế nào để chọn mức độ phức tạp phù hợp và ước lượng khả năng dự đoán của mô hình trên dữ liệu mới? Kiểm định chéo (cross-validation) và các phương pháp lấy mẫu lại (resampling) giải quyết câu hỏi này theo hướng thực nghiệm: thay vì phụ thuộc hoàn toàn vào các công thức lý thuyết, ta sử dụng chính dữ liệu hiện có để ước lượng sai số kiểm tra (test error) một cách đáng tin cậy.

---

## Vấn đề: Sai số huấn luyện và sai số kiểm tra

Sai số huấn luyện (training error) không phản ánh chính xác chất lượng dự đoán thực tế của mô hình:
- Sai số huấn luyện luôn giảm khi mô hình trở nên phức tạp hơn.
- Đại lượng ta thực sự quan tâm là sai số kiểm tra (test error) trên dữ liệu mới chưa từng xuất hiện khi huấn luyện.

Trong thực tế, ta không có một tập kiểm tra vô hạn để đánh giá mọi mô hình ứng viên. Giải pháp là tận dụng dữ liệu huấn luyện hiện có để **ước lượng** sai số kiểm tra thông qua việc chia tách và lấy mẫu lại.

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

Thay vì chỉ chia dữ liệu một lần, phương pháp K-Fold chia tập dữ liệu thành $$K$$ phần (fold) có kích thước xấp xỉ bằng nhau (thường chọn $$K=5$$ hoặc $$K=10$$) và thực hiện theo quy trình:

1. Chia ngẫu nhiên tập dữ liệu thành $$K$$ phần bằng nhau.
2. Lặp $$K$$ lần: ở lần thứ $$k$$, dùng phần thứ $$k$$ làm tập validation, huấn luyện mô hình trên $$K-1$$ phần còn lại và tính sai số validation.
3. Lấy trung bình cộng của $$K$$ sai số validation để thu được ước lượng sai số CV:

$$\text{CV}_{(K)} = \frac{1}{K}\sum_{k=1}^K \text{MSE}_k$$

**Ưu điểm:**
- Sử dụng dữ liệu hiệu quả hơn vì mỗi quan sát đều được dùng cho cả huấn luyện và kiểm định.
- Cho ước lượng ổn định hơn và có phương sai thấp hơn so với chia train/test một lần.

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

LOOCV là trường hợp đặc biệt của K-Fold khi $$K = n$$: ở mỗi lần lặp, ta giữ lại đúng 1 quan sát làm tập validation và huấn luyện mô hình trên $$n-1$$ quan sát còn lại, lặp lại đủ $$n$$ lần.

$$\text{CV}_{(n)} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$

trong đó $$\hat{y}_i$$ là giá trị dự đoán cho quan sát thứ $$i$$ từ mô hình được huấn luyện khi loại bỏ chính quan sát đó.

**Ưu điểm:**
- Tận dụng gần như toàn bộ dữ liệu cho mỗi lần huấn luyện ($$n-1$$ điểm).
- Cho ước lượng gần như không chệch (nearly unbiased) đối với sai số kiểm tra.

**Nhược điểm:**
- Chi phí tính toán cao vì phải huấn luyện mô hình $$n$$ lần.
- Phương sai của ước lượng có thể cao do các tập huấn luyện trùng lặp nhau gần như hoàn toàn.

**Khi nào nên dùng LOOCV?** Phương pháp này phù hợp khi kích thước mẫu $$n$$ nhỏ (ví dụ $$n < 100$$) hoặc khi thời gian huấn luyện mô hình rất nhanh (như các mô hình tuyến tính).

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

## Stratified K-Fold (cho bài toán phân loại)

Trong bài toán phân loại có dữ liệu mất cân bằng lớp (imbalanced classes), việc chia ngẫu nhiên thông thường có thể khiến một số fold chứa quá ít mẫu của lớp thiểu số. **Stratified K-Fold** khắc phục điều này bằng cách chia dữ liệu sao cho tỷ lệ các lớp trong từng fold xấp xỉ tỷ lệ các lớp trong toàn bộ tập dữ liệu.

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

Bootstrap là một phương pháp lấy mẫu lại phổ biến khác, đặc biệt hữu ích để ước lượng độ bất định của tham số và sai số dự đoán.

**Quy trình:**
1. Từ tập dữ liệu gốc gồm $$n$$ quan sát, lấy mẫu **có hoàn lại** $$n$$ quan sát để tạo một mẫu bootstrap.
2. Huấn luyện mô hình trên mẫu bootstrap vừa tạo.
3. Đánh giá mô hình trên các quan sát **không** xuất hiện trong mẫu bootstrap đó (tập out-of-bag).
4. Lặp lại $$B$$ lần (thường $$B = 100$$ hoặc $$B = 1000$$).

**Ước lượng sai số kiểm tra bằng Bootstrap (out-of-bag):**

$$\text{Err}_{\text{boot}} = \frac{1}{B}\sum_{b=1}^B \frac{1}{|C^{-b}|}\sum_{i \in C^{-b}} L(y_i, \hat{f}^{*b}(x_i))$$

trong đó $$C^{-b}$$ là tập các quan sát không xuất hiện trong mẫu bootstrap thứ $$b$$.

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

## Lựa chọn phương pháp kiểm định chéo

| Phương pháp | Khi nào nên dùng | Ưu điểm | Nhược điểm |
|-------------|------------------|---------|------------|
| **Train/Test Split** | Tạo mốc cơ sở nhanh | Nhanh, đơn giản | Phương sai ước lượng cao |
| **5-Fold CV** | Lựa chọn mặc định cho đa số bài toán | Cân bằng tốt giữa độ chệch và phương sai | — |
| **10-Fold CV** | Tập dữ liệu kích thước trung bình | Độ chệch thấp hơn 5-Fold | Tốn thời gian tính toán hơn |
| **LOOCV** | Tập dữ liệu nhỏ ($$n < 100$$) | Độ chệch rất thấp | Phương sai cao, tốn chi phí tính toán |
| **Stratified CV** | Phân loại mất cân bằng lớp | Bảo toàn tỷ lệ các lớp ở mỗi fold | — |
| **Bootstrap** | Ước lượng độ bất định | Linh hoạt | Cài đặt phức tạp hơn |

## Video tham khảo

- [Máy Học Ứng Dụng - Bài 11. Phương Pháp Đánh Giá Chéo (Cross-Validation) Trong Máy Học](https://www.youtube.com/watch?v=aKVgXrDBqgk) — OVNE-Giáo Dục Việt, tiếng Việt. **Xem trước.** Giải thích phương pháp đánh giá chéo bằng tiếng Việt: luân phiên các fold để ước lượng sai số trên dữ liệu mới thay vì chỉ dựa vào sai số huấn luyện.
- [Machine Learning Fundamentals: Cross Validation](https://www.youtube.com/watch?v=fSytzGwwBVw) — StatQuest with Josh Starmer, tiếng Anh. Minh họa trực quan K-Fold và Leave-One-Out, cùng cách dùng cross-validation để chọn siêu tham số.
- [Bootstrapping Main Ideas!!!](https://www.youtube.com/watch?v=Xz0x-8-cgaQ) — StatQuest with Josh Starmer, tiếng Anh. Trình bày cơ chế lấy mẫu có hoàn lại và tập out-of-bag, tương ứng mục Bootstrap ở cuối bài.

## Bài tập thực hành

**Bài 1: Cài đặt K-Fold từ đầu**
Viết hàm `my_kfold_cv(X, y, model, k=5)` không sử dụng hàm chia fold có sẵn của scikit-learn.

**Bài 2: Kiểm định chéo lồng nhau (Nested CV)**
Cài đặt nested cross-validation để vừa tinh chỉnh siêu tham số vừa đánh giá mô hình mà tránh bị chệch lạc quan.

**Bài 3: Kiểm định chéo cho chuỗi thời gian (Time Series CV)**
Tìm hiểu `TimeSeriesSplit` và áp dụng cho dữ liệu chuỗi thời gian (không được xáo trộn thứ tự thời gian).
