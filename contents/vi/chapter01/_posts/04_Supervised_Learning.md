---
layout: post
title: 01-02-00 Tổng Quan về Supervised Learning
chapter: "01"
order: 2
owner: nglelinh
lang: vi
categories:
- chapter01
lesson_type: required
---

Supervised learning là trái tim của machine learning hiện đại, nơi chúng ta học từ dữ liệu có nhãn để dự đoán kết quả cho dữ liệu mới. Trong bài học này, chúng ta sẽ xây dựng khung sườn tư duy về statistical learning, hiểu rõ sự khác biệt giữa regression và classification, và đặc biệt quan trọng - nắm vững khái niệm bias-variance tradeoff, chìa khóa để hiểu tại sao mô hình phức tạp không phải lúc nào cũng tốt hơn.

---

## Supervised Learning Là Gì?

Trong supervised learning, chúng ta có một tập dữ liệu huấn luyện gồm các cặp $$(X, Y)$$:
- $$X = (X_1, X_2, \ldots, X_p)$$: Vector đặc trưng (features/predictors)
- $$Y$$: Biến mục tiêu (target/response)

**Mục tiêu:** Học một hàm $$f$$ sao cho $$Y \approx f(X)$$, để có thể dự đoán $$Y$$ cho các giá trị $$X$$ mới chưa từng thấy.

### Hai Loại Bài Toán Chính

**1. Regression (Hồi quy):** $$Y$$ là biến liên tục
- Ví dụ: Dự đoán giá nhà, nhiệt độ, doanh số bán hàng

**2. Classification (Phân loại):** $$Y$$ là biến rời rạc (categorical)
- Ví dụ: Phân loại email spam/không spam, nhận diện chữ số viết tay

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split

def visualize_supervised_learning():
    """
    Minh họa sự khác biệt giữa regression và classification
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Regression example
    X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
    axes[0].scatter(X_reg, y_reg, alpha=0.6, s=50)
    axes[0].set_xlabel('Feature X')
    axes[0].set_ylabel('Target Y (continuous)')
    axes[0].set_title('Regression: Dự đoán giá trị liên tục')
    axes[0].grid(True, alpha=0.3)
    
    # Classification example
    X_clf, y_clf = make_classification(n_samples=200, n_features=2, n_redundant=0,
                                       n_informative=2, n_clusters_per_class=1,
                                       random_state=42)
    colors = ['red' if label == 0 else 'blue' for label in y_clf]
    axes[1].scatter(X_clf[:, 0], X_clf[:, 1], c=colors, alpha=0.6, s=50)
    axes[1].set_xlabel('Feature X₁')
    axes[1].set_ylabel('Feature X₂')
    axes[1].set_title('Classification: Phân loại nhị phân')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()

# visualize_supervised_learning()
```

## Loss Function và Training Error

Để đánh giá mô hình, chúng ta cần định nghĩa **loss function** $$L(Y, f(X))$$ đo lường sai số giữa giá trị thực $$Y$$ và dự đoán $$f(X)$$.

### Loss Functions Phổ Biến

**Regression:**
- **Squared Error Loss (L2):** $$L(Y, f(X)) = (Y - f(X))^2$$
- **Absolute Error Loss (L1):** $$L(Y, f(X)) = |Y - f(X)|$$

**Classification:**
- **0-1 Loss:** $$L(Y, f(X)) = \mathbb{1}_{Y \neq f(X)}$$
- **Cross-Entropy Loss:** $$L(Y, f(X)) = -\sum_k y_k \log f_k(X)$$

**Training Error (Empirical Risk):**

$$\text{Err}_{\text{train}} = \frac{1}{n} \sum_{i=1}^n L(y_i, f(x_i))$$

Đây là sai số trên dữ liệu huấn luyện - dữ liệu mà mô hình đã "nhìn thấy".

**Test Error (Expected Prediction Error):**

$$\text{Err}_{\text{test}} = E[L(Y, f(X))]$$

Đây là sai số kỳ vọng trên dữ liệu mới - thước đo thực sự về hiệu suất mô hình.

**Vấn đề cốt lõi:** Training error luôn giảm khi mô hình phức tạp hơn, nhưng test error có thể tăng (overfitting)!

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

def demonstrate_overfitting(degree_max=15):
    """
    Minh họa overfitting với polynomial regression
    """
    # Sinh dữ liệu
    np.random.seed(42)
    X = np.linspace(0, 1, 20).reshape(-1, 1)
    y_true = np.sin(2 * np.pi * X).ravel()
    y = y_true + np.random.normal(0, 0.2, X.shape[0])
    
    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Thử các độ phức tạp khác nhau
    degrees = range(1, degree_max + 1)
    train_errors = []
    test_errors = []
    
    for degree in degrees:
        # Tạo polynomial features
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        
        # Huấn luyện mô hình
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        
        # Tính sai số
        train_pred = model.predict(X_train_poly)
        test_pred = model.predict(X_test_poly)
        
        train_errors.append(mean_squared_error(y_train, train_pred))
        test_errors.append(mean_squared_error(y_test, test_pred))
    
    # Vẽ biểu đồ
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(degrees, train_errors, 'o-', label='Training Error', linewidth=2)
    plt.plot(degrees, test_errors, 's-', label='Test Error', linewidth=2)
    plt.xlabel('Độ phức tạp mô hình (Polynomial Degree)')
    plt.ylabel('Mean Squared Error')
    plt.title('Training Error vs Test Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Vẽ mô hình với độ phức tạp khác nhau
    plt.subplot(1, 2, 2)
    X_plot = np.linspace(0, 1, 100).reshape(-1, 1)
    
    for degree in [1, 3, 9, 15]:
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_plot_poly = poly.transform(X_plot)
        
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        y_plot = model.predict(X_plot_poly)
        
        plt.plot(X_plot, y_plot, label=f'Degree {degree}', linewidth=2)
    
    plt.scatter(X_train, y_train, color='black', s=50, alpha=0.5, label='Training data')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Các Mô Hình với Độ Phức Tạp Khác Nhau')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()
    
    # Tìm độ phức tạp tối ưu
    optimal_degree = degrees[np.argmin(test_errors)]
    print(f"Độ phức tạp tối ưu (test error thấp nhất): {optimal_degree}")
    print(f"Test error tối thiểu: {min(test_errors):.4f}")

# demonstrate_overfitting()
```

## Bias-Variance Tradeoff

Đây là một trong những khái niệm quan trọng nhất trong machine learning. Test error có thể phân tích thành ba thành phần:

$$\text{Expected Test Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

### Giải Thích Các Thành Phần

**1. Bias (Độ chệch):**
- Sai số do mô hình quá đơn giản, không thể nắm bắt được mối quan hệ thực trong dữ liệu
- Mô hình có bias cao: **underfitting**
- Ví dụ: Dùng đường thẳng để fit dữ liệu phi tuyến

**2. Variance (Phương sai):**
- Sai số do mô hình quá nhạy cảm với nhiễu trong dữ liệu huấn luyện
- Mô hình có variance cao: **overfitting**
- Ví dụ: Polynomial bậc cao fit hoàn hảo training data nhưng dự đoán kém trên test data

**3. Irreducible Error:**
- Nhiễu vốn có trong dữ liệu, không thể giảm bằng mô hình tốt hơn

**Tradeoff:** Khi giảm bias (mô hình phức tạp hơn), variance tăng. Khi giảm variance (mô hình đơn giản hơn), bias tăng.

```python
def bias_variance_decomposition(n_simulations=100):
    """
    Mô phỏng bias-variance decomposition
    """
    # Hàm thực f(x) = sin(2πx)
    def true_function(x):
        return np.sin(2 * np.pi * x)
    
    # Điểm test cố định
    x_test = np.array([0.5])
    y_true = true_function(x_test)
    
    degrees = [1, 3, 9, 15]
    results = {deg: {'predictions': [], 'bias': 0, 'variance': 0} for deg in degrees}
    
    for _ in range(n_simulations):
        # Sinh training data mới mỗi lần
        X_train = np.random.uniform(0, 1, 20).reshape(-1, 1)
        y_train = true_function(X_train).ravel() + np.random.normal(0, 0.2, 20)
        
        for degree in degrees:
            # Train model
            poly = PolynomialFeatures(degree=degree)
            X_train_poly = poly.fit_transform(X_train)
            x_test_poly = poly.transform(x_test.reshape(-1, 1))
            
            model = LinearRegression()
            model.fit(X_train_poly, y_train)
            
            # Dự đoán
            pred = model.predict(x_test_poly)[0]
            results[degree]['predictions'].append(pred)
    
    # Tính bias và variance
    for degree in degrees:
        preds = np.array(results[degree]['predictions'])
        mean_pred = preds.mean()
        
        results[degree]['bias'] = (mean_pred - y_true[0])**2
        results[degree]['variance'] = preds.var()
        results[degree]['mse'] = results[degree]['bias'] + results[degree]['variance']
    
    # Vẽ biểu đồ
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Phân phối dự đoán
    axes[0].axhline(y_true[0], color='green', linestyle='--', linewidth=2, label='True value')
    for degree in degrees:
        preds = results[degree]['predictions']
        axes[0].hist(preds, bins=20, alpha=0.5, label=f'Degree {degree}')
    axes[0].set_xlabel('Predicted value at x=0.5')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Phân Phối Dự Đoán qua Nhiều Training Sets')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Bias-Variance decomposition
    x_pos = np.arange(len(degrees))
    bias_vals = [results[d]['bias'] for d in degrees]
    var_vals = [results[d]['variance'] for d in degrees]
    
    axes[1].bar(x_pos - 0.2, bias_vals, 0.4, label='Bias²', alpha=0.8)
    axes[1].bar(x_pos + 0.2, var_vals, 0.4, label='Variance', alpha=0.8)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f'Degree {d}' for d in degrees])
    axes[1].set_ylabel('Error')
    axes[1].set_title('Bias-Variance Decomposition')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    # plt.show()
    
    # In kết quả
    print("Bias-Variance Analysis:")
    print("-" * 50)
    for degree in degrees:
        print(f"Degree {degree}:")
        print(f"  Bias²:    {results[degree]['bias']:.4f}")
        print(f"  Variance: {results[degree]['variance']:.4f}")
        print(f"  MSE:      {results[degree]['mse']:.4f}")
        print()

# bias_variance_decomposition()
```

## Curse of Dimensionality

Khi số lượng features $$p$$ tăng lên, không gian dữ liệu trở nên "thưa thớt" (sparse). Điều này gây ra nhiều vấn đề:

1. **Cần nhiều dữ liệu hơn:** Để có cùng mật độ dữ liệu trong không gian $$p$$ chiều, cần $$n \propto 2^p$$ mẫu
2. **Khoảng cách mất ý nghĩa:** Trong không gian cao chiều, hầu hết các điểm đều xa nhau như nhau
3. **Overfitting dễ xảy ra:** Mô hình có thể fit noise thay vì signal

```python
def demonstrate_curse_of_dimensionality():
    """
    Minh họa curse of dimensionality
    """
    dimensions = [1, 2, 5, 10, 20, 50]
    n_samples = 1000
    
    avg_distances = []
    std_distances = []
    
    for dim in dimensions:
        # Sinh dữ liệu ngẫu nhiên trong không gian dim chiều
        data = np.random.uniform(0, 1, (n_samples, dim))
        
        # Tính khoảng cách từ điểm đầu tiên đến các điểm khác
        distances = np.sqrt(((data[0] - data[1:])**2).sum(axis=1))
        
        avg_distances.append(distances.mean())
        std_distances.append(distances.std())
    
    # Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    plt.errorbar(dimensions, avg_distances, yerr=std_distances, 
                marker='o', capsize=5, linewidth=2, markersize=8)
    plt.xlabel('Số chiều (p)')
    plt.ylabel('Khoảng cách trung bình')
    plt.title('Curse of Dimensionality: Khoảng cách tăng theo số chiều')
    plt.grid(True, alpha=0.3)
    # plt.show()
    
    print("Khoảng cách trung bình theo số chiều:")
    for dim, avg, std in zip(dimensions, avg_distances, std_distances):
        print(f"  p={dim:2d}: {avg:.4f} ± {std:.4f}")

# demonstrate_curse_of_dimensionality()
```

## Bài Tập Thực Hành

**Bài 1: Phân Tích Overfitting**
Sử dụng dataset Boston Housing (có sẵn trong sklearn). Thử polynomial regression với các degree khác nhau và vẽ đồ thị training/test error.

**Bài 2: Bias-Variance Tradeoff**
Tạo dữ liệu synthetic với hàm thực $$f(x) = x^2$$. Mô phỏng bias-variance decomposition cho linear regression và polynomial regression degree 2.

**Bài 3: High-Dimensional Data**
Sinh dữ liệu ngẫu nhiên với $$p = 100$$ features nhưng chỉ 5 features thực sự có ích. Huấn luyện mô hình và quan sát hiện tượng overfitting.
