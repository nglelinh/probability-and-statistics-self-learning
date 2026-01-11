---
layout: post
title: 01-06-00 Logistic Regression và Phân Loại Tuyến Tính
chapter: "07"
order: 5
owner: nglelinh
lang: vi
categories:
- chapter01
lesson_type: required
---

Chuyển từ regression (dự đoán giá trị liên tục) sang classification (dự đoán nhãn rời rạc) đòi hỏi một cách tiếp cận khác. Logistic regression là cầu nối hoàn hảo: nó sử dụng framework tuyến tính nhưng áp dụng cho bài toán phân loại thông qua một biến đổi thông minh - hàm sigmoid. Trong bài học này, chúng ta sẽ hiểu tại sao không thể dùng linear regression cho classification, cách logistic regression giải quyết vấn đề này, và làm thế nào để ước lượng parameters bằng maximum likelihood.

---

## Tại Sao Không Dùng Linear Regression Cho Classification?

Giả sử chúng ta có bài toán phân loại nhị phân: $$Y \in \{0, 1\}$$. Nếu dùng linear regression:

$$P(Y=1|X) = \beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p$$

**Vấn đề:** Vế phải có thể nhận giá trị ngoài $$[0, 1]$$, không phải xác suất hợp lệ!

**Giải pháp:** Cần một hàm biến đổi $$g: \mathbb{R} \to [0,1]$$.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

def why_not_linear_for_classification():
    """
    Minh họa tại sao linear regression không phù hợp cho classification
    """
    # Sinh dữ liệu classification đơn giản
    np.random.seed(42)
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    
    # Tạo labels với boundary tại x=0
    y_true_prob = 1 / (1 + np.exp(-2*X))  # Logistic function
    y = (y_true_prob.ravel() > 0.5).astype(int)
    
    # Linear regression
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    y_pred_linear = lin_reg.predict(X)
    
    # Logistic regression
    log_reg = LogisticRegression()
    log_reg.fit(X, y)
    y_pred_logistic = log_reg.predict_proba(X)[:, 1]
    
    # Vẽ biểu đồ
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.6, s=50, label='Data')
    plt.plot(X, y_pred_linear, 'r-', linewidth=2, label='Linear Regression')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.axhline(y=1, color='black', linestyle='--', linewidth=1)
    plt.xlabel('X')
    plt.ylabel('Predicted Probability')
    plt.title('Linear Regression (SAI!)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.5, 1.5)
    
    plt.subplot(1, 2, 2)
    plt.scatter(X, y, alpha=0.6, s=50, label='Data')
    plt.plot(X, y_pred_logistic, 'g-', linewidth=2, label='Logistic Regression')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.axhline(y=1, color='black', linestyle='--', linewidth=1)
    plt.xlabel('X')
    plt.ylabel('Predicted Probability')
    plt.title('Logistic Regression (ĐÚNG!)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)
    
    plt.tight_layout()
    # plt.show()
    
    print("Linear regression predictions:")
    print(f"  Min: {y_pred_linear.min():.4f} (< 0, không hợp lệ!)")
    print(f"  Max: {y_pred_linear.max():.4f} (> 1, không hợp lệ!)")
    print("\nLogistic regression predictions:")
    print(f"  Min: {y_pred_logistic.min():.4f}")
    print(f"  Max: {y_pred_logistic.max():.4f}")

# why_not_linear_for_classification()
```

## Mô Hình Logistic Regression

Logistic regression mô hình hóa **log-odds** (logit) như một hàm tuyến tính:

$$\log\left(\frac{P(Y=1|X)}{1-P(Y=1|X)}\right) = \beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p$$

Giải ra để tìm $$P(Y=1|X)$$:

$$P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p)}} = \frac{1}{1 + e^{-\mathbf{x}^T\boldsymbol{\beta}}}$$

Đây chính là **hàm sigmoid** (hoặc logistic function):

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Tính chất của sigmoid:**
- $$\sigma(z) \in (0, 1)$$ với mọi $$z \in \mathbb{R}$$
- $$\sigma(0) = 0.5$$
- $$\sigma(-z) = 1 - \sigma(z)$$
- Đạo hàm: $$\sigma'(z) = \sigma(z)(1-\sigma(z))$$

```python
def visualize_sigmoid():
    """
    Trực quan hóa hàm sigmoid và các tính chất
    """
    z = np.linspace(-6, 6, 200)
    sigmoid = 1 / (1 + np.exp(-z))
    sigmoid_derivative = sigmoid * (1 - sigmoid)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sigmoid function
    axes[0].plot(z, sigmoid, 'b-', linewidth=2, label='σ(z)')
    axes[0].axhline(y=0.5, color='r', linestyle='--', linewidth=1, label='Decision boundary')
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=1)
    axes[0].set_xlabel('z = β₀ + β₁X₁ + ... + βₚXₚ')
    axes[0].set_ylabel('P(Y=1|X)')
    axes[0].set_title('Hàm Sigmoid (Logistic Function)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Derivative
    axes[1].plot(z, sigmoid_derivative, 'g-', linewidth=2, label="σ'(z) = σ(z)(1-σ(z))")
    axes[1].set_xlabel('z')
    axes[1].set_ylabel("σ'(z)")
    axes[1].set_title('Đạo Hàm của Sigmoid')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()

# visualize_sigmoid()
```

## Maximum Likelihood Estimation

Không giống OLS (minimize RSS), logistic regression sử dụng **maximum likelihood**.

Giả sử có $$n$$ quan sát độc lập. Likelihood function:

$$L(\boldsymbol{\beta}) = \prod_{i=1}^n P(Y=y_i|X=\mathbf{x}_i) = \prod_{i=1}^n p_i^{y_i}(1-p_i)^{1-y_i}$$

trong đó $$p_i = P(Y=1|\mathbf{x}_i) = \sigma(\mathbf{x}_i^T\boldsymbol{\beta})$$.

**Log-likelihood:**

$$\ell(\boldsymbol{\beta}) = \sum_{i=1}^n [y_i \log p_i + (1-y_i)\log(1-p_i)]$$

**Mục tiêu:** $$\hat{\boldsymbol{\beta}} = \arg\max_{\boldsymbol{\beta}} \ell(\boldsymbol{\beta})$$

**Vấn đề:** Không có closed-form solution! Phải dùng numerical optimization (Newton-Raphson, gradient descent).

```python
from scipy.optimize import minimize

def implement_logistic_regression_from_scratch():
    """
    Implement logistic regression bằng maximum likelihood
    """
    # Sinh dữ liệu
    np.random.seed(42)
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1, random_state=42)
    
    # Thêm intercept
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    
    # Sigmoid function
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip để tránh overflow
    
    # Negative log-likelihood (để minimize)
    def neg_log_likelihood(beta, X, y):
        z = X @ beta
        p = sigmoid(z)
        # Thêm epsilon để tránh log(0)
        epsilon = 1e-15
        p = np.clip(p, epsilon, 1 - epsilon)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    
    # Gradient
    def gradient(beta, X, y):
        z = X @ beta
        p = sigmoid(z)
        return -X.T @ (y - p)
    
    # Optimize
    initial_beta = np.zeros(X_with_intercept.shape[1])
    result = minimize(neg_log_likelihood, initial_beta, 
                     args=(X_with_intercept, y),
                     jac=gradient, method='BFGS')
    
    beta_mle = result.x
    
    # So sánh với sklearn
    log_reg_sklearn = LogisticRegression(penalty=None, solver='lbfgs')
    log_reg_sklearn.fit(X, y)
    
    print("Coefficients từ MLE:")
    print(f"  Intercept: {beta_mle[0]:.4f}")
    print(f"  β₁: {beta_mle[1]:.4f}")
    print(f"  β₂: {beta_mle[2]:.4f}")
    print("\nCoefficients từ sklearn:")
    print(f"  Intercept: {log_reg_sklearn.intercept_[0]:.4f}")
    print(f"  β₁: {log_reg_sklearn.coef_[0, 0]:.4f}")
    print(f"  β₂: {log_reg_sklearn.coef_[0, 1]:.4f}")
    
    # Vẽ decision boundary
    plot_decision_boundary(X, y, beta_mle)

def plot_decision_boundary(X, y, beta):
    """
    Vẽ decision boundary
    """
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    colors = ['red' if label == 0 else 'blue' for label in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6, s=50)
    
    # Decision boundary: β₀ + β₁X₁ + β₂X₂ = 0
    # => X₂ = -(β₀ + β₁X₁) / β₂
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x1_boundary = np.linspace(x1_min, x1_max, 100)
    x2_boundary = -(beta[0] + beta[1] * x1_boundary) / beta[2]
    
    plt.plot(x1_boundary, x2_boundary, 'g-', linewidth=2, label='Decision Boundary')
    plt.xlabel('X₁')
    plt.ylabel('X₂')
    plt.title('Logistic Regression Decision Boundary')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()

# implement_logistic_regression_from_scratch()
```

## Regularized Logistic Regression

Giống như linear regression, logistic regression cũng có thể overfitting. Thêm regularization:

**Ridge (L2):**

$$\min_{\boldsymbol{\beta}} \left[-\ell(\boldsymbol{\beta}) + \lambda \sum_{j=1}^p \beta_j^2\right]$$

**Lasso (L1):**

$$\min_{\boldsymbol{\beta}} \left[-\ell(\boldsymbol{\beta}) + \lambda \sum_{j=1}^p |\beta_j|\right]$$

```python
from sklearn.linear_model import LogisticRegressionCV

def regularized_logistic_regression():
    """
    So sánh logistic regression với và không có regularization
    """
    # Sinh dữ liệu với nhiều features
    np.random.seed(42)
    X, y = make_classification(n_samples=200, n_features=20, n_informative=10,
                               n_redundant=5, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Models
    models = {
        'No regularization': LogisticRegression(penalty=None, max_iter=1000),
        'L2 (Ridge)': LogisticRegression(penalty='l2', C=1.0, max_iter=1000),
        'L1 (Lasso)': LogisticRegression(penalty='l1', C=1.0, solver='liblinear', max_iter=1000)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        n_nonzero = np.sum(np.abs(model.coef_) > 1e-5)
        
        results[name] = {
            'train_acc': train_score,
            'test_acc': test_score,
            'n_features': n_nonzero
        }
    
    # In kết quả
    print("Comparison of Regularization Methods:")
    print("-" * 60)
    for name, res in results.items():
        print(f"{name}:")
        print(f"  Train Accuracy: {res['train_acc']:.4f}")
        print(f"  Test Accuracy:  {res['test_acc']:.4f}")
        print(f"  Non-zero coefs: {res['n_features']}/20")
        print()

# regularized_logistic_regression()
```

## Multi-Class Classification

Logistic regression có thể mở rộng cho multi-class ($$K > 2$$ classes) bằng **Softmax Regression**:

$$P(Y=k|X) = \frac{e^{\mathbf{x}^T\boldsymbol{\beta}_k}}{\sum_{j=1}^K e^{\mathbf{x}^T\boldsymbol{\beta}_j}}$$

Sklearn tự động xử lý multi-class với `multi_class='multinomial'`.

```python
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def multiclass_logistic_regression():
    """
    Logistic regression cho multi-class classification
    """
    # Load Iris dataset (3 classes)
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = LogisticRegression(multi_class='multinomial', max_iter=1000)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=iris.target_names,
               yticklabels=iris.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Iris Dataset')
    # plt.show()
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

# multiclass_logistic_regression()
```

## Đánh Giá Mô Hình Classification

### Accuracy
$$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}$$

**Vấn đề:** Không tốt khi classes imbalanced!

### Precision, Recall, F1-Score

- **Precision:** Trong số dự đoán positive, bao nhiêu % đúng?
- **Recall (Sensitivity):** Trong số thực tế positive, bao nhiêu % được phát hiện?
- **F1-Score:** Harmonic mean của Precision và Recall

### ROC Curve và AUC

```python
from sklearn.metrics import roc_curve, auc, roc_auc_score

def plot_roc_curve():
    """
    Vẽ ROC curve
    """
    # Sinh dữ liệu
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()
    
    print(f"AUC Score: {roc_auc:.4f}")

# plot_roc_curve()
```

## Bài Tập Thực Hành

**Bài 1: Implement Gradient Descent**
Viết thuật toán gradient descent để train logistic regression thay vì dùng BFGS.

**Bài 2: Feature Importance**
Sử dụng L1 regularization để chọn features quan trọng nhất trong dataset phân loại.

**Bài 3: Imbalanced Classification**
Tạo dataset imbalanced (90% class 0, 10% class 1). So sánh accuracy vs F1-score.
