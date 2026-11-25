---
layout: post
title: 01-06-00 Logistic Regression và Phân Loại Tuyến Tính
chapter: "07"
order: 5
owner: nglelinh
lang: vi
categories:
- chapter07
lesson_type: required
---

Một ngân hàng muốn dự đoán xác suất một khách hàng sẽ vỡ nợ thẻ tín dụng ($$Y = 1$$) hay trả đúng hạn ($$Y = 0$$) dựa trên tỷ lệ nợ trên thu nhập. Nếu thử ép hồi quy tuyến tính OLS vào bài toán này, đường thẳng dự báo sẽ hồn nhiên chạy vọt lên $$1{,}35$$ (xác suất 135%!) hoặc tụt xuống $$-0{,}20$$ (xác suất âm!), đồng thời sai số nhị phân không bao giờ tuân theo phân phối chuẩn đồng phương sai.

**Hồi quy Logistic (Logistic Regression)** giải quyết bài toán phân loại bằng một ý tưởng cực kỳ thanh lịch: giữ nguyên biểu thức tuyến tính $$z = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p$$, nhưng đưa nó đi qua hàm cong chữ S **sigmoid** $$\sigma(z) = \frac{1}{1+e^{-z}}$$ để nén gọn vào khoảng $$(0, 1)$$. Không chỉ là mô hình phân loại tuyến tính nền tảng, hồi quy logistic thực chất chính là một **nơ-ron đơn lẻ** với hàm kích hoạt sigmoid và hàm mất mát cross-entropy — viên gạch khởi nguyên của cả học sâu hiện đại.

---

## Kiến thức cần nhớ: Bốn trụ cột toán học của Hồi quy Logistic

Để không bao giờ nhầm hồi quy logistic chỉ là "một hàm vẽ đường cong chữ S", hãy nắm vững bốn nền tảng sau:

1. **Tỷ lệ cược (Odds) và Hàm liên kết Logit:** Nếu xác suất xảy ra sự kiện là $$p$$, thì **odds** là $$\frac{p}{1-p} \in (0, \infty)$$ và **log-odds (logit)** là $$\ln\frac{p}{1-p} \in (-\infty, +\infty)$$. Hồi quy logistic mô hình hóa tuyến tính chính log-odds này: $$\ln\frac{p}{1-p} = \mathbf{w}^T\mathbf{x}$$. Do đó mỗi hệ số $$\beta_j$$ có ý nghĩa diễn giải trực tiếp: tăng $$x_j$$ lên 1 đơn vị sẽ **nhân** odds lên $$e^{\beta_j}$$ lần (tỷ số odds — *odds ratio*).
2. **MLE = Cực tiểu hóa Cross-Entropy (Log-Loss):** Khác với OLS, hồi quy logistic không dùng sai số bình phương (vì bình phương của sigmoid là hàm không lồi, dễ mắc kẹt cực trị địa phương). Thay vào đó, cực đại hóa log-likelihood Bernoulli sinh ra hàm mất mát **Binary Cross-Entropy** — hoàn toàn **lồi toàn cục**, đảm bảo Gradient Descent hay Newton–Raphson luôn hội tụ về đúng một nghiệm tối ưu duy nhất!
3. **Biên quyết định tuyến tính (Linear Decision Boundary):** Ngưỡng phân loại $$p = 0{,}5$$ tương đương chính xác với siêu phẳng $$\mathbf{w}^T\mathbf{x} = 0$$. Mọi xác suất dự báo đều được quyết định bởi khoảng cách có dấu từ điểm dữ liệu đến siêu phẳng này.
4. **Đánh giá khi dữ liệu mất cân bằng:** Khi chỉ có 1% giao dịch là gian lận, một mô hình ngốc nghếch luôn đoán "không gian lận" vẫn đạt **Accuracy 99%**. Đó là lý do các bài toán phân loại thực tế phải dùng **Precision, Recall, F1** và **ROC–AUC / PR–AUC**.

---

## Tại Sao Không Dùng Hồi Quy Tuyến Tính Cho Phân Loại?

Xét bài toán phân loại nhị phân với $$Y \in \{0, 1\}$$. Nếu áp dụng trực tiếp hồi quy tuyến tính:

$$P(Y=1|X) = \beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p$$

**Vấn đề:** Vế phải có thể nhận giá trị nằm ngoài đoạn $$[0, 1]$$, nên không thể diễn giải như một xác suất hợp lệ.

**Giải pháp:** Cần một hàm biến đổi $$g: \mathbb{R} \to [0,1]$$ để ánh xạ tổ hợp tuyến tính vào khoảng giá trị của xác suất.

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

## Mô Hình Hồi Quy Logistic

Hồi quy logistic mô hình hóa **log-odds** (logit) dưới dạng một hàm tuyến tính của các đặc trưng:

$$\log\left(\frac{P(Y=1|X)}{1-P(Y=1|X)}\right) = \beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p$$

Giải phương trình trên theo $$P(Y=1|X)$$, ta thu được:

$$P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p)}} = \frac{1}{1 + e^{-\mathbf{x}^T\boldsymbol{\beta}}}$$

Đây chính là **hàm sigmoid** (hay hàm logistic):

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Các tính chất cơ bản của hàm sigmoid:**
- $$\sigma(z) \in (0, 1)$$ với mọi $$z \in \mathbb{R}$$.
- $$\sigma(0) = 0.5$$.
- $$\sigma(-z) = 1 - \sigma(z)$$.
- Đạo hàm: $$\sigma'(z) = \sigma(z)(1-\sigma(z))$$.

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

## Ước Lượng Hợp Lý Cực Đại (Maximum Likelihood Estimation)

Khác với OLS (tối thiểu hóa RSS), hồi quy logistic ước lượng tham số bằng phương pháp **hợp lý cực đại (maximum likelihood)**.

Giả sử có $$n$$ quan sát độc lập, hàm hợp lý (likelihood function) được viết là:

$$L(\boldsymbol{\beta}) = \prod_{i=1}^n P(Y=y_i|X=\mathbf{x}_i) = \prod_{i=1}^n p_i^{y_i}(1-p_i)^{1-y_i}$$

trong đó $$p_i = P(Y=1|\mathbf{x}_i) = \sigma(\mathbf{x}_i^T\boldsymbol{\beta})$$.

**Hàm log-likelihood:**

$$\ell(\boldsymbol{\beta}) = \sum_{i=1}^n [y_i \log p_i + (1-y_i)\log(1-p_i)]$$

**Mục tiêu:** Tìm $$\hat{\boldsymbol{\beta}} = \arg\max_{\boldsymbol{\beta}} \ell(\boldsymbol{\beta})$$.

**Lưu ý:** Bài toán tối ưu này không có nghiệm dạng đóng nên cần giải bằng các phương pháp tối ưu số như Newton-Raphson hoặc gradient descent.

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

## Hồi Quy Logistic Có Điều Chuẩn

Tương tự hồi quy tuyến tính, hồi quy logistic cũng có thể gặp hiện tượng quá khớp (overfitting). Ta khắc phục bằng cách thêm số hạng điều chuẩn vào hàm negative log-likelihood:

**Điều chuẩn Ridge (L2):**

$$\min_{\boldsymbol{\beta}} \left[-\ell(\boldsymbol{\beta}) + \lambda \sum_{j=1}^p \beta_j^2\right]$$

**Điều chuẩn Lasso (L1):**

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

## Phân Loại Đa Lớp (Multi-Class Classification)

Hồi quy logistic có thể mở rộng cho bài toán đa lớp ($$K > 2$$ lớp) bằng **hồi quy Softmax (Softmax Regression)**:

$$P(Y=k|X) = \frac{e^{\mathbf{x}^T\boldsymbol{\beta}_k}}{\sum_{j=1}^K e^{\mathbf{x}^T\boldsymbol{\beta}_j}}$$

Thư viện `scikit-learn` hỗ trợ trực tiếp bài toán đa lớp thông qua tham số `multi_class='multinomial'`.

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

## Đánh Giá Mô Hình Phân Loại

### Độ Chính Xác (Accuracy)

$$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}$$

**Hạn chế:** Chỉ số này dễ gây hiểu nhầm khi dữ liệu mất cân bằng lớp (imbalanced classes).

### Precision, Recall và F1-Score

- **Precision:** Trong số các mẫu được dự đoán là dương tính (positive), tỷ lệ dự đoán đúng là bao nhiêu.
- **Recall (Sensitivity):** Trong số các mẫu thực sự là dương tính, mô hình phát hiện được bao nhiêu phần trăm.
- **F1-Score:** Trung bình điều hòa (harmonic mean) giữa Precision và Recall.

### Đường Cong ROC và Diện Tích AUC

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

---

### Ví dụ tính tay làm từng bước: Tính xác suất, Odds Ratio và 1 bước Gradient Descent trên mô hình Logistic

Giả sử một mô hình logistic dự đoán xác suất sinh viên đỗ kỳ thi chứng chỉ theo số giờ ôn tập $$x$$ có phương trình log-odds:

$$z = -2{,}0 + 0{,}5x \implies p(x) = \sigma(z) = \frac{1}{1 + e^{-z}}.$$

1. **Tính xác suất và tỷ lệ cược tại hai mốc giờ ôn:**
   - Tại $$x = 4$$ giờ: $$z = -2{,}0 + 0{,}5(4) = 0 \implies p(4) = \frac{1}{1+e^0} = 0{,}50$$ (đây chính là ngưỡng ranh giới quyết định 50/50!).
   - Tại $$x = 6$$ giờ: $$z = -2{,}0 + 0{,}5(6) = +1{,}0 \implies p(6) = \frac{1}{1+e^{-1}} \approx \frac{1}{1{,}3679} \approx 0{,}731$$.
   - Tỷ lệ cược (odds) tại $$x = 6$$ giờ là $$\frac{p}{1-p} = e^{1{,}0} \approx 2{,}718$$ (cứ gần 2,72 khả năng đỗ thì mới có 1 khả năng trượt).
2. **Diễn giải hệ số $$\beta_1 = 0{,}5$$ qua Odds Ratio:**
   Mỗi khi ôn thêm **1 giờ**, log-odds tăng thêm $$0{,}5$$, tức tỷ lệ cược đỗ được **nhân lên $$e^{0{,}5} \approx 1{,}649$$ lần** (tăng khoảng $$64{,}9\%$$ odds, bất kể bạn đang ở mốc giờ ôn nào!).
3. **Tính hàm mất mát Cross-Entropy và gradient cho 1 quan sát $$(x=6,\; y=1)$$:**
   - Mất mát log-loss: $$L = -\bigl[y\ln p + (1-y)\ln(1-p)\bigr] = -\ln(0{,}731) \approx 0{,}313.$$
   - Đạo hàm của $$L$$ theo $$z$$ rút gọn đẹp đến bất ngờ (nhờ tính chất $$\sigma'(z) = \sigma(z)(1-\sigma(z))$$):
     $$\frac{\partial L}{\partial z} = p - y = 0{,}731 - 1 = -0{,}269.$$
   - Theo quy tắc dây chuyền, $$\frac{\partial L}{\partial \beta_1} = (p - y)x = (-0{,}269)\times 6 = -1{,}614$$. Với tốc độ học $$\eta = 0{,}1$$, bước cập nhật Gradient Descent kéo $$\beta_1$$ tăng lên:
     $$\beta_1^{\text{mới}} = 0{,}5 - 0{,}1(-1{,}614) = 0{,}6614,$$
     đẩy xác suất dự báo cho sinh viên này tiến sát về nhãn thật $$y = 1$$!

### Giải thích: Bốn điều cần nắm vững khi triển khai Hồi quy Logistic

1. **Dùng Logistic để dự báo *xác suất*, đừng chỉ cắt cứng ở 0,5:** Ngưỡng $$0{,}5$$ chỉ tối ưu khi chi phí báo động giả và bỏ lọt bằng nhau. Trong tầm soát ung thư hay phát hiện gian lận, bạn hoàn toàn có thể hạ ngưỡng xuống $$0{,}15$$ hoặc $$0{,}20$$ dựa trên đường cong Precision–Recall.
2. **Coi chừng hiện tượng tách biệt hoàn toàn (Complete Separation):** Nếu có một đường thẳng tách hoàn toàn lớp 0 và lớp 1 không sai điểm nào, MLE không điều chuẩn sẽ đẩy $$\|\boldsymbol{\beta}\| \to \infty$$ để ép $$p\to 1$$! Đó là lý do `LogisticRegression` trong scikit-learn luôn bật mặc định điều chuẩn L2 (`C=1.0`).
3. **Gradient của Cross-Entropy gọn hệt như OLS:** Công thức $$\nabla_{\boldsymbol{\beta}} L = \mathbf{X}^T(\mathbf{p} - \mathbf{y})$$ có hình thức giống hệt gradient của bình phương tối thiểu, chỉ thay dự báo tuyến tính $$\hat{\mathbf{y}}$$ bằng xác suất $$\mathbf{p} = \sigma(\mathbf{X}\boldsymbol{\beta})$$ — một sự thống nhất tuyệt đẹp của họ Mô hình Tuyến tính Tổng quát (GLM).
4. **Hồi quy Softmax mở rộng tự nhiên cho $$K$$ lớp:** Thay sigmoid 2 lớp bằng $$P(Y=k\mid\mathbf{x}) = \frac{e^{\mathbf{w}_k^T\mathbf{x}}}{\sum_j e^{\mathbf{w}_j^T\mathbf{x}}}$$, ta thu được chính lớp đầu ra của mọi mạng phân loại đa lớp và mô hình ngôn ngữ hiện đại.

**Kiểm tra nhanh.** Với mô hình $$z = -2{,}0 + 0{,}5x$$ ở trên, nếu một sinh viên ôn $$x = 2$$ giờ, xác suất đỗ $$p(2)$$ bằng bao nhiêu — và nếu thực tế sinh viên đó *trượt* ($$y = 0$$), đạo hàm $$\frac{\partial L}{\partial z} = p - y$$ mang dấu dương hay âm?

---

## Video tham khảo

- [StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8) — StatQuest with Josh Starmer, tiếng Anh. **Xem trước.** Giải thích lý do không dùng đường thẳng cho bài toán phân loại, sau đó giới thiệu hàm sigmoid và log-odds.
- [Logistic Regression Details Pt 2: Maximum Likelihood](https://www.youtube.com/watch?v=BfKanl1aSG0) — StatQuest with Josh Starmer, tiếng Anh. Trình bày phương pháp ước lượng hợp lý cực đại cho các hệ số của hồi quy logistic.
- [ROC and AUC, Clearly Explained!](https://www.youtube.com/watch?v=4jRBRDbJemM) — StatQuest with Josh Starmer, tiếng Anh. Giải thích đường cong ROC và diện tích AUC trong đánh giá mô hình phân loại.

## Bài Tập Thực Hành

**Bài 1: Cài đặt Gradient Descent**
Viết thuật toán gradient descent để huấn luyện hồi quy logistic thay vì sử dụng phương pháp BFGS.

**Bài 2: Đánh giá tầm quan trọng của đặc trưng (Feature Importance)**
Sử dụng điều chuẩn L1 để chọn lọc các đặc trưng quan trọng nhất trên một tập dữ liệu phân loại.

**Bài 3: Phân loại dữ liệu mất cân bằng (Imbalanced Classification)**
Tạo tập dữ liệu mất cân bằng (90% lớp 0, 10% lớp 1) và so sánh chỉ số Accuracy với F1-score.
