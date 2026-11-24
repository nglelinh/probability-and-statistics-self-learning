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

Một sinh viên chỉ học thuộc lòng đáp án đề thi năm ngoái có thể đạt 10/10 khi tự làm lại đề cũ ở nhà, nhưng lại trượt thảm hại khi bước vào phòng thi thật với bộ câu hỏi mới. Trong học máy, hiện tượng "học vẹt" đó có tên gọi chính thức là **quá khớp (overfitting)** — và toàn bộ ngành **học có giám sát (supervised learning)** xoay quanh một mục tiêu tối thượng duy nhất: không phải khớp hoàn hảo dữ liệu quá khứ, mà là **tổng quát hóa tốt sang dữ liệu tương lai chưa từng thấy**.

Bài học này đặt khung toán học rõ ràng cho học có giám sát: từ bài toán hồi quy và phân loại, hàm mất mát, cho đến định lý nền tảng nhất về đánh đổi **Độ chệch – Phương sai (Bias–Variance Tradeoff)** và **Lời nguyền số chiều**.

---

## Bản đồ kiến thức nền: Học có giám sát là gì và nghiên cứu những gì?

Từ góc nhìn xác suất – thống kê, **học có giám sát** giả định thế giới sinh ra các cặp dữ liệu $$(X, Y)$$ từ một phân phối ẩn chưa biết $$P(X, Y)$$, và ta cần học một hàm dự đoán $$\hat f(X)$$ từ mẫu hữu hạn $$n$$ điểm. Ngành này đứng trên **bốn trụ cột**:

1. **Hồi quy vs. Phân loại:** Đầu ra $$Y$$ là số liên tục (giá nhà, nhiệt độ) → Hồi quy (thường dùng mất mát bình phương MSE); đầu ra $$Y$$ là nhãn rời rạc (thư rác/thường, bệnh/khỏe) → Phân loại (dùng cross-entropy hoặc 0–1 loss).
2. **Rủi ro thực nghiệm vs. Rủi ro kỳ vọng:** Máy tính chỉ nhìn thấy lỗi trung bình trên tập huấn luyện, trong khi mục tiêu thật là kỳ vọng lỗi trên phân phối tổng thể $$P(X, Y)$$.
3. **Phân rã Độ chệch – Phương sai (Bias–Variance Decomposition):** Mọi sai số dự đoán bình phương đều tách chính xác thành 3 phần: *Độ chệch bình phương* (mô hình quá đơn giản, bỏ lỡ quy luật thật) + *Phương sai* (mô hình quá nhạy, lắc lư mạnh mỗi khi đổi mẫu dữ liệu) + *Nhiễu bất khả kháng* của tự nhiên.
4. **Hình học cao chiều (Lời nguyền số chiều):** Khi số đặc trưng $$p$$ tăng cao, dữ liệu trở nên cực kỳ thưa thớt, buộc ta phải dùng điều chuẩn (regularization) hoặc khai thác cấu trúc ẩn.

---

## Học có giám sát (Supervised Learning) là gì?

Trong học có giám sát, ta có một tập dữ liệu huấn luyện gồm các cặp quan sát $$(X, Y)$$:
- $$X = (X_1, X_2, \ldots, X_p)$$: vector đặc trưng (features/predictors).
- $$Y$$: biến mục tiêu (target/response).

**Mục tiêu:** Học một hàm $$f$$ sao cho $$Y \approx f(X)$$ để dự đoán $$Y$$ cho các giá trị $$X$$ mới chưa từng quan sát.

### Hai loại bài toán chính

1. **Hồi quy (Regression):** $$Y$$ là biến liên tục. Ví dụ: dự đoán giá nhà, nhiệt độ hoặc doanh số bán hàng.
2. **Phân loại (Classification):** $$Y$$ là biến rời rạc (biến phân loại/categorical). Ví dụ: phân loại thư rác (spam/không spam) hoặc nhận diện chữ số viết tay.

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

## Hàm mất mát (Loss Function) và sai số huấn luyện

Để đánh giá chất lượng dự đoán của mô hình, ta định nghĩa **hàm mất mát (loss function)** $$L(Y, f(X))$$ nhằm đo độ sai lệch giữa giá trị thực $$Y$$ và giá trị dự đoán $$f(X)$$.

### Các hàm mất mát phổ biến

**Bài toán hồi quy:**
- **Sai số bình phương (Squared Error Loss — L2):** $$L(Y, f(X)) = (Y - f(X))^2$$
- **Sai số tuyệt đối (Absolute Error Loss — L1):** $$L(Y, f(X)) = |Y - f(X)|$$

**Bài toán phân loại:**
- **Hàm mất mát 0–1 (0-1 Loss):** $$L(Y, f(X)) = \mathbb{1}_{Y \neq f(X)}$$
- **Hàm mất mát Cross-Entropy:** $$L(Y, f(X)) = -\sum_k y_k \log f_k(X)$$

**Sai số huấn luyện (Training Error / Empirical Risk):**

$$\text{Err}_{\text{train}} = \frac{1}{n} \sum_{i=1}^n L(y_i, f(x_i))$$

Đây là sai số trung bình trên tập dữ liệu huấn luyện — dữ liệu mà mô hình đã được dùng để ước lượng tham số.

**Sai số kiểm tra (Test Error / Expected Prediction Error):**

$$\text{Err}_{\text{test}} = E[L(Y, f(X))]$$

Đây là sai số kỳ vọng trên dữ liệu mới và là thước đo thực sự cho khả năng tổng quát hóa của mô hình.

**Vấn đề cốt lõi:** Khi mô hình càng phức tạp, sai số huấn luyện luôn có xu hướng giảm, nhưng sai số kiểm tra có thể tăng lên do hiện tượng quá khớp (overfitting).

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

## Đánh đổi giữa độ chệch và phương sai (Bias–Variance Tradeoff)

Với sai số bình phương, sai số kiểm tra kỳ vọng có thể được phân tách thành ba thành phần:

$$\text{Expected Test Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

### Ý nghĩa của từng thành phần

1. **Độ chệch (Bias):** Sai số phát sinh khi mô hình quá đơn giản, không đủ khả năng nắm bắt quy luật thực của dữ liệu. Mô hình có độ chệch cao thường rơi vào tình trạng **thiếu khớp (underfitting)** — ví dụ dùng đường thẳng để khớp dữ liệu có quan hệ phi tuyến.
2. **Phương sai (Variance):** Sai số phát sinh khi mô hình quá nhạy với nhiễu trong tập huấn luyện, khiến dự đoán biến động mạnh nếu thay đổi tập dữ liệu huấn luyện. Mô hình có phương sai cao thường rơi vào tình trạng **quá khớp (overfitting)** — ví dụ dùng đa thức bậc cao khớp rất sát tập huấn luyện nhưng dự đoán kém trên tập kiểm tra.
3. **Sai số không thể khử (Irreducible Error):** Phần nhiễu vốn có trong dữ liệu, không thể giảm xuống bằng cách thay đổi mô hình.

**Sự đánh đổi:** Tăng độ phức tạp của mô hình thường làm giảm độ chệch nhưng lại làm tăng phương sai; ngược lại, đơn giản hóa mô hình giúp giảm phương sai nhưng làm tăng độ chệch. Mục tiêu của lựa chọn mô hình là tìm điểm cân bằng để tổng sai số kiểm tra đạt mức thấp nhất.

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

## Lời nguyền số chiều (Curse of Dimensionality)

Khi số đặc trưng $$p$$ tăng lên, không gian dữ liệu trở nên ngày càng thưa thớt (sparse), kéo theo ba hệ quả chính:

1. **Nhu cầu dữ liệu tăng theo hàm mũ:** Để duy trì cùng mật độ điểm trong không gian $$p$$ chiều, số lượng mẫu cần thiết tăng theo tỷ lệ $$n \propto 2^p$$.
2. **Khoảng cách kém phân biệt:** Trong không gian nhiều chiều, khoảng cách giữa các cặp điểm có xu hướng trở nên xấp xỉ nhau, làm giảm hiệu quả của các phương pháp dựa trên khoảng cách.
3. **Dễ xảy ra quá khớp:** Khi có quá nhiều đặc trưng so với số mẫu, mô hình dễ khớp vào nhiễu thay vì tín hiệu thực.

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

### Khung giải thích: tự tay kiểm chứng phân rã $$\mathrm{MSE} = \mathrm{Bias}^2 + \mathrm{Var}$$ bằng con số cụ thể (nên đọc chậm)

Khẩu hiệu "mô hình phức tạp giảm Bias nhưng tăng Variance" sẽ sáng tỏ ngay lập tức khi bạn tự tính sai số của 3 mô hình trên cùng một điểm kiểm tra $$x_0$$ có giá trị thật $$f(x_0) = 10$$ (bỏ qua nhiễu ngoài $$\sigma^2 = 0$$ để dễ tính). Giả sử ta huấn luyện mỗi mô hình trên **3 tập dữ liệu ngẫu nhiên khác nhau** và thu được 3 dự đoán $$\hat f^{(1)}, \hat f^{(2)}, \hat f^{(3)}$$:

- **Mô hình A (quá đơn giản — Underfitting):** Dự đoán $$\{7.9,\, 8.0,\, 8.1\}$$ (trung bình $$\bar{\hat f}_A = 8.0$$).
  - $$\mathrm{Bias}^2 = (8.0 - 10)^2 = 4.00$$.
  - $$\mathrm{Var} = \frac{(-0.1)^2 + 0^2 + (0.1)^2}{3} \approx 0.0067$$.
  - $$\mathrm{MSE}_A = 4.00 + 0.0067 = 4.0067$$ (rất ổn định nhưng luôn lệch xa đáp án thật!).
- **Mô hình B (quá phức tạp — Overfitting):** Dự đoán $$\{6,\, 10,\, 14\}$$ (trung bình $$\bar{\hat f}_B = 10.0$$).
  - $$\mathrm{Bias}^2 = (10 - 10)^2 = 0$$ (trung bình không hề chệch!).
  - $$\mathrm{Var} = \frac{(-4)^2 + 0^2 + 4^2}{3} = \frac{32}{3} \approx 10.67$$.
  - $$\mathrm{MSE}_B = 0 + 10.67 = 10.67$$ (nhảy loạn xạ tùy mẫu huấn luyện, tệ gấp đôi mô hình A!).
- **Mô hình C (cân bằng tốt):** Dự đoán $$\{9,\, 10,\, 11\}$$ (trung bình $$\bar{\hat f}_C = 10.0$$).
  - $$\mathrm{Bias}^2 = 0,\quad \mathrm{Var} = \frac{(-1)^2 + 0^2 + 1^2}{3} \approx 0.67 \;\Longrightarrow\; \mathrm{MSE}_C \approx 0.67$$ — **thắng áp đảo cả A lẫn B!**

Hãy tự kiểm tra lại bằng cách tính trực tiếp trung bình bình phương sai số với $$10$$ cho Mô hình B: $$\frac{(6-10)^2 + (10-10)^2 + (14-10)^2}{3} = \frac{32}{3} \approx 10.67$$ — khớp tuyệt đối với $$\mathrm{Bias}^2 + \mathrm{Var}$$!

## Video tham khảo

- [A Gentle Introduction to Machine Learning](https://www.youtube.com/watch?v=Gv9_4yMHFhI) — StatQuest with Josh Starmer, tiếng Anh. **Xem trước.** Giới thiệu khung học có giám sát — hồi quy khi $$Y$$ liên tục và phân loại khi $$Y$$ rời rạc — trước khi đi vào phần bias–variance.
- [Machine Learning Fundamentals: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA) — StatQuest with Josh Starmer, tiếng Anh. Trực quan hóa độ chệch và phương sai trên đường khớp dữ liệu, giải thích vì sao mô hình phức tạp hơn không phải lúc nào cũng dự đoán tốt hơn.
- [Statistical Learning: 2.3 Model Selection and Bias Variance Tradeoff](https://www.youtube.com/watch?v=pvcEQfcO3pk) — Stanford Online, tiếng Anh. Trình bày sự đánh đổi bias–variance trong bối cảnh học thống kê, kết nối trực tiếp với bài chọn mô hình ở phần sau.

## Bài tập thực hành

**Bài 1: Phân tích hiện tượng quá khớp (Overfitting)**
Sử dụng tập dữ liệu Boston Housing (trong scikit-learn). Thử hồi quy đa thức với các bậc khác nhau và vẽ đồ thị so sánh sai số huấn luyện và sai số kiểm tra.

**Bài 2: Phân tích đánh đổi Bias–Variance**
Tạo dữ liệu mô phỏng từ hàm thực $$f(x) = x^2$$. Thực hiện mô phỏng phân rã bias–variance cho mô hình hồi quy tuyến tính và hồi quy đa thức bậc 2.

**Bài 3: Dữ liệu nhiều chiều (High-Dimensional Data)**
Sinh dữ liệu ngẫu nhiên gồm $$p = 100$$ đặc trưng nhưng chỉ có 5 đặc trưng thực sự liên quan đến biến mục tiêu. Huấn luyện mô hình và quan sát hiện tượng quá khớp.
