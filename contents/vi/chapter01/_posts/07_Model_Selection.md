---
layout: post
title: 01-08-00 Tiêu Chí Lựa Chọn Mô Hình
chapter: "01"
order: 8
owner: nglelinh
lang: vi
categories:
- chapter01
lesson_type: required
---

Làm thế nào để chọn giữa một mô hình đơn giản và một mô hình phức tạp? Trong khi kiểm định chéo (cross-validation) đưa ra câu trả lời bằng thực nghiệm, các tiêu chí thống kê như AIC và BIC cung cấp khung lý thuyết để cân bằng giữa mức độ khớp dữ liệu (goodness-of-fit) và độ phức tạp của mô hình. Bài học này giới thiệu các tiêu chí phổ biến dùng cho việc lựa chọn mô hình một cách có nguyên tắc.

---

## Bài toán lựa chọn mô hình

Khi so sánh các mô hình ứng viên, ta luôn đối mặt với sự đánh đổi cơ bản:
- **Mô hình phức tạp** khớp sát dữ liệu huấn luyện nhưng dễ bị quá khớp (overfitting).
- **Mô hình đơn giản** dễ bị thiếu khớp (underfitting) nhưng thường có khả năng tổng quát hóa ổn định hơn.

Do đó, ta cần một tiêu chí định lượng cân bằng giữa độ khớp dữ liệu và độ phức tạp mô hình.

## Tiêu chí thông tin Akaike (AIC)

AIC (Akaike Information Criterion) ước lượng sai số dự đoán kỳ vọng của mô hình theo công thức:

$$\text{AIC} = -2\log L(\hat{\theta}) + 2k$$

trong đó:
- $$L(\hat{\theta})$$: giá trị hàm hợp lý cực đại (maximum likelihood).
- $$k$$: số lượng tham số được ước lượng của mô hình.

**Nguyên tắc:** Chọn mô hình có giá trị AIC **nhỏ nhất**.

**Ý nghĩa hai thành phần:**
- Thành phần thứ nhất $$-2\log L(\hat{\theta})$$ đo mức độ khớp dữ liệu (giá trị càng nhỏ nghĩa là mô hình khớp dữ liệu huấn luyện càng sát).
- Thành phần thứ hai $$2k$$ là đại lượng phạt cho độ phức tạp, tăng tuyến tính theo số tham số $$k$$.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

def calculate_aic(y_true, y_pred, n_params):
    """
    Tính AIC cho linear model
    """
    n = len(y_true)
    mse = mean_squared_error(y_true, y_pred)
    
    # Log-likelihood (giả định Gaussian errors)
    log_likelihood = -n/2 * np.log(2*np.pi*mse) - n/2
    
    aic = -2 * log_likelihood + 2 * n_params
    return aic

def model_selection_with_aic():
    """
    So sánh models bằng AIC
    """
    # Sinh dữ liệu
    np.random.seed(42)
    X = np.linspace(0, 1, 50).reshape(-1, 1)
    y = np.sin(2*np.pi*X).ravel() + np.random.normal(0, 0.2, 50)
    
    degrees = range(1, 11)
    aics = []
    bics = []
    
    for degree in degrees:
        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        
        n_params = X_poly.shape[1]
        aic = calculate_aic(y, y_pred, n_params)
        
        # BIC
        n = len(y)
        log_likelihood = -n/2 * np.log(2*np.pi*mean_squared_error(y, y_pred)) - n/2
        bic = -2 * log_likelihood + n_params * np.log(n)
        
        aics.append(aic)
        bics.append(bic)
    
    # Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, aics, 'o-', label='AIC', linewidth=2)
    plt.plot(degrees, bics, 's-', label='BIC', linewidth=2)
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Information Criterion')
    plt.title('Model Selection: AIC vs BIC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()
    
    optimal_aic = degrees[np.argmin(aics)]
    optimal_bic = degrees[np.argmin(bics)]
    
    print(f"Optimal degree (AIC): {optimal_aic}")
    print(f"Optimal degree (BIC): {optimal_bic}")

# model_selection_with_aic()
```

## Tiêu chí thông tin Bayes (BIC)

BIC (Bayesian Information Criterion) có cấu trúc tương tự AIC nhưng áp dụng mức phạt độ phức tạp phụ thuộc vào cỡ mẫu $$n$$:

$$\text{BIC} = -2\log L(\hat{\theta}) + k\log(n)$$

**So sánh AIC và BIC:**
- Khi $$n > 7$$ (tức $$\log n > 2$$), BIC phạt độ phức tạp mạnh hơn AIC.
- Do mức phạt lớn hơn, BIC có xu hướng chọn mô hình đơn giản hơn.
- AIC thường phù hợp hơn khi mục tiêu chính là dự đoán; BIC thường phù hợp hơn khi mục tiêu là xác định đúng mô hình sinh dữ liệu ("true model").

## Hệ số xác định hiệu chỉnh (Adjusted $$R^2$$)

$$R^2_{\text{adj}} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$$

trong đó $$n$$ là cỡ mẫu và $$p$$ là số biến giải thích của mô hình.

- **Ưu điểm:** Công thức đơn giản, dễ tính toán và trực quan.
- **Nhược điểm:** Chỉ áp dụng trực tiếp cho các mô hình hồi quy tuyến tính.

## Tiêu chí Mallows' $$C_p$$

$$C_p = \frac{\text{RSS}_p}{\hat{\sigma}^2} - n + 2p$$

trong đó $$\text{RSS}_p$$ là tổng bình phương phần dư của mô hình con có $$p$$ tham số và $$\hat{\sigma}^2$$ là ước lượng phương sai nhiễu lấy từ mô hình đầy đủ (full model).

**Nguyên tắc:** Chọn mô hình có $$C_p$$ nhỏ và xấp xỉ $$p$$ ($$C_p \approx p$$).

## Video tham khảo

- [Model selection with AIC and AICc](https://www.youtube.com/watch?v=HOqHI53x9Go) — TileStats, tiếng Anh. **Xem trước.** Giải thích cách AIC cộng thêm thành phần phạt theo số tham số và chọn mô hình có tiêu chí nhỏ nhất, khớp với công thức và quy tắc trong bài.
- [Machine Learning Fundamentals: Cross Validation](https://www.youtube.com/watch?v=fSytzGwwBVw) — StatQuest with Josh Starmer, tiếng Anh. Trình bày cách tiếp cận thực nghiệm được nhắc ở đầu bài: ước lượng sai số dự đoán bằng cross-validation thay vì tiêu chí phạt giải tích.

## Bài tập

**Bài 1:** Cài đặt hàm tính BIC từ đầu và so sánh kết quả chọn bậc đa thức giữa BIC và AIC trên bài toán hồi quy đa thức.

**Bài 2:** Sử dụng tiêu chí AIC để lựa chọn tập đặc trưng trong hồi quy tuyến tính theo phương pháp chọn tiến (forward selection) hoặc loại lùi (backward selection).
