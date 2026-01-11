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

Làm thế nào để chọn giữa mô hình đơn giản và mô hình phức tạp? Cross-validation cho câu trả lời thực nghiệm, nhưng các tiêu chí thống kê như AIC và BIC cung cấp framework lý thuyết để cân bằng giữa goodness-of-fit và complexity. Bài học này giới thiệu các công cụ quan trọng để model selection một cách có nguyên tắc.

---

## Vấn Đề Model Selection

**Trade-off cơ bản:**
- Mô hình phức tạp: fit tốt training data nhưng có thể overfit
- Mô hình đơn giản: underfitting nhưng generalize tốt hơn

**Cần:** Một tiêu chí cân bằng giữa fit và complexity.

## Akaike Information Criterion (AIC)

AIC ước lượng expected prediction error:

$$\text{AIC} = -2\log L(\hat{\theta}) + 2k$$

trong đó:
- $$L(\hat{\theta})$$: Maximum likelihood
- $$k$$: Số parameters

**Nguyên tắc:** Chọn model có AIC **nhỏ nhất**.

**Ý nghĩa:**
- Term 1: Đo goodness-of-fit (càng nhỏ càng tốt)
- Term 2: Penalty cho complexity

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

## Bayesian Information Criterion (BIC)

BIC tương tự AIC nhưng penalty mạnh hơn:

$$\text{BIC} = -2\log L(\hat{\theta}) + k\log(n)$$

**So sánh AIC vs BIC:**
- BIC penalty mạnh hơn khi $$n > 7$$ (vì $$\log n > 2$$)
- BIC xu hướng chọn mô hình đơn giản hơn
- AIC tốt hơn cho prediction
- BIC tốt hơn cho model selection (tìm "true model")

## Adjusted R²

$$R^2_{\text{adj}} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$$

**Ưu điểm:** Đơn giản, dễ hiểu
**Nhược điểm:** Chỉ áp dụng cho linear models

## Mallows' Cp

$$C_p = \frac{\text{RSS}_p}{\hat{\sigma}^2} - n + 2p$$

trong đó $$\hat{\sigma}^2$$ từ full model.

**Nguyên tắc:** Chọn model có $$C_p \approx p$$.

## Bài Tập

**Bài 1:** Implement BIC từ đầu và so sánh với AIC trên polynomial regression.

**Bài 2:** Sử dụng AIC để chọn features trong linear regression (forward/backward selection).
