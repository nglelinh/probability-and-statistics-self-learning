---
layout: post
title: 01-09-00 Inference cho Mô Hình Tuyến Tính
chapter: "06"
order: 2
owner: nglelinh
lang: vi
categories:
- chapter06
lesson_type: required
---

Sau khi train mô hình, chúng ta thường muốn trả lời các câu hỏi: "Feature này có thực sự quan trọng không?", "Khoảng tin cậy của coefficient là bao nhiêu?", "Dự đoán cho điểm mới có độ tin cậy như thế nào?". Đây là lĩnh vực của statistical inference - không chỉ dự đoán mà còn định lượng uncertainty.

---

## Confidence Intervals cho Coefficients

Trong linear regression, giả định $$\epsilon \sim \mathcal{N}(0, \sigma^2)$$:

$$\hat{\beta}_j \sim \mathcal{N}\left(\beta_j, \sigma^2 (X^TX)^{-1}_{jj}\right)$$

**Confidence interval 95%:**

$$\hat{\beta}_j \pm t_{0.025, n-p-1} \cdot \text{SE}(\hat{\beta}_j)$$

trong đó $$\text{SE}(\hat{\beta}_j) = \hat{\sigma}\sqrt{(X^TX)^{-1}_{jj}}$$

```python
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

def coefficient_inference(X, y, alpha=0.05):
    """
    Tính confidence intervals và p-values cho coefficients
    """
    n, p = X.shape
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Estimate sigma
    residuals = y - y_pred
    sigma_squared = np.sum(residuals**2) / (n - p)
    
    # Variance-covariance matrix
    X_with_intercept = np.column_stack([np.ones(n), X])
    var_covar = sigma_squared * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
    
    # Standard errors
    se = np.sqrt(np.diag(var_covar))
    
    # t-statistics
    coefs = np.concatenate([[model.intercept_], model.coef_])
    t_stats = coefs / se
    
    # p-values
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n-p))
    
    # Confidence intervals
    t_critical = stats.t.ppf(1 - alpha/2, df=n-p)
    ci_lower = coefs - t_critical * se
    ci_upper = coefs + t_critical * se
    
    # Print results
    print("Coefficient Inference:")
    print("=" * 70)
    print(f"{'Variable':<15} {'Coef':>10} {'SE':>10} {'t-stat':>10} {'p-value':>10}")
    print("-" * 70)
    
    names = ['Intercept'] + [f'X{i+1}' for i in range(p-1)]
    for i, name in enumerate(names):
        print(f"{name:<15} {coefs[i]:>10.4f} {se[i]:>10.4f} {t_stats[i]:>10.4f} {p_values[i]:>10.4f}")
    
    print("\n95% Confidence Intervals:")
    for i, name in enumerate(names):
        print(f"{name:<15} [{ci_lower[i]:>8.4f}, {ci_upper[i]:>8.4f}]")

# Example
# X = np.random.randn(100, 3)
# y = 2 + 3*X[:, 0] - 1.5*X[:, 1] + 0.5*X[:, 2] + np.random.randn(100)
# coefficient_inference(X, y)
```

## Hypothesis Testing

**Null hypothesis:** $$H_0: \beta_j = 0$$ (feature không có ảnh hưởng)

**Test statistic:**

$$t = \frac{\hat{\beta}_j}{\text{SE}(\hat{\beta}_j)} \sim t_{n-p-1}$$

**Decision:** Reject $$H_0$$ nếu $$|t| > t_{\alpha/2, n-p-1}$$ hoặc p-value $$< \alpha$$.

## F-test cho Overall Significance

Test xem **tất cả** coefficients có bằng 0 không:

$$H_0: \beta_1 = \beta_2 = \cdots = \beta_p = 0$$

**F-statistic:**

$$F = \frac{(\text{TSS} - \text{RSS})/p}{\text{RSS}/(n-p-1)} \sim F_{p, n-p-1}$$

```python
def f_test(X, y):
    """
    F-test cho overall significance
    """
    n, p = X.shape
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Sum of squares
    tss = np.sum((y - y.mean())**2)
    rss = np.sum((y - y_pred)**2)
    
    # F-statistic
    f_stat = ((tss - rss) / p) / (rss / (n - p))
    
    # p-value
    p_value = 1 - stats.f.cdf(f_stat, p, n - p)
    
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {p_value:.6f}")
    
    if p_value < 0.05:
        print("Kết luận: Reject H₀ - Model có ý nghĩa thống kê")
    else:
        print("Kết luận: Không đủ bằng chứng để bác bỏ H₀")

# f_test(X, y)
```

## Prediction Intervals

**Confidence interval cho E[Y|X]:** Uncertainty về mean prediction
**Prediction interval cho Y|X:** Uncertainty về individual prediction (rộng hơn)

$$\hat{y} \pm t_{\alpha/2, n-p-1} \cdot \hat{\sigma}\sqrt{1 + \mathbf{x}_0^T(X^TX)^{-1}\mathbf{x}_0}$$

## Bài Tập

**Bài 1:** Implement prediction intervals và visualize chúng.

**Bài 2:** Multiple testing correction (Bonferroni) khi test nhiều coefficients.
