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

Sau khi huấn luyện một mô hình hồi quy tuyến tính, chúng ta thường muốn trả lời các câu hỏi suy diễn như: đặc trưng này có thực sự có ý nghĩa thống kê không, khoảng tin cậy của từng hệ số hồi quy là bao nhiêu, và dự đoán cho một quan sát mới có độ bất định ra sao? Đây là trọng tâm của suy diễn thống kê (statistical inference) — không chỉ đưa ra giá trị dự đoán mà còn định lượng độ bất định đi kèm.

---

## Khoảng Tin Cậy Cho Các Hệ Số Hồi Quy

Trong mô hình hồi quy tuyến tính với giả định nhiễu chuẩn $$\epsilon \sim \mathcal{N}(0, \sigma^2)$$, phân phối của hệ số ước lượng thỏa mãn:

$$\hat{\beta}_j \sim \mathcal{N}\left(\beta_j, \sigma^2 (X^TX)^{-1}_{jj}\right)$$

**Khoảng tin cậy 95% cho $$\beta_j$$:**

$$\hat{\beta}_j \pm t_{0.025, n-p-1} \cdot \text{SE}(\hat{\beta}_j)$$

trong đó sai số chuẩn là $$\text{SE}(\hat{\beta}_j) = \hat{\sigma}\sqrt{(X^TX)^{-1}_{jj}}$$, với $$p$$ là số biến đặc trưng (không tính hệ số chặn) và bậc tự do là $$n - p - 1$$.

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
    
    # Estimate sigma (n - p - 1 bậc tự do vì mô hình có thêm hệ số chặn)
    residuals = y - y_pred
    sigma_squared = np.sum(residuals**2) / (n - p - 1)
    
    # Variance-covariance matrix
    X_with_intercept = np.column_stack([np.ones(n), X])
    var_covar = sigma_squared * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
    
    # Standard errors
    se = np.sqrt(np.diag(var_covar))
    
    # t-statistics
    coefs = np.concatenate([[model.intercept_], model.coef_])
    t_stats = coefs / se
    
    # p-values
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n-p-1))
    
    # Confidence intervals
    t_critical = stats.t.ppf(1 - alpha/2, df=n-p-1)
    ci_lower = coefs - t_critical * se
    ci_upper = coefs + t_critical * se
    
    # Print results
    print("Coefficient Inference:")
    print("=" * 70)
    print(f"{'Variable':<15} {'Coef':>10} {'SE':>10} {'t-stat':>10} {'p-value':>10}")
    print("-" * 70)
    
    names = ['Intercept'] + [f'X{i+1}' for i in range(p)]
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

## Kiểm Định Giả Thuyết Cho Từng Hệ Số

**Giả thuyết không:** $$H_0: \beta_j = 0$$ (biến đặc trưng thứ $$j$$ không có tác động tuyến tính lên biến mục tiêu khi giữ cố định các biến khác).

**Thống kê kiểm định:**

$$t = \frac{\hat{\beta}_j}{\text{SE}(\hat{\beta}_j)} \sim t_{n-p-1}$$

**Quy tắc quyết định:** Bác bỏ $$H_0$$ nếu $$|t| > t_{\alpha/2, n-p-1}$$ hoặc p-value $$< \alpha$$.

## Kiểm Định F Cho Ý Nghĩa Tổng Thể (Overall Significance)

Kiểm định đồng thời xem **tất cả** các hệ số độ dốc có cùng bằng 0 hay không:

$$H_0: \beta_1 = \beta_2 = \cdots = \beta_p = 0$$

**Thống kê F:**

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
    
    # F-statistic (n - p - 1 bậc tự do cho phần dư khi có hệ số chặn)
    f_stat = ((tss - rss) / p) / (rss / (n - p - 1))
    
    # p-value
    p_value = 1 - stats.f.cdf(f_stat, p, n - p - 1)
    
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {p_value:.6f}")
    
    if p_value < 0.05:
        print("Kết luận: Reject H₀ - Model có ý nghĩa thống kê")
    else:
        print("Kết luận: Không đủ bằng chứng để bác bỏ H₀")

# f_test(X, y)
```

## Khoảng Dự Báo (Prediction Intervals)

Cần phân biệt hai loại khoảng bất định khi dự đoán tại điểm mới:
- **Khoảng tin cậy cho E[Y|X]:** Đo lường độ bất định của giá trị dự đoán trung bình.
- **Khoảng dự báo cho Y|X:** Đo lường độ bất định cho một quan sát cá lẻ mới, do đó luôn rộng hơn khoảng tin cậy vì phải tính thêm phương sai nhiễu $$\sigma^2$$ của quan sát mới.

$$\hat{y} \pm t_{\alpha/2, n-p-1} \cdot \hat{\sigma}\sqrt{1 + \mathbf{x}_0^T(X^TX)^{-1}\mathbf{x}_0}$$

## Video Tham Khảo

- [Linear Regression, Clearly Explained!!!](https://www.youtube.com/watch?v=7ArmBVF2dCs) — StatQuest with Josh Starmer, tiếng Anh. **Nên xem trước.** Trình bày suy diễn cho hệ số hồi quy: khoảng tin cậy và p-value của hệ số góc, tương ứng với phần đầu của bài.
- [Multiple Regression, Clearly Explained!!!](https://www.youtube.com/watch?v=zITIFTsivN8) — StatQuest with Josh Starmer, tiếng Anh. Giải thích kiểm định F để đánh giá ý nghĩa tổng thể khi bổ sung biến vào mô hình.
- [Prediction Intervals vs. Confidence Intervals](https://www.youtube.com/watch?v=nA0BnTm_bU0) — Stats with Brian, tiếng Anh. Phân biệt khoảng tin cậy cho giá trị trung bình có điều kiện và khoảng dự báo cho một quan sát mới.

## Bài Tập

**Bài 1:** Cài đặt hàm tính khoảng dự báo (prediction intervals) cho mô hình hồi quy tuyến tính và vẽ biểu đồ minh họa so sánh với khoảng tin cậy.

**Bài 2:** Áp dụng hiệu chỉnh đa kiểm định (Bonferroni) khi thực hiện kiểm định đồng thời nhiều hệ số hồi quy.
