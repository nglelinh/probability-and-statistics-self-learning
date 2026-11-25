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

Chỉ cần hai dòng lệnh `LinearRegression().fit(X, y)` trong Python, máy tính sẽ lập tức in ra một hệ số rất đẹp mắt: chẳng hạn mỗi triệu đồng tăng thêm cho ngân sách quảng cáo giúp doanh số tăng thêm **2,40 triệu đồng**. Nhưng nếu dừng lại ở con số đó, bạn đang lái xe mà không nhìn kính chắn gió: con số `2,40` kia là một quy luật vững chắc ($$2{,}40 \pm 0{,}30$$) hay chỉ là dao động vô nghĩa quanh số 0 ($$2{,}40 \pm 5{,}00$$)?

**Suy diễn thống kê cho mô hình tuyến tính** bổ sung đúng phần còn thiếu giữa "khớp đường thẳng" và "ra quyết định khoa học". Bài học này giải mã bốn công cụ xuất hiện trong mọi bảng tóm tắt hồi quy (`statsmodels.OLS.summary()`): **khoảng tin cậy của từng hệ số**, **kiểm định $$t$$ cho từng biến**, **kiểm định $$F$$ cho ý nghĩa tổng thể của cả mô hình**, và sự khác biệt quan trọng giữa **khoảng tin cậy cho trung bình** với **khoảng dự báo cho một quan sát mới**.

---

## Kiến thức cần nhớ: Bốn mảnh ghép trong bảng suy diễn hồi quy tuyến tính

Khi đọc một báo cáo hồi quy tuyến tính cổ điển (OLS dưới giả định nhiễu chuẩn $$\epsilon \sim \mathcal{N}(0, \sigma^2 I)$$), mọi con số đều xoay quanh bốn trụ cột sau:

1. **Ma trận hiệp phương sai của ước lượng hệ số:** Nhờ tính tuyến tính của $$\hat{\beta} = (X^T X)^{-1} X^T y$$, phân phối lấy mẫu của các hệ số là phân phối chuẩn đa biến với hiệp phương sai $$\sigma^2 (X^T X)^{-1}$$. Các phần tử chéo cho ta sai số chuẩn $$\mathrm{SE}(\hat{\beta}_j)$$ — thước đo độ rung lắc của từng hệ số (và khi các biến bị đa cộng tuyến, $$(X^TX)^{-1}_{jj}$$ phình to khiến $$\mathrm{SE}$$ tăng vọt!).
2. **Kiểm định $$t$$ riêng lẻ vs. Kiểm định $$F$$ tổng thể:** Kiểm định $$t$$ hỏi *"giữ nguyên các biến khác, biến $$X_j$$ có đóng góp gì thêm không?"* ($$H_0: \beta_j = 0$$). Kiểm định $$F$$ hỏi *"cả tập $$p$$ biến cùng lúc có giải thích được gì hơn một đường ngang trung bình $$\bar{y}$$ không?"* ($$H_0: \beta_1 = \cdots = \beta_p = 0$$). Trong hồi quy đa biến có tương quan cao, hoàn toàn có thể xảy ra chuyện kiểm định $$F$$ cực kỳ có ý nghĩa nhưng từng kiểm định $$t$$ riêng lẻ lại không có ý nghĩa!
3. **Bậc tự do $$n - p - 1$$:** Ta bắt đầu với $$n$$ quan sát độc lập và phải ước lượng $$p + 1$$ tham số ($$p$$ hệ số góc và 1 hệ số chặn $$\beta_0$$), nên phương sai nhiễu $$\hat{\sigma}^2 = \frac{\mathrm{RSS}}{n-p-1}$$ chỉ còn $$n - p - 1$$ bậc tự do.
4. **Bất định nhận thức (epistemic) vs. Bất định ngẫu nhiên (aleatoric):** Dự báo *giá trị trung bình* $$\mathbb{E}[Y \mid x_0]$$ chỉ chịu bất định từ việc ước lượng đường hồi quy (biến mất khi $$n\to\infty$$). Dự báo *một quan sát cá biệt mới* $$Y_{\text{new}}$$ phải gánh thêm nhiễu nội tại $$\epsilon_{\text{new}} \sim \mathcal{N}(0, \sigma^2)$$ không bao giờ biến mất — khiến **khoảng dự báo (Prediction Interval)** luôn rộng hơn hẳn **khoảng tin cậy (Confidence Interval)**.

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

---

### Ví dụ tính tay làm từng bước: Giải mã trọn vẹn một bảng hồi quy $$n = 22$$ tháng

Giả sử ta hồi quy doanh số hàng tháng $$Y$$ theo ngân sách quảng cáo $$X$$ ($$p = 1$$ biến, $$n = 22$$ tháng, bậc tự do $$df = n - p - 1 = 20$$) và thu được:
- Hệ số góc ước lượng: $$\hat{\beta}_1 = 2{,}40$$, sai số chuẩn $$\mathrm{SE}(\hat{\beta}_1) = 0{,}80$$.
- Độ lệch chuẩn phần dư: $$\hat{\sigma} = 4{,}0$$ triệu đồng.

**1. Kiểm định $$t$$ và khoảng tin cậy 95% cho hệ số góc $$\beta_1$$:**
- Thống kê $$t$$ cho $$H_0: \beta_1 = 0$$:
  $$t = \frac{\hat{\beta}_1 - 0}{\mathrm{SE}(\hat{\beta}_1)} = \frac{2{,}40}{0{,}80} = 3{,}00.$$
- Tra phân phối $$t$$ hai phía ở $$df = 20$$: $$t_{0{,}025;\,20} \approx 2{,}086$$ (và $$p\text{-value} \approx 0{,}0071 < 0{,}05$$). Vì $$3{,}00 > 2{,}086$$, ta bác bỏ $$H_0$$: quảng cáo có tác động dương có ý nghĩa thống kê.
- Khoảng tin cậy 95% cho $$\beta_1$$:
  $$2{,}40 \pm 2{,}086 \times 0{,}80 = 2{,}40 \pm 1{,}669 = [0{,}731,\; 4{,}069].$$
  *(Nhận xét: khoảng không chứa số 0 — luôn tương đương tuyệt đối với việc kiểm định $$t$$ hai phía bác bỏ $$H_0: \beta_1 = 0$$ ở cùng mức $$\alpha = 0{,}05$$!)*

**2. So sánh trực tiếp Khoảng tin cậy vs. Khoảng dự báo tại điểm trung bình $$x_0 = \bar{x}$$:**
- Giả sử giá trị dự báo điểm tại $$\bar{x}$$ là $$\hat{y}_0 = 50{,}0$$ triệu đồng.
- **Khoảng tin cậy 95% cho doanh số trung bình** tại mức ngân sách $$\bar{x}$$:
  $$\mathrm{SE}_{\text{mean}} = \frac{\hat{\sigma}}{\sqrt{n}} = \frac{4{,}0}{\sqrt{22}} \approx 0{,}853 \implies 50{,}0 \pm 2{,}086(0{,}853) = [48{,}22,\; 51{,}78].$$
- **Khoảng dự báo 95% cho doanh số của MỘT tháng mới cụ thể** ở mức ngân sách $$\bar{x}$$:
  $$\mathrm{SE}_{\text{pred}} = \hat{\sigma}\sqrt{1 + \frac{1}{n}} = 4{,}0\sqrt{1 + \frac{1}{22}} \approx 4{,}090 \implies 50{,}0 \pm 2{,}086(4{,}090) = [41{,}47,\; 58{,}53].$$

Khoảng dự báo rộng tới **$$\pm 8{,}53$$ triệu** — gần gấp **5 lần** khoảng tin cậy của trung bình ($$\pm 1{,}78$$ triệu) — vì ngay cả khi ta biết chính xác 100% đường hồi quy thật, doanh số từng tháng đơn lẻ vẫn dao động ngẫu nhiên với độ lệch chuẩn $$\sigma = 4{,}0$$!

### Giải thích: Bốn lưu ý quan trọng khi suy diễn trên mô hình tuyến tính

1. **Nhìn cặp ($$\hat{\beta}_j$$, $$\mathrm{SE}$$) trước khi nhìn dấu sao p-value:** Khoảng tin cậy $$[0{,}731,\; 4{,}069]$$ nói cho bạn nhiều thông tin hơn hẳn dòng chữ "$$p < 0{,}01$$" — nó cho thấy tác động thật có thể khiêm tốn ở mức $$0{,}73$$ hoặc rất lớn ở mức $$4{,}07$$.
2. **Đừng bỏ qua kiểm định $$F$$ khi có nhiều biến:** Nếu mô hình có 50 biến độc lập không liên quan gì đến $$Y$$, trung bình sẽ có 2–3 biến tình cờ đạt $$p < 0{,}05$$ ở kiểm định $$t$$ riêng lẻ. Kiểm định $$F$$ tổng thể đóng vai trò "cổng kiểm soát" đầu tiên chống lại ảo giác này.
3. **Chọn đúng khoảng cho đúng câu hỏi nghiệp vụ:** Báo cáo *hiệu quả trung bình* của chiến dịch dài hạn dùng **Confidence Interval**; lập kế hoạch tồn kho hay ngân sách dự phòng cho *một cửa hàng / một tháng cụ thể* bắt buộc dùng **Prediction Interval**.
4. **Nhớ kiểm tra giả định nhiễu:** Các công thức $$t$$ và $$F$$ trên dựa vào giả định phần dư độc lập và có phương sai không đổi (homoscedasticity). Nếu phương sai nhiễu thay đổi theo $$X$$, cần dùng sai số chuẩn bền vững (Huber–White / sandwich SE) hoặc Bootstrap.

**Kiểm tra nhanh.** Trong ví dụ $$n = 22$$ ở trên, giả sử ta thu thập thêm dữ liệu lên tới $$n = 22{,}000$$ tháng (với cùng $$\hat{\sigma} = 4{,}0$$). Độ rộng của **khoảng tin cậy cho trung bình** sẽ tiến về đâu, còn độ rộng của **khoảng dự báo cho một tháng mới** sẽ tiến về xấp xỉ bao nhiêu ($$\pm ?$$)?

---

## Video Tham Khảo

- [Linear Regression, Clearly Explained!!!](https://www.youtube.com/watch?v=7ArmBVF2dCs) — StatQuest with Josh Starmer, tiếng Anh. **Nên xem trước.** Trình bày suy diễn cho hệ số hồi quy: khoảng tin cậy và p-value của hệ số góc, tương ứng với phần đầu của bài.
- [Multiple Regression, Clearly Explained!!!](https://www.youtube.com/watch?v=zITIFTsivN8) — StatQuest with Josh Starmer, tiếng Anh. Giải thích kiểm định F để đánh giá ý nghĩa tổng thể khi bổ sung biến vào mô hình.
- [Prediction Intervals vs. Confidence Intervals](https://www.youtube.com/watch?v=nA0BnTm_bU0) — Stats with Brian, tiếng Anh. Phân biệt khoảng tin cậy cho giá trị trung bình có điều kiện và khoảng dự báo cho một quan sát mới.

## Bài Tập

**Bài 1:** Cài đặt hàm tính khoảng dự báo (prediction intervals) cho mô hình hồi quy tuyến tính và vẽ biểu đồ minh họa so sánh với khoảng tin cậy.

**Bài 2:** Áp dụng hiệu chỉnh đa kiểm định (Bonferroni) khi thực hiện kiểm định đồng thời nhiều hệ số hồi quy.
