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

Nếu bạn thêm một cột dữ liệu hoàn toàn ngẫu nhiên — chẳng hạn số đo cỡ giày của người bán nhà — vào mô hình hồi quy dự đoán giá nhà, hệ số xác định $$R^2$$ trên tập huấn luyện sẽ **không bao giờ giảm, mà gần như luôn tăng lên**! Quy luật trớ trêu ấy nhắc ta nhớ đến **Dao cạo Occam** trong khoa học: một mô hình phức tạp hơn luôn khớp dữ liệu cũ tốt hơn, nhưng liệu phần khớp thêm đó là quy luật thật hay chỉ đang ôm thêm nhiễu?

Thay vì phải chạy kiểm định chéo tốn kém hàng chục lần, thống kê toán học cung cấp các **tiêu chí lựa chọn mô hình** (AIC, BIC, Adjusted $$R^2$$, Mallows' $$C_p$$): cân đo trực tiếp giữa **độ khớp dữ liệu** và **án phạt cho số tham số** ngay trên cùng một công thức gọn gàng.

---

## Kiến thức cần nhớ: Lựa chọn mô hình và các tiêu chí thông tin là gì?

Các tiêu chí lựa chọn mô hình trong bài này đều mang chung cấu trúc $$\text{Điểm phạt} = \text{Sai số khớp} + \text{Án phạt độ phức tạp}$$, chia thành **bốn công cụ chính**:

1. **AIC (Akaike Information Criterion — $$2k - 2\ln\hat L$$):** Xuất phát từ lý thuyết thông tin (phân kỳ KL), hướng tới mục tiêu **dự đoán tốt nhất trên dữ liệu mới** (tiệm cận tương đương LOOCV).
2. **BIC (Bayesian Information Criterion — $$k\ln n - 2\ln\hat L$$):** Xuất phát từ xấp xỉ tích phân Bayes, phạt nặng hơn AIC khi $$n \ge 8$$ (vì $$\ln n > 2$$), ưu tiên tìm ra **mô hình đúng tối giản** khi cỡ mẫu lớn.
3. **Adjusted $$R^2$$:** Chỉnh lại $$R^2$$ cổ điển theo bậc tự do $$(n - k - 1)$$, chỉ tăng khi biến mới cải thiện mô hình đủ tốt hơn mức ngẫu nhiên.
4. **Mallows' $$C_p$$:** Ước lượng trực tiếp sai số dự đoán bình phương ngoài mẫu trong hồi quy tuyến tính; mô hình tốt thường có $$C_p \approx p$$.

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

### Giải thích: tự tay tính AIC và BIC để thấy hai tiêu chí chọn khác nhau khi nào

Cùng một bộ dữ liệu gồm $$n = 100$$ quan sát ($$\ln 100 \approx 4.605$$), xét hai mô hình hồi quy:
- **Mô hình 1 (đơn giản):** $$k_1 = 2$$ tham số, log-likelihood cực đại $$\ln \hat L_1 = -50.0$$.
- **Mô hình 2 (phức tạp hơn):** thêm 2 biến mới nên $$k_2 = 4$$ tham số, giúp log-likelihood cải thiện lên $$\ln \hat L_2 = -47.5$$ (tăng thêm $$\Delta\ln\hat L = +2.5$$).

Hãy tự thay số vào hai công thức (giá trị càng **nhỏ** càng tốt):

1. **Tính AIC ($$2k - 2\ln\hat L$$):**
   - $$\mathrm{AIC}_1 = 2(2) - 2(-50.0) = 4 + 100 = 104.0$$.
   - $$\mathrm{AIC}_2 = 2(4) - 2(-47.5) = 8 + 95 = 103.0$$.
   Vì $$103.0 < 104.0$$, **AIC chọn Mô hình 2** (phần thưởng khớp thêm $$2\Delta\ln\hat L = 5$$ lớn hơn án phạt thêm $$2\Delta k = 4$$).

2. **Tính BIC ($$k\ln n - 2\ln\hat L$$):**
   - $$\mathrm{BIC}_1 = 2(4.605) - 2(-50.0) = 9.21 + 100 = 109.21$$.
   - $$\mathrm{BIC}_2 = 4(4.605) - 2(-47.5) = 18.42 + 95 = 113.42$$.
   Vì $$109.21 < 113.42$$, **BIC lại chọn Mô hình 1** (với $$n=100$$, mỗi tham số mới bị BIC phạt tới $$4.605$$ thay vì $$2$$, nên 2 tham số mới bị phạt $$9.21 > 5$$)!

**Bài học thực hành:** Nếu mục tiêu của bạn là *dự đoán sát nhất*, hãy nghiêng về AIC (hoặc CV); nếu mục tiêu là *giải thích khoa học, chọn tập biến gọn gàng nhất*, hãy ưu tiên BIC.

## Video tham khảo

- [Model selection with AIC and AICc](https://www.youtube.com/watch?v=HOqHI53x9Go) — TileStats, tiếng Anh. **Xem trước.** Giải thích cách AIC cộng thêm thành phần phạt theo số tham số và chọn mô hình có tiêu chí nhỏ nhất, khớp với công thức và quy tắc trong bài.
- [Machine Learning Fundamentals: Cross Validation](https://www.youtube.com/watch?v=fSytzGwwBVw) — StatQuest with Josh Starmer, tiếng Anh. Trình bày cách tiếp cận thực nghiệm được nhắc ở đầu bài: ước lượng sai số dự đoán bằng cross-validation thay vì tiêu chí phạt giải tích.

## Bài tập

**Bài 1:** Cài đặt hàm tính BIC từ đầu và so sánh kết quả chọn bậc đa thức giữa BIC và AIC trên bài toán hồi quy đa thức.

**Bài 2:** Sử dụng tiêu chí AIC để lựa chọn tập đặc trưng trong hồi quy tuyến tính theo phương pháp chọn tiến (forward selection) hoặc loại lùi (backward selection).
