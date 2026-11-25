---
layout: post
title: 01-04-00 Hồi Quy Tuyến Tính và Phương Pháp Bình Phương Tối Thiểu
chapter: "07"
order: 3
owner: nglelinh
lang: vi
categories:
- chapter07
lesson_type: required
---

Một nhà phân tích dữ liệu hào hứng khoe với sếp: *"Sau khi thêm 15 biến mới vào mô hình dự báo doanh thu, hệ số $$R^2$$ đã tăng vọt từ 0,80 lên 0,94!"* Nhưng khi đưa mô hình lên dữ liệu tháng sau, sai số dự báo lại tệ hơn cả mô hình ban đầu. Lý do nằm ở một tính chất toán học bị nhiều người quên mất: **cứ thêm bất kỳ biến nào — kể cả cột số ngẫu nhiên từ bảng xổ số — vào mô hình OLS, $$R^2$$ trên tập huấn luyện không bao giờ giảm!**

Bài học này đào sâu vào nền tảng đánh giá và kiểm chứng mô hình hồi quy tuyến tính: chứng minh trực giác **Định lý Gauss–Markov**, so sánh **$$R^2$$ thô**, **$$R^2$$ hiệu chỉnh (Adjusted $$R^2$$)** và **Sai số chuẩn phần dư (RSE)**, đồng thời hệ thống hóa trọn bộ giả định để bạn biết chính xác khi nào OLS đáng tin và khi nào nó đang đánh lừa bạn.

---

## Kiến thức cần nhớ: Bốn thước đo và giả định quyết định chất lượng mô hình OLS

Để đánh giá một mô hình hồi quy đa biến một cách trưởng thành, hãy nắm chắc bốn trụ cột sau:

1. **Tính tối ưu Gauss–Markov (BLUE):** Dưới 4 giả định tuyến tính, lấy mẫu ngẫu nhiên, không đa cộng tuyến hoàn toàn, và nhiễu đồng phương sai có kỳ vọng 0, OLS đạt phương sai nhỏ nhất trong mọi bộ ước lượng tuyến tính không chệch. Khi một giả định vỡ (ví dụ phương sai nhiễu thay đổi — heteroscedasticity), OLS vẫn *không chệch* nhưng không còn hiệu quả nhất và sai số chuẩn cổ điển bị sai lệch.
2. **$$R^2$$ vs. Adjusted $$R^2$$:** $$R^2 = 1 - \mathrm{RSS}/\mathrm{TSS}$$ chỉ đo mức khớp trên dữ liệu đã thấy nên luôn ưu ái mô hình càng nhiều biến càng tốt. **Adjusted $$R^2$$** thay tổng bình phương bằng phương sai trung bình chia cho bậc tự do ($$\mathrm{RSS}/(n-p-1)$$ và $$\mathrm{TSS}/(n-1)$$) — nó chỉ tăng khi biến mới cải thiện mô hình *nhiều hơn mức may rủi ngẫu nhiên*.
3. **Sai số chuẩn phần dư (RSE):** $$\mathrm{RSE} = \sqrt{\frac{\mathrm{RSS}}{n-p-1}}$$ đo sai số điển hình của mô hình theo **đúng đơn vị gốc của $$Y$$** (triệu đồng, giây, kg) — giúp đánh giá trực tiếp ý nghĩa thực tiễn thay vì chỉ nhìn con số không đơn vị $$R^2$$.
4. **Chẩn đoán giả định & Ngoại lệ có đòn bẩy (Leverage):** Một điểm dữ liệu có giá trị $$X$$ nằm xa trung tâm có **đòn bẩy cao** — chỉ riêng một điểm đó có thể xoay nghiêng cả đường hồi quy. Kiểm tra phần dư chuẩn hóa và khoảng cách Cook là bước không thể thiếu.

---

## Mô Hình Hồi Quy Tuyến Tính

Giả sử ta có $$n$$ quan sát với $$p$$ đặc trưng (features). Mô hình hồi quy tuyến tính có dạng:

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p + \epsilon$$

hoặc viết gọn dưới dạng ma trận:

$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$

trong đó:
- $$\mathbf{y}$$: vector $$n \times 1$$ chứa các giá trị biến mục tiêu.
- $$\mathbf{X}$$: ma trận $$n \times (p+1)$$ chứa các đặc trưng (cột đầu tiên gồm toàn giá trị 1 tương ứng với hệ số chặn).
- $$\boldsymbol{\beta}$$: vector $$(p+1) \times 1$$ chứa các hệ số hồi quy.
- $$\boldsymbol{\epsilon}$$: vector $$n \times 1$$ chứa các sai số ngẫu nhiên, giả định $$\epsilon_i \sim \mathcal{N}(0, \sigma^2)$$ độc lập.

## Phương Pháp Bình Phương Tối Thiểu (Ordinary Least Squares - OLS)

**Ý tưởng:** Tìm vector hệ số $$\boldsymbol{\beta}$$ sao cho tổng bình phương phần dư (Residual Sum of Squares - RSS) đạt giá trị nhỏ nhất:

$$\text{RSS}(\boldsymbol{\beta}) = \sum_{i=1}^n (y_i - \mathbf{x}_i^T\boldsymbol{\beta})^2 = ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2$$

**Lời giải giải tích:**

Lấy đạo hàm của $$\text{RSS}$$ theo $$\boldsymbol{\beta}$$ và đặt bằng 0:

$$\frac{\partial \text{RSS}}{\partial \boldsymbol{\beta}} = -2\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) = 0$$

Giải phương trình trên, ta thu được nghiệm dạng đóng (closed-form solution) của OLS từ **phương trình pháp tuyến (normal equation)**:

$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def implement_ols_from_scratch():
    """
    Lập trình OLS từ đầu và so sánh với sklearn
    """
    # Sinh dữ liệu
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=1, noise=10)
    
    # Thêm cột intercept
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    
    # OLS từ đầu: β = (X'X)^(-1) X'y
    beta_ols = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
    y_pred_ols = X_with_intercept @ beta_ols
    
    # So sánh với sklearn
    model_sklearn = LinearRegression()
    model_sklearn.fit(X, y)
    y_pred_sklearn = model_sklearn.predict(X)
    
    # Vẽ biểu đồ
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.6, s=50, label='Data')
    plt.plot(X, y_pred_ols, 'r-', linewidth=2, label='OLS từ đầu')
    plt.plot(X, y_pred_sklearn, 'g--', linewidth=2, label='sklearn')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('So Sánh OLS Implementation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Residuals
    residuals = y - y_pred_ols
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred_ols, residuals, alpha=0.6, s=50)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()
    
    print("Coefficients từ OLS:")
    print(f"  Intercept: {beta_ols[0]:.4f}")
    print(f"  Slope:     {beta_ols[1]:.4f}")
    print("\nCoefficients từ sklearn:")
    print(f"  Intercept: {model_sklearn.intercept_:.4f}")
    print(f"  Slope:     {model_sklearn.coef_[0]:.4f}")
    print(f"\nR²: {r2_score(y, y_pred_ols):.4f}")
    print(f"MSE: {mean_squared_error(y, y_pred_ols):.4f}")

# implement_ols_from_scratch()
```

## Ý Nghĩa Hình Học

Phương pháp OLS có cách diễn giải hình học trực quan thông qua phép chiếu vuông góc:

- $$\mathbf{y}$$ là một vector trong không gian $$\mathbb{R}^n$$.
- Các cột của $$\mathbf{X}$$ xác định một không gian con (column space) của $$\mathbb{R}^n$$.
- Vector dự đoán $$\hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}}$$ là **hình chiếu vuông góc** của $$\mathbf{y}$$ lên không gian cột của $$\mathbf{X}$$.
- Vector phần dư $$\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$$ vuông góc với không gian cột này.

**Ma trận chiếu (Hat Matrix):**

$$\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$$

$$\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$$

Ma trận $$\mathbf{H}$$ được gọi là "hat matrix" vì phép nhân với $$\mathbf{H}$$ biến $$\mathbf{y}$$ thành $$\hat{\mathbf{y}}$$ (thêm ký hiệu dấu mũ lên $$\mathbf{y}$$).

## Định Lý Gauss-Markov

Dưới các giả định:
1. $$E[\boldsymbol{\epsilon}] = 0$$ (sai số có kỳ vọng bằng 0).
2. $$\text{Var}(\boldsymbol{\epsilon}) = \sigma^2 \mathbf{I}$$ (phương sai đồng nhất và các sai số không tương quan với nhau).

khi đó **ước lượng OLS là ước lượng tuyến tính không chệch tốt nhất (Best Linear Unbiased Estimator — BLUE)**:
- **Không chệch (Unbiased):** $$E[\hat{\boldsymbol{\beta}}] = \boldsymbol{\beta}$$.
- **Tốt nhất (Best):** Có phương sai nhỏ nhất trong tất cả các ước lượng tuyến tính không chệch.

```python
def verify_gauss_markov(n_simulations=1000):
    """
    Kiểm chứng tính unbiased của OLS estimator
    """
    true_beta = np.array([2.0, 3.5])  # [intercept, slope]
    n_samples = 50
    
    beta_estimates = []
    
    for _ in range(n_simulations):
        # Sinh dữ liệu
        X = np.random.uniform(0, 10, n_samples)
        epsilon = np.random.normal(0, 2, n_samples)
        y = true_beta[0] + true_beta[1] * X + epsilon
        
        # Ước lượng OLS
        X_with_intercept = np.column_stack([np.ones(n_samples), X])
        beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
        beta_estimates.append(beta_hat)
    
    beta_estimates = np.array(beta_estimates)
    
    # Vẽ biểu đồ
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for i, param_name in enumerate(['Intercept', 'Slope']):
        axes[i].hist(beta_estimates[:, i], bins=50, density=True, alpha=0.7)
        axes[i].axvline(true_beta[i], color='red', linestyle='--', linewidth=2, 
                       label=f'True {param_name} = {true_beta[i]}')
        axes[i].axvline(beta_estimates[:, i].mean(), color='green', linestyle='--', linewidth=2,
                       label=f'Mean estimate = {beta_estimates[:, i].mean():.4f}')
        axes[i].set_xlabel(param_name)
        axes[i].set_ylabel('Density')
        axes[i].set_title(f'Phân Phối của {param_name} Estimator')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()
    
    print("Kiểm chứng Unbiased Property:")
    print(f"True β₀: {true_beta[0]:.4f}, Mean estimate: {beta_estimates[:, 0].mean():.4f}")
    print(f"True β₁: {true_beta[1]:.4f}, Mean estimate: {beta_estimates[:, 1].mean():.4f}")

# verify_gauss_markov()
```

## Đánh Giá Mô Hình

### Hệ Số Xác Định $$R^2$$ (Coefficient of Determination)

$$R^2 = 1 - \frac{\text{RSS}}{\text{TSS}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

- Với mô hình OLS có hệ số chặn trên tập huấn luyện, $$R^2 \in [0, 1]$$.
- $$R^2 = 1$$: Mô hình khớp hoàn toàn với dữ liệu.
- $$R^2 = 0$$: Mô hình không giải thích tốt hơn việc dự đoán bằng trung bình mẫu $$\bar{y}$$.

**Hạn chế:** $$R^2$$ không bao giờ giảm (và thường tăng) khi thêm đặc trưng mới vào mô hình, ngay cả khi đặc trưng đó không liên quan đến biến mục tiêu.

### $$R^2$$ Hiệu Chỉnh (Adjusted $$R^2$$)

$$R^2_{\text{adj}} = 1 - \frac{\text{RSS}/(n-p-1)}{\text{TSS}/(n-1)}$$

Chỉ số này hiệu chỉnh theo số lượng tham số $$p$$, qua đó phạt mô hình khi thêm biến không mang lại cải thiện tương xứng.

### Sai Số Chuẩn Phần Dư (Residual Standard Error - RSE)

$$\text{RSE} = \sqrt{\frac{\text{RSS}}{n-p-1}}$$

Đại lượng này là ước lượng cho độ lệch chuẩn $$\sigma$$ của sai số ngẫu nhiên.

```python
from sklearn.datasets import load_diabetes

def comprehensive_model_evaluation():
    """
    Đánh giá toàn diện mô hình linear regression
    """
    # Load dataset
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Tính các metrics
    n, p = X.shape
    rss = np.sum((y - y_pred)**2)
    tss = np.sum((y - y.mean())**2)
    r2 = 1 - rss/tss
    adj_r2 = 1 - (rss/(n-p-1))/(tss/(n-1))
    rse = np.sqrt(rss/(n-p-1))
    
    print("Model Evaluation Metrics:")
    print(f"  R²:          {r2:.4f}")
    print(f"  Adjusted R²: {adj_r2:.4f}")
    print(f"  RSE:         {rse:.4f}")
    print(f"  MSE:         {mean_squared_error(y, y_pred):.4f}")
    
    # Diagnostic plots
    residuals = y - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Residuals vs Fitted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6, s=30)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Fitted values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Q-Q plot
    from scipy import stats as sp_stats
    sp_stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Scale-Location
    standardized_residuals = residuals / residuals.std()
    axes[1, 0].scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), alpha=0.6, s=30)
    axes[1, 0].set_xlabel('Fitted values')
    axes[1, 0].set_ylabel('√|Standardized residuals|')
    axes[1, 0].set_title('Scale-Location')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Residuals histogram
    axes[1, 1].hist(residuals, bins=30, density=True, alpha=0.7)
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Histogram of Residuals')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()

# comprehensive_model_evaluation()
```

## Các Giả Định Của Hồi Quy Tuyến Tính

1. **Tính tuyến tính (Linearity):** Mối quan hệ giữa $$X$$ và $$Y$$ là tuyến tính.
2. **Tính độc lập (Independence):** Các quan sát độc lập với nhau.
3. **Phương sai đồng nhất (Homoscedasticity):** Phương sai của sai số không đổi.
4. **Tính chuẩn (Normality):** Sai số tuân theo phân phối chuẩn.
5. **Không có đa cộng tuyến mạnh (No multicollinearity):** Các đặc trưng không tương quan cao với nhau.

**Cách kiểm tra các giả định:**
- Tính tuyến tính: Biểu đồ phần dư (Residual plot).
- Phương sai đồng nhất: Biểu đồ Scale-Location.
- Tính chuẩn: Biểu đồ Q-Q plot, kiểm định Shapiro-Wilk.
- Đa cộng tuyến: Hệ số phóng đại phương sai VIF (Variance Inflation Factor).

---

### Ví dụ tính tay làm từng bước: Khi một biến "rác" đánh lừa $$R^2$$ nhưng bị Adjusted $$R^2$$ vạch mặt

Giả sử bạn xây dựng mô hình hồi quy trên $$n = 12$$ quan sát với tổng biến thiên $$\mathrm{TSS} = 100$$:

- **Mô hình 1 (chỉ dùng $$p = 1$$ biến thực sự có ý nghĩa):** Tổng bình phương phần dư $$\mathrm{RSS}_1 = 20{,}0$$.
  - $$R^2_1 = 1 - \frac{20{,}0}{100} = 0{,}8000.$$
  - $$R^2_{\text{adj},1} = 1 - \frac{\mathrm{RSS}_1 / (n - p - 1)}{\mathrm{TSS} / (n - 1)} = 1 - \frac{20{,}0 / 10}{100 / 11} = 1 - \frac{2{,}000}{9{,}091} = 0{,}7800.$$
  - $$\mathrm{RSE}_1 = \sqrt{20{,}0 / 10} = \sqrt{2{,}0} \approx 1{,}414.$$

- **Mô hình 2 (nhồi thêm 1 biến nhiễu ngẫu nhiên $$\implies p = 2$$):** Nhờ tình cờ khớp chút nhiễu trên 12 điểm, $$\mathrm{RSS}_2$$ nhích nhẹ xuống $$19{,}2$$.
  - $$R^2_2 = 1 - \frac{19{,}2}{100} = 0{,}8080$$ (**tăng** thêm $$+0{,}008$$ — nhìn tưởng tốt hơn!).
  - $$R^2_{\text{adj},2} = 1 - \frac{19{,}2 / (12 - 2 - 1)}{100 / 11} = 1 - \frac{19{,}2 / 9}{9{,}091} = 1 - \frac{2{,}133}{9{,}091} = 0{,}7653$$ (**GIẢM** mạnh!).
  - $$\mathrm{RSE}_2 = \sqrt{19{,}2 / 9} = \sqrt{2{,}133} \approx 1{,}461$$ (**tăng**, tức sai số dự báo ước lượng *xấu đi*!).

Chỉ một phép chia đơn giản cho bậc tự do $$n - p - 1$$ (từ 10 xuống 9) đã phơi bày ngay sự thật: biến thứ hai không tạo đủ giảm RSS để bù đắp 1 bậc tự do bị tiêu tốn!

### Giải thích: Bốn nguyên tắc vàng khi đánh giá mô hình hồi quy đa biến

1. **Không bao giờ dùng $$R^2$$ trên tập train để so sánh mô hình có số biến $$p$$ khác nhau:** Hãy dùng **Adjusted $$R^2$$**, **AIC/BIC**, hoặc tốt nhất là **Cross-Validation (kiểm chứng chéo)** trên dữ liệu chưa từng nhìn thấy.
2. **Đọc RSE song song với $$R^2$$:** Một mô hình dự đoán giá nhà có $$R^2 = 0{,}85$$ nghe rất cao, nhưng nếu $$\mathrm{RSE} = 1{,}5$$ tỷ đồng thì vẫn quá thô cho giao dịch thực tế.
3. **Hiểu rõ hậu quả khi từng giả định bị vi phạm:** Thiếu biến phi tuyến làm hệ số **chệch**; phương sai thay đổi hay tự tương quan không làm hệ số chệch nhưng làm **p-value và khoảng tin cậy sai lệch**.
4. **Kiểm tra đa cộng tuyến bằng VIF:** Khi hai biến giải thích tương quan quá sát, dự báo $$\hat{y}$$ tổng thể vẫn có thể ổn nhưng từng hệ số $$\hat{\beta}_j$$ riêng lẻ lại nhảy loạn xạ — lúc đó cần chuyển sang Ridge/Lasso ở bài tiếp theo.

**Kiểm tra nhanh.** Trong ví dụ $$n = 12$$ ở trên, biến thứ hai cần kéo $$\mathrm{RSS}_2$$ xuống dưới mức bao nhiêu (so với $$\mathrm{RSS}_1 = 20{,}0$$) để $$R^2_{\text{adj},2}$$ thực sự *tăng* cao hơn $$0{,}7800$$? *(Gợi ý: cần $$\mathrm{RSS}_2 / 9 < 20{,}0 / 10 = 2{,}0$$.)*

---

## Video tham khảo

- [Linear Regression, Clearly Explained!!!](https://www.youtube.com/watch?v=7ArmBVF2dCs) — StatQuest with Josh Starmer, tiếng Anh. **Xem trước.** Trình bày hồi quy tuyến tính từ phần dư đến $$R^2$$ và giá trị $$p$$, tương ứng với nội dung đánh giá mô hình trong bài.
- [Simple Linear Regression:  Checking Assumptions with Residual Plots](https://www.youtube.com/watch?v=iMdtTCX2Q70) — jbstatistics, tiếng Anh. Hướng dẫn dùng đồ thị phần dư để kiểm tra tính tuyến tính, phương sai đồng nhất và các điểm ngoại lai.
- [R-squared, Clearly Explained!!!](https://www.youtube.com/watch?v=2AQKmw14mHM) — StatQuest with Josh Starmer, tiếng Anh. Giải thích $$R^2$$ như tỷ lệ phương sai quanh giá trị trung bình được đường hồi quy giải thích.

## Bài Tập Thực Hành

**Bài 1: Cài đặt OLS bằng phân rã QR (QR Decomposition)**
Thay vì tính nghịch đảo trực tiếp $$(X^TX)^{-1}$$, hãy sử dụng phân rã QR để giải bài toán OLS nhằm tăng độ ổn định số học.

**Bài 2: Phát hiện đa cộng tuyến (Multicollinearity Detection)**
Tải một tập dữ liệu có các đặc trưng tương quan cao, tính chỉ số VIF và quan sát ảnh hưởng của đa cộng tuyến lên các hệ số ước lượng.

**Bài 3: Hồi quy đa thức (Polynomial Regression)**
Tự cài đặt hồi quy đa thức bằng cách tạo các đặc trưng đa thức và so sánh kết quả với `PolynomialFeatures` của `scikit-learn`.
