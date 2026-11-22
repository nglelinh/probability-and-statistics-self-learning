---
layout: post
title: 02-04-00 Các Phân Phối Xác Suất Quan Trọng
chapter: "04"
order: 3
owner: nglelinh
lang: vi
categories:
- chapter02
lesson_type: required
---

## Phân Phối Rời Rạc

### 1. Phân Phối Bernoulli (Bernoulli Distribution)

**Mô tả:** Mô hình hóa một phép thử chỉ có 2 kết quả (thành công hoặc thất bại).

**Tham số:** $$p$$ (xác suất thành công).

**PMF:**
$$P(X = k) = \begin{cases} p & \text{if } k=1 \\ 1-p & \text{if } k=0 \end{cases}$$

**Kỳ vọng:** $$E[X] = p$$

**Phương sai:** $$\text{Var}(X) = p(1-p)$$

**Ứng dụng:** Tung đồng xu, nhấp chuột hay không nhấp chuột (click/no-click), chuyển đổi (conversion).

### 2. Phân Phối Nhị Thức (Binomial Distribution)

**Mô tả:** Đếm số lần thành công trong $$n$$ phép thử Bernoulli độc lập có cùng xác suất thành công.

**Tham số:** $$n$$ (số phép thử), $$p$$ (xác suất thành công ở mỗi phép thử).

**PMF:**
$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

**Kỳ vọng:** $$E[X] = np$$

**Phương sai:** $$\text{Var}(X) = np(1-p)$$

**Ứng dụng:** Thử nghiệm A/B (A/B testing), kiểm soát chất lượng, số người nhấp chuột trong $$n$$ lượt truy cập.

### 3. Phân Phối Poisson (Poisson Distribution)

**Mô tả:** Đếm số sự kiện xảy ra trong một khoảng thời gian hoặc vùng không gian cố định.

**Tham số:** $$\lambda$$ (tần suất trung bình).

**PMF:**
$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

**Kỳ vọng:** $$E[X] = \lambda$$

**Phương sai:** $$\text{Var}(X) = \lambda$$

**Ứng dụng:** Số email nhận được mỗi giờ, số khách hàng đến cửa hàng trong một khung giờ, các sự kiện hiếm gặp.

### 4. Phân Phối Hình Học (Geometric Distribution)

**Mô tả:** Đếm số phép thử Bernoulli độc lập cần thực hiện để đạt được lần thành công đầu tiên.

**Tham số:** $$p$$ (xác suất thành công ở mỗi phép thử).

**PMF:**
$$P(X = k) = (1-p)^{k-1} p$$

**Kỳ vọng:** $$E[X] = \frac{1}{p}$$

**Phương sai:** $$\text{Var}(X) = \frac{1-p}{p^2}$$

**Ứng dụng:** Số bước chờ đến sự kiện đầu tiên, số lần thử cho đến khi thành công.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def visualize_discrete_distributions():
    """
    Visualize các phân phối rời rạc quan trọng
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Bernoulli
    p_bern = 0.7
    x_bern = [0, 1]
    pmf_bern = [1-p_bern, p_bern]
    
    axes[0, 0].bar(x_bern, pmf_bern, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('P(X = x)')
    axes[0, 0].set_title(f'Bernoulli(p={p_bern})\nE[X]={p_bern}, Var(X)={p_bern*(1-p_bern):.3f}')
    axes[0, 0].set_xticks(x_bern)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Binomial
    n_binom, p_binom = 20, 0.3
    x_binom = np.arange(0, n_binom+1)
    pmf_binom = stats.binom.pmf(x_binom, n_binom, p_binom)
    
    axes[0, 1].bar(x_binom, pmf_binom, alpha=0.7, color='coral', edgecolor='black')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('P(X = x)')
    axes[0, 1].set_title(f'Binomial(n={n_binom}, p={p_binom})\nE[X]={n_binom*p_binom}, Var(X)={n_binom*p_binom*(1-p_binom):.1f}')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Poisson
    lam = 5
    x_pois = np.arange(0, 20)
    pmf_pois = stats.poisson.pmf(x_pois, lam)
    
    axes[1, 0].bar(x_pois, pmf_pois, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('P(X = x)')
    axes[1, 0].set_title(f'Poisson(λ={lam})\nE[X]={lam}, Var(X)={lam}')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Geometric
    p_geom = 0.2
    x_geom = np.arange(1, 25)
    pmf_geom = stats.geom.pmf(x_geom, p_geom)
    
    axes[1, 1].bar(x_geom, pmf_geom, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('P(X = x)')
    axes[1, 1].set_title(f'Geometric(p={p_geom})\nE[X]={1/p_geom}, Var(X)={(1-p_geom)/p_geom**2:.1f}')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    # plt.show()

# visualize_discrete_distributions()
```

## Phân Phối Liên Tục

### 1. Phân Phối Đều (Uniform Distribution)

**Mô tả:** Mọi giá trị trong đoạn $$[a, b]$$ đều có mật độ xác suất như nhau.

**Tham số:** $$a, b$$ (cận dưới và cận trên).

**PDF:**
$$f(x) = \begin{cases} \frac{1}{b-a} & \text{if } a \leq x \leq b \\ 0 & \text{otherwise} \end{cases}$$

**Kỳ vọng:** $$E[X] = \frac{a+b}{2}$$

**Phương sai:** $$\text{Var}(X) = \frac{(b-a)^2}{12}$$

**Ứng dụng:** Sinh số ngẫu nhiên, phân phối tiên nghiệm (prior) trong thống kê Bayes.

### 2. Phân Phối Chuẩn (Normal / Gaussian Distribution)

**Mô tả:** Phân phối dạng hình chuông, giữ vai trò trung tâm trong lý thuyết xác suất và thống kê.

**Tham số:** $$\mu$$ (kỳ vọng), $$\sigma^2$$ (phương sai).

**PDF:**
$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**Kỳ vọng:** $$E[X] = \mu$$

**Phương sai:** $$\text{Var}(X) = \sigma^2$$

**Tính chất đặc biệt:**
- Đối xứng quanh $$\mu$$.
- Tuân theo quy tắc 68–95–99.7.
- Tổng các biến ngẫu nhiên chuẩn độc lập vẫn tuân theo phân phối chuẩn.

**Ứng dụng:** Định lý Giới hạn Trung tâm (CLT), mô hình hóa sai số đo lường, hiện tượng tự nhiên, các giả định trong học máy.

### 3. Phân Phối Mũ (Exponential Distribution)

**Mô tả:** Mô hình hóa thời gian chờ đến khi sự kiện đầu tiên xảy ra (phiên bản liên tục của phân phối Hình học).

**Tham số:** $$\lambda$$ (tốc độ hay tần suất xảy ra sự kiện).

**PDF:**
$$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$

**Kỳ vọng:** $$E[X] = \frac{1}{\lambda}$$

**Phương sai:** $$\text{Var}(X) = \frac{1}{\lambda^2}$$

**Tính chất:** Có tính chất không nhớ (memoryless property).

**Ứng dụng:** Thời gian giữa hai sự kiện liên tiếp, phân tích tuổi thọ thiết bị (lifetime analysis).

### 4. Phân Phối Chi-Bình Phương (Chi-Squared Distribution)

**Mô tả:** Tổng bình phương của $$k$$ biến ngẫu nhiên chuẩn tắc $$\mathcal{N}(0, 1)$$ độc lập.

**Tham số:** $$k$$ (số bậc tự do — degrees of freedom).

**PDF:** Có dạng giải tích phức tạp (dùng hàm Gamma), thường tra bảng hoặc tính bằng phần mềm.

**Kỳ vọng:** $$E[X] = k$$

**Phương sai:** $$\text{Var}(X) = 2k$$

**Ứng dụng:** Kiểm định giả thuyết, kiểm định độ phù hợp (goodness-of-fit tests).

### 5. Phân Phối Student's t (Student's t-Distribution)

**Mô tả:** Có dạng hình chuông đối xứng tương tự phân phối Chuẩn nhưng có phần "đuôi dày" hơn.

**Tham số:** $$\nu$$ (số bậc tự do — degrees of freedom).

**Kỳ vọng:** $$E[X] = 0$$ (với $$\nu > 1$$)

**Phương sai:** $$\text{Var}(X) = \frac{\nu}{\nu-2}$$ (với $$\nu > 2$$)

**Tính chất:** Khi $$\nu \to \infty$$, phân phối t hội tụ về phân phối chuẩn tắc Normal(0, 1).

**Ứng dụng:** Kiểm định t (t-tests), xây dựng khoảng tin cậy khi chưa biết phương sai tổng thể.

```python
def visualize_continuous_distributions():
    """
    Visualize các phân phối liên tục quan trọng
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1. Uniform
    a, b = 0, 1
    x_unif = np.linspace(-0.5, 1.5, 1000)
    pdf_unif = stats.uniform.pdf(x_unif, a, b-a)
    
    axes[0, 0].plot(x_unif, pdf_unif, linewidth=2, color='steelblue')
    axes[0, 0].fill_between(x_unif, pdf_unif, alpha=0.3)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('f(x)')
    axes[0, 0].set_title(f'Uniform({a}, {b})\nE[X]={(a+b)/2}, Var(X)={(b-a)**2/12:.3f}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Normal
    mu, sigma = 0, 1
    x_norm = np.linspace(-4, 4, 1000)
    pdf_norm = stats.norm.pdf(x_norm, mu, sigma)
    
    axes[0, 1].plot(x_norm, pdf_norm, linewidth=2, color='coral')
    axes[0, 1].fill_between(x_norm, pdf_norm, alpha=0.3)
    # 68-95-99.7 rule
    axes[0, 1].axvline(mu, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].axvspan(mu-sigma, mu+sigma, alpha=0.2, color='yellow', label='68%')
    axes[0, 1].axvspan(mu-2*sigma, mu+2*sigma, alpha=0.1, color='orange', label='95%')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('f(x)')
    axes[0, 1].set_title(f'Normal(μ={mu}, σ²={sigma**2})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Exponential
    lam = 1
    x_exp = np.linspace(0, 5, 1000)
    pdf_exp = stats.expon.pdf(x_exp, scale=1/lam)
    
    axes[0, 2].plot(x_exp, pdf_exp, linewidth=2, color='green')
    axes[0, 2].fill_between(x_exp, pdf_exp, alpha=0.3)
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('f(x)')
    axes[0, 2].set_title(f'Exponential(λ={lam})\nE[X]={1/lam}, Var(X)={1/lam**2}')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Chi-squared
    x_chi = np.linspace(0, 20, 1000)
    for df in [2, 5, 10]:
        pdf_chi = stats.chi2.pdf(x_chi, df)
        axes[1, 0].plot(x_chi, pdf_chi, linewidth=2, label=f'df={df}')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('f(x)')
    axes[1, 0].set_title('Chi-Squared Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Student's t
    x_t = np.linspace(-4, 4, 1000)
    for df in [1, 5, 30]:
        pdf_t = stats.t.pdf(x_t, df)
        axes[1, 1].plot(x_t, pdf_t, linewidth=2, label=f'df={df}')
    # So sánh với Normal
    axes[1, 1].plot(x_t, stats.norm.pdf(x_t), 'k--', linewidth=2, label='Normal(0,1)')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('f(x)')
    axes[1, 1].set_title('Student\'s t Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Beta
    x_beta = np.linspace(0, 1, 1000)
    for alpha, beta in [(0.5, 0.5), (2, 2), (2, 5)]:
        pdf_beta = stats.beta.pdf(x_beta, alpha, beta)
        axes[1, 2].plot(x_beta, pdf_beta, linewidth=2, label=f'α={alpha}, β={beta}')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('f(x)')
    axes[1, 2].set_title('Beta Distribution')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()

# visualize_continuous_distributions()
```

## Mối Quan Hệ Giữa Các Phân Phối

```python
def demonstrate_distribution_relationships():
    """
    Minh họa mối quan hệ giữa các phân phối
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Binomial → Normal (CLT)
    n_values = [10, 30, 100]
    p = 0.5
    x = np.linspace(0, 100, 1000)
    
    for n in n_values:
        x_binom = np.arange(0, n+1)
        pmf_binom = stats.binom.pmf(x_binom, n, p)
        axes[0, 0].plot(x_binom, pmf_binom, 'o-', label=f'Binomial(n={n})', alpha=0.7)
        
        # Normal approximation
        mu, sigma = n*p, np.sqrt(n*p*(1-p))
        pdf_norm = stats.norm.pdf(x, mu, sigma)
        axes[0, 0].plot(x, pdf_norm, '--', label=f'Normal approx (n={n})', alpha=0.7)
    
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('Probability/Density')
    axes[0, 0].set_title('Binomial → Normal (khi n lớn)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Poisson → Normal
    lam_values = [5, 20, 50]
    
    for lam in lam_values:
        x_pois = np.arange(0, int(lam + 4*np.sqrt(lam)))
        pmf_pois = stats.poisson.pmf(x_pois, lam)
        axes[0, 1].plot(x_pois, pmf_pois, 'o-', label=f'Poisson(λ={lam})', alpha=0.7)
        
        # Normal approximation
        x_norm = np.linspace(0, lam + 4*np.sqrt(lam), 1000)
        pdf_norm = stats.norm.pdf(x_norm, lam, np.sqrt(lam))
        axes[0, 1].plot(x_norm, pdf_norm, '--', alpha=0.7)
    
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('Probability/Density')
    axes[0, 1].set_title('Poisson → Normal (khi λ lớn)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. t → Normal
    x_t = np.linspace(-4, 4, 1000)
    df_values = [1, 3, 10, 30]
    
    for df in df_values:
        pdf_t = stats.t.pdf(x_t, df)
        axes[1, 0].plot(x_t, pdf_t, linewidth=2, label=f't(df={df})')
    
    pdf_norm = stats.norm.pdf(x_t)
    axes[1, 0].plot(x_t, pdf_norm, 'k--', linewidth=3, label='Normal(0,1)')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('f(x)')
    axes[1, 0].set_title('t → Normal (khi df → ∞)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Sum of Normals is Normal
    n_samples = 10000
    X1 = np.random.normal(2, 1, n_samples)
    X2 = np.random.normal(-1, 1.5, n_samples)
    X_sum = X1 + X2
    
    axes[1, 1].hist(X1, bins=50, alpha=0.3, density=True, label='X₁ ~ N(2, 1)')
    axes[1, 1].hist(X2, bins=50, alpha=0.3, density=True, label='X₂ ~ N(-1, 2.25)')
    axes[1, 1].hist(X_sum, bins=50, alpha=0.5, density=True, label='X₁ + X₂', edgecolor='black')
    
    # Theoretical sum
    mu_sum = 2 + (-1)
    var_sum = 1 + 2.25
    x_theory = np.linspace(-10, 10, 1000)
    pdf_theory = stats.norm.pdf(x_theory, mu_sum, np.sqrt(var_sum))
    axes[1, 1].plot(x_theory, pdf_theory, 'r-', linewidth=2, label=f'N({mu_sum}, {var_sum:.2f})')
    
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Tổng các biến Normal độc lập vẫn là Normal')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()

# demonstrate_distribution_relationships()
```

## Video Tham Khảo

- [The Binomial Distribution and Test, Clearly Explained!!!](https://www.youtube.com/watch?v=J8jNoF-K8E8) — StatQuest with Josh Starmer, tiếng Anh. **Nên xem trước.** Giới thiệu phân phối nhị thức và kiểm định đi kèm, thuộc nhóm phân phối rời rạc đã trình bày trong bài.
- [The Normal Distribution, Clearly Explained!!!](https://www.youtube.com/watch?v=rzFX5NWojp0) — StatQuest with Josh Starmer, tiếng Anh. Trình bày phân phối chuẩn (hình dạng, trung bình và độ lệch chuẩn) cho phần phân phối liên tục.
- [An Introduction to the Poisson Distribution](https://www.youtube.com/watch?v=jmqZG6roVqU) — jbstatistics, tiếng Anh. Giải thích phân phối Poisson cho dữ liệu đếm và mối liên hệ xấp xỉ từ phân phối nhị thức khi n lớn, p nhỏ — phù hợp với mục mối quan hệ giữa các phân phối.

## Bài Tập Thực Hành

**Bài 1: So sánh phân phối Nhị thức và Poisson**
Khi $$n$$ lớn và $$p$$ nhỏ sao cho $$np = \lambda$$, phân phối Binomial(n, p) có thể xấp xỉ bởi Poisson(λ).
- Thử nghiệm với n = 100, p = 0.05 (λ = 5).
- Vẽ đồ thị cả hai phân phối trên cùng hệ trục và so sánh.
- Tính sai số giữa hai phân phối.

**Bài 2: Tính chất không nhớ (Memoryless Property)**
Phân phối Mũ có tính chất không nhớ: P(X > s + t | X > s) = P(X > t).
- Kiểm chứng tính chất này bằng mô phỏng.
- So sánh với một phân phối không có tính chất không nhớ (ví dụ: phân phối Đều — Uniform).

**Bài 3: Quy tắc 68–95–99.7**
Với phân phối Normal(μ, σ²):
- Mô phỏng 100,000 mẫu.
- Tính tỷ lệ mẫu rơi vào các khoảng [μ - σ, μ + σ], [μ - 2σ, μ + 2σ] và [μ - 3σ, μ + 3σ].
- So sánh kết quả thực nghiệm với các mức lý thuyết 68%, 95% và 99.7%.

**Bài 4: Định lý Giới hạn Trung tâm (Central Limit Theorem)**
Lấy mẫu từ phân phối Exponential(λ = 1), tính trung bình của n = 30 mẫu, và lặp lại quá trình này 10,000 lần.
- Vẽ biểu đồ tần suất (histogram) của các trung bình mẫu thu được.
- So sánh với phân phối Normal(1/λ, (1/λ²)/n).
- Lặp lại thí nghiệm với các phân phối gốc khác (Uniform, Poisson).
