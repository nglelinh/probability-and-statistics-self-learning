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

### 1. Bernoulli Distribution

**Mô tả:** Thí nghiệm có 2 kết quả (thành công/thất bại)

**Tham số:** $$p$$ (xác suất thành công)

**PMF:** 
$$P(X = k) = \begin{cases} p & \text{if } k=1 \\ 1-p & \text{if } k=0 \end{cases}$$

**Kỳ vọng:** $$E[X] = p$$

**Phương sai:** $$\text{Var}(X) = p(1-p)$$

**Ứng dụng:** Tung đồng xu, click/no-click, conversion

### 2. Binomial Distribution

**Mô tả:** Số lần thành công trong $$n$$ phép thử Bernoulli độc lập

**Tham số:** $$n$$ (số phép thử), $$p$$ (xác suất thành công)

**PMF:**
$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

**Kỳ vọng:** $$E[X] = np$$

**Phương sai:** $$\text{Var}(X) = np(1-p)$$

**Ứng dụng:** A/B testing, quality control, số người click trong n visitors

### 3. Poisson Distribution

**Mô tả:** Số sự kiện xảy ra trong một khoảng thời gian/không gian cố định

**Tham số:** $$\lambda$$ (tỷ lệ trung bình)

**PMF:**
$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

**Kỳ vọng:** $$E[X] = \lambda$$

**Phương sai:** $$\text{Var}(X) = \lambda$$

**Ứng dụng:** Số email nhận được mỗi giờ, số khách hàng đến cửa hàng, rare events

### 4. Geometric Distribution

**Mô tả:** Số phép thử cần thiết để có thành công đầu tiên

**Tham số:** $$p$$ (xác suất thành công)

**PMF:**
$$P(X = k) = (1-p)^{k-1} p$$

**Kỳ vọng:** $$E[X] = \frac{1}{p}$$

**Phương sai:** $$\text{Var}(X) = \frac{1-p}{p^2}$$

**Ứng dụng:** Thời gian chờ đến sự kiện đầu tiên, số lần thử đến khi thành công

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

### 1. Uniform Distribution

**Mô tả:** Mọi giá trị trong khoảng $$[a, b]$$ đều có khả năng như nhau

**Tham số:** $$a, b$$ (biên dưới và biên trên)

**PDF:**
$$f(x) = \begin{cases} \frac{1}{b-a} & \text{if } a \leq x \leq b \\ 0 & \text{otherwise} \end{cases}$$

**Kỳ vọng:** $$E[X] = \frac{a+b}{2}$$

**Phương sai:** $$\text{Var}(X) = \frac{(b-a)^2}{12}$$

**Ứng dụng:** Random number generation, prior trong Bayesian

### 2. Normal (Gaussian) Distribution

**Mô tả:** Phân phối "chuông", quan trọng nhất trong thống kê

**Tham số:** $$\mu$$ (mean), $$\sigma^2$$ (variance)

**PDF:**
$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**Kỳ vọng:** $$E[X] = \mu$$

**Phương sai:** $$\text{Var}(X) = \sigma^2$$

**Tính chất đặc biệt:**
- Symmetric around $$\mu$$
- 68-95-99.7 rule
- Tổng các biến chuẩn độc lập vẫn là chuẩn

**Ứng dụng:** CLT, errors, natural phenomena, ML assumptions

### 3. Exponential Distribution

**Mô tả:** Thời gian chờ đến sự kiện đầu tiên (liên tục của Geometric)

**Tham số:** $$\lambda$$ (rate)

**PDF:**
$$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$

**Kỳ vọng:** $$E[X] = \frac{1}{\lambda}$$

**Phương sai:** $$\text{Var}(X) = \frac{1}{\lambda^2}$$

**Tính chất:** Memoryless property

**Ứng dụng:** Thời gian giữa các sự kiện, lifetime analysis

### 4. Chi-Squared Distribution

**Mô tả:** Tổng bình phương của $$k$$ biến chuẩn độc lập

**Tham số:** $$k$$ (degrees of freedom)

**PDF:** (Phức tạp, không cần nhớ)

**Kỳ vọng:** $$E[X] = k$$

**Phương sai:** $$\text{Var}(X) = 2k$$

**Ứng dụng:** Hypothesis testing, goodness-of-fit tests

### 5. Student's t-Distribution

**Mô tả:** Giống Normal nhưng có "đuôi dày" hơn

**Tham số:** $$\nu$$ (degrees of freedom)

**Kỳ vọng:** $$E[X] = 0$$ (nếu $$\nu > 1$$)

**Phương sai:** $$\text{Var}(X) = \frac{\nu}{\nu-2}$$ (nếu $$\nu > 2$$)

**Tính chất:** Khi $$\nu \to \infty$$, hội tụ về Normal(0,1)

**Ứng dụng:** t-tests, confidence intervals khi không biết variance

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

## Bài Tập Thực Hành

**Bài 1: So Sánh Binomial và Poisson**
Khi $$n$$ lớn và $$p$$ nhỏ sao cho $$np = \lambda$$, Binomial(n, p) ≈ Poisson(λ).
- Thử với n=100, p=0.05 (λ=5)
- Vẽ cả hai phân phối và so sánh
- Tính sai số giữa hai phân phối

**Bài 2: Memoryless Property**
Exponential distribution có tính chất memoryless: P(X > s+t | X > s) = P(X > t)
- Verify bằng mô phỏng
- So sánh với phân phối không có tính chất này (ví dụ: Uniform)

**Bài 3: 68-95-99.7 Rule**
Với Normal(μ, σ²):
- Mô phỏng 100,000 mẫu
- Tính tỷ lệ mẫu trong [μ-σ, μ+σ], [μ-2σ, μ+2σ], [μ-3σ, μ+3σ]
- So sánh với 68%, 95%, 99.7%

**Bài 4: Central Limit Theorem**
Sample từ Exponential(λ=1), tính mean của n=30 mẫu. Lặp lại 10,000 lần.
- Vẽ histogram của sample means
- So sánh với Normal(1/λ, (1/λ²)/n)
- Thử với các phân phối khác (Uniform, Poisson)
