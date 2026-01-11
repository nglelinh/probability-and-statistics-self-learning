---
layout: post
title: 01-03-00 Phân Phối Xác Suất và Lấy Mẫu
chapter: "04"
order: 4
owner: nglelinh
lang: vi
categories:
- chapter01
lesson_type: required
---

Trong bài học trước, chúng ta đã tìm hiểu về Bias-Variance Tradeoff. Để hiểu sâu hơn về cách đánh giá mô hình và thực hiện suy diễn thống kê (inference), chúng ta cần nắm vững Định lý Giới hạn Trung tâm (CLT) và phân phối lấy mẫu (sampling distribution).

> **Lưu ý:** Nếu bạn chưa nắm vững các khái niệm cơ bản về xác suất và các phân phối thông dụng (Bernoulli, Binomial, Normal, Poisson...), hãy xem lại **[Chapter 03: Xác Suất](../../chapter03/_posts/01_Concept_of_Probability.md)** trước khi tiếp tục. Bài học này sẽ giả định bạn đã quen thuộc với các phân phối đó và tập trung vào ứng dụng của chúng trong việc lấy mẫu.

---

## Ôn Tập Nhanh: Phân Phối Chuẩn (Normal Distribution)

Trong Statistical Learning, phân phối Chuẩn (Gaussian) đóng vai trò trung tâm vì:
1.  **Central Limit Theorem:** Tổng/trung bình của nhiều biến ngẫu nhiên độc lập hội tụ về phân phối chuẩn.
2.  **Least Squares:** Phương pháp bình phương tối thiểu (OLS) tương đương với Maximum Likelihood Estimation (MLE) khi nhiễu tuân theo phân phối chuẩn.

$$X \sim \mathcal{N}(\mu, \sigma^2)$$

PDF: $$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

(Xem chi tiết về các phân phối khác như Binomial, Poisson, t-distribution tại [Bài 04-02](02_Discrete_Continuous_Distributions.md))

## Định Lý Giới Hạn Trung Tâm (CLT)

**Phát biểu:** Cho $$X_1, \ldots, X_n$$ iid với $$E[X_i] = \mu$$, $$\text{Var}(X_i) = \sigma^2 < \infty$$:

$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)$$

**Ý nghĩa:** Bất kể phân phối gốc, trung bình mẫu sẽ gần phân phối Chuẩn khi $$n$$ lớn.

```python
def demonstrate_clt(distribution='exponential', n_samples=30, n_simulations=10000):
    """
    Minh họa CLT với các phân phối khác nhau
    """
    sample_means = []
    
    for _ in range(n_simulations):
        if distribution == 'exponential':
            sample = np.random.exponential(1, n_samples)
            mu, sigma = 1.0, 1.0
        elif distribution == 'uniform':
            sample = np.random.uniform(0, 1, n_samples)
            mu, sigma = 0.5, np.sqrt(1/12)
        
        sample_means.append(sample.mean())
    
    sample_means = np.array(sample_means)
    standardized = (sample_means - mu) / (sigma / np.sqrt(n_samples))
    
    # Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    plt.hist(standardized, bins=50, density=True, alpha=0.7, label='Standardized means')
    
    x = np.linspace(-4, 4, 100)
    plt.plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2, label='N(0,1)')
    
    plt.xlabel('Standardized value')
    plt.ylabel('Density')
    plt.title(f'CLT Demo: {distribution.capitalize()} distribution, n={n_samples}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()

# demonstrate_clt('exponential', n_samples=30)
```

## Sampling Distributions

**Sampling distribution:** Phân phối của một thống kê (như $$\bar{X}$$) khi lấy mẫu nhiều lần.

**Quan trọng cho:**
- Xây dựng confidence intervals
- Hypothesis testing
- Hiểu uncertainty trong estimates

```python
def sampling_distribution_demo():
    """
    Minh họa sampling distribution của trung bình
    """
    # Population
    population = np.random.normal(100, 15, size=10000)
    true_mean = population.mean()
    
    # Lấy nhiều mẫu
    sample_size = 30
    n_samples = 1000
    sample_means = []
    
    for _ in range(n_samples):
        sample = np.random.choice(population, size=sample_size, replace=False)
        sample_means.append(sample.mean())
    
    sample_means = np.array(sample_means)
    
    # Vẽ
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(population, bins=50, density=True, alpha=0.7)
    plt.axvline(true_mean, color='r', linestyle='--', linewidth=2, label=f'μ = {true_mean:.2f}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Population Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(sample_means, bins=50, density=True, alpha=0.7)
    plt.axvline(true_mean, color='r', linestyle='--', linewidth=2, label=f'μ = {true_mean:.2f}')
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    plt.title(f'Sampling Distribution (n={sample_size})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()
    
    print(f"Standard Error (lý thuyết): {15/np.sqrt(sample_size):.4f}")
    print(f"Standard Error (thực nghiệm): {sample_means.std():.4f}")

# sampling_distribution_demo()
```

## Ứng Dụng Trong Machine Learning

**1. Confidence Intervals:**
CLT cho phép xây dựng CI cho model parameters.

**2. Bootstrap:**
Resampling để ước lượng sampling distribution.

**3. Hypothesis Testing:**
Test xem model có tốt hơn baseline không.

## Bài Tập

**Bài 1:** Mô phỏng CLT với phân phối Chi-squared. Quan sát tốc độ hội tụ.

**Bài 2:** Tính confidence interval 95% cho trung bình bằng CLT và Bootstrap. So sánh kết quả.
