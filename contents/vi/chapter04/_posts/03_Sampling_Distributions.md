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

Trong các bài học trước, chúng ta đã tìm hiểu về sự đánh đổi giữa độ chệch và phương sai (Bias–Variance Tradeoff). Để đánh giá mô hình một cách có cơ sở và thực hiện suy diễn thống kê (statistical inference), chúng ta cần nắm vững Định lý Giới hạn Trung tâm (CLT) và khái niệm phân phối lấy mẫu (sampling distribution).

> **Lưu ý:** Nếu bạn chưa nắm vững các khái niệm xác suất cơ bản và các phân phối thông dụng (Bernoulli, Nhị thức, Chuẩn, Poisson...), hãy xem lại **[Chapter 03: Xác Suất](../../chapter03/_posts/01_Concept_of_Probability.md)** trước khi tiếp tục. Bài học này giả định bạn đã quen thuộc với các phân phối đó và tập trung vào ứng dụng của chúng trong việc lấy mẫu.

---

## Ôn Tập Nhanh: Phân Phối Chuẩn (Normal Distribution)

Trong học thống kê (Statistical Learning), phân phối Chuẩn (Gaussian) giữ vai trò trung tâm vì hai lý do chính:
1. **Định lý Giới hạn Trung tâm (Central Limit Theorem):** Tổng hoặc trung bình của nhiều biến ngẫu nhiên độc lập hội tụ về phân phối chuẩn khi cỡ mẫu đủ lớn.
2. **Bình phương tối thiểu (Least Squares):** Phương pháp bình phương tối thiểu thông thường (OLS) tương đương với ước lượng hợp lý cực đại (Maximum Likelihood Estimation — MLE) khi nhiễu tuân theo phân phối chuẩn.

$$X \sim \mathcal{N}(\mu, \sigma^2)$$

PDF: $$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

*(Xem chi tiết về các phân phối khác như Nhị thức, Poisson, phân phối t tại [Bài 04-02](02_Discrete_Continuous_Distributions.md).)*

## Định Lý Giới Hạn Trung Tâm (CLT)

**Phát biểu:** Cho $$X_1, \ldots, X_n$$ là các biến ngẫu nhiên độc lập và có cùng phân phối (i.i.d.) với kỳ vọng $$E[X_i] = \mu$$ và phương sai hữu hạn $$\text{Var}(X_i) = \sigma^2 < \infty$$ (với $$\sigma > 0$$). Khi đó:

$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)$$

**Ý nghĩa:** Bất kể phân phối gốc của tổng thể có hình dạng ra sao (miễn là có phương sai hữu hạn), phân phối của trung bình mẫu sẽ xấp xỉ phân phối Chuẩn khi cỡ mẫu $$n$$ đủ lớn.

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

## Phân Phối Lấy Mẫu (Sampling Distributions)

**Phân phối lấy mẫu:** Là phân phối xác suất của một thống kê mẫu (chẳng hạn trung bình mẫu $$\bar{X}$$) khi ta lặp lại quá trình lấy mẫu nhiều lần từ cùng một tổng thể.

**Vai trò trong thống kê:**
- Xây dựng khoảng tin cậy (confidence intervals).
- Thực hiện kiểm định giả thuyết (hypothesis testing).
- Định lượng độ bất định (uncertainty) của các ước lượng.

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

## Ứng Dụng Trong Học Máy (Machine Learning)

**1. Khoảng tin cậy (Confidence Intervals):**
Định lý Giới hạn Trung tâm cho phép xây dựng khoảng tin cậy cho các tham số và chỉ số đánh giá của mô hình.

**2. Bootstrap:**
Kỹ thuật tái lấy mẫu (resampling) giúp ước lượng phân phối lấy mẫu trực tiếp từ dữ liệu hiện có mà không cần biết dạng phân phối gốc.

**3. Kiểm định giả thuyết (Hypothesis Testing):**
Đánh giá xem một mô hình mới có thực sự cải thiện so với mô hình cơ sở (baseline) hay sự chênh lệch chỉ do nhiễu ngẫu nhiên.

## Video Tham Khảo

- [But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) — 3Blue1Brown, tiếng Anh. **Nên xem trước.** Minh họa Định lý Giới hạn Trung tâm qua bảng Galton và tổng các biến ngẫu nhiên: trung bình mẫu hội tụ về dạng phân phối chuẩn ngay cả khi từng quan sát riêng lẻ không có phân phối chuẩn.
- [The Central Limit Theorem, Clearly Explained!!!](https://www.youtube.com/watch?v=YAlJCEDH2uY) — StatQuest with Josh Starmer, tiếng Anh. Trình bày ngắn gọn hơn về cùng định lý và giải thích cơ sở cho kiểm định t được nhắc đến trong phần ứng dụng.
- [The Sampling Distribution of the Sample Mean (fast version)](https://www.youtube.com/watch?v=0zqNGDVNKgA) — jbstatistics, tiếng Anh. Giải thích khái niệm phân phối lấy mẫu của trung bình mẫu — đối tượng được mô phỏng trong bài trước khi áp dụng CLT.

## Bài Tập

**Bài 1:** Mô phỏng Định lý Giới hạn Trung tâm với phân phối Chi-bình phương (Chi-squared) và quan sát tốc độ hội tụ khi tăng cỡ mẫu.

**Bài 2:** Tính khoảng tin cậy 95% cho giá trị trung bình bằng hai phương pháp: CLT và Bootstrap, sau đó so sánh hai kết quả.
