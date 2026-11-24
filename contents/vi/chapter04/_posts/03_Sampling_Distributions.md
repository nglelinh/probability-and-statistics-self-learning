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

Làm thế nào một viện thăm dò dư luận ở một quốc gia 100 triệu dân chỉ cần phỏng vấn ngẫu nhiên đúng **1.000 người** mà vẫn dự báo được tỷ lệ ủng hộ toàn quốc với sai số chỉ khoảng $$\pm 3\%$$? Tại sao ta không cần hỏi 1 triệu người hay 10 triệu người?

Câu trả lời nằm ở một trong những định lý kỳ diệu nhất của toàn bộ toán học: **Định lý Giới hạn Trung tâm (Central Limit Theorem — CLT)** và lý thuyết **Phân phối Lấy mẫu (Sampling Distributions)**. Bất kể phân phối gốc của từng cá thể méo mó, lệch lạc hay rời rạc ra sao, chỉ cần bạn lấy trung bình cộng của một mẫu ngẫu nhiên đủ lớn, bản thân con số trung bình mẫu $$\bar X$$ sẽ tụ quanh giá trị thật theo đúng hình chiếc chuông phân phối Chuẩn với độ rộng co lại chính xác theo quy luật $$1/\sqrt{n}$$!

---

## Bản đồ kiến thức nền: Phân phối lấy mẫu và Định lý Giới hạn Trung tâm là gì?

Bài học này là chiếc cầu nối quyết định từ *Xác suất* (Chương 3–4) sang *Suy luận Thống kê* (Chương 5–6), đứng trên **ba trụ cột**:

1. **Thống kê mẫu là một biến ngẫu nhiên:** Nếu bạn rút một mẫu 100 người khác, trung bình mẫu $$\bar X$$ sẽ ra một số hơi khác. Phân phối xác suất của $$\bar X$$ qua vô số lần lấy mẫu lặp lại gọi là **phân phối lấy mẫu**.
2. **Quy luật Sai số chuẩn (Standard Error — $$\mathrm{SE} = \sigma/\sqrt{n}$$):** Độ phân tán của $$\bar X$$ nhỏ hơn độ phân tán của từng cá thể đúng $$\sqrt{n}$$ lần — giải thích vì sao muốn giảm sai số đi một nửa, ta phải tăng cỡ mẫu lên **gấp 4 lần**.
3. **Định lý Giới hạn Trung tâm (CLT):** Với $$n$$ đủ lớn, $$\bar X \approx \mathcal{N}\bigl(\mu,\, \sigma^2/n\bigr)$$ bất kể hình dạng phân phối gốc (miễn phương sai hữu hạn) — nền tảng của khoảng tin cậy, kiểm định giả thuyết, A/B testing và nhiễu mini-batch trong SGD!

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

### Khung giải thích: tự tay kiểm tra CLT trên 2 viên xúc xắc và tính sai số cuộc thăm dò $$n = 1{,}000$$ (nên đọc chậm)

1. **Xem hình chiếc chuông bắt đầu hiện ra chỉ từ $$n = 2$$ viên xúc xắc:**
   - Tung **1 viên xúc xắc** $$X_1$$: phân phối phẳng lì hình chữ nhật (mỗi mặt $$1..6$$ xác suất $$1/6$$), kỳ vọng $$\mu = 3.5$$, phương sai $$\sigma^2 = \frac{35}{12} \approx 2.917$$.
   - Lấy trung bình **2 viên xúc xắc** $$\bar X_2 = \frac{X_1 + X_2}{2}$$: có $$36$$ cặp đồng khả năng. Xác suất nhận hai giá trị cực biên giảm thê thảm: $$P(\bar X_2 = 1) = \frac{1}{36}$$ và $$P(\bar X_2 = 6) = \frac{1}{36}$$, trong khi đỉnh nhọn vọt lên ngay tại trung tâm: $$P(\bar X_2 = 3.5) = \frac{6}{36} = \frac{1}{6}$$ (gấp 6 lần hai đầu mút!).
   - Phương sai co lại đúng một nửa: $$\mathrm{Var}(\bar X_2) = \frac{\sigma^2}{2} = \frac{35}{24} \approx 1.458$$. Chỉ với $$n = 2$$, hình chữ nhật phẳng đã biến thành hình tam giác cân; đến $$n = 30$$, nó trở thành đường cong chuông Gauss gần như hoàn hảo!

2. **Tính tay biên độ sai số của cuộc thăm dò $$n = 1{,}000$$ cử tri:**
   Giả sử tỷ lệ ủng hộ thật quanh $$p = 0.50$$ (phương sai lớn nhất $$p(1-p) = 0.25$$). Với mẫu ngẫu nhiên $$n = 1{,}000$$:
   $$
   \mathrm{SE}(\hat p) = \sqrt{\frac{0.5 \times 0.5}{1000}} = \sqrt{0.00025} \approx 0.0158.
   $$
   Theo CLT, biên độ sai số ở mức tin cậy 95% là:
   $$
   \pm 1.96 \times 0.0158 \approx \pm 0.031 = \pm 3.1\%!
   $$
   Điều đáng kinh ngạc nhất trong công thức trên: **dân số tổng thể $$N$$ (1 triệu hay 100 triệu người) hoàn toàn không xuất hiện!** Miễn mẫu được chọn *ngẫu nhiên không thiên lệch*, 1.000 người là đủ đo mạch đập của cả một quốc gia.

## Video Tham Khảo

- [But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) — 3Blue1Brown, tiếng Anh. **Nên xem trước.** Minh họa Định lý Giới hạn Trung tâm qua bảng Galton và tổng các biến ngẫu nhiên: trung bình mẫu hội tụ về dạng phân phối chuẩn ngay cả khi từng quan sát riêng lẻ không có phân phối chuẩn.
- [The Central Limit Theorem, Clearly Explained!!!](https://www.youtube.com/watch?v=YAlJCEDH2uY) — StatQuest with Josh Starmer, tiếng Anh. Trình bày ngắn gọn hơn về cùng định lý và giải thích cơ sở cho kiểm định t được nhắc đến trong phần ứng dụng.
- [The Sampling Distribution of the Sample Mean (fast version)](https://www.youtube.com/watch?v=0zqNGDVNKgA) — jbstatistics, tiếng Anh. Giải thích khái niệm phân phối lấy mẫu của trung bình mẫu — đối tượng được mô phỏng trong bài trước khi áp dụng CLT.

## Bài Tập

**Bài 1:** Mô phỏng Định lý Giới hạn Trung tâm với phân phối Chi-bình phương (Chi-squared) và quan sát tốc độ hội tụ khi tăng cỡ mẫu.

**Bài 2:** Tính khoảng tin cậy 95% cho giá trị trung bình bằng hai phương pháp: CLT và Bootstrap, sau đó so sánh hai kết quả.
