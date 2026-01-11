---
layout: post
title: 02-07-00 Kiểm Định Giả Thuyết
chapter: "06"
order: 1
owner: nglelinh
lang: vi
categories:
- chapter02
lesson_type: required
---

Kiểm định giả thuyết (hypothesis testing) là công cụ để đưa ra quyết định dựa trên dữ liệu: Liệu một drug mới có hiệu quả hơn placebo? Liệu conversion rate đã tăng sau khi thay đổi UI? Bài học này sẽ giúp bạn hiểu framework của hypothesis testing, ý nghĩa của p-value (và những hiểu lầm phổ biến), các loại sai lầm, và các tests phổ biến nhất.

---

## Framework của Hypothesis Testing

**Null Hypothesis (H₀):** Giả thuyết "không có gì xảy ra", status quo
**Alternative Hypothesis (H₁ hoặc Hₐ):** Giả thuyết chúng ta muốn chứng minh

**Ví dụ:**
- H₀: Drug mới không hiệu quả hơn placebo (μ_drug = μ_placebo)
- H₁: Drug mới hiệu quả hơn (μ_drug > μ_placebo)

**Quy trình:**
1. Đặt H₀ và H₁
2. Chọn test statistic
3. Tính p-value
4. Quyết định: Reject H₀ hoặc Fail to reject H₀

### Câu Chuyện Kinh Điển: The Lady Tasting Tea
> Dựa trên *The Lady Tasting Tea* của David Salsburg.

Vào một buổi chiều hè ở Cambridge năm 1920, một nhóm các nhà khoa học, trong đó có Ronald Fisher, đang uống trà. Một quý cô tuyên bố rằng cô có thể phân biệt được trà được rót vào sữa hay sữa được rót vào trà.

![Lady Tasting Tea](../img/lady_tasting_tea.jpg)
*Minh họa quý cô nếm trà (Ảnh: Wikimedia Commons).*

Fisher đã thiết kế một thí nghiệm:
- Chuẩn bị 8 tách trà: 4 tách trà trước, 4 tách sữa trước.
- Thứ tự ngẫu nhiên.
- Quý cô nếm và gọi tên từng tách.

**Giả thuyết:**
- $$H_0$$: Quý cô đoán mò (không có khả năng phân biệt).
- $$H_1$$: Quý cô có khả năng phân biệt thật sự.

Nếu cô ấy đoán đúng cả 8 tách, xác suất đoán mò là $$\frac{1}{\binom{8}{4}} = \frac{1}{70} \approx 0.014$$.
Vì $$0.014 < 0.05$$, Fisher kết luận: Có bằng chứng để bác bỏ $$H_0$$. Quý cô thực sự có tài năng!

Đây chính là khởi nguồn của **Randomized Controlled Trial** và kiểm định giả thuyết hiện đại.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def hypothesis_testing_framework():
    """
    Minh họa framework của hypothesis testing
    """
    # Giả sử: Test xem mean có = 100 không
    # H₀: μ = 100
    # H₁: μ ≠ 100
    
    # Dữ liệu
    np.random.seed(42)
    true_mean = 105  # Thực tế khác 100!
    data = np.random.normal(true_mean, 15, 50)
    
    # Test statistic
    hypothesized_mean = 100
    sample_mean = data.mean()
    sample_std = data.std(ddof=1)
    n = len(data)
    
    # t-statistic
    t_stat = (sample_mean - hypothesized_mean) / (sample_std / np.sqrt(n))
    
    # p-value (two-tailed)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Data
    axes[0].hist(data, bins=20, alpha=0.7, edgecolor='black')
    axes[0].axvline(sample_mean, color='blue', linestyle='-', linewidth=2, 
                   label=f'Sample mean = {sample_mean:.2f}')
    axes[0].axvline(hypothesized_mean, color='red', linestyle='--', linewidth=2,
                   label=f'H₀: μ = {hypothesized_mean}')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Data và Hypothesized Mean')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # t-distribution
    x = np.linspace(-4, 4, 1000)
    pdf = stats.t.pdf(x, df=n-1)
    axes[1].plot(x, pdf, linewidth=2, color='steelblue', label='t-distribution')
    axes[1].fill_between(x, pdf, alpha=0.3)
    
    # Rejection regions (α = 0.05)
    alpha = 0.05
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    
    # Shade rejection regions
    x_left = x[x < -t_critical]
    x_right = x[x > t_critical]
    axes[1].fill_between(x_left, stats.t.pdf(x_left, df=n-1), alpha=0.5, color='red', label='Rejection region')
    axes[1].fill_between(x_right, stats.t.pdf(x_right, df=n-1), alpha=0.5, color='red')
    
    # Observed t-statistic
    axes[1].axvline(t_stat, color='green', linestyle='--', linewidth=2, 
                   label=f't-stat = {t_stat:.2f}')
    axes[1].axvline(-t_critical, color='orange', linestyle=':', linewidth=1.5)
    axes[1].axvline(t_critical, color='orange', linestyle=':', linewidth=1.5, 
                   label=f't-critical = ±{t_critical:.2f}')
    
    axes[1].set_xlabel('t-statistic')
    axes[1].set_ylabel('Density')
    axes[1].set_title(f'p-value = {p_value:.4f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()
    
    print("Hypothesis Testing Framework:")
    print("=" * 60)
    print(f"H₀: μ = {hypothesized_mean}")
    print(f"H₁: μ ≠ {hypothesized_mean}")
    print()
    print(f"Sample mean: {sample_mean:.2f}")
    print(f"Sample std: {sample_std:.2f}")
    print(f"n = {n}")
    print()
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print()
    if p_value < alpha:
        print(f"✓ Reject H₀ (p < {alpha})")
        print(f"  Kết luận: Có bằng chứng rằng μ ≠ {hypothesized_mean}")
    else:
        print(f"✗ Fail to reject H₀ (p ≥ {alpha})")
        print(f"  Kết luận: Không đủ bằng chứng để bác bỏ H₀")

# hypothesis_testing_framework()
```

## P-value: Ý Nghĩa và Hiểu Lầm

**Định nghĩa ĐÚNG:** P-value là xác suất quan sát được dữ liệu cực đoan như vậy (hoặc cực đoan hơn), GIẢ SỬ H₀ đúng.

**Hiểu lầm PHỔ BIẾN (SAI):**
- ❌ "P-value là xác suất H₀ đúng"
- ❌ "1 - p-value là xác suất H₁ đúng"
- ❌ "p < 0.05 nghĩa là kết quả quan trọng"

```python
def p_value_interpretation():
    """
    Minh họa ý nghĩa của p-value
    """
    # H₀: μ = 0
    # Observed: sample mean = 2, n = 25, s = 5
    
    hypothesized_mean = 0
    observed_mean = 2
    sample_std = 5
    n = 25
    
    # t-statistic
    t_obs = (observed_mean - hypothesized_mean) / (sample_std / np.sqrt(n))
    
    # p-value (one-tailed: H₁: μ > 0)
    p_value = 1 - stats.t.cdf(t_obs, df=n-1)
    
    # Simulation: Nếu H₀ đúng, phân phối của sample mean là gì?
    n_sims = 10000
    sample_means = []
    
    for _ in range(n_sims):
        # Sinh dữ liệu từ H₀: μ = 0
        sample = np.random.normal(hypothesized_mean, sample_std, n)
        sample_means.append(sample.mean())
    
    sample_means = np.array(sample_means)
    
    # p-value empirical
    p_value_empirical = np.mean(sample_means >= observed_mean)
    
    # Visualize
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(sample_means, bins=50, density=True, alpha=0.7, edgecolor='black',
             label='Sampling distribution\n(nếu H₀ đúng)')
    
    # Theoretical
    x = np.linspace(sample_means.min(), sample_means.max(), 1000)
    se = sample_std / np.sqrt(n)
    plt.plot(x, stats.norm.pdf(x, hypothesized_mean, se), 'r-', linewidth=2,
             label='Theoretical')
    
    # Observed
    plt.axvline(observed_mean, color='green', linestyle='--', linewidth=2,
               label=f'Observed mean = {observed_mean}')
    
    # Shade p-value region
    x_shade = x[x >= observed_mean]
    plt.fill_between(x_shade, stats.norm.pdf(x_shade, hypothesized_mean, se),
                     alpha=0.5, color='red', label=f'p-value ≈ {p_value:.3f}')
    
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    plt.title('P-value: Xác suất quan sát ≥ 2 nếu H₀ đúng')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # P-value interpretation
    plt.subplot(1, 2, 2)
    plt.text(0.5, 0.7, 'P-value Interpretation', ha='center', fontsize=16, fontweight='bold')
    plt.text(0.5, 0.55, f'p-value = {p_value:.4f}', ha='center', fontsize=14, color='blue')
    plt.text(0.1, 0.4, '✓ ĐÚNG:', fontsize=12, fontweight='bold', color='green')
    plt.text(0.15, 0.32, f'Nếu H₀ đúng (μ=0), xác suất\nquan sát mean ≥ {observed_mean} là {p_value:.3f}', 
             fontsize=10, va='top')
    plt.text(0.1, 0.15, '✗ SAI:', fontsize=12, fontweight='bold', color='red')
    plt.text(0.15, 0.07, f'Xác suất H₀ đúng là {p_value:.3f}', fontsize=10, va='top')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    plt.tight_layout()
    # plt.show()
    
    print("P-value Interpretation:")
    print("=" * 60)
    print(f"Observed sample mean: {observed_mean}")
    print(f"p-value (theoretical): {p_value:.4f}")
    print(f"p-value (simulation):  {p_value_empirical:.4f}")
    print()
    print("✓ Ý nghĩa: Nếu H₀ đúng, có {:.2f}% cơ hội quan sát được".format(p_value*100))
    print("  dữ liệu cực đoan như vậy hoặc cực đoan hơn.")

# p_value_interpretation()
```

# p_value_interpretation()
```

## Mặt Tối của P-value: P-hacking và Data Dredging

> *"If you torture the data long enough, it will confess."* - Ronald Coase

Trong thực tế, nhiều nghiên cứu mắc phải lỗi **P-hacking**: Thử hàng trăm phân tích khác nhau, lọc dữ liệu, đổi test statistic... cho đến khi tìm được một kết quả có $$p < 0.05$$ và công bố kết quả đó.

**Nghịch lý:** Nếu bạn thực hiện 20 tests trên dữ liệu hoàn toàn ngẫu nhiên (không có hiệu ứng thật), xác suất có ít nhất một test cho kết quả "significant" (p < 0.05) là:

$$1 - (1 - 0.05)^{20} \approx 64\%$$

Điều này dẫn đến cuộc **khủng hoảng tái lập (replication crisis)** trong khoa học.

**Lời khuyên từ *Naked Statistics*:**
- Hãy trung thực với quy trình phân tích.
- Xác định trước (preregister) các giả thuyết và phương pháp phân tích.
- Luôn nghi ngờ những kết quả "vừa đủ" significant (p ≈ 0.049).

## Type I và Type II Errors

|                | H₀ đúng        | H₀ sai         |
|----------------|----------------|----------------|
| **Reject H₀**  | Type I Error (α) | Correct ✓      |
| **Fail to reject H₀** | Correct ✓      | Type II Error (β) |

**Type I Error (False Positive):** Reject H₀ khi H₀ đúng
- Significance level α = P(Type I Error)
- Thường chọn α = 0.05

**Type II Error (False Negative):** Fail to reject H₀ khi H₀ sai
- β = P(Type II Error)
- Power = 1 - β = P(Reject H₀ | H₀ sai)

```python
def type_i_ii_errors():
    """
    Minh họa Type I và Type II errors
    """
    # Setup
    null_mean = 0
    true_mean = 2  # H₀ SAI!
    sigma = 5
    n = 25
    alpha = 0.05
    
    # Critical value
    se = sigma / np.sqrt(n)
    z_critical = stats.norm.ppf(1 - alpha)
    critical_value = null_mean + z_critical * se
    
    # Visualize
    x = np.linspace(-3, 5, 1000)
    
    # Distribution under H₀
    pdf_h0 = stats.norm.pdf(x, null_mean, se)
    
    # Distribution under H₁ (true)
    pdf_h1 = stats.norm.pdf(x, true_mean, se)
    
    plt.figure(figsize=(12, 6))
    
    # Plot distributions
    plt.plot(x, pdf_h0, linewidth=2, label='Distribution nếu H₀ đúng (μ=0)', color='blue')
    plt.plot(x, pdf_h1, linewidth=2, label=f'Distribution thực tế (μ={true_mean})', color='red')
    
    # Critical value
    plt.axvline(critical_value, color='green', linestyle='--', linewidth=2,
               label=f'Critical value = {critical_value:.2f}')
    
    # Type I Error (α)
    x_type1 = x[x >= critical_value]
    plt.fill_between(x_type1, stats.norm.pdf(x_type1, null_mean, se),
                     alpha=0.3, color='blue', label=f'Type I Error (α = {alpha})')
    
    # Type II Error (β)
    x_type2 = x[x < critical_value]
    plt.fill_between(x_type2, stats.norm.pdf(x_type2, true_mean, se),
                     alpha=0.3, color='red')
    
    # Power
    x_power = x[x >= critical_value]
    plt.fill_between(x_power, stats.norm.pdf(x_power, true_mean, se),
                     alpha=0.5, color='green', label='Power (1-β)')
    
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    plt.title('Type I Error, Type II Error, và Power')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()
    
    # Calculate β
    beta = stats.norm.cdf(critical_value, true_mean, se)
    power = 1 - beta
    
    print("Type I và Type II Errors:")
    print("=" * 60)
    print(f"H₀: μ = {null_mean}")
    print(f"H₁: μ = {true_mean} (thực tế)")
    print(f"α = {alpha}")
    print()
    print(f"Critical value: {critical_value:.2f}")
    print(f"Type I Error (α): {alpha:.3f}")
    print(f"Type II Error (β): {beta:.3f}")
    print(f"Power (1-β): {power:.3f}")
    print()
    print("⚠️  Trade-off: Giảm α → Tăng β (và ngược lại)")
    print("✓  Tăng n → Giảm cả α và β!")

# type_i_ii_errors()
```

## Common Tests

### 1. One-Sample t-test

Test xem mean có bằng một giá trị cụ thể không.

```python
def one_sample_t_test():
    """
    One-sample t-test
    """
    # H₀: μ = 100
    # H₁: μ ≠ 100
    
    hypothesized_mean = 100
    data = np.array([102, 98, 105, 99, 103, 101, 97, 104, 100, 102])
    
    # Manual calculation
    sample_mean = data.mean()
    sample_std = data.std(ddof=1)
    n = len(data)
    t_stat = (sample_mean - hypothesized_mean) / (sample_std / np.sqrt(n))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
    
    # Using scipy
    t_stat_scipy, p_value_scipy = stats.ttest_1samp(data, hypothesized_mean)
    
    print("One-Sample t-test:")
    print("=" * 60)
    print(f"H₀: μ = {hypothesized_mean}")
    print(f"Data: {data}")
    print()
    print(f"Sample mean: {sample_mean:.2f}")
    print(f"Sample std: {sample_std:.2f}")
    print()
    print(f"t-statistic: {t_stat:.4f} (scipy: {t_stat_scipy:.4f})")
    print(f"p-value: {p_value:.4f} (scipy: {p_value_scipy:.4f})")
    print()
    if p_value < 0.05:
        print("✓ Reject H₀ at α=0.05")
    else:
        print("✗ Fail to reject H₀ at α=0.05")

# one_sample_t_test()
```

### 2. Two-Sample t-test

Test xem hai nhóm có mean khác nhau không.

```python
def two_sample_t_test():
    """
    Two-sample t-test (A/B test)
    """
    # H₀: μ_A = μ_B
    # H₁: μ_A ≠ μ_B
    
    # Group A (control)
    group_a = np.random.normal(100, 15, 50)
    
    # Group B (treatment) - slightly higher mean
    group_b = np.random.normal(105, 15, 50)
    
    # t-test
    t_stat, p_value = stats.ttest_ind(group_a, group_b)
    
    # Visualize
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(group_a, bins=20, alpha=0.7, label=f'Group A (mean={group_a.mean():.2f})', edgecolor='black')
    plt.hist(group_b, bins=20, alpha=0.7, label=f'Group B (mean={group_b.mean():.2f})', edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Two Groups')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot([group_a, group_b], labels=['Group A', 'Group B'])
    plt.ylabel('Value')
    plt.title(f't-test: p-value = {p_value:.4f}')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    # plt.show()
    
    print("Two-Sample t-test:")
    print("=" * 60)
    print(f"Group A: n={len(group_a)}, mean={group_a.mean():.2f}, std={group_a.std():.2f}")
    print(f"Group B: n={len(group_b)}, n={len(group_b)}, mean={group_b.mean():.2f}, std={group_b.std():.2f}")
    print()
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print()
    if p_value < 0.05:
        print("✓ Reject H₀: Hai nhóm có mean khác nhau (α=0.05)")
    else:
        print("✗ Fail to reject H₀: Không đủ bằng chứng về sự khác biệt")

# two_sample_t_test()
```

### 3. Chi-Squared Test

Test xem phân phối quan sát có khớp với phân phối kỳ vọng không.

```python
def chi_squared_test():
    """
    Chi-squared goodness-of-fit test
    """
    # Xúc xắc có fair không?
    # H₀: Xúc xắc fair (mỗi mặt có P = 1/6)
    
    # Observed frequencies
    observed = np.array([45, 52, 48, 55, 50, 50])  # 300 rolls
    
    # Expected frequencies (nếu fair)
    n_total = observed.sum()
    expected = np.array([n_total/6] * 6)
    
    # Chi-squared test
    chi2_stat, p_value = stats.chisquare(observed, expected)
    
    # Visualize
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    x = np.arange(1, 7)
    width = 0.35
    plt.bar(x - width/2, observed, width, label='Observed', alpha=0.7, edgecolor='black')
    plt.bar(x + width/2, expected, width, label='Expected (fair)', alpha=0.7, edgecolor='black')
    plt.xlabel('Die Face')
    plt.ylabel('Frequency')
    plt.title(f'Chi-squared Test: p-value = {p_value:.4f}')
    plt.xticks(x)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Chi-squared distribution
    plt.subplot(1, 2, 2)
    df = 5  # 6 categories - 1
    x_chi = np.linspace(0, 20, 1000)
    pdf = stats.chi2.pdf(x_chi, df)
    plt.plot(x_chi, pdf, linewidth=2, label=f'χ²(df={df})')
    plt.axvline(chi2_stat, color='red', linestyle='--', linewidth=2,
               label=f'Observed χ² = {chi2_stat:.2f}')
    
    # Shade p-value
    x_shade = x_chi[x_chi >= chi2_stat]
    plt.fill_between(x_shade, stats.chi2.pdf(x_shade, df), alpha=0.5, color='red')
    
    plt.xlabel('χ² statistic')
    plt.ylabel('Density')
    plt.title('Chi-squared Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()
    
    print("Chi-Squared Goodness-of-Fit Test:")
    print("=" * 60)
    print(f"H₀: Xúc xắc fair (mỗi mặt P=1/6)")
    print(f"Observed: {observed}")
    print(f"Expected: {expected}")
    print()
    print(f"χ² statistic: {chi2_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print()
    if p_value < 0.05:
        print("✓ Reject H₀: Xúc xắc KHÔNG fair")
    else:
        print("✗ Fail to reject H₀: Không đủ bằng chứng xúc xắc không fair")

# chi_squared_test()
```

## Multiple Testing Problem

Khi thực hiện nhiều tests, xác suất ít nhất một False Positive tăng lên!

**Bonferroni Correction:** Dùng α/m thay vì α (với m là số tests)

```python
def multiple_testing_problem():
    """
    Minh họa multiple testing problem
    """
    n_tests = 20
    alpha = 0.05
    n_simulations = 10000
    
    # Simulation: Tất cả H₀ đều ĐÚNG
    false_positives = []
    
    for _ in range(n_simulations):
        # Sinh dữ liệu từ H₀
        p_values = []
        for _ in range(n_tests):
            data = np.random.normal(0, 1, 30)  # H₀: μ = 0
            _, p = stats.ttest_1samp(data, 0)
            p_values.append(p)
        
        # Đếm số false positives (p < α)
        n_false_pos = np.sum(np.array(p_values) < alpha)
        false_positives.append(n_false_pos)
    
    false_positives = np.array(false_positives)
    
    # Probability of at least one false positive
    prob_at_least_one = np.mean(false_positives > 0)
    
    # Theoretical
    prob_at_least_one_theory = 1 - (1 - alpha)**n_tests
    
    print("Multiple Testing Problem:")
    print("=" * 60)
    print(f"Số tests: {n_tests}")
    print(f"α per test: {alpha}")
    print()
    print(f"P(ít nhất 1 false positive) simulation: {prob_at_least_one:.3f}")
    print(f"P(ít nhất 1 false positive) theory:     {prob_at_least_one_theory:.3f}")
    print()
    print(f"⚠️  Với {n_tests} tests, có {prob_at_least_one*100:.1f}% cơ hội có false positive!")
    print()
    print("Giải pháp: Bonferroni Correction")
    print(f"  Dùng α_corrected = {alpha}/{n_tests} = {alpha/n_tests:.4f}")
    
    # Visualize
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(false_positives, bins=range(n_tests+2), alpha=0.7, edgecolor='black', density=True)
    plt.axvline(false_positives.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean = {false_positives.mean():.2f}')
    plt.xlabel('Number of False Positives')
    plt.ylabel('Probability')
    plt.title(f'Distribution of False Positives\n({n_tests} tests, all H₀ true)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    n_tests_range = range(1, 51)
    prob_fp = [1 - (1 - alpha)**m for m in n_tests_range]
    plt.plot(n_tests_range, prob_fp, linewidth=2)
    plt.axhline(alpha, color='red', linestyle='--', linewidth=2, label=f'α = {alpha}')
    plt.xlabel('Number of Tests')
    plt.ylabel('P(at least one false positive)')
    plt.title('Multiple Testing Increases False Positive Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()

# multiple_testing_problem()
```

## Bài Tập Thực Hành

**Bài 1: Power Analysis**
Với α=0.05, n=25, σ=10:
- Tính power cho effect sizes: 0.2, 0.5, 0.8, 1.0
- Vẽ power curve
- Tìm n cần thiết để đạt power=0.8 với effect size=0.5

**Bài 2: A/B Test Simulation**
Mô phỏng A/B test với:
- Control: conversion rate = 10%
- Treatment: conversion rate = 12%
- n = 1000 mỗi nhóm
- Chạy 1000 simulations, tính power

**Bài 3: Multiple Comparisons**
Có 5 nhóm, muốn test tất cả các cặp (10 comparisons).
- Không correction: tính family-wise error rate
- Bonferroni correction
- Holm-Bonferroni method

**Bài 4: Permutation Test**
Implement permutation test cho two-sample comparison.
- So sánh với t-test
- Thử với non-normal data
