---
layout: post
title: 02-07-00 Kiểm Định Giả Thuyết
chapter: "06"
order: 1
owner: nglelinh
lang: vi
categories:
- chapter06
lesson_type: required
---

Một nhà máy đóng gói cà phê in đậm con số **500 g** trên bao bì. Sáng hôm đó, kiểm định viên cân ngẫu nhiên 16 gói từ dây chuyền và thấy khối lượng trung bình chỉ đạt **494 g**. Liệu dây chuyền đóng gói thực sự đang bị lệch định lượng và cần dừng máy sửa ngay, hay 6 gam hụt đi kia chỉ là dao động ngẫu nhiên vô hại của mẫu nhỏ?

Câu hỏi đó — *tín hiệu thật hay nhiễu ngẫu nhiên?* — là cốt lõi của **kiểm định giả thuyết thống kê (hypothesis testing)**. Giống như nguyên tắc "suy đoán vô tội" trong phòng xử án, thống kê đặt ra giả thuyết mặc định $$H_0$$ ("không có sai lệch / không có hiệu ứng"), rồi đo xem dữ liệu quan sát được **bất thường đến mức nào** nếu $$H_0$$ thật sự đúng. Bài học này đi từ câu chuyện kinh điển *Quý cô nếm trà* của Ronald Fisher, bóc tách ý nghĩa thật sự của **p-value** (và cạm bẫy *p-hacking*), cân bằng hai loại sai lầm Loại I / Loại II, cho tới các phép kiểm định trụ cột ($$t$$-test, $$\chi^2$$) và bài toán đa kiểm định.

---

## Bản đồ kiến thức nền: Kiến trúc logic của Kiểm định Giả thuyết

Nhiều người bấm nút chạy kiểm định trong Python nhưng lại diễn giải sai hoàn toàn kết quả vì bỏ qua bốn nền tảng logic sau:

1. **Logic chứng minh phản chứng có xác suất:** Kiểm định không bao giờ "chứng minh $$H_0$$ đúng". Ta tạm giả sử $$H_0$$ đúng, tính xác suất xuất hiện dữ liệu cực đoan như (hoặc hơn) mẫu thực tế — đó chính là **p-value**. Nếu p-value quá nhỏ (ví dụ $$< 0{,}05$$), ta kết luận một trong hai điều: hoặc vừa xảy ra chuyện hiếm có khó tin, hoặc giả định $$H_0$$ đã sai — nên ta **bác bỏ $$H_0$$**.
2. **Đánh đổi Sai lầm Loại I ($$\alpha$$) và Sai lầm Loại II ($$\beta$$):** Báo động giả (bác bỏ $$H_0$$ khi $$H_0$$ đúng, xác suất $$\alpha$$) và bỏ lọt tín hiệu (không bác bỏ $$H_0$$ khi $$H_1$$ đúng, xác suất $$\beta$$, với lực kiểm định $$\text{Power} = 1 - \beta$$) luôn kéo nhau về hai phía. Giảm $$\alpha$$ quá gắt sẽ làm tăng $$\beta$$ trừ khi bạn tăng cỡ mẫu $$n$$.
3. **Ý nghĩa thống kê vs. Ý nghĩa thực tiễn:** Với 10 triệu người dùng, một sự thay đổi tỷ lệ click nhỏ xíu $$0{,}01\%$$ cũng có thể cho $$p < 0{,}0001$$ nhưng chẳng mang lại giá trị kinh doanh nào. Luôn báo cáo **độ lớn hiệu ứng (effect size)** và **khoảng tin cậy** đi kèm p-value.
4. **Lạm phát sai lầm trong Đa kiểm định (Multiple Testing):** Khi bạn kiểm định 20 giả thuyết vô thưởng vô phạt ở ngưỡng $$\alpha = 0{,}05$$, xác suất có ít nhất một báo động giả nhảy vọt lên hơn 64%! Đó là lý do các hiệu chỉnh như Bonferroni hay Benjamini–Hochberg (FDR) là bắt buộc trong khoa học dữ liệu và y sinh hiện đại.

---

## Khung Lý Thuyết Của Kiểm Định Giả Thuyết

**Giả thuyết không (Null Hypothesis — H₀):** Giả thuyết mặc định, thể hiện trạng thái "không có sự khác biệt" hay "không có hiệu ứng" (status quo).
**Giả thuyết đối (Alternative Hypothesis — H₁ hoặc Hₐ):** Giả thuyết nghiên cứu mà ta muốn tìm bằng chứng để ủng hộ.

**Ví dụ:**
- H₀: Thuốc mới không hiệu quả hơn giả dược (μ_drug = μ_placebo).
- H₁: Thuốc mới hiệu quả hơn giả dược (μ_drug > μ_placebo).

**Quy trình cơ bản:**
1. Phát biểu H₀ và H₁.
2. Chọn thống kê kiểm định (test statistic) phù hợp.
3. Tính giá trị p-value.
4. Đưa ra quyết định: bác bỏ H₀ (reject H₀) hoặc chưa đủ bằng chứng bác bỏ H₀ (fail to reject H₀).

### Câu Chuyện Kinh Điển: Quý Cô Nếm Trà (The Lady Tasting Tea)
> Dựa trên *The Lady Tasting Tea* của David Salsburg.

Vào một buổi chiều hè ở Cambridge năm 1920, một nhóm nhà khoa học, trong đó có Ronald Fisher, đang uống trà. Một quý cô khẳng định bà có thể phân biệt được tách trà được rót trà trước hay rót sữa trước.

![Lady Tasting Tea](../img/lady_tasting_tea.jpg)
*Minh họa câu chuyện quý cô nếm trà (Ảnh: Wikimedia Commons).*

Fisher đã thiết kế một thí nghiệm có kiểm soát:
- Chuẩn bị 8 tách trà: 4 tách rót trà trước, 4 tách rót sữa trước.
- Sắp xếp theo thứ tự ngẫu nhiên.
- Quý cô nếm thử và xác định cách pha của từng tách.

**Giả thuyết:**
- $$H_0$$: Quý cô chỉ đoán ngẫu nhiên (không có khả năng phân biệt).
- $$H_1$$: Quý cô thực sự có khả năng phân biệt.

Nếu quý cô đoán đúng cả 8 tách, xác suất xảy ra điều đó chỉ do may mắn là $$\frac{1}{\binom{8}{4}} = \frac{1}{70} \approx 0.014$$. Vì $$0.014 < 0.05$$, Fisher kết luận có đủ bằng chứng thống kê để bác bỏ $$H_0$$.

Thí nghiệm này được xem là một trong những khởi nguồn của **thử nghiệm đối chứng ngẫu nhiên (Randomized Controlled Trial)** và kiểm định giả thuyết hiện đại.

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

## P-value: Ý Nghĩa và Những Hiểu Lầm Phổ Biến

**Định nghĩa đúng:** P-value là xác suất quan sát được một kết quả cực đoan như mẫu hiện tại (hoặc cực đoan hơn), **với giả định rằng H₀ đúng**.

**Các hiểu lầm phổ biến (sai):**
- "P-value là xác suất để H₀ đúng."
- "1 − p-value là xác suất để H₁ đúng."
- "p < 0.05 đồng nghĩa kết quả có ý nghĩa quan trọng về mặt thực tiễn."

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

## Mặt Tối Của P-value: P-hacking và Data Dredging

> *"If you torture the data long enough, it will confess."* — Ronald Coase

Trong thực tế, nhiều nghiên cứu mắc lỗi **p-hacking** (hay data dredging): thử nhiều cách phân tích khác nhau, lọc bỏ dữ liệu hoặc thay đổi thống kê kiểm định cho đến khi thu được một kết quả có $$p < 0.05$$ rồi mới công bố.

**Hệ quả thống kê:** Nếu thực hiện 20 kiểm định độc lập trên dữ liệu hoàn toàn ngẫu nhiên (không có hiệu ứng thực), xác suất xuất hiện ít nhất một kiểm định "có ý nghĩa thống kê" ở mức p < 0.05 lên tới:

$$1 - (1 - 0.05)^{20} \approx 64\%$$

Hiện tượng này là một trong những nguyên nhân chính dẫn đến **khủng hoảng tái lập (replication crisis)** trong nghiên cứu khoa học.

**Khuyến nghị từ *Naked Statistics*:**
- Minh bạch và trung thực với toàn bộ quy trình phân tích.
- Đăng ký trước (preregister) các giả thuyết và phương pháp phân tích trước khi xem dữ liệu.
- Thận trọng trước những kết quả chỉ vừa sát ngưỡng có ý nghĩa thống kê (chẳng hạn p ≈ 0.049).

## Sai Lầm Loại I và Sai Lầm Loại II

|                       | H₀ đúng           | H₀ sai            |
|-----------------------|-------------------|-------------------|
| **Bác bỏ H₀**         | Sai lầm loại I (α) | Quyết định đúng ✓ |
| **Không bác bỏ H₀**   | Quyết định đúng ✓ | Sai lầm loại II (β) |

**Sai lầm loại I (Type I Error — Dương tính giả / False Positive):** Bác bỏ H₀ khi H₀ thực tế đúng.
- Mức ý nghĩa α = P(Sai lầm loại I).
- Thường được chọn ở mức α = 0.05.

**Sai lầm loại II (Type II Error — Âm tính giả / False Negative):** Không bác bỏ H₀ khi H₀ thực tế sai.
- β = P(Sai lầm loại II).
- Lực kiểm định (Power) = 1 − β = P(Bác bỏ H₀ | H₀ sai).

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

## Các Phép Kiểm Định Phổ Biến

### 1. Kiểm Định t Một Mẫu (One-Sample t-test)

Kiểm định xem trung bình tổng thể có bằng một giá trị giả định cụ thể hay không.

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

### 2. Kiểm Định t Hai Mẫu (Two-Sample t-test)

Kiểm định xem trung bình của hai nhóm độc lập có khác nhau hay không.

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

### 3. Kiểm Định Chi-Bình Phương (Chi-Squared Test)

Kiểm định xem phân phối tần suất quan sát được có phù hợp với phân phối kỳ vọng theo giả thuyết hay không.

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

## Bài Toán Đa Kiểm Định (Multiple Testing Problem)

Khi thực hiện đồng thời nhiều kiểm định giả thuyết, xác suất mắc ít nhất một sai lầm loại I (dương tính giả) tăng lên đáng kể.

**Hiệu chỉnh Bonferroni (Bonferroni Correction):** Thay vì dùng ngưỡng α cho mỗi kiểm định, ta điều chỉnh xuống ngưỡng α/m (với m là số lượng kiểm định).

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

---

### Ví dụ tính tay cầm tay chỉ việc: Kiểm định $$t$$ một mẫu cho dây chuyền cà phê 500 g và bẫy đa kiểm định

Quay lại dây chuyền đóng gói cà phê với khối lượng chuẩn $$\mu_0 = 500\text{ g}$$. Cân thử $$n = 16$$ gói độc lập, ta thu được trung bình mẫu $$\bar{x} = 494\text{ g}$$ và độ lệch chuẩn mẫu $$s = 8\text{ g}$$. Kiểm định ở mức ý nghĩa $$\alpha = 0{,}05$$ (hai phía):

1. **Phát biểu giả thuyết:**
   $$H_0: \mu = 500\text{ g} \qquad \text{vs.} \qquad H_1: \mu \neq 500\text{ g}.$$
2. **Tính sai số chuẩn và thống kê kiểm định $$t$$:**
   $$\mathrm{SE} = \frac{s}{\sqrt{n}} = \frac{8}{\sqrt{16}} = 2{,}0\text{ g}, \qquad t = \frac{\bar{x} - \mu_0}{\mathrm{SE}} = \frac{494 - 500}{2{,}0} = -3{,}00.$$
3. **So sánh với phân phối $$t$$ tại bậc tự do $$df = n - 1 = 15$$:**
   - Giá trị tới hạn hai phía ở mức $$\alpha = 0{,}05$$: $$t_{0{,}025;\,15} \approx 2{,}131$$.
   - Giá trị p-value chính xác: $$p = 2\,P(T_{15} \le -3{,}00) \approx 0{,}0090$$.
4. **Kết luận:** Vì $$\lvert -3{,}00\rvert > 2{,}131$$ (hay $$p = 0{,}0090 < 0{,}05$$), ta **bác bỏ $$H_0$$** — dây chuyền đang đóng gói thiếu trọng lượng có ý nghĩa thống kê (và mức thiếu 6 g trên 500 g cũng đáng kể về mặt thực tiễn!).

**Cộng thêm một phép tính nhỏ về Đa kiểm định:** Giả sử nhà máy kiểm tra đồng thời **20 dây chuyền hoàn toàn bình thường** (tất cả $$H_0$$ đều đúng) ở ngưỡng $$\alpha = 0{,}05$$ độc lập nhau. Xác suất *không có dây chuyền nào* báo động giả là $$(1 - 0{,}05)^{20} = 0{,}95^{20} \approx 0{,}358$$. Suy ra xác suất có **ít nhất 1 báo động giả** lên tới:

$$P(\text{ít nhất 1 sai lầm Loại I}) = 1 - 0{,}95^{20} \approx 64{,}2\%!$$

Nếu áp dụng hiệu chỉnh **Bonferroni**, ngưỡng mới cho từng dây chuyền phải siết xuống $$\alpha' = 0{,}05 / 20 = 0{,}0025$$.

### Khung giải thích: Bốn bước đọc hiểu kết quả kiểm định không bị lừa bởi p-value (nên đọc chậm)

1. **Nhớ đúng định nghĩa xác suất có điều kiện của p-value:** p-value là $$P(\text{dữ liệu cực đoan} \mid H_0\text{ đúng})$$, **tuyệt đối không phải** $$P(H_0\text{ đúng} \mid \text{dữ liệu})$$. Nhầm hai vế này là lỗi suy luận đảo xác suất có điều kiện (tương tự lỗi ở định lý Bayes).
2. **Đừng bao giờ nói "chấp nhận $$H_0$$" khi $$p > 0{,}05$$:** Không đủ bằng chứng kết tội không có nghĩa là bị cáo chắc chắn vô tội — rất có thể cỡ mẫu $$n$$ quá nhỏ nên lực kiểm định (power) yếu, chưa đủ nhìn thấy hiệu ứng thật.
3. **Kiểm tra xem có bao nhiêu phép thử đã chạy ngầm:** Một giá trị $$p = 0{,}03$$ rút ra từ 1 giả thuyết đặt trước có trọng lượng hoàn toàn khác với $$p = 0{,}03$$ được "đãi cát" sau khi thử 30 phân khúc khách hàng khác nhau (*p-hacking*).
4. **Đặt p-value cạnh khoảng tin cậy và độ lớn hiệu ứng:** Trong ví dụ cà phê trên, khoảng tin cậy 95% là $$494 \pm 2{,}131 \times 2 = [489{,}74;\; 498{,}26]\text{ g}$$ — không chứa 500 g và cho biết ngay mức hụt nằm khoảng 2 đến 10 g.

**Kiểm tra nhanh.** Trong ví dụ cà phê trên ($$\bar{x} = 494, s = 8$$), giả sử kiểm định viên chỉ cân có $$n = 4$$ gói thay vì 16 gói. Hãy tính nhanh lại thống kê $$t$$ — liệu lúc đó ta còn đủ bằng chứng bác bỏ $$H_0$$ ở mức $$\alpha = 0{,}05$$ ($$t_{0{,}025;\,3} \approx 3{,}182$$) không?

---

## Video Tham Khảo

- [929. Ý tưởng của kiểm định giả thuyết thống kê | Xác suất và thống kê | Khan Academy](https://www.youtube.com/watch?v=JwlCt6roH6c) — Khan Academy Vietnam, tiếng Việt. **Nên xem trước.** Giới thiệu tư duy kiểm định giả thuyết: giả sử H₀ đúng rồi đánh giá mức độ bất thường của dữ liệu quan sát.
- [p-values: What they are and how to interpret them](https://www.youtube.com/watch?v=vemZtEM63GY) — StatQuest with Josh Starmer, tiếng Anh. Làm rõ p-value là xác suất quan sát dữ liệu cực đoan dưới H₀, không phải xác suất H₀ đúng — khớp với phần cảnh báo trong bài.
- [Type I Errors, Type II Errors, and the Power of the Test](https://www.youtube.com/watch?v=7mE-K_w1v90) — jbstatistics, tiếng Anh. Giải thích sai lầm loại I, sai lầm loại II và lực kiểm định — bổ trợ cho bảng tổng hợp trước các kiểm định t và chi-bình phương.

## Bài Tập Thực Hành

**Bài 1: Phân tích lực kiểm định (Power Analysis)**
Với α = 0.05, n = 25, σ = 10:
- Tính lực kiểm định (power) ứng với các mức độ ảnh hưởng (effect size) 0.2, 0.5, 0.8 và 1.0.
- Vẽ đường cong lực kiểm định (power curve).
- Tìm cỡ mẫu n tối thiểu để đạt lực kiểm định 0.8 với mức độ ảnh hưởng 0.5.

**Bài 2: Mô phỏng thử nghiệm A/B**
Mô phỏng một thử nghiệm A/B với các thông số:
- Nhóm đối chứng (Control): tỷ lệ chuyển đổi = 10%.
- Nhóm thử nghiệm (Treatment): tỷ lệ chuyển đổi = 12%.
- Cỡ mẫu n = 1,000 cho mỗi nhóm.
- Chạy 1,000 lần mô phỏng và ước lượng lực kiểm định.

**Bài 3: So sánh bội (Multiple Comparisons)**
Cho 5 nhóm cần kiểm định so sánh từng cặp (tổng cộng 10 phép so sánh):
- Tính xác suất sai lầm trên cả họ (family-wise error rate) khi không hiệu chỉnh.
- Áp dụng hiệu chỉnh Bonferroni.
- Áp dụng phương pháp Holm–Bonferroni.

**Bài 4: Kiểm định hoán vị (Permutation Test)**
Cài đặt kiểm định hoán vị cho bài toán so sánh hai mẫu:
- So sánh kết quả với kiểm định t (t-test).
- Thử nghiệm trên dữ liệu không tuân theo phân phối chuẩn.
