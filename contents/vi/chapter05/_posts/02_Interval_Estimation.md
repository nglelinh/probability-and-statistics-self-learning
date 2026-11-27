---
layout: post
title: 02-06-00 Ước Lượng Khoảng
chapter: "05"
order: 2
owner: nglelinh
lang: vi
categories:
- chapter05
lesson_type: required
---

Giả sử hai công ty khảo sát độc lập cùng công bố một dòng tiêu đề giống hệt nhau: *"52% khách hàng ưa thích giao diện mới."* Thoạt nghe, cả hai đều đưa ra cùng một ước lượng điểm. Nhưng nhìn xuống dòng chú thích nhỏ ở cuối trang, bạn giật mình: Công ty A chỉ hỏi **40 người**, còn Công ty B hỏi **4.000 người**. Con số 52% của Công ty A có thể nhảy từ 36% đến 68% chỉ vì may rủi ngẫu nhiên, trong khi con số 52% của Công ty B ổn định trong khoảng hẹp 50,5% đến 53,5%. Một ước lượng điểm mà thiếu thước đo bất định chẳng khác nào một phép đo vật lý không ghi sai số!

**Ước lượng khoảng (interval estimation)** và **khoảng tin cậy (Confidence Interval — CI)** sinh ra để trả lại sự trung thực cho mọi con số thống kê. Bài học này tháo gỡ hiểu lầm kinh điển nhất về ý nghĩa của "độ tin cậy 95%", hướng dẫn cách tính chính xác khoảng tin cậy cho trung bình (dùng $$z$$ và $$t$$) và tỷ lệ, rồi giới thiệu vũ khí hiện đại **Bootstrap** — phương pháp tái lấy mẫu bằng máy tính giúp bạn dựng khoảng tin cậy cho cả những thống kê phức tạp không hề có công thức giải tích.

---

## Kiến thức cần nhớ: Khoảng tin cậy đứng ở đâu trong suy diễn thống kê?

Để không bao giờ diễn giải sai một khoảng tin cậy trong báo cáo hay bài báo khoa học, hãy nắm vững bốn trụ cột sau:

1. **Lượng xoay trục (Pivotal Quantity):** Bí quyết dựng khoảng tin cậy cổ điển là tìm một đại lượng kết hợp dữ liệu với tham số chưa biết — như $$Z = \frac{\bar{X}-\mu}{\sigma/\sqrt{n}}$$ hay $$T = \frac{\bar{X}-\mu}{S/\sqrt{n}}$$ — sao cho phân phối của nó **không phụ thuộc vào tham số ẩn**. Từ đó ta đảo bất đẳng thức để "giam" tham số vào giữa hai đầu mút ngẫu nhiên.
2. **Phân phối Student-$$t$$ và cái giá của việc không biết $$\sigma$$:** Khi phải thay độ lệch chuẩn tổng thể $$\sigma$$ bằng độ lệch chuẩn mẫu $$S$$, ta đưa thêm một nguồn bất định mới vào mẫu số. Phân phối $$t$$ có đuôi dày hơn phân phối chuẩn (đặc biệt khi bậc tự do $$n-1$$ nhỏ) để bù đắp đúng phần bất định này.
3. **Diễn giải tần suất (Frequentist) vs. Bayes:** Trong trường phái tần suất, tham số thật $$\theta$$ là một **hằng số cố định**, còn chiếc lưới $$[L, U]$$ mới là thứ nhảy ngẫu nhiên mỗi lần lấy mẫu. Mức 95% là tỷ lệ thành công dài hạn của *quy trình tung lưới*, không phải xác suất $$\theta$$ rơi vào một chiếc lưới đã tung xong (khái niệm đó thuộc về *khoảng khả tín — Credible Interval* trong thống kê Bayes).
4. **Tái lấy mẫu phi tham số (Bootstrap):** Khi thống kê quan tâm là trung vị, phân vị 95%, hay hệ số AUC của mô hình học máy, không có công thức $$t$$ nào dùng được. Bootstrap dùng chính phân phối thực nghiệm của mẫu để mô phỏng hàng nghìn lần lấy mẫu lại có hoàn lại, ước lượng trực tiếp phân phối lấy mẫu mà không cần giả định phân phối chuẩn.

---

## Khoảng Tin Cậy (Confidence Interval) Là Gì?

<div class="textbook-definition" markdown="1">
**Định nghĩa:** Khoảng tin cậy 95% cho tham số $$\theta$$ là một khoảng ngẫu nhiên $$[L, U]$$ (trong đó $$L$$ và $$U$$ được tính từ dữ liệu mẫu) thỏa mãn:

$$P(L \leq \theta \leq U) = 0.95$$
</div>

**Cách hiểu đúng:**
Nếu ta lặp lại quá trình lấy mẫu 100 lần độc lập và tính 100 khoảng tin cậy 95% tương ứng, thì **khoảng 95 trong số 100 khoảng đó** sẽ chứa tham số thực $$\mu$$. Cách diễn giải phổ biến rằng "có 95% xác suất tham số thực nằm trong một khoảng cụ thể vừa tính được" là **không chính xác** theo trường phái tần suất, vì tham số $$\theta$$ là hằng số cố định (không phải biến ngẫu nhiên) và khoảng đã tính cũng đã cố định.

![Confidence Interval Visualization]({{ site.baseurl }}/img/course/confidence_interval_viz.svg)
*Minh họa: 20 mẫu thử được rút ra. Các đoạn thẳng màu xanh chứa giá trị thực (Mean = 100), còn đoạn màu đỏ không chứa. Với mức tin cậy 95%, phần lớn các khoảng thu được sẽ chứa giá trị thực.*

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def visualize_ci_concept():
    """
    Minh họa ý nghĩa của confidence interval
    """
    # True parameter
    true_mean = 10
    true_std = 2
    
    # Lấy nhiều mẫu và tính CI cho mỗi mẫu
    n_samples = 30
    n_experiments = 100
    confidence_level = 0.95
    
    # Z-score cho 95% CI
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    # Lưu các CI
    intervals = []
    contains_true = []
    
    for _ in range(n_experiments):
        sample = np.random.normal(true_mean, true_std, n_samples)
        sample_mean = sample.mean()
        sample_std = sample.std(ddof=1)
        
        # CI: mean ± z * (std / sqrt(n))
        margin = z_score * (sample_std / np.sqrt(n_samples))
        lower = sample_mean - margin
        upper = sample_mean + margin
        
        intervals.append((lower, upper))
        contains_true.append(lower <= true_mean <= upper)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot intervals
    ax1 = axes[0]
    for i, ((lower, upper), contains) in enumerate(zip(intervals[:50], contains_true[:50])):
        color = 'green' if contains else 'red'
        ax1.plot([lower, upper], [i, i], color=color, linewidth=1.5, alpha=0.7)
        ax1.plot([(lower + upper)/2], [i], 'o', color=color, markersize=3)
    
    ax1.axvline(true_mean, color='blue', linestyle='--', linewidth=2, label=f'True μ = {true_mean}')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Experiment number')
    ax1.set_title(f'50 Confidence Intervals (95% level)\nGreen: chứa true μ, Red: không chứa')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Coverage
    coverage = np.mean(contains_true)
    ax2 = axes[1]
    ax2.bar(['Chứa true μ', 'Không chứa'], 
            [coverage * 100, (1-coverage) * 100],
            color=['green', 'red'], alpha=0.7, edgecolor='black')
    ax2.axhline(95, color='blue', linestyle='--', linewidth=2, label='Expected 95%')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title(f'Coverage: {coverage*100:.1f}% ({n_experiments} experiments)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    # plt.show()
    
    print("Confidence Interval Concept:")
    print("=" * 60)
    print(f"True mean: {true_mean}")
    print(f"Confidence level: {confidence_level*100}%")
    print(f"Number of experiments: {n_experiments}")
    print(f"Coverage (% intervals containing true mean): {coverage*100:.1f}%")
    print()
    print("⚠️  Ý nghĩa ĐÚNG: 95% các intervals (từ các mẫu khác nhau) chứa true parameter")
    print("⚠️  KHÔNG phải: Xác suất để true parameter nằm trong một interval cụ thể là 95%")

# visualize_ci_concept()
```

## Khoảng Tin Cậy Cho Trung Bình (Khi Đã Biết Phương Sai)

Nếu $$X_1, \ldots, X_n \sim N(\mu, \sigma^2)$$ và đã biết độ lệch chuẩn tổng thể $$\sigma$$, khoảng tin cậy mức $$(1 - \alpha)$$ cho $$\mu$$ là:

$$\bar{X} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

Trong đó $$z_{\alpha/2}$$ là giá trị tới hạn đuôi trên mức $$\alpha/2$$ (tức phân vị mức $$1 - \alpha/2$$) của phân phối chuẩn tắc N(0, 1):
- Khoảng tin cậy 90%: $$z_{0.05} = 1.645$$
- Khoảng tin cậy 95%: $$z_{0.025} = 1.96$$
- Khoảng tin cậy 99%: $$z_{0.005} = 2.576$$

```python
def ci_known_variance():
    """
    CI cho mean khi biết variance
    """
    # Sinh dữ liệu
    true_mean, true_std = 100, 15
    n_samples = 50
    data = np.random.normal(true_mean, true_std, n_samples)
    
    # Tính CI với các confidence levels khác nhau
    sample_mean = data.mean()
    confidence_levels = [0.90, 0.95, 0.99]
    
    print("CI cho Mean (Biết σ):")
    print("=" * 60)
    print(f"True mean: {true_mean}, True std: {true_std}")
    print(f"Sample mean: {sample_mean:.2f}, n = {n_samples}")
    print()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, conf_level in enumerate(confidence_levels):
        z_score = stats.norm.ppf((1 + conf_level) / 2)
        margin = z_score * (true_std / np.sqrt(n_samples))
        lower = sample_mean - margin
        upper = sample_mean + margin
        
        print(f"{conf_level*100}% CI: [{lower:.2f}, {upper:.2f}]")
        print(f"  Width: {upper - lower:.2f}")
        print(f"  Margin of error: ±{margin:.2f}")
        print()
        
        # Vẽ
        y_pos = i
        ax.plot([lower, upper], [y_pos, y_pos], linewidth=3, label=f'{conf_level*100}% CI')
        ax.plot([sample_mean], [y_pos], 'o', markersize=10, color='black')
    
    ax.axvline(true_mean, color='red', linestyle='--', linewidth=2, label=f'True μ = {true_mean}')
    ax.set_yticks(range(len(confidence_levels)))
    ax.set_yticklabels([f'{cl*100}%' for cl in confidence_levels])
    ax.set_xlabel('Value')
    ax.set_title('Confidence Intervals với Các Mức Tin Cậy Khác Nhau')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    # plt.show()

# ci_known_variance()
```

## Khoảng Tin Cậy Cho Trung Bình (Khi Chưa Biết Phương Sai) — Phân Phối t

Khi chưa biết $$\sigma$$, ta thay bằng độ lệch chuẩn mẫu $$s$$ và sử dụng phân phối Student's t với $$n - 1$$ bậc tự do:

$$\bar{X} \pm t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}$$

```python
def ci_unknown_variance():
    """
    CI cho mean khi KHÔNG biết variance (dùng t-distribution)
    """
    # Sinh dữ liệu
    true_mean, true_std = 100, 15
    n_samples = 20  # Mẫu nhỏ
    data = np.random.normal(true_mean, true_std, n_samples)
    
    sample_mean = data.mean()
    sample_std = data.std(ddof=1)
    
    # So sánh Z-based vs t-based CI
    confidence_level = 0.95
    
    # Z-based (SAI khi không biết σ!)
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    margin_z = z_score * (sample_std / np.sqrt(n_samples))
    ci_z = (sample_mean - margin_z, sample_mean + margin_z)
    
    # t-based (ĐÚNG)
    t_score = stats.t.ppf((1 + confidence_level) / 2, df=n_samples-1)
    margin_t = t_score * (sample_std / np.sqrt(n_samples))
    ci_t = (sample_mean - margin_t, sample_mean + margin_t)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # So sánh t vs Z distribution
    x = np.linspace(-4, 4, 1000)
    axes[0].plot(x, stats.norm.pdf(x), linewidth=2, label='Z ~ N(0,1)')
    axes[0].plot(x, stats.t.pdf(x, df=n_samples-1), linewidth=2, label=f't (df={n_samples-1})')
    axes[0].axvline(z_score, color='blue', linestyle='--', label=f'z = {z_score:.3f}')
    axes[0].axvline(t_score, color='red', linestyle='--', label=f't = {t_score:.3f}')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('Density')
    axes[0].set_title('t-distribution có đuôi dày hơn Z')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # So sánh CI
    axes[1].plot(ci_z, [0, 0], linewidth=3, label=f'Z-based: [{ci_z[0]:.2f}, {ci_z[1]:.2f}]')
    axes[1].plot(ci_t, [1, 1], linewidth=3, label=f't-based: [{ci_t[0]:.2f}, {ci_t[1]:.2f}]')
    axes[1].plot([sample_mean, sample_mean], [0, 1], 'ko', markersize=10)
    axes[1].axvline(true_mean, color='green', linestyle='--', linewidth=2, label=f'True μ = {true_mean}')
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(['Z-based\n(SAI!)', 't-based\n(ĐÚNG)'])
    axes[1].set_xlabel('Value')
    axes[1].set_title(f'95% CI (n={n_samples})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    # plt.show()
    
    print("CI cho Mean (Không biết σ):")
    print("=" * 60)
    print(f"Sample size: n = {n_samples}")
    print(f"Sample mean: {sample_mean:.2f}")
    print(f"Sample std: {sample_std:.2f}")
    print()
    print(f"Z-based CI (SAI!): [{ci_z[0]:.2f}, {ci_z[1]:.2f}], width = {ci_z[1]-ci_z[0]:.2f}")
    print(f"t-based CI (ĐÚNG): [{ci_t[0]:.2f}, {ci_t[1]:.2f}], width = {ci_t[1]-ci_t[0]:.2f}")
    print()
    print(f"⚠️  t-based CI rộng hơn vì account for uncertainty trong estimate của σ")

# ci_unknown_variance()
```

## Khoảng Tin Cậy Cho Tỷ Lệ (Proportion)

Cho $$X \sim \text{Binomial}(n, p)$$ với ước lượng tỷ lệ mẫu $$\hat{p} = X/n$$:

**Khoảng tin cậy Wald (xấp xỉ chuẩn):**
$$\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$

**Khoảng tin cậy Wilson:** Có công thức phức tạp hơn nhưng cho tỷ lệ bao phủ (coverage) chính xác hơn khi cỡ mẫu $$n$$ nhỏ hoặc $$p$$ gần 0 hoặc 1.

```python
def ci_proportion():
    """
    CI cho proportion
    """
    # Dữ liệu: 45 successes trong 100 trials
    n_trials = 100
    n_success = 45
    p_hat = n_success / n_trials
    
    confidence_level = 0.95
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    # Wald CI
    se = np.sqrt(p_hat * (1 - p_hat) / n_trials)
    margin = z_score * se
    ci_wald = (p_hat - margin, p_hat + margin)
    
    # Wilson CI (better for small n or extreme p)
    # Formula: (p̂ + z²/(2n) ± z√(p̂(1-p̂)/n + z²/(4n²))) / (1 + z²/n)
    z2 = z_score**2
    denominator = 1 + z2/n_trials
    center = (p_hat + z2/(2*n_trials)) / denominator
    margin_wilson = z_score * np.sqrt((p_hat*(1-p_hat)/n_trials + z2/(4*n_trials**2))) / denominator
    ci_wilson = (center - margin_wilson, center + margin_wilson)
    
    print("CI cho Proportion:")
    print("=" * 60)
    print(f"Data: {n_success} successes in {n_trials} trials")
    print(f"p̂ = {p_hat:.3f}")
    print()
    print(f"Wald CI:   [{ci_wald[0]:.3f}, {ci_wald[1]:.3f}]")
    print(f"Wilson CI: [{ci_wilson[0]:.3f}, {ci_wilson[1]:.3f}]")
    print()
    print("⚠️  Wilson CI thường tốt hơn, đặc biệt khi n nhỏ hoặc p gần 0/1")
    
    # Verify coverage bằng simulation
    true_p = 0.45
    n_sims = 10000
    wald_coverage = 0
    wilson_coverage = 0
    
    for _ in range(n_sims):
        sample = np.random.binomial(n_trials, true_p)
        p_hat_sim = sample / n_trials
        
        # Wald
        se_sim = np.sqrt(p_hat_sim * (1 - p_hat_sim) / n_trials)
        margin_sim = z_score * se_sim
        ci_wald_sim = (p_hat_sim - margin_sim, p_hat_sim + margin_sim)
        if ci_wald_sim[0] <= true_p <= ci_wald_sim[1]:
            wald_coverage += 1
        
        # Wilson
        center_sim = (p_hat_sim + z2/(2*n_trials)) / denominator
        margin_wilson_sim = z_score * np.sqrt((p_hat_sim*(1-p_hat_sim)/n_trials + z2/(4*n_trials**2))) / denominator
        ci_wilson_sim = (center_sim - margin_wilson_sim, center_sim + margin_wilson_sim)
        if ci_wilson_sim[0] <= true_p <= ci_wilson_sim[1]:
            wilson_coverage += 1
    
    print(f"\nCoverage Simulation (true p = {true_p}):")
    print(f"Wald CI:   {wald_coverage/n_sims*100:.2f}%")
    print(f"Wilson CI: {wilson_coverage/n_sims*100:.2f}%")
    print(f"Expected:  {confidence_level*100}%")

# ci_proportion()
```

## Khoảng Tin Cậy Bootstrap (Bootstrap Confidence Intervals)

Bootstrap là kỹ thuật tái lấy mẫu cho phép xây dựng khoảng tin cậy cho nhiều loại thống kê khác nhau mà không cần giả định dạng phân phối gốc của dữ liệu.

**Ý tưởng:** Lặp lại việc lấy mẫu có hoàn lại (resampling with replacement) từ tập dữ liệu ban đầu với cùng cỡ mẫu, sau đó tính thống kê quan tâm trên từng mẫu tái chọn để ước lượng phân phối lấy mẫu của nó.

```python
def bootstrap_ci():
    """
    Bootstrap confidence intervals
    """
    # Dữ liệu (không chuẩn!)
    np.random.seed(42)
    data = np.concatenate([
        np.random.exponential(2, 50),
        np.random.normal(10, 1, 30)
    ])
    
    # Statistic: median
    observed_median = np.median(data)
    
    # Bootstrap
    n_bootstrap = 10000
    bootstrap_medians = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        resample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_medians.append(np.median(resample))
    
    bootstrap_medians = np.array(bootstrap_medians)
    
    # CI methods
    confidence_level = 0.95
    alpha = 1 - confidence_level
    
    # 1. Percentile method
    ci_percentile = np.percentile(bootstrap_medians, [alpha/2*100, (1-alpha/2)*100])
    
    # 2. Normal approximation
    se_bootstrap = bootstrap_medians.std()
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    ci_normal = (observed_median - z_score*se_bootstrap, observed_median + z_score*se_bootstrap)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Original data
    axes[0].hist(data, bins=30, alpha=0.7, edgecolor='black')
    axes[0].axvline(observed_median, color='red', linestyle='--', linewidth=2, 
                   label=f'Median = {observed_median:.2f}')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Original Data (Non-normal!)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Bootstrap distribution
    axes[1].hist(bootstrap_medians, bins=50, alpha=0.7, edgecolor='black', density=True)
    axes[1].axvline(observed_median, color='red', linestyle='--', linewidth=2, label='Observed median')
    axes[1].axvline(ci_percentile[0], color='blue', linestyle=':', linewidth=2)
    axes[1].axvline(ci_percentile[1], color='blue', linestyle=':', linewidth=2, 
                   label=f'95% CI: [{ci_percentile[0]:.2f}, {ci_percentile[1]:.2f}]')
    axes[1].set_xlabel('Bootstrap Median')
    axes[1].set_ylabel('Density')
    axes[1].set_title(f'Bootstrap Distribution ({n_bootstrap:,} resamples)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()
    
    print("Bootstrap Confidence Intervals:")
    print("=" * 60)
    print(f"Observed median: {observed_median:.2f}")
    print(f"Bootstrap SE: {se_bootstrap:.2f}")
    print()
    print(f"Percentile CI:  [{ci_percentile[0]:.2f}, {ci_percentile[1]:.2f}]")
    print(f"Normal approx:  [{ci_normal[0]:.2f}, {ci_normal[1]:.2f}]")
    print()
    print("✓ Bootstrap hoạt động cho BẤT KỲ statistic nào!")
    print("✓ Không cần giả định về phân phối!")

# bootstrap_ci()
```

---

### Ví dụ tính tay làm từng bước: Dựng khoảng tin cậy 95% cho thời gian tải trang web

Một kỹ sư hiệu năng đo thời gian tải trang trên $$n = 9$$ lượt truy cập độc lập (giả sử xấp xỉ chuẩn, chưa biết phương sai tổng thể $$\sigma^2$$), thu được trung bình mẫu $$\bar{x} = 2{,}40\text{ giây}$$ và độ lệch chuẩn mẫu $$s = 0{,}60\text{ giây}$$. Hãy tự tay tính khoảng tin cậy 95% cho thời gian tải trung bình thật $$\mu$$:

1. **Tính sai số chuẩn (Standard Error):**
   $$\mathrm{SE} = \frac{s}{\sqrt{n}} = \frac{0{,}60}{\sqrt{9}} = 0{,}20\text{ giây}.$$
2. **Xác định giá trị tới hạn $$t$$:** Với bậc tự do $$df = n - 1 = 8$$ và mức tin cậy 95% (mỗi đuôi $$\alpha/2 = 0{,}025$$), tra bảng phân phối $$t$$ ta được:
   $$t_{0{,}025;\,8} \approx 2{,}306$$
   *(lưu ý: lớn hơn đáng kể so với $$z_{0{,}025} = 1{,}960$$ vì cỡ mẫu nhỏ $$n=9$$ làm đuôi $$t$$ dày hơn)*.
3. **Tính biên sai số (Margin of Error) và hai đầu mút:**
   $$\mathrm{ME} = 2{,}306 \times 0{,}20 = 0{,}461\text{ giây},$$
   $$\mathrm{CI}_{95\%} = [2{,}40 - 0{,}461,\; 2{,}40 + 0{,}461] = [1{,}939,\; 2{,}861]\text{ giây}.$$
4. **Thử nghiệm tư duy — Tăng cỡ mẫu lên gấp 4 lần ($$n = 36$$):** Nếu vẫn thu được $$\bar{x} = 2{,}40$$ và $$s = 0{,}60$$, khi đó $$\mathrm{SE} = 0{,}60/\sqrt{36} = 0{,}10$$ và $$t_{0{,}025;\,35} \approx 2{,}030$$, nên biên sai số thu hẹp xuống còn $$\mathrm{ME} = 0{,}203\text{ giây}$$ (khoảng $$[2{,}197,\; 2{,}603]$$). Muốn giảm một nửa độ rộng khoảng tin cậy, bạn phải trả giá bằng **gấp 4 lần dữ liệu** (quy luật $$1/\sqrt{n}$$)!

### Giải thích: Ba câu hỏi cần tự kiểm trước khi đọc một khoảng tin cậy

1. **Khoảng này đo bất định của cái gì?** Sai số chuẩn $$s/\sqrt{n}$$ đo độ dao động của *trung bình mẫu*, KHÔNG phải độ trải rộng của từng cá thể dữ liệu (độ lệch chuẩn $$s$$). Đừng nhầm khoảng tin cậy 95% của trung bình với khoảng chứa 95% dữ liệu!
2. **Độ tin cậy và độ rộng luôn đánh đổi nhau:** Với cùng một bộ dữ liệu, muốn tăng độ tin cậy từ 95% lên 99% ($$t_{0{,}005;\,8} \approx 3{,}355$$), khoảng bắt buộc phải phình rộng thêm gần 45%. Con đường duy nhất vừa tin cậy cao vừa hẹp là thu thập thêm mẫu hoặc giảm nhiễu đo lường.
3. **Diễn giải đúng tinh thần tần suất:** Khoảng $$[1{,}939,\; 2{,}861]$$ được sinh ra bởi một quy trình mà nếu lặp lại vô số lần, 95% số khoảng sẽ phủ trúng $$\mu$$. Hãy coi nó là tập hợp những giá trị $$\mu$$ "tương thích hợp lý" với dữ liệu hiện tại.
4. **Khi nào chuyển sang Bootstrap?** Khi dữ liệu lệch mạnh, mẫu vừa phải, hoặc thống kê bạn quan tâm không phải trung bình (ví dụ: trung vị thời gian phản hồi p50 hay p99), hãy dùng Bootstrap thay vì ép công thức $$t$$ đối xứng.

**Kiểm tra nhanh.** Nếu mục tiêu SLA của trang web là thời gian tải trung bình phải nhỏ hơn $$2{,}0\text{ giây}$$, nhìn vào khoảng tin cậy 95% ở trên ($$[1{,}939,\; 2{,}861]$$ khi $$n=9$$ so với $$[2{,}197,\; 2{,}603]$$ khi $$n=36$$), trong trường hợp nào ta đã đủ bằng chứng khẳng định trang web đang *vi phạm* SLA ở mức ý nghĩa 5%?

---

## Video Tham Khảo

- [Khoảng tin cậy (confidence interval) là gì? | Phân Tích Thống Kê 15 | Learn to do SCIENCE](https://www.youtube.com/watch?v=cr9QjU0yDt0) — Learn to do SCIENCE, tiếng Việt. **Nên xem trước.** Làm rõ bản chất khoảng tin cậy 95% là tính chất của quy trình lấy mẫu lặp lại, không phải xác suất tham số nằm trong một khoảng cụ thể đã tính.
- [Introduction to the t Distribution (non-technical)](https://www.youtube.com/watch?v=Uv6nGIgZMVw) — jbstatistics, tiếng Anh. Giới thiệu phân phối t khi chưa biết phương sai tổng thể — bổ trợ cho mục thay thế giá trị tới hạn z bằng phân vị t.
- [Bootstrapping Main Ideas!!!](https://www.youtube.com/watch?v=Xz0x-8-cgaQ) — StatQuest with Josh Starmer, tiếng Anh. Giải thích phương pháp Bootstrap lấy phân vị của thống kê trên các mẫu tái chọn — tương ứng với mục khoảng tin cậy Bootstrap trong bài.

## Bài Tập Thực Hành

**Bài 1: Mô phỏng tỷ lệ bao phủ (Coverage Simulation)**
Sinh 1,000 mẫu từ phân phối N(50, 10²) với cỡ mẫu n = 25. Với mỗi mẫu:
- Tính khoảng tin cậy 95% bằng phân phối t.
- Kiểm tra xem khoảng tin cậy đó có chứa trung bình thực hay không.
- Tính tỷ lệ bao phủ thực nghiệm và so sánh với mức lý thuyết 95%.

**Bài 2: Độ rộng khoảng tin cậy theo cỡ mẫu**
Với mức tin cậy cố định 95%:
- Tính độ rộng khoảng tin cậy ứng với n = 10, 20, 50, 100, 200.
- Vẽ biểu đồ độ rộng khoảng tin cậy theo n.
- Kiểm chứng mối quan hệ tỷ lệ nghịch với √n (độ rộng ∝ 1/√n).

**Bài 3: Bootstrap cho hệ số tương quan**
Sinh dữ liệu (X, Y) có hệ số tương quan ρ = 0.6.
- Tính khoảng tin cậy Bootstrap cho hệ số tương quan.
- So sánh với khoảng tin cậy dùng phép biến đổi Fisher z (Fisher's z-transformation).
- Thử nghiệm với các cỡ mẫu n = 20, 50, 100.

**Bài 4: Minh họa hiểu lầm phổ biến về khoảng tin cậy**
Tạo biểu đồ trực quan phân biệt hai phát biểu:
- Quy trình xây dựng khoảng tin cậy 95% tạo ra các khoảng chứa tham số thực với tỷ lệ 95% khi lặp lại nhiều lần (đúng).
- Tham số thực có 95% xác suất rơi vào một khoảng tin cậy cụ thể đã tính (sai).
