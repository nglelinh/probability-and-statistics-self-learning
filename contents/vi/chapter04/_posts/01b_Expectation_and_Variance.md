---
layout: post
title: 02-03-00 Kỳ Vọng và Phương Sai
chapter: "04"
order: 2
owner: nglelinh
lang: vi
categories:
- chapter02
lesson_type: required
---

Giả sử hai quỹ đầu tư cùng quảng cáo mức **lợi nhuận kỳ vọng +10%/năm**. Quỹ A dao động êm ái giữa +8% và +12%; còn quỹ B lúc thì vọt lên +60%, lúc lại lao dốc -40% khiến bạn mất gần nửa tài sản! Nếu chỉ nhìn mỗi con số "kỳ vọng", hai quỹ trông hệt như nhau — nhưng độ rủi ro của chúng thuộc về hai thế giới hoàn toàn khác biệt.

Biến ngẫu nhiên cho ta bức tranh toàn bộ phân phối, còn **Kỳ vọng (Expectation)**, **Phương sai (Variance)** và **Hiệp phương sai (Covariance)** là ba tọa độ tóm lược quyền lực nhất: trọng tâm phân phối nằm đâu, nó lắc lư mạnh cỡ nào, và hai biến ngẫu nhiên có "nhảy múa" cùng nhịp hay ngược nhịp nhau không.

---

## Kiến thức cần nhớ: Kỳ vọng, Phương sai và Hiệp phương sai nghiên cứu những gì?

1. **Kỳ vọng $$\mathbb{E}[X]$$ (Trọng tâm & Tính tuyến tính):** Trung bình có trọng số theo xác suất. Tính chất siêu mạnh là **tính tuyến tính $$\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$$ luôn đúng ngay cả khi $$X, Y$$ phụ thuộc nhau!**
2. **Phương sai $$\mathrm{Var}(X)$$ & Độ lệch chuẩn $$\sigma$$:** Đo mức phân tán trung bình bình phương quanh kỳ vọng ($$\mathrm{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$). Khi nhân biến với hằng số $$a$$, phương sai nhân với $$a^2$$ (còn cộng hằng số $$b$$ thì phương sai không đổi!).
3. **Hiệp phương sai $$\mathrm{Cov}(X, Y)$$ & Hệ số tương quan $$\rho$$:** Đo mức độ đồng biến động tuyến tính giữa hai biến; chuẩn hóa về $$[-1, +1]$$ để không phụ thuộc đơn vị đo.
4. **Quy tắc cộng phương sai & Đa dạng hóa rủi ro:** $$\mathrm{Var}(aX + bY) = a^2\mathrm{Var}(X) + b^2\mathrm{Var}(Y) + 2ab\,\mathrm{Cov}(X,Y)$$ — chính số hạng chéo $$2ab\,\mathrm{Cov}$$ là nền tảng toán học của lý thuyết danh mục đầu tư hiện đại (Markowitz) và giảm phương sai trong Ensemble Learning (Random Forest).

---

## Kỳ Vọng (Expected Value)

<div class="textbook-definition" markdown="1">
**Định nghĩa:** Kỳ vọng (hay giá trị trung bình lý thuyết) của biến ngẫu nhiên $$X$$ được định nghĩa như sau:
</div>

**Trường hợp rời rạc:**
$$E[X] = \sum_x x \cdot P(X = x) = \sum_x x \cdot p_X(x)$$

**Trường hợp liên tục:**
$$E[X] = \int_{-\infty}^{\infty} x \cdot f_X(x) \, dx$$

**Ký hiệu:** $$E[X] = \mu = \mu_X$$

### Ý Nghĩa Trực Quan

Kỳ vọng có thể được hình dung như **trọng tâm** của phân phối — điểm tựa giúp phân phối giữ thăng bằng nếu ta đặt toàn bộ khối lượng xác suất lên một thanh ngang.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def visualize_expectation():
    """
    Minh họa kỳ vọng như trung tâm khối lượng
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Phân phối rời rạc
    x_discrete = np.array([1, 2, 3, 4, 5, 6])
    pmf = np.array([0.1, 0.15, 0.25, 0.25, 0.15, 0.1])
    expected_value = np.sum(x_discrete * pmf)
    
    axes[0].stem(x_discrete, pmf, basefmt=' ', linefmt='steelblue', markerfmt='o')
    axes[0].axvline(expected_value, color='red', linestyle='--', linewidth=2, 
                   label=f'E[X] = {expected_value:.2f}')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('P(X = x)')
    axes[0].set_title('Kỳ Vọng của Phân Phối Rời Rạc')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Phân phối liên tục (Beta skewed)
    x_continuous = np.linspace(0, 1, 1000)
    pdf = stats.beta.pdf(x_continuous, 2, 5)
    expected_value_cont = 2 / (2 + 5)  # E[Beta(α,β)] = α/(α+β)
    
    axes[1].plot(x_continuous, pdf, linewidth=2, color='steelblue')
    axes[1].fill_between(x_continuous, pdf, alpha=0.3)
    axes[1].axvline(expected_value_cont, color='red', linestyle='--', linewidth=2,
                   label=f'E[X] = {expected_value_cont:.3f}')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('f(x)')
    axes[1].set_title('Kỳ Vọng của Phân Phối Liên Tục (Beta)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()

# visualize_expectation()
```

## Cạm Bẫy Trung Bình (The Average Trap)

> *"The well-wrapped average."* — Darrell Huff, *How to Lie with Statistics*

Giá trị trung bình (Mean) là thước đo vị trí quan trọng, nhưng đôi khi có thể gây hiểu nhầm, đặc biệt khi dữ liệu có phân phối lệch (skewed) hoặc chứa các giá trị ngoại lai (outliers).

### Ví Dụ: Mức Lương "Trung Bình"

Giả sử người quản lý tuyên bố mức lương trung bình tại công ty là $5,000/tháng, trong khi hầu hết nhân viên chỉ nhận được $2,000/tháng. Xét cơ cấu lương thực tế:
- Quản lý: $100,000 (1 người)
- Nhân viên: $2,000 (30 người)

$$Mean = \frac{100,000 + 30 \times 2,000}{31} \approx 5,161$$

Trong trường hợp này, **trung vị (Median)** phản ánh sát thực tế hơn ($2,000).

*   **Trung bình (Mean):** Nhạy cảm với giá trị ngoại lai và dễ bị kéo về phía đuôi dài của phân phối.
*   **Trung vị (Median):** Ít chịu ảnh hưởng bởi giá trị ngoại lai (có tính vững — robust).

![Average Trap Income](../img/average_trap_income.png)
*Phân phối thu nhập lệch phải (right-skewed): giá trị trung bình bị kéo về phía các mức thu nhập cực lớn, trong khi trung vị phản ánh sát hơn thu nhập của đa số.*

```python
def average_trap_demo():
    """
    Minh họa Mean vs Median trong phân phối thu nhập (skewed)
    """
    # Tạo phân phối thu nhập: Đa số thấp, một số ít cực cao
    np.random.seed(42)
    n_employees = 1000
    
    # Lognormal distribution thường dùng để mô phỏng thu nhập
    incomes = np.random.lognormal(mean=7.5, sigma=1.0, size=n_employees)
    
    # Thêm vài 'sếp' lương khủng
    bosses = np.array([50000, 100000, 200000]) * 10
    incomes = np.concatenate([incomes, bosses])
    
    mean_val = np.mean(incomes)
    median_val = np.median(incomes)
    
    plt.figure(figsize=(10, 6))
    plt.hist(incomes, bins=50, range=(0, 20000), alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean (Kỳ vọng) = ${mean_val:,.0f}')
    plt.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Median (Trung vị) = ${median_val:,.0f}')
    
    plt.xlabel('Thu nhập')
    plt.ylabel('Số lượng nhân viên')
    plt.title('Cạm Bẫy Trung Bình: Thu Nhập Tại Công Ty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()
    
    print("So sánh Mean và Median:")
    print("=" * 40)
    print(f"Mean Income:   ${mean_val:,.2f} (Bị kéo bởi sếp)")
    print(f"Median Income: ${median_val:,.2f} (Phản ánh đại đa số)")
    print(f"Chênh lệch:    {mean_val/median_val:.1f} lần!")

# average_trap_demo()
```

### Kiểm Chứng Bằng Mô Phỏng

```python
def verify_expectation_by_simulation():
    """
    Verify định nghĩa kỳ vọng bằng Luật Số Lớn
    """
    # Phân phối: Binomial(n=10, p=0.3)
    n, p = 10, 0.3
    expected_theoretical = n * p  # E[Binomial] = np
    
    # Mô phỏng
    n_simulations = [10, 100, 1000, 10000, 100000]
    sample_means = []
    
    for n_sim in n_simulations:
        samples = np.random.binomial(n, p, n_sim)
        sample_means.append(samples.mean())
    
    # Vẽ biểu đồ
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.semilogx(n_simulations, sample_means, 'o-', linewidth=2, markersize=8, label='Sample mean')
    plt.axhline(expected_theoretical, color='red', linestyle='--', linewidth=2, label=f'E[X] = {expected_theoretical}')
    plt.xlabel('Số lần mô phỏng')
    plt.ylabel('Sample mean')
    plt.title('Luật Số Lớn: Sample Mean → E[X]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Histogram với nhiều mẫu
    samples_large = np.random.binomial(n, p, 100000)
    plt.subplot(1, 2, 2)
    plt.hist(samples_large, bins=np.arange(-0.5, n+1.5, 1), density=True, alpha=0.7, edgecolor='black')
    plt.axvline(samples_large.mean(), color='blue', linestyle='-', linewidth=2, label=f'Sample mean = {samples_large.mean():.3f}')
    plt.axvline(expected_theoretical, color='red', linestyle='--', linewidth=2, label=f'E[X] = {expected_theoretical}')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title(f'Phân phối với {len(samples_large):,} mẫu')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()
    
    print("Verify E[X] bằng mô phỏng:")
    print("=" * 50)
    print(f"Phân phối: Binomial(n={n}, p={p})")
    print(f"E[X] lý thuyết = np = {expected_theoretical}")
    print()
    for n_sim, mean in zip(n_simulations, sample_means):
        print(f"  n={n_sim:6d}: Sample mean = {mean:.4f}")

# verify_expectation_by_simulation()
```

## Tính Chất Của Kỳ Vọng

### 1. Tính Tuyến Tính (Linearity)

$$E[aX + b] = aE[X] + b$$

$$E[X + Y] = E[X] + E[Y]$$ (luôn đúng, không đòi hỏi hai biến phải độc lập).

```python
def verify_linearity():
    """
    Verify tính tuyến tính của kỳ vọng
    """
    n_samples = 100000
    
    # Tạo hai biến ngẫu nhiên PHỤ THUỘC
    X = np.random.normal(0, 1, n_samples)
    Y = 2*X + np.random.normal(0, 0.5, n_samples)  # Y phụ thuộc vào X!
    
    # Tính kỳ vọng
    a, b = 3, 5
    
    # E[aX + b]
    lhs1 = (a*X + b).mean()
    rhs1 = a * X.mean() + b
    
    # E[X + Y]
    lhs2 = (X + Y).mean()
    rhs2 = X.mean() + Y.mean()
    
    print("Verify Tính Tuyến Tính của Kỳ Vọng:")
    print("=" * 60)
    print(f"Với a={a}, b={b}")
    print()
    print(f"E[{a}X + {b}] = {lhs1:.6f}")
    print(f"{a}E[X] + {b} = {rhs1:.6f}")
    print(f"Sai số: {abs(lhs1 - rhs1):.8f}")
    print()
    print(f"E[X + Y] = {lhs2:.6f}")
    print(f"E[X] + E[Y] = {rhs2:.6f}")
    print(f"Sai số: {abs(lhs2 - rhs2):.8f}")
    print()
    print("⚠️  Lưu ý: E[X + Y] = E[X] + E[Y] ngay cả khi X và Y PHỤ THUỘC!")

# verify_linearity()
```

### 2. Định Lý LOTUS (Law of the Unconscious Statistician)

Nếu $$Y = g(X)$$, thì kỳ vọng của $$g(X)$$ có thể tính trực tiếp từ phân phối của $$X$$:

**Trường hợp rời rạc:** $$E[g(X)] = \sum_x g(x) \cdot p_X(x)$$

**Trường hợp liên tục:** $$E[g(X)] = \int_{-\infty}^{\infty} g(x) \cdot f_X(x) \, dx$$

**Ý nghĩa:** Ta không cần tìm phân phối của $$Y$$ mà chỉ cần sử dụng trực tiếp phân phối của $$X$$.

```python
def demonstrate_lotus():
    """
    Minh họa LOTUS
    """
    # X ~ Uniform(0, 1)
    # Tính E[X²]
    
    # Phương pháp 1: LOTUS - dùng phân phối của X
    from scipy.integrate import quad
    f_x = lambda x: 1  # PDF của Uniform(0,1)
    g = lambda x: x**2
    expected_lotus, _ = quad(lambda x: g(x) * f_x(x), 0, 1)
    
    # Phương pháp 2: Mô phỏng
    n_samples = 100000
    X = np.random.uniform(0, 1, n_samples)
    Y = X**2
    expected_sim = Y.mean()
    
    # Phương pháp 3: Lý thuyết
    # E[X²] cho Uniform(0,1) = 1/3
    expected_theory = 1/3
    
    print("LOTUS: Tính E[g(X)] mà không cần phân phối của g(X)")
    print("=" * 60)
    print("X ~ Uniform(0, 1), g(X) = X²")
    print()
    print(f"E[X²] bằng LOTUS (tích phân):  {expected_lotus:.6f}")
    print(f"E[X²] bằng mô phỏng:           {expected_sim:.6f}")
    print(f"E[X²] lý thuyết:               {expected_theory:.6f}")
    
    # Visualize
    x = np.linspace(0, 1, 1000)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, np.ones_like(x), linewidth=2, label='f(x) = 1')
    plt.fill_between(x, np.ones_like(x), alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('PDF của X ~ Uniform(0, 1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(Y, bins=50, density=True, alpha=0.7, edgecolor='black', label='Histogram của Y=X²')
    plt.axvline(expected_sim, color='red', linestyle='--', linewidth=2, label=f'E[Y] = {expected_sim:.3f}')
    plt.xlabel('y')
    plt.ylabel('Density')
    plt.title('Phân phối của Y = X²')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()

# demonstrate_lotus()
```

## Phương Sai (Variance)

<div class="textbook-definition" markdown="1">
**Định nghĩa:** Phương sai đo lường mức độ phân tán của biến ngẫu nhiên xung quanh kỳ vọng của nó:

$$\text{Var}(X) = E[(X - \mu)^2] = E[X^2] - (E[X])^2$$
</div>

**Độ lệch chuẩn:** $$\sigma = \sqrt{\text{Var}(X)}$$

**Ký hiệu:** $$\text{Var}(X) = \sigma^2 = \sigma_X^2$$

```python
def visualize_variance():
    """
    Minh họa ý nghĩa của phương sai
    """
    # Hai phân phối cùng kỳ vọng nhưng khác phương sai
    mu = 0
    sigma1, sigma2 = 0.5, 2
    
    x = np.linspace(-6, 6, 1000)
    pdf1 = stats.norm.pdf(x, mu, sigma1)
    pdf2 = stats.norm.pdf(x, mu, sigma2)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, pdf1, linewidth=2, label=f'σ = {sigma1} (Var = {sigma1**2})')
    plt.plot(x, pdf2, linewidth=2, label=f'σ = {sigma2} (Var = {sigma2**2})')
    plt.axvline(mu, color='red', linestyle='--', alpha=0.5, label=f'E[X] = {mu}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Cùng Kỳ Vọng, Khác Phương Sai')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Mô phỏng
    n_samples = 10000
    samples1 = np.random.normal(mu, sigma1, n_samples)
    samples2 = np.random.normal(mu, sigma2, n_samples)
    
    plt.subplot(1, 2, 2)
    plt.hist(samples1, bins=50, density=True, alpha=0.5, label=f'σ = {sigma1}', edgecolor='black')
    plt.hist(samples2, bins=50, density=True, alpha=0.5, label=f'σ = {sigma2}', edgecolor='black')
    plt.axvline(mu, color='red', linestyle='--', linewidth=2, label=f'E[X] = {mu}')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title(f'Histogram ({n_samples:,} mẫu)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()
    
    print("Phương sai đo mức độ phân tán:")
    print("=" * 50)
    print(f"Phân phối 1: E[X] = {samples1.mean():.4f}, Var(X) = {samples1.var():.4f}")
    print(f"Phân phối 2: E[X] = {samples2.mean():.4f}, Var(X) = {samples2.var():.4f}")

# visualize_variance()
```

## Tính Chất Của Phương Sai

### 1. Công Thức Tính Nhanh

$$\text{Var}(X) = E[X^2] - (E[X])^2$$

Trong thực hành, công thức này thường thuận tiện để tính toán hơn so với định nghĩa gốc.

### 2. Biến Đổi Tuyến Tính

$$\text{Var}(aX + b) = a^2 \text{Var}(X)$$

**Lưu ý:** Hằng số cộng thêm $$b$$ chỉ dịch chuyển phân phối chứ không làm thay đổi mức độ phân tán, nên không ảnh hưởng đến phương sai.

### 3. Tổng Các Biến Độc Lập

Nếu $$X$$ và $$Y$$ **độc lập**, thì:

$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$$

**Lưu ý:** Hệ thức này nói chung **không đúng** nếu $$X$$ và $$Y$$ phụ thuộc vào nhau.

```python
def verify_variance_properties():
    """
    Verify các tính chất của phương sai
    """
    n_samples = 100000
    
    # Tạo biến ngẫu nhiên
    X = np.random.normal(5, 2, n_samples)  # E[X]=5, Var(X)=4
    
    a, b = 3, 10
    
    # Tính chất 1: Var(aX + b) = a² Var(X)
    Y = a*X + b
    var_y_empirical = Y.var()
    var_y_theoretical = a**2 * X.var()
    
    print("Tính Chất của Phương Sai:")
    print("=" * 60)
    print(f"X ~ N(5, 4), a={a}, b={b}")
    print()
    print(f"Var({a}X + {b}) empirical:  {var_y_empirical:.4f}")
    print(f"{a}² × Var(X) theoretical: {var_y_theoretical:.4f}")
    print()
    
    # Tính chất 2: Var(X + Y) cho biến độc lập vs phụ thuộc
    # Độc lập
    Z_indep = np.random.normal(3, 1.5, n_samples)  # Độc lập với X
    var_sum_indep = (X + Z_indep).var()
    var_sum_theory_indep = X.var() + Z_indep.var()
    
    # Phụ thuộc
    Z_dep = 2*X + np.random.normal(0, 0.5, n_samples)  # Phụ thuộc vào X
    var_sum_dep = (X + Z_dep).var()
    var_sum_theory_wrong = X.var() + Z_dep.var()  # Công thức SAI!
    
    print("Var(X + Z) khi X và Z độc lập:")
    print(f"  Var(X + Z) empirical:  {var_sum_indep:.4f}")
    print(f"  Var(X) + Var(Z):       {var_sum_theory_indep:.4f}")
    print()
    print("Var(X + Z) khi X và Z PHỤ THUỘC:")
    print(f"  Var(X + Z) empirical:  {var_sum_dep:.4f}")
    print(f"  Var(X) + Var(Z):       {var_sum_theory_wrong:.4f} ❌ SAI!")
    print()
    print("⚠️  Var(X + Y) = Var(X) + Var(Y) CHỈ đúng khi X và Y độc lập!")

# verify_variance_properties()
```

## Hiệp Phương Sai (Covariance) và Hệ Số Tương Quan (Correlation)

### Hiệp Phương Sai (Covariance)

$$\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]$$

**Ý nghĩa:**
- $$\text{Cov}(X, Y) > 0$$: $$X$$ và $$Y$$ có xu hướng cùng tăng hoặc cùng giảm.
- $$\text{Cov}(X, Y) < 0$$: $$X$$ tăng thì $$Y$$ có xu hướng giảm và ngược lại.
- $$\text{Cov}(X, Y) = 0$$: Hai biến không có mối quan hệ tuyến tính (tuy nhiên vẫn có thể tồn tại quan hệ phi tuyến).

**Tính chất:** Nếu $$X$$ và $$Y$$ độc lập, thì $$\text{Cov}(X, Y) = 0$$; chiều ngược lại nhìn chung **không đúng**.

### Hệ Số Tương Quan (Correlation)

$$\rho_{X,Y} = \text{Corr}(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

**Tính chất:** $$-1 \leq \rho \leq 1$$

```python
def visualize_correlation():
    """
    Minh họa correlation với các mức độ khác nhau
    """
    n_samples = 1000
    
    # Tạo dữ liệu với correlation khác nhau
    correlations = [0.9, 0.5, 0, -0.5, -0.9]
    
    fig, axes = plt.subplots(1, 5, figsize=(18, 3))
    
    for i, rho in enumerate(correlations):
        # Tạo dữ liệu với correlation rho
        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        X, Y = np.random.multivariate_normal(mean, cov, n_samples).T
        
        # Tính correlation empirical
        rho_empirical = np.corrcoef(X, Y)[0, 1]
        
        axes[i].scatter(X, Y, alpha=0.5, s=10)
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        axes[i].set_title(f'ρ = {rho}\n(empirical: {rho_empirical:.3f})')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(-3, 3)
        axes[i].set_ylim(-3, 3)
    
    plt.tight_layout()
    # plt.show()

# visualize_correlation()
```

### Cảnh Báo: Tương Quan Không Đồng Nghĩa Với Nhân Quả (Correlation ≠ Causation)

```python
def correlation_not_causation():
    """
    Minh họa: Correlation không có nghĩa là causation
    """
    n = 1000
    
    # Biến ẩn: Nhiệt độ
    temperature = np.random.uniform(15, 35, n)
    
    # Ice cream sales phụ thuộc vào nhiệt độ
    ice_cream = 100 + 10 * temperature + np.random.normal(0, 20, n)
    
    # Drowning incidents cũng phụ thuộc vào nhiệt độ
    drowning = 5 + 0.5 * temperature + np.random.normal(0, 2, n)
    
    # Tính correlation
    corr_ice_drowning = np.corrcoef(ice_cream, drowning)[0, 1]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    # Ice cream vs Drowning (có correlation!)
    axes[0].scatter(ice_cream, drowning, alpha=0.5, s=20)
    axes[0].set_xlabel('Ice Cream Sales')
    axes[0].set_ylabel('Drowning Incidents')
    axes[0].set_title(f'Correlation = {corr_ice_drowning:.3f}\n⚠️ Nhưng KHÔNG có causation!')
    axes[0].grid(True, alpha=0.3)
    
    # Temperature vs Ice cream
    axes[1].scatter(temperature, ice_cream, alpha=0.5, s=20, color='coral')
    axes[1].set_xlabel('Temperature')
    axes[1].set_ylabel('Ice Cream Sales')
    axes[1].set_title('Nhiệt độ → Ice Cream\n(Causation thực sự)')
    axes[1].grid(True, alpha=0.3)
    
    # Temperature vs Drowning
    axes[2].scatter(temperature, drowning, alpha=0.5, s=20, color='green')
    axes[2].set_xlabel('Temperature')
    axes[2].set_ylabel('Drowning Incidents')
    axes[2].set_title('Nhiệt độ → Drowning\n(Causation thực sự)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()
    
    print("Ví Dụ: Correlation ≠ Causation")
    print("=" * 60)
    print(f"Correlation(Ice Cream, Drowning) = {corr_ice_drowning:.3f}")
    print()
    print("Kết luận SAI: Ice cream gây ra đuối nước!")
    print("Kết luận ĐÚNG: Cả hai đều do nhiệt độ cao (confounding variable)")

# correlation_not_causation()
```

### Giải thích: tự tay tính sức mạnh giảm rủi ro khi ghép hai tài sản

Tại sao câu tục ngữ *"đừng bỏ hết trứng vào một giỏ"* lại là một định lý chính xác của phương sai và hiệp phương sai? Hãy tự kiểm tra bằng số cụ thể với hai cổ phiếu $$X, Y$$ đều có lợi nhuận kỳ vọng $$\mathbb{E}[X] = \mathbb{E}[Y] = 10\%$$ và độ lệch chuẩn rủi ro $$\sigma_X = \sigma_Y = 20\%$$ (tức $$\mathrm{Var}(X) = \mathrm{Var}(Y) = 400$$).

Bạn chia đôi vốn 50/50 lập danh mục $$W = 0.5X + 0.5Y$$:

1. **Lợi nhuận kỳ vọng của danh mục (luôn giữ nguyên):**
   $$\mathbb{E}[W] = 0.5(10\%) + 0.5(10\%) = 10\%.$$
2. **Rủi ro khi hai cổ phiếu ĐỘC LẬP ($$\rho = 0 \Rightarrow \mathrm{Cov}(X,Y) = 0$$):**
   $$
   \mathrm{Var}(W) = (0.5)^2(400) + (0.5)^2(400) + 0 = 100 + 100 = 200
   \;\Longrightarrow\;
   \sigma_W = \sqrt{200} \approx 14.14\%!
   $$
   Chỉ bằng việc chia đôi vốn, **rủi ro giảm gần 30% mà lợi nhuận kỳ vọng không mất một xu nào**!
3. **Rủi ro khi hai cổ phiếu TƯƠNG QUAN ÂM ($$\rho = -0.5 \Rightarrow \mathrm{Cov}(X,Y) = (-0.5)(20)(20) = -200$$):**
   $$
   \mathrm{Var}(W) = 100 + 100 + 2(0.5)(0.5)(-200) = 200 - 100 = 100
   \;\Longrightarrow\;
   \sigma_W = \sqrt{100} = 10.0\%!
   $$
   **Rủi ro bị cắt đi đúng một nửa** (từ $$20\%$$ xuống $$10\%$$)! Ngược lại, nếu $$\rho = +1$$ (hai cổ phiếu đi hệt nhau), $$\mathrm{Var}(W) = 200 + 200 = 400 \Rightarrow \sigma_W = 20\%$$ — đa dạng hóa hoàn toàn vô tác dụng. Đó cũng là lý do trong Random Forest, ta cố tình làm các cây quyết định ít tương quan nhau nhất có thể!

## Video Tham Khảo

- [Expected Values, Main Ideas!!!](https://www.youtube.com/watch?v=KLs_7b7SKi4) — StatQuest with Josh Starmer, tiếng Anh. **Nên xem trước.** Giới thiệu kỳ vọng của biến rời rạc như trung bình có trọng số theo xác suất, tương ứng với phần định nghĩa mở đầu bài.
- [Covariance, Clearly Explained!!!](https://www.youtube.com/watch?v=qtaqvPAeEJY) — StatQuest with Josh Starmer, tiếng Anh. Giải thích hiệp phương sai và hệ số tương quan, bổ trợ cho mục cuối bài trước khi bàn về quan hệ nhân quả.
- [Correlation vs Causation Explained: Why Patterns Can Mislead Us](https://www.youtube.com/watch?v=ofweqU0Lz4I) — Sprouts, tiếng Anh. Giải thích vì sao hệ số tương quan cao không đủ để kết luận quan hệ nguyên nhân — kết quả, khớp với cảnh báo trong bài.

## Bài Tập Thực Hành

**Bài 1: Tính kỳ vọng và phương sai**
Cho biến ngẫu nhiên rời rạc có PMF:
- P(X = -1) = 0.2
- P(X = 0) = 0.5
- P(X = 2) = 0.3

Tính E[X], Var(X), E[X²], E[3X + 5] và kiểm chứng bằng mô phỏng.

**Bài 2: Ứng dụng LOTUS**
Cho X ~ Uniform(0, 2). Tính E[X³] bằng hai cách và so sánh kết quả:
- Dùng định lý LOTUS (tích phân).
- Dùng mô phỏng.

**Bài 3: Phương sai danh mục đầu tư (Portfolio Variance)**
Bạn phân bổ 50% vốn vào cổ phiếu A (độ lệch chuẩn σ_A = 20%) và 50% vào cổ phiếu B (độ lệch chuẩn σ_B = 30%).
- Nếu A và B độc lập, hãy tính phương sai của danh mục đầu tư.
- Nếu hệ số tương quan Corr(A, B) = 0.5, hãy tính lại phương sai của danh mục.
- Kiểm chứng cả hai trường hợp bằng mô phỏng.

**Bài 4: Bộ tứ Anscombe (Anscombe's Quartet)**
Tìm hiểu về Anscombe's Quartet — 4 tập dữ liệu có cùng trung bình, phương sai và hệ số tương quan nhưng hình dạng phân bố hoàn toàn khác nhau khi trực quan hóa. Hãy tái tạo 4 tập dữ liệu này và vẽ biểu đồ minh họa.
