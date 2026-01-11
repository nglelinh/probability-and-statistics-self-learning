---
layout: post
title: "Bài 2: Tính Toán và Phân Tích ANOVA Một Yếu Tố"
chapter: "08"
order: 2
owner: "nglelinh"
---

## 1. Mục Tiêu Bài Học

Sau khi hoàn thành bài học này, sinh viên sẽ có thể:
- Thực hiện các bước **Kiểm định giả thuyết ANOVA một yếu tố (One-Way ANOVA)**.
- Tính toán và diễn giải **bảng ANOVA** (Degrees of Freedom, Mean Squares, F-statistic, p-value).
- Thực hiện **Phân tích hậu kiểm (Post-hoc Analysis)** sử dụng kiểm định **Tukey's HSD** để xác định cụ thể các nhóm khác biệt.
- Sử dụng thư viện `scipy.stats` và `statsmodels` cho quy trình phân tích đầy đủ.

## 2. Kiến Thức Tiền Đề

- Bài 1: Giới thiệu về ANOVA.
- Thư viện `statsmodels` (cần cài đặt: `pip install statsmodels`).

## 3. Động Lực và Giới Thiệu

Ở bài trước, ta đã biết ANOVA dùng F-statistic (tỷ lệ Tín hiệu/Nhiễu) để kiểm tra sự khác biệt. Trong bài này, ta sẽ đi vào chi tiết **cách tính toán** con số đó và quan trọng hơn: **làm gì sau khi biết có sự khác biệt?**.

Nếu Sếp hỏi: "Chiến dịch Marketing nào hiệu quả nhất: Email, Facebook hay TikTok?", và bạn trả lời "Có sự khác biệt giữa chúng" (kết quả ANOVA), Sếp sẽ sa thải bạn. Bạn cần trả lời: "Facebook hiệu quả hơn Email, nhưng ngang bằng TikTok" (kết quả Post-hoc).

## 4. Các Khái Niệm Cốt Lõi và Quy Trình

### 4.1 Giả Thuyết Của One-Way ANOVA

-   $H_0$: $\mu_1 = \mu_2 = ... = \mu_k$ (Trung bình tất cả các nhóm đều bằng nhau).
-   $H_a$: Có ít nhất một cặp trung bình ($\mu_i, \mu_j$) khác nhau.

### 4.2 Bảng ANOVA (The ANOVA Table)

Đây là cách tiêu chuẩn để trình bày kết quả:

| Nguồn Biến Động (Source) | Bậc Tự Do (DF) | Tổng Bình Phương (SS) | Trung Bình Bình Phương (MS) | F-statistic | p-value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Between Groups** | $k - 1$ | $SS_{Between}$ | $MS_{B} = SS_{B} / DF_{B}$ | $F = MS_{B} / MS_{W}$ | Prob(>F) |
| **Within Groups (Error)** | $N - k$ | $SS_{Within}$ | $MS_{W} = SS_{W} / DF_{W}$ | | |
| **Total** | $N - 1$ | $SS_{Total}$ | | | |

-   $k$: Số lượng nhóm.
-   $N$: Tổng số quan sát.
-   Nếu $p\_value < \alpha$ (thường là 0.05) -> Bác bỏ $H_0$.

### 4.3 Phân Tích Hậu Kiểm (Post-hoc Analysis)

Khi bác bỏ $H_0$, ta biết "có sự khác biệt", nhưng chưa biết "đâu là sự khác biệt". Ta dùng Post-hoc tests.
-   **Tukey's HSD (Honest Significant Difference)**: Phổ biến nhất, kiểm soát tốt sai lầm loại I khi so sánh mọi cặp (Pairwise comparisons).
-   **Bonferroni Correction**: Đơn giản nhưng bảo thủ (khó tìm ra khác biệt hơn).

## 5. Ví Dụ Minh Họa

Quay lại ví dụ Phân bón A, B, C.
Giả sử $F = 5.2$, $p\_value = 0.01$. -> Kết luận: Hiệu quả phân bón có khác nhau.
Chạy Tukey's HSD:
-   A vs B: $p = 0.5$ (Không khác biệt).
-   B vs C: $p = 0.04$ (Khác biệt).
-   A vs C: $p = 0.02$ (Khác biệt).
-> Kết luận cuối cùng: C tốt hơn A và B; A và B tương đương nhau.

## 6. Triển Khai trên Python

Ta sẽ dùng lại ví dụ giả lập 3 nhóm A, B, C từ bài trước.

```python
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Tạo lại dữ liệu (để code độc lập)
group_a = stats.norm.rvs(loc=50, scale=5, size=30, random_state=1)
group_b = stats.norm.rvs(loc=55, scale=5, size=30, random_state=1)
group_c = stats.norm.rvs(loc=65, scale=10, size=30, random_state=1)

df = pd.DataFrame({
    'Value': list(group_a) + list(group_b) + list(group_c),
    'Group': ['A']*30 + ['B']*30 + ['C']*30
})

# CÁCH 1: Dùng Scipy (Nhanh gọn, chỉ ra F và p)
f_stat, p_val = stats.f_oneway(group_a, group_b, group_c)
print(f"Scipy One-Way ANOVA: F={f_stat:.3f}, p={p_val:.5f}")

# CÁCH 2: Dùng Statsmodels (Xuất Bảng ANOVA chi tiết - Khuyên dùng)
# Cú pháp giống R: Value phụ thuộc vào Group
model = ols('Value ~ C(Group)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("\n--- Bảng ANOVA (Statsmodels) ---")
print(anova_table)

# Kiểm tra kết luận
if p_val < 0.05:
    print("\n -> Có sự khác biệt có ý nghĩa thống kê giữa các nhóm.")
    print(" -> Tiến hành Post-hoc Test (Tukey's HSD)...")
    
    # Post-hoc Test
    tukey = pairwise_tukeyhsd(endog=df['Value'],     # Dữ liệu đo lường
                              groups=df['Group'],    # Nhóm
                              alpha=0.05)            # Mức ý nghĩa
    
    print("\n--- Kết quả Tukey HSD ---")
    print(tukey)
    
    # Vẽ biểu đồ khoảng tin cậy cho sự khác biệt
    # tukey.plot_simultaneous() 
else:
    print("\n -> Không đủ bằng chứng để bác bỏ H0. Các nhóm coi như ngang nhau.")

```

## 7. Giải Thích và Các Lỗi Thường Gặp

-   **Lỗi**: Dùng t-test thay vì Post-hoc.
    -   Như đã nói, điều này làm tăng rủi ro sai lầm. Tukey's HSD đã điều chỉnh ("phạt") p-value để giữ cho rủi ro tổng thể luôn ở mức 5%.
-   **Đọc nhầm bảng Tukey**:
    -   Nhìn vào cột `reject` (True/False). True nghĩa là có sự khác biệt.
    -   Nhìn `meandiff`: Sự khác biệt trung bình.

## 8. Bài Tập

1.  (Python) Tải tập dữ liệu `Diet` (hoặc tạo giả lập 3 chế độ ăn kiêng khác nhau).
    -   Biến phụ thuộc: Số cân giảm được (`weight_loss`).
    -   Biến phân loại: Loại chế độ ăn (`diet_type`: 1, 2, 3).
    -   Thực hiện ANOVA để xem chế độ ăn nào hiệu quả nhất.
    -   Nếu có khác biệt, dùng Tukey để chỉ ra sự khác biệt đó.

## 9. Tài Liệu Tham Khảo

-   [1] Chapter 12, *OpenIntro Statistics*.
-   [2] Statsmodels Documentation: [ANOVA](https://www.statsmodels.org/stable/anova.html).
