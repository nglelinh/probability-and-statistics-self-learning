---
layout: post
title: "Bài 2: Tính Toán và Phân Tích ANOVA Một Yếu Tố"
chapter: "08"
order: 2
owner: nglelinh
lang: vi
categories:
- chapter08
lesson_type: required
---

## 1. Mục Tiêu Bài Học

Sau khi hoàn thành bài học này, sinh viên có thể:
- Thực hiện đầy đủ các bước **kiểm định giả thuyết ANOVA một yếu tố (One-Way ANOVA)**.
- Tính toán và diễn giải **bảng ANOVA** gồm bậc tự do (Degrees of Freedom), trung bình bình phương (Mean Squares), thống kê $F$ và giá trị $p$.
- Thực hiện **phân tích hậu kiểm (Post-hoc Analysis)** bằng kiểm định **Tukey's HSD** để xác định cụ thể các cặp nhóm khác biệt.
- Sử dụng thư viện `scipy.stats` và `statsmodels` cho quy trình phân tích hoàn chỉnh.

## 2. Kiến Thức Tiền Đề

- Bài 1: Giới thiệu về ANOVA.
- Thư viện `statsmodels` (cài đặt bằng lệnh `pip install statsmodels`).

## 3. Động Lực và Giới Thiệu

Ở bài trước, ta đã biết ANOVA sử dụng thống kê $F$ (tỷ số giữa biến động tín hiệu và biến động nhiễu) để kiểm tra xem có sự khác biệt giữa các nhóm hay không. Trong bài này, ta sẽ đi vào chi tiết **cách tính toán** bảng ANOVA và câu hỏi quan trọng tiếp theo: **cần làm gì sau khi phát hiện có sự khác biệt?**

Chẳng hạn, khi cần trả lời câu hỏi chiến dịch marketing nào hiệu quả hơn giữa Email, Facebook và TikTok, kết luận dừng ở mức "có sự khác biệt giữa các kênh" (kết quả ANOVA tổng quát) là chưa đủ để ra quyết định. Ta cần kết luận cụ thể hơn, ví dụ "Facebook hiệu quả hơn Email nhưng không khác biệt đáng kể so với TikTok" — và đó là vai trò của phân tích hậu kiểm (post-hoc).

## 4. Các Khái Niệm Cốt Lõi và Quy Trình

### 4.1 Giả Thuyết Của ANOVA Một Yếu Tố

-   $H_0$: $\mu_1 = \mu_2 = ... = \mu_k$ (trung bình tổng thể của tất cả các nhóm bằng nhau).
-   $H_a$: Có ít nhất một cặp trung bình ($\mu_i, \mu_j$) khác nhau.

### 4.2 Bảng ANOVA (The ANOVA Table)

Kết quả ANOVA thường được trình bày theo bảng chuẩn sau:

| Nguồn Biến Động (Source) | Bậc Tự Do (DF) | Tổng Bình Phương (SS) | Trung Bình Bình Phương (MS) | F-statistic | p-value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Between Groups** | $k - 1$ | $SS_{Between}$ | $MS_{B} = SS_{B} / DF_{B}$ | $F = MS_{B} / MS_{W}$ | Prob(>F) |
| **Within Groups (Error)** | $N - k$ | $SS_{Within}$ | $MS_{W} = SS_{W} / DF_{W}$ | | |
| **Total** | $N - 1$ | $SS_{Total}$ | | | |

Trong đó:
-   $k$: Số lượng nhóm.
-   $N$: Tổng số quan sát.
-   Nếu $p\_value < \alpha$ (thường chọn $\alpha = 0.05$), ta bác bỏ $H_0$.

### 4.3 Phân Tích Hậu Kiểm (Post-hoc Analysis)

Khi bác bỏ $H_0$, ta mới biết có sự khác biệt giữa các nhóm nhưng chưa biết cụ thể cặp nhóm nào khác nhau. Các kiểm định hậu kiểm phổ biến gồm:
-   **Tukey's HSD (Honest Significant Difference):** Kiểm định phổ biến nhất cho so sánh mọi cặp (pairwise comparisons), kiểm soát tốt xác suất sai lầm loại I tổng thể.
-   **Hiệu chỉnh Bonferroni (Bonferroni Correction):** Cách tính đơn giản nhưng mang tính bảo thủ cao, nên lực kiểm định thấp hơn (khó phát hiện khác biệt thực sự hơn).

## 5. Ví Dụ Minh Họa

Quay lại ví dụ so sánh 3 loại phân bón A, B và C. Giả sử kết quả ANOVA cho $F = 5.2$ và $p\_value = 0.01 < 0.05$, nên ta bác bỏ $H_0$ và kết luận hiệu quả của các loại phân bón có sự khác biệt. Tiếp theo, ta chạy kiểm định Tukey's HSD:
-   A với B: $p = 0.5$ (không có khác biệt có ý nghĩa thống kê).
-   B với C: $p = 0.04$ (có khác biệt có ý nghĩa thống kê).
-   A với C: $p = 0.02$ (có khác biệt có ý nghĩa thống kê).

Nếu trung bình của nhóm C cao hơn A và B, kết luận cuối cùng là phân bón C cho hiệu quả tốt hơn A và B, trong khi A và B không khác biệt đáng kể.

## 6. Triển Khai Trên Python

Ta sử dụng lại bộ dữ liệu giả lập gồm 3 nhóm A, B, C từ bài trước.

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

-   **Dùng kiểm định $t$ lặp lại thay cho hậu kiểm:** Như đã phân tích, việc chạy nhiều kiểm định $t$ không hiệu chỉnh sẽ làm tăng xác suất sai lầm loại I tổng thể. Tukey's HSD tự động hiệu chỉnh giá trị $p$ để giữ mức sai lầm loại I tổng thể (family-wise error rate) ở ngưỡng $\alpha$ (thường là 5%).
-   **Đọc nhầm bảng kết quả Tukey:**
    -   Cột `reject` (`True`/`False`) cho biết có bác bỏ giả thuyết hai nhóm bằng nhau hay không (`True` nghĩa là có khác biệt có ý nghĩa thống kê).
    -   Cột `meandiff` thể hiện hiệu trung bình giữa hai nhóm đang so sánh.

## Video tham khảo

- [Phân tích phương sai một yếu tố (One-Way ANOVA) | Phân Tích Thống Kê 23 | Learn to do SCIENCE](https://www.youtube.com/watch?v=JWiznZZEjBw) — Learn to do SCIENCE, tiếng Việt. **Xem trước.** Hướng dẫn tính ANOVA một yếu tố bằng tiếng Việt, tương ứng với bảng kết quả `statsmodels` trong bài.
- [Using Linear Models for t-tests and ANOVA, Clearly Explained!!!](https://www.youtube.com/watch?v=NF5_btOaCig) — StatQuest with Josh Starmer, tiếng Anh. Diễn giải kiểm định $F$ như một trường hợp của mô hình tuyến tính với ma trận thiết kế, kết nối ANOVA với hồi quy.
- [Analysis of Variance (ANOVA) and F statistics .... MADE EASY!!!](https://www.youtube.com/watch?v=YllG5Cy6OFw) — Stats with Brian, tiếng Anh. Trình bày thống kê $F$ dưới dạng tỷ số giữa phương sai giữa nhóm và phương sai trong nhóm.

## 8. Bài Tập

1.  (Python) Tải tập dữ liệu `Diet` (hoặc tạo dữ liệu giả lập cho 3 chế độ ăn kiêng khác nhau):
    -   Biến phụ thuộc: Số cân nặng giảm được (`weight_loss`).
    -   Biến phân loại: Loại chế độ ăn (`diet_type`: 1, 2, 3).
    -   Thực hiện kiểm định ANOVA để kiểm tra xem có sự khác biệt giữa các chế độ ăn hay không.
    -   Nếu có khác biệt, sử dụng kiểm định Tukey's HSD để xác định cặp chế độ ăn nào khác nhau.

## 9. Tài Liệu Tham Khảo

-   [1] Chapter 12, *OpenIntro Statistics*.
-   [2] Statsmodels Documentation: [ANOVA](https://www.statsmodels.org/stable/anova.html).
