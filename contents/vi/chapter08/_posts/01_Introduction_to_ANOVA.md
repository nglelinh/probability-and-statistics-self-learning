---
layout: post
title: "Bài 1: Giới Thiệu về ANOVA"
chapter: "08"
order: 1
owner: nglelinh
lang: vi
categories:
- chapter08
lesson_type: required
---

## 1. Mục Tiêu Bài Học

Sau khi hoàn thành bài học này, sinh viên sẽ có thể:
- Hiểu khái niệm **Phân tích Phương sai (Analysis of Variance - ANOVA)** và lý do tại sao nó được đặt tên như vậy.
- Giải thích tại sao **không nên dùng nhiều kiểm định t (t-tests)** để so sánh nhiều nhóm (vấn đề tăng lạm sai lầm loại I).
- Nắm vững nguyên lý **Phân rã Phương sai (Assumptions of Variance Decomposition)**: $SS_{Total} = SS_{Between} + SS_{Within}$.
- Kiểm tra các **giả định (assumptions)** của ANOVA bằng Python.

## 2. Kiến Thức Tiền Đề

- Kiểm định giả thuyết (Chapter 6).
- Phương sai và Độ lệch chuẩn (Chapter 2).
- Phân phối chuẩn (Chapter 4).

## 3. Động Lực và Giới Thiệu

Hãy tưởng tượng bạn là một kỹ sư nông nghiệp muốn so sánh hiệu quả của 3 loại phân bón: A, B, và C.
- Nếu chỉ có A và B, bạn dùng **t-test** (Chapter 6).
- Nhưng với A, B, và C, bạn phải so sánh A-B, B-C, và A-C.
- Nếu có 5 loại phân bón, bạn cần 10 cặp so sánh!

Tại sao không cứ chạy 10 cái t-test? Vì mỗi lần kiểm định, bạn chấp nhận một rủi ro sai lầm (thường là $\alpha = 0.05$). Chạy càng nhiều, rủi ro tích lũy càng lớn (giống như tung xúc xắc nhiều lần thì khả năng mặt sấp xuất hiện ít nhất một lần sẽ tăng lên). ANOVA ra đời để giải quyết vấn đề này: **So sánh đồng thời nhiều trung bình chỉ bằng một kiểm định duy nhất**.

Tên gọi "Phân tích Phương sai" nghe có vẻ lạ khi mục tiêu là "So sánh Trung bình". Bài học này sẽ giải thích nghịch lý thú vị đó.

## 4. Các Khái Niệm Cốt Lõi và Lý Thuyết

### 4.1 Tại Sao "Phân Tích Phương Sai" Lại Dùng Để So Sánh Trung Bình?

Tư tưởng chính của ANOVA là so sánh **hai nguồn biến động**:
1.  **Biến động GIỮA các nhóm (Between-Group Variability)**: Do sự khác biệt thực sự giữa các loại phân bón (tín hiệu - Signal).
2.  **Biến động TRONG các nhóm (Within-Group Variability)**: Do nhiễu ngẫu nhiên, sai số đo đạc, sự khác biệt cá thể (nhiễu - Noise).

Nếu **Signal >> Noise**, ta kết luận các nhóm khác nhau.
Tỷ số này chính là thống kê F:
$$F = \frac{\text{Variability Between Groups}}{\text{Variability Within Groups}}$$

### 4.2 Phân Rã Tổng Bình Phương (Sum of Squares Decomposition)

Phương sai tổng thể được tách thành hai phần:
$$SS_{Total} = SS_{Between} + SS_{Within}$$

-   $SS_{Total}$: Tổng độ lệch bình phương của mọi điểm dữ liệu so với trung bình chung (Grand Mean).
-   $SS_{Between}$ (hay $SS_{Tr}$ - Treatment): Tổng độ lệch bình phương của trung bình nhóm so với trung bình chung (đại diện cho hiệu ứng xử lý).
-   $SS_{Within}$ (hay $SS_{E}$ - Error): Tổng độ lệch bình phương của từng điểm dữ liệu so với trung bình của nhóm nó.

### 4.3 Các Giả Định Của ANOVA

Để kết quả ANOVA đáng tin cậy, dữ liệu phải thỏa mãn 3 giả định:
1.  **Tính Chuẩn (Normality)**: Dữ liệu trong mỗi nhóm tuân theo phân phối chuẩn. (Kiểm tra bằng Shapiro-Wilk test hoặc Q-Q plot).
2.  **Tính Đồng Nhất Phương Sai (Homogeneity of Variance / Homoscedasticity)**: Phương sai của các nhóm phải tương đương nhau. (Kiểm tra bằng Levene's test hoặc Bartlett's test).
3.  **Tính Độc Lập (Independence)**: Các quan sát phải độc lập với nhau.

## 5. Ví Dụ Minh Họa - Trực Giác Qua Hình Ảnh

Xét 3 nhóm học sinh học 3 phương pháp khác nhau.
-   **Trường hợp 1**: Các nhóm tách biệt rõ ràng. $SS_{Between}$ lớn, $SS_{Within}$ nhỏ. -> F lớn -> Bác bỏ $H_0$. (Phương pháp dạy có tác dụng).
-   **Trường hợp 2**: Các nhóm chồng lấn lên nhau. $SS_{Between}$ nhỏ, $SS_{Within}$ lớn. -> F nhỏ -> Chấp nhận $H_0$. (Khác biệt chỉ là do ngẫu nhiên).

## 6. Triển Khai trên Python

Trước khi chạy ANOVA, ta phải kiểm tra giả định và trực quan hóa dữ liệu.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# Tạo dữ liệu giả lập
# Nhóm A: Hiệu quả trung bình, Phương sai nhỏ
group_a = stats.norm.rvs(loc=50, scale=5, size=30, random_state=1)
# Nhóm B: Hiệu quả cao hơn chút, Phương sai nhỏ
group_b = stats.norm.rvs(loc=55, scale=5, size=30, random_state=1)
# Nhóm C: Hiệu quả rất cao, Phương sai lớn hơn
group_c = stats.norm.rvs(loc=65, scale=10, size=30, random_state=1)

data = pd.DataFrame({
    'Value': list(group_a) + list(group_b) + list(group_c),
    'Group': ['A']*30 + ['B']*30 + ['C']*30
})

# 1. Trực quan hóa bằng Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x='Group', y='Value', data=data)
plt.title('So sánh phân phối 3 nhóm')
plt.show()
# Nhận xét: Nhóm C có vẻ cao hơn hẳn A và B. Boxplot giúp ta có cảm nhận ban đầu.

# 2. Kiểm tra giả định Homogeneity of Variance (Levene's Test)
# H0: Phương sai các nhóm bằng nhau
stat, p_value = stats.levene(group_a, group_b, group_c)
print(f"Levene's Test: Statistic={stat:.3f}, p-value={p_value:.3f}")

if p_value > 0.05:
    print("Không đủ bằng chứng bác bỏ H0 -> Phương sai đồng nhất.")
else:
    print("Bác bỏ H0 -> Phương sai không đồng nhất (Vi phạm giả định ANOVA).")

# 3. Kiểm tra giả định Normality (Shapiro-Wilk) cho một nhóm (ví dụ nhóm A)
# H0: Dữ liệu tuân theo phân phối chuẩn
stat, p_value = stats.shapiro(group_a)
print(f"Shapiro-Wilk for Group A: p-value={p_value:.3f}")
```

## 7. Giải Thích và Các Lỗi Thường Gặp

-   **Lầm tưởng**: "ANOVA cho biết nhóm nào lớn nhất".
    -   **Sự thật**: ANOVA chỉ cho biết "Có ít nhất một nhóm khác biệt với các nhóm còn lại". Nó **không** chỉ ra cụ thể là nhóm nào. Để biết nhóm nào, ta cần bước hậu kiểm (Post-hoc analysis) sẽ học ở bài sau.
-   **Bỏ qua giả định**: Rất nhiều người chạy ANOVA mà không kiểm tra phương sai đồng nhất. Nếu phương sai khác nhau quá nhiều, kết quả F-test có thể sai lệch. Trong trường hợp đó, nên dùng **Welch's ANOVA**.

## 8. Bài Tập

1.  Tại sao khi so sánh 3 nhóm thuốc, ta không dùng 3 phép kiểm định t-test (A vs B, B vs C, A vs C)?
2.  Biến động "Within-Group" đại diện cho điều gì trong ngữ cảnh thí nghiệm khoa học?
3.  (Python) Tải tập dữ liệu `Iris`.
    -   Vẽ boxplot so sánh `sepal_width` giữa 3 loài hoa (`species`).
    -   Dùng Levene's test kiểm tra xem phương sai của `sepal_width` giữa 3 loài có bằng nhau không.

## 9. Tài Liệu Tham Khảo

-   [1] Chapter 12, *OpenIntro Statistics*.
-   [2] Chapter 12, Mario F. Triola, *Elementary Statistics*.
