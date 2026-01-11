---
layout: post
title: "Bài 2: Các Đặc Trưng Số (Numerical Measures)"
chapter: "02"
order: 2
owner: "nglelinh"
---

## 1. Mục Tiêu Bài Học

Sau khi hoàn thành bài học này, sinh viên sẽ có thể:
- Hiểu và tính toán các đại lượng đo lường **khuynh hướng tâm** (Mean, Median, Mode).
- Hiểu và tính toán các đại lượng đo lường **mức độ phân tán** (Variance, Standard Deviation, Range, IQR).
- Phân biệt sự khác nhau giữa tham số quần thể (Parameter) và thống kê mẫu (Statistic).
- Áp dụng **Python (NumPy, Pandas)** để tính toán các chỉ số này một cách nhanh chóng.

## 2. Kiến Thức Tiền Đề

- Bài 1: Các loại dữ liệu.
- Kỹ năng Python cơ bản, thư viện `numpy`.

## 3. Động Lực và Giới Thiệu

Khi bạn có một tập dữ liệu lớn với hàng triệu dòng, bạn không thể nhìn từng dòng để hiểu dữ liệu nói gì. Bạn cần những con số "đại diện" tóm tắt toàn bộ dữ liệu đó. 
- "Doanh thu trung bình tháng này là bao nhiêu?" -> **Mean**.
- "Mức lương mà một nửa số nhân viên đạt được là bao nhiêu?" -> **Median**.
- "Rủi ro đầu tư vào cổ phiếu này cao hay thấp?" -> **Standard Deviation** (Biến động càng cao, rủi ro càng lớn).

Các đặc trưng số (Numerical Measures) là công cụ mạnh mẽ nhất để thực hiện Thống kê Mô tả (Descriptive Statistics).

## 4. Các Khái Niệm Cốt Lõi và Lý Thuyết

### 4.1 Khuynh Hướng Tâm (Measures of Central Tendency)

Đại diện cho "trung tâm" của dữ liệu.

1.  **Trung bình cộng (Mean - $\bar{x}$ hoặc $\mu$)**:
    -   Công thức mẫu: $\bar{x} = \frac{\sum x_i}{n}$
    -   Nhạy cảm với giá trị ngoại lai (outliers). Nếu Elon Musk bước vào quán bar, thu nhập trung bình của mọi người trong quán sẽ tăng vọt, nhưng thu nhập thực tế của họ không đổi.

2.  **Trung vị (Median)**:
    -   Giá trị nằm chính giữa khi dữ liệu đã được sắp xếp.
    -   Ít bị ảnh hưởng bởi outliers hơn Mean. Thường dùng cho dữ liệu thu nhập, giá nhà, tuổi thọ.

3.  **Yếu vị (Mode)**:
    -   Giá trị xuất hiện nhiều nhất.
    -   Dùng được cho cả dữ liệu định tính (Nominal).

### 4.2 Mức Độ Phân Tán (Measures of Dispersion)

Đo lường sự "trải rộng" hay biến động của dữ liệu.

1.  **Khoảng biến thiên (Range)**: $Max - Min$.
2.  **Phương sai (Variance - $s^2$ hoặc $\sigma^2$)**:
    -   Trung bình của bình phương độ lệch so với Mean.
    -   Công thức mẫu (hiệu chỉnh Bessel): $s^2 = \frac{\sum (x_i - \bar{x})^2}{n - 1}$
3.  **Độ lệch chuẩn (Standard Deviation - $s$ hoặc $\sigma$)**:
    -   Căn bậc hai của Phương sai: $s = \sqrt{s^2}$.
    -   Cùng đơn vị đo với dữ liệu gốc, dễ diễn giải hơn Phương sai.
4.  **Tứ phân vị (Quartiles) và IQR (Interquartile Range)**:
    -   $Q_1$ (25%), $Q_2$ (50% - Median), $Q_3$ (75%).
    -   $IQR = Q_3 - Q_1$: Chứa 50% dữ liệu ở giữa, rất bền vững với outliers.

### 4.3 Quần Thể vs Mẫu (Population vs Sample)

-   **Tham số (Parameter)**: Con số mô tả cả quần thể (thường không biết, ký hiệu Hy Lạp: $\mu, \sigma$).
-   **Thống kê (Statistic)**: Con số tính từ một mẫu (biết được, tính toán được, ký hiệu Latin: $\bar{x}, s$).
-   *Lưu ý quan trọng*: Khi tính phương sai mẫu, ta chia cho $n-1$ thay vì $n$ để có ước lượng không chệch (Unbiased Estimator).

## 5. Ví Dụ Minh Họa

Xét tập dữ liệu điểm thi của 5 sinh viên: $A = \{5, 7, 8, 9, 100\}$.
(Giả sử 100 là do nhập liệu sai hoặc thang điểm lạ, đây là outlier).

-   **Mean** = $(5+7+8+9+100)/5 = 129/5 = 25.8$. (Quá cao so với phần lớn sinh viên).
-   **Median**: Sắp xếp $\{5, 7, 8, 9, 100\}$ -> Số giữa là **8**. (Phản ánh đúng thực tế hơn).

## 6. Triển Khai trên Python

```python
import numpy as np
import pandas as pd
from scipy import stats

# Tập dữ liệu có outlier
data = np.array([5, 7, 8, 9, 100])

print(f"Dữ liệu: {data}")

# 1. Khuynh hướng tâm
mean_val = np.mean(data)
median_val = np.median(data)
mode_val = stats.mode(data, keepdims=True).mode[0]

print(f"Mean: {mean_val}")      # 25.8
print(f"Median: {median_val}")  # 8.0
print(f"Mode: {mode_val}")

# 2. Phân tán
# ddof=1 nghĩa là chia cho n-1 (Sample Variance/Std)
var_val = np.var(data, ddof=1) 
std_val = np.std(data, ddof=1)
iqr_val = stats.iqr(data)

print(f"Sample Variance: {var_val:.2f}")
print(f"Sample Std Dev: {std_val:.2f}")
print(f"IQR: {iqr_val}")

# 3. Sử dụng Pandas describe()
df = pd.DataFrame({'Score': data})
print("\n--- Pandas Describe ---")
print(df.describe())
```

## 7. Giải Thích và Các Lỗi Thường Gặp

### Lỗi/Ngộ nhận: "Mean luôn tốt hơn Median"
-   **Thực tế**: Trong phân phối chuẩn (hình chuông), Mean = Median. Nhưng với phân phối lệch (như thu nhập), Median đại diện tốt hơn. Luôn kiểm tra cả hai.

### Lỗi: Quên ddof=1
-   Trong Python, `numpy.std()` mặc định `ddof=0` (chia cho $n$ - Population Std).
-   `pandas.std()` mặc định `ddof=1` (chia cho $n-1$ - Sample Std).
-   Khi làm việc với dữ liệu mẫu, luôn nhớ dùng `ddof=1` (Degrees of Freedom).

## 8. Bài Tập

1.  Cho tập dữ liệu: $X = \{12, 15, 12, 89, 14, 12, 16\}$.
    -   Tính Mean, Median, Mode.
    -   Giá trị nào đại diện tốt nhất? Tại sao?
    -   Tính Range và Standard Deviation.
    
2.  (Python) Tải tập dữ liệu `Titanic` (có sẵn trên mạng hoặc thư viện `seaborn`).
    -   Tính độ tuổi trung bình (Mean Age) của hành khách.
    -   So sánh Median Age của hành khách nam và nữ.
    -   Giá vé (Fare) có phân phối lệch không? So sánh Mean và Median của giá vé.

## 9. Tài Liệu Tham Khảo

-   [1] Chapter 2, Mario F. Triola, *Elementary Statistics*.
-   [2] Chapter 1, Allen B. Downey, *Think Stats*.
-   [3] *How to Lie with Statistics* (Huff) - Chapter "The Well-Chosen Average".
