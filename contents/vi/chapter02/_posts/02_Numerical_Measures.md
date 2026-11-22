---
layout: post
title: "Bài 2: Các Đặc Trưng Số (Numerical Measures)"
chapter: "02"
order: 2
owner: "nglelinh"
---

## 1. Mục tiêu bài học

Sau khi hoàn thành bài học này, người học có thể:
- Hiểu và tính toán các đại lượng đo **khuynh hướng trung tâm** (Mean, Median, Mode).
- Hiểu và tính toán các đại lượng đo **mức độ phân tán** (Variance, Standard Deviation, Range, IQR).
- Phân biệt tham số tổng thể (Parameter) với thống kê mẫu (Statistic).
- Sử dụng **Python (NumPy, Pandas)** để tính toán nhanh các đặc trưng số trên dữ liệu thực tế.

## 2. Kiến thức tiền đề

- Bài 1: Các loại dữ liệu và đặc trưng.
- Kiến thức Python cơ bản và thư viện `numpy`.

## 3. Động lực và giới thiệu

Khi làm việc với một tập dữ liệu lớn gồm hàng triệu dòng, ta không thể đọc từng quan sát để nắm bắt thông tin mà cần các con số tóm tắt đại diện:
- "Doanh thu trung bình tháng này là bao nhiêu?" → **Trung bình cộng (Mean)**.
- "Mức lương chia đôi tập nhân viên thành hai nửa bằng nhau là bao nhiêu?" → **Trung vị (Median)**.
- "Mức độ biến động của cổ phiếu này lớn hay nhỏ?" → **Độ lệch chuẩn (Standard Deviation)** — biến động càng cao, rủi ro càng lớn.

Các đặc trưng số (Numerical Measures) là công cụ cơ bản của Thống kê Mô tả (Descriptive Statistics).

## 4. Các khái niệm cốt lõi và lý thuyết

### 4.1 Khuynh hướng trung tâm (Measures of Central Tendency)

Các đại lượng này mô tả vị trí trung tâm của phân phối dữ liệu:

1. **Trung bình cộng (Mean — $\bar{x}$ hoặc $\mu$)**:
   - Công thức mẫu: $\bar{x} = \frac{\sum x_i}{n}$.
   - Nhạy cảm với giá trị ngoại lai (outliers). Chẳng hạn, nếu Elon Musk bước vào một quán bar, thu nhập trung bình của những người trong quán sẽ tăng vọt dù thu nhập thực tế của đa số mọi người không hề thay đổi.

2. **Trung vị (Median)**:
   - Giá trị nằm chính giữa dãy dữ liệu sau khi đã sắp xếp theo thứ tự.
   - Ít bị ảnh hưởng bởi giá trị ngoại lai hơn trung bình cộng, nên thường được dùng cho dữ liệu lệch như thu nhập, giá nhà hoặc tuổi thọ.

3. **Yếu vị (Mode)**:
   - Giá trị xuất hiện với tần suất cao nhất trong tập dữ liệu.
   - Có thể áp dụng cho cả dữ liệu định tính (thang đo định danh).

### 4.2 Mức độ phân tán (Measures of Dispersion)

Các đại lượng này đo lường mức độ trải rộng hay biến động của dữ liệu:

1. **Khoảng biến thiên (Range)**: $Max - Min$.
2. **Phương sai (Variance — $s^2$ hoặc $\sigma^2$)**:
   - Đo trung bình bình phương độ lệch của các quan sát so với trung bình cộng.
   - Công thức phương sai mẫu (với hiệu chỉnh Bessel): $s^2 = \frac{\sum (x_i - \bar{x})^2}{n - 1}$.
3. **Độ lệch chuẩn (Standard Deviation — $s$ hoặc $\sigma$)**:
   - Bằng căn bậc hai của phương sai: $s = \sqrt{s^2}$.
   - Có cùng đơn vị đo với dữ liệu gốc nên dễ diễn giải trực tiếp hơn phương sai.
4. **Tứ phân vị (Quartiles) và khoảng tứ phân vị (IQR — Interquartile Range)**:
   - $Q_1$ (phân vị 25%), $Q_2$ (phân vị 50%, tức trung vị), $Q_3$ (phân vị 75%).
   - $IQR = Q_3 - Q_1$: bao phủ 50% dữ liệu ở vùng giữa và ít chịu ảnh hưởng của giá trị ngoại lai.

### 4.3 Tổng thể và mẫu (Population vs. Sample)

- **Tham số tổng thể (Parameter)**: Đại lượng mô tả toàn bộ tổng thể, thường chưa biết và ký hiệu bằng chữ cái Hy Lạp ($\mu, \sigma$).
- **Thống kê mẫu (Statistic)**: Đại lượng tính từ một mẫu quan sát được, ký hiệu bằng chữ cái Latinh ($\bar{x}, s$).
- *Lưu ý*: Khi tính phương sai mẫu, ta chia cho $n-1$ thay vì $n$ để thu được ước lượng không chệch (unbiased estimator) cho phương sai tổng thể $\sigma^2$.

## 5. Ví dụ minh họa

Xét tập dữ liệu điểm thi của 5 sinh viên: $A = \{5, 7, 8, 9, 100\}$ (giả sử giá trị 100 là lỗi nhập liệu, tức một giá trị ngoại lai).

- **Mean**: $(5+7+8+9+100)/5 = 129/5 = 25.8$, cao hơn nhiều so với mặt bằng chung của đa số sinh viên.
- **Median**: Sắp xếp $\{5, 7, 8, 9, 100\}$, giá trị chính giữa là **8**, phản ánh sát thực tế hơn.

## 6. Triển khai trên Python

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

## 7. Giải thích và các lỗi thường gặp

### Ngộ nhận: "Trung bình cộng luôn tốt hơn trung vị"
Trong phân phối đối xứng (như phân phối chuẩn), trung bình cộng và trung vị trùng nhau. Tuy nhiên, với phân phối lệch (như thu nhập), trung vị đại diện cho mức điển hình tốt hơn. Vì vậy, nên xem xét cả hai chỉ số cùng lúc.

### Lỗi: Quên đặt `ddof=1` trong NumPy
- `numpy.std()` mặc định `ddof=0` (chia cho $n$, tương ứng độ lệch chuẩn tổng thể).
- `pandas.std()` mặc định `ddof=1` (chia cho $n-1$, tương ứng độ lệch chuẩn mẫu).
- Khi tính trên dữ liệu mẫu bằng NumPy, cần truyền rõ tham số `ddof=1` (bậc tự do điều chỉnh).

## Video tham khảo

- [Calculating the Mean, Variance and Standard Deviation, Clearly Explained!!!](https://www.youtube.com/watch?v=SzZ6GpcfoQY) — StatQuest with Josh Starmer, tiếng Anh. **Xem trước.** Giải thích cách tính trung bình, phương sai và độ lệch chuẩn mẫu, khớp với các đại lượng được tính trong bài.
- [Review and intuition why we divide by n-1 for the unbiased sample | Khan Academy](https://www.youtube.com/watch?v=KkaU2ur3Ymw) — Khan Academy, tiếng Anh. Lý giải trực quan vì sao phương sai mẫu chia cho $n-1$, tương ứng tham số `ddof=1` trong NumPy.
- [Population and Estimated Parameters, Clearly Explained!!!](https://www.youtube.com/watch?v=vikkiwjQqfU) — StatQuest with Josh Starmer, tiếng Anh. Phân biệt tham số tổng thể với ước lượng từ mẫu, làm rõ bản chất của trung bình và độ lệch chuẩn mẫu.

## 8. Bài tập

1. Cho tập dữ liệu: $X = \{12, 15, 12, 89, 14, 12, 16\}$.
   - Tính Mean, Median và Mode.
   - Đại lượng nào phản ánh trung tâm dữ liệu tốt nhất? Vì sao?
   - Tính Range và Standard Deviation (mẫu).

2. (Python) Tải tập dữ liệu `Titanic` (từ thư viện `seaborn` hoặc nguồn công khai).
   - Tính độ tuổi trung bình (`Age`) của hành khách.
   - So sánh trung vị độ tuổi giữa hành khách nam và nữ.
   - Kiểm tra xem giá vé (`Fare`) có phân phối lệch không bằng cách so sánh Mean và Median.

## 9. Tài liệu tham khảo

- [1] Chapter 2, Mario F. Triola, *Elementary Statistics*.
- [2] Chapter 1, Allen B. Downey, *Think Stats*.
- [3] *How to Lie with Statistics* (Huff) — Chapter "The Well-Chosen Average".
