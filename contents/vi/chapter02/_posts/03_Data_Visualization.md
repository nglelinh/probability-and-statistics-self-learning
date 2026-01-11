---
layout: post
title: "Bài 3: Trực Quan Hóa Dữ Liệu (Data Visualization)"
chapter: "02"
order: 3
owner: "nglelinh"
---

## 1. Mục Tiêu Bài Học

Sau khi hoàn thành bài học này, sinh viên sẽ có thể:
- Hiểu vai trò của Trực quan hóa dữ liệu trong Exploratory Data Analysis (EDA).
- Tạo và giải thích các biểu đồ cơ bản: **Histogram** (Phân phối), **Boxplot** (Phát hiện outlier), **Scatter plot** (Tương quan).
- Nhận biết các biểu đồ gây hiểu lầm (Misleading Graphs).
- Sử dụng **Matplotlib** và **Seaborn** để vẽ biểu đồ đẹp và đúng chuẩn trong Python.

## 2. Kiến Thức Tiền Đề

- Bài 1 & 2 (Dữ liệu và Các đặc trưng số).
- Thư viện `matplotlib` và `seaborn`.

## 3. Động Lực và Giới Thiệu

"Một bức tranh đáng giá ngàn lời nói". Trong dữ liệu, một biểu đồ tốt đáng giá ngàn con số.
Các chỉ số như Mean hay Variance chỉ cho ta cái nhìn tổng quát, nhưng đôi khi chúng che giấu bản chất thực sự của dữ liệu (ví dụ: **Anscombe's Quartet** - 4 tập dữ liệu có cùng Mean, Variance, Correlation nhưng hình dáng hoàn toàn khác nhau).
Trực quan hóa giúp ta "nhìn thấy" dữ liệu, phát hiện xu hướng, patterns, và các điểm bất thường mà các con số đơn lẻ không thể hiện được.

## 4. Các Khái Niệm Cốt Lõi và Biểu Đồ Quan Trọng

### 4.1 Biểu Đồ Phân Phối (Distribution)

1.  **Histogram (Biểu đồ tần suất)**:
    -   Chia dữ liệu liên tục thành các khoảng (bins) và đếm số lượng điểm dữ liệu trong mỗi khoảng.
    -   Giúp hình dung: Dữ liệu có đối xứng không? Có lệch (skewed) không? Đỉnh ở đâu?
    
2.  **Kernel Density Estimation (KDE)**:
    -   Làm trơn histogram để ước lượng hàm mật độ xác suất.

### 4.2 Biểu Đồ So Sánh và Outlier

1.  **Boxplot (Biểu đồ hộp)**:
    -   Dựa trên 5 con số tóm tắt: Min, Q1, Median, Q3, Max.
    -   Rất mạnh để phát hiện **Outliers** (các điểm nằm ngoài hai "râu" của hộp).
    -   So sánh phân phối giữa các nhóm (ví dụ: Lương theo Giới tính).

### 4.3 Biểu Đồ Mối Quan Hệ (Relationship)

1.  **Scatter Plot (Biểu đồ tán xạ)**:
    -   Biểu diễn mối quan hệ giữa 2 biến liên tục (trục X và Y).
    -   Giúp nhận diện tương quan: Tích cực (cùng tăng), Tiêu cực (ngược chiều), hay Không tương quan.

## 5. Ví Dụ Minh Họa

-   **Histogram**: Cho thấy chiều cao người trưởng thành tuân theo phân phối chuẩn (hình chuông).
-   **Scatter Plot**: Vẽ Giá nhà (Y) theo Diện tích (X). Thường thấy xu hướng đi lên (Diện tích tăng -> Giá tăng).

## 6. Triển Khai trên Python

Chúng ta sẽ sử dụng bộ dữ liệu kinh điển `Iris` hoặc tạo dữ liệu giả lập.

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Cài đặt style cho đẹp
sns.set_theme(style="whitegrid")

# Tạo dữ liệu giả lập: Điểm thi 2 môn Toán và Văn
np.random.seed(42)
n_students = 200
math_scores = np.random.normal(loc=7.0, scale=1.5, size=n_students)
lit_scores = 0.6 * math_scores + np.random.normal(loc=2.0, scale=1.0, size=n_students)
# Giới hạn điểm trong [0, 10]
math_scores = np.clip(math_scores, 0, 10)
lit_scores = np.clip(lit_scores, 0, 10)

df = pd.DataFrame({'Math': math_scores, 'Literature': lit_scores})

# 1. Histogram & KDE (Phân phối điểm Toán)
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x="Math", kde=True, bins=15, color="skyblue")
plt.title("Phân phối điểm môn Toán")
plt.xlabel("Điểm số")
plt.ylabel("Số lượng sinh viên")
plt.show()

# 2. Boxplot (So sánh điểm Toán và Văn)
plt.figure(figsize=(8, 6))
sns.boxplot(data=df)
plt.title("So sánh phân phối điểm Toán và Văn")
plt.ylabel("Điểm số")
plt.show()

# 3. Scatter Plot (Tương quan Toán vs Văn)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Math", y="Literature")
plt.title("Mối quan hệ giữa điểm Toán và Văn")
plt.show()
```

## 7. Giải Thích và Các Lỗi Thường Gặp (Misleading Graphs)

1.  **Trục Y không bắt đầu từ 0**:
    -   Làm phóng đại sự khác biệt nhỏ. Truyền thông thường dùng chiêu này để tạo scandal.
    -   *Lời khuyên*: Luôn kiểm tra trục tung (Y-axis).
    
2.  **Dùng biểu đồ tròn (Pie Chart) quá nhiều**:
    -   Mắt người rất tệ trong việc so sánh diện tích/góc.
    -   *Lời khuyên*: Dùng Bar Chart thay thế Pie Chart trong hầu hết trường hợp.

3.  **Spurious Correlations (Tương quan giả)**:
    -   Vẽ biểu đồ thấy hai đường đi cùng nhau không có nghĩa là cái này gây ra cái kia (Correlation $\neq$ Causation). Ví dụ: Doanh số bán kem và Số vụ chết đuối cùng tăng vào mùa hè (biến ẩn là Nhiệt độ).

## 8. Bài Tập

1.  (Python) Tải tập dữ liệu `Tips` từ seaborn (`df = sns.load_dataset('tips')`).
    -   Vẽ Histogram của cột `total_bill`. Phân phối có lệch không?
    -   Vẽ Boxplot của `total_bill` phân nhóm theo `day` (Thứ trong tuần). Ngày nào khách chi tiêu nhiều nhất? Có nhiều outliers không?
    -   Vẽ Scatter plot giữa `total_bill` và `tip`. Có mối quan hệ gì không?
    -   (Nâng cao) Dùng `hue="smoker"` trong scatter plot để xem người hút thuốc có tip hào phóng hơn không.

## 9. Tài Liệu Tham Khảo

-   [1] Chapter 2, *OpenIntro Statistics*.
-   [2] *Data Visualization: A Practical Introduction* (Kieran Healy).
-   [3] Edward Tufte, *The Visual Display of Quantitative Information* (Sách kinh điển về Data Viz).
