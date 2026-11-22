---
layout: post
title: "Bài 3: Trực Quan Hóa Dữ Liệu (Data Visualization)"
chapter: "02"
order: 3
owner: "nglelinh"
---

## 1. Mục tiêu bài học

Sau khi hoàn thành bài học này, người học có thể:
- Hiểu vai trò của trực quan hóa dữ liệu trong quá trình khám phá dữ liệu (Exploratory Data Analysis — EDA).
- Vẽ và diễn giải các biểu đồ cơ bản: **Histogram** (phân phối), **Boxplot** (so sánh và phát hiện giá trị ngoại lai), **Scatter plot** (mối quan hệ giữa hai biến).
- Nhận diện các dạng biểu đồ gây hiểu lầm (Misleading Graphs).
- Sử dụng **Matplotlib** và **Seaborn** để tạo biểu đồ rõ ràng và đúng chuẩn trong Python.

## 2. Kiến thức tiền đề

- Bài 1 và Bài 2 (Các loại dữ liệu và Các đặc trưng số).
- Thư viện `matplotlib` và `seaborn`.

## 3. Động lực và giới thiệu

Các chỉ số tóm tắt như trung bình cộng hay phương sai cung cấp cái nhìn khái quát, nhưng đôi khi không thể hiện được cấu trúc thực sự của dữ liệu. Một ví dụ kinh điển là **Bộ tứ Anscombe (Anscombe's Quartet)**: bốn tập dữ liệu có cùng trung bình, phương sai và hệ số tương quan nhưng hình dạng phân bố hoàn toàn khác nhau.

Trực quan hóa giúp ta quan sát trực tiếp hình dạng dữ liệu, phát hiện xu hướng, quy luật (patterns) và các điểm bất thường mà con số tóm tắt đơn lẻ dễ bỏ sót.

## 4. Các khái niệm cốt lõi và các loại biểu đồ cơ bản

### 4.1 Biểu đồ phân phối (Distribution)

1. **Biểu đồ tần suất (Histogram)**:
   - Chia miền giá trị của biến liên tục thành các khoảng (bins) và đếm số quan sát rơi vào mỗi khoảng.
   - Giúp nhận biết hình dạng phân phối: đối xứng hay lệch (skewed), có một đỉnh hay nhiều đỉnh.

2. **Ước lượng mật độ hạt nhân (Kernel Density Estimation — KDE)**:
   - Làm trơn histogram để ước lượng đường cong hàm mật độ xác suất liên tục.

### 4.2 Biểu đồ so sánh và phát hiện ngoại lai

1. **Biểu đồ hộp (Boxplot)**:
   - Tóm tắt phân phối qua các mốc chính: giá trị nhỏ nhất (trong phạm vi râu), $Q_1$, trung vị (Median), $Q_3$ và giá trị lớn nhất (trong phạm vi râu).
   - Hữu ích để phát hiện **giá trị ngoại lai (outliers)** — các điểm nằm ngoài hai "râu" (whiskers) của hộp — và so sánh nhanh phân phối giữa các nhóm (chẳng hạn thu nhập theo giới tính).

### 4.3 Biểu đồ mối quan hệ (Relationship)

1. **Biểu đồ tán xạ (Scatter Plot)**:
   - Biểu diễn đồng thời hai biến liên tục trên trục hoành (X) và trục tung (Y).
   - Giúp quan sát chiều hướng và mức độ tương quan: tương quan dương (cùng tăng), tương quan âm (ngược chiều) hoặc không có tương quan rõ rệt.

## 5. Ví dụ minh họa

- **Histogram**: Biểu diễn phân phối chiều cao người trưởng thành thường có dạng hình chuông đối xứng (phân phối chuẩn).
- **Scatter Plot**: Biểu diễn giá nhà (Y) theo diện tích (X) thường cho thấy xu hướng đồng biến (diện tích tăng thì giá tăng).

## 6. Triển khai trên Python

Dưới đây là ví dụ minh họa ba loại biểu đồ trên tập dữ liệu giả lập gồm điểm hai môn Toán và Văn (tương tự cách thực hành trên bộ dữ liệu kinh điển `Iris`).

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

## 7. Các lỗi thường gặp và biểu đồ gây hiểu lầm (Misleading Graphs)

1. **Cắt trục tung (trục Y không bắt đầu từ 0)**:
   - Việc cắt bớt gốc 0 trên biểu đồ cột dễ phóng đại một chênh lệch nhỏ thành sự khác biệt lớn. Cần luôn kiểm tra phạm vi trục tung khi đọc biểu đồ.

2. **Lạm dụng biểu đồ tròn (Pie Chart)**:
   - Mắt người khó so sánh chính xác góc và diện tích, đặc biệt khi có nhiều nhóm có tỷ lệ gần nhau. Trong hầu hết trường hợp, biểu đồ cột (Bar Chart) truyền tải thông tin rõ ràng hơn.

3. **Tương quan giả (Spurious Correlations)**:
   - Hai đại lượng biến thiên cùng chiều trên biểu đồ không đồng nghĩa với quan hệ nhân quả (Correlation $\neq$ Causation). Ví dụ: doanh số bán kem và số vụ đuối nước cùng tăng vào mùa hè do cùng chịu tác động của biến ẩn là nhiệt độ.

## Video tham khảo

- [StatQuest: Histograms, Clearly Explained](https://www.youtube.com/watch?v=qBigTkBLU6g) — StatQuest with Josh Starmer, tiếng Anh. **Xem trước.** Giải thích cách histogram phản ánh hình dạng phân phối, tương ứng biểu đồ đầu tiên trong ví dụ điểm Toán.
- [Boxplots are Awesome!!!](https://www.youtube.com/watch?v=fHLhBnmwUM0) — StatQuest with Josh Starmer, tiếng Anh. Trình bày cách dùng boxplot để so sánh nhóm và nhận diện giá trị ngoại lai, khớp với ví dụ so sánh điểm Toán và Văn.
- [How to spot a misleading graph - Lea Gaslowitz](https://www.youtube.com/watch?v=E91bGT9BjYk) — TED-Ed, tiếng Anh. Phân tích các cách cắt trục hoặc chọn tỷ lệ làm sai lệch thông điệp, liên quan trực tiếp đến mục biểu đồ gây hiểu lầm.

## 8. Bài tập

1. (Python) Tải tập dữ liệu `Tips` từ seaborn (`df = sns.load_dataset('tips')`) và thực hiện các yêu cầu sau:
   - Vẽ Histogram của cột `total_bill` và nhận xét phân phối có bị lệch hay không.
   - Vẽ Boxplot của `total_bill` phân nhóm theo `day` (thứ trong tuần). Xác định ngày có mức chi tiêu cao nhất và kiểm tra xem có nhiều giá trị ngoại lai không.
   - Vẽ Scatter plot giữa `total_bill` và `tip` để quan sát mối quan hệ giữa hai biến.
   - (Nâng cao) Thêm tham số `hue="smoker"` vào scatter plot để so sánh xu hướng tiền tip giữa nhóm có hút thuốc và không hút thuốc.

## 9. Tài liệu tham khảo

- [1] Chapter 2, *OpenIntro Statistics*.
- [2] *Data Visualization: A Practical Introduction* (Kieran Healy).
- [3] Edward Tufte, *The Visual Display of Quantitative Information*.
