---
layout: post
title: "Bài 3: Trực Quan Hóa Dữ Liệu (Data Visualization)"
chapter: "02"
order: 3
owner: "nglelinh"
---

Năm 1973, nhà thống kê Francis Anscombe công bố bốn bộ dữ liệu nhỏ mang tên **Bộ tứ Anscombe** khiến cả giới khoa học dữ liệu phải giật mình: cả bốn bộ đều có *trung bình $$x$$ bằng nhau (9.0)*, *trung bình $$y$$ bằng nhau (7.5)*, *độ lệch chuẩn bằng nhau*, *hệ số tương quan giống hệt nhau ($$r = 0.816$$)* và *đường hồi quy tuyến tính y hệt nhau ($$y = 3 + 0.5x$$)*. Thế nhưng ngay khi vẽ lên mặt phẳng tọa độ, một bộ là đường thẳng nhiễu, một bộ là đường cong parabol hoàn hảo, một bộ là đường thẳng bị một điểm ngoại lai duy nhất kéo lệch, và bộ cuối cùng là một cột thẳng đứng cộng thêm một điểm lạc loài!

Bài học bất hủ ấy khẳng định: **đừng bao giờ tin các con số tóm tắt khi chưa nhìn thấy biểu đồ của dữ liệu**. Trực quan hóa không phải khâu "trang trí cho slide đẹp mắt", mà là kính hiển vi đầu tiên giúp phát hiện cấu trúc ẩn, điểm ngoại lai và những cái bẫy thị giác.

---

## Kiến thức cần nhớ: Trực quan hóa dữ liệu trong thống kê nghiên cứu những gì?

Chọn đúng biểu đồ là một quyết định toán học gắn liền với kiểu dữ liệu và câu hỏi phân tích, gồm **bốn trụ cột**:

1. **Phân phối 1 biến (Distribution):** Histogram, KDE (mật độ trơn) và Boxplot — cho thấy đỉnh phân phối (một đỉnh hay hai đỉnh?), độ lệch và các điểm ngoại lai theo quy tắc $$1.5 \times \mathrm{IQR}$$.
2. **Quan hệ giữa 2+ biến (Relationships):** Biểu đồ phân tán (Scatter plot) phát hiện quan hệ phi tuyến; biểu đồ đường (Line chart) cho chuỗi thời gian; Heatmap cho ma trận tương quan.
3. **So sánh theo nhóm (Comparison):** Bar chart và Violin/Boxplot nhóm giúp đối chiếu nhanh phân phối giữa các phân khúc.
4. **Độ trung thực thị giác (Visual Integrity):** Hiểu **Hệ số nói dối (Lie Factor)** của Edward Tufte để không bị đánh lừa bởi trục tung bị cắt ngang, biểu đồ 3D méo góc nhìn hay quy mô diện tích sai tỉ lệ.

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

### Giải thích: tự tay tính Hệ số nói dối (Lie Factor) khi biểu đồ cắt trục tung

Bạn rất hay gặp những biểu đồ quảng cáo cho thấy doanh số hay hiệu năng "tăng vọt gấp ba lần", nhưng nhìn kỹ hóa ra chỉ tăng có 2%! Hãy dùng công thức **Lie Factor** của Edward Tufte để vạch trần trò ảo thuật thị giác đó bằng số cụ thể:

$$
\text{Lie Factor} = \frac{\text{Mức thay đổi thể hiện trên hình vẽ (\%)}}{\text{Mức thay đổi thực tế của dữ liệu (\%)}}.
$$

Giả sử tỷ lệ khách hàng hài lòng tăng từ **98%** năm ngoái lên **100%** năm nay:
- **Mức tăng thực tế của dữ liệu:** $$\frac{100 - 98}{98} \approx +2.04\%$$.
- **Biểu đồ trung thực (trục tung bắt đầu từ $$0$$):** Chiều cao hai cột là $$98$$ và $$100$$ → cột mới cao hơn đúng $$+2.04\%$$ → $$\text{Lie Factor} = \frac{2.04\%}{2.04\%} = 1.0$$ (hoàn toàn trung thực!).
- **Biểu đồ cắt trục tung (trục tung bắt đầu từ $$97$$):**
  - Chiều cao hiển thị của cột 1: $$98 - 97 = 1$$ đơn vị.
  - Chiều cao hiển thị của cột 2: $$100 - 97 = 3$$ đơn vị (**cao gấp 3 lần**, tức tăng $$+200\%$$ trên hình vẽ!).
  - Tính **Lie Factor**:
    $$
    \text{Lie Factor} = \frac{200\%}{2.04\%} \approx 98!
    $$
  Chỉ bằng một thao tác cắt trục tung, biểu đồ đã **phóng đại sự thật lên gần 100 lần**!

**Quy tắc thực hành:** Với **biểu đồ cột (Bar chart)** mã hóa giá trị bằng *độ dài cột*, trục tung **bắt buộc phải bắt đầu từ 0**. (Chỉ biểu đồ đường — Line chart — mới được phép thu hẹp trục tung để soi xu hướng nhỏ, và phải ghi chú rõ phạm vi trục.)

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
