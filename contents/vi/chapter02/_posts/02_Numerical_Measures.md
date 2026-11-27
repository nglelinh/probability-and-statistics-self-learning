---
layout: post
title: "Bài 2: Các Đặc Trưng Số (Numerical Measures)"
chapter: "02"
order: 2
owner: "nglelinh"
---

Một câu chuyện cười nổi tiếng trong giới thống kê kể rằng: có 9 người bình thường đang ngồi uống cà phê với thu nhập trung bình 15 triệu đồng/tháng. Bỗng một tỷ phú bước vào quán — ngay lập tức, **thu nhập trung bình của cả quán vọt lên hàng trăm tỷ đồng**, dù chẳng có ai trong 9 vị khách ban đầu giàu thêm một xu nào!

Câu chuyện ấy tóm gọn linh hồn của bài học này: một bảng dữ liệu có hàng nghìn con số cần được nén lại thành vài **đặc trưng số** dễ hiểu, nhưng nếu bạn chỉ nhìn mỗi số "trung bình" mà quên đi **trung vị (median)**, **độ lệch chuẩn (SD)** và **khoảng tứ phân vị (IQR)**, những con số tóm tắt đó rất dễ nói dối bạn.

---

## Kiến thức cần nhớ: Các đặc trưng số trong thống kê mô tả đo những chiều kích nào?

![Các Đại lượng Đặc trưng: Vị trí, Phân tán & Hình dáng]({{ site.baseurl }}/img/course/ch02_02_numerical_measures.svg)

Để mô tả trọn vẹn một tập dữ liệu số mà không bị một con số đơn lẻ đánh lừa, thống kê mô tả luôn đi theo **bốn trụ cột bổ trợ nhau**:

1. **Xu hướng trung tâm (Center):** *Dữ liệu tụ quanh đâu?* Gồm Trung bình (Mean — nhạy với ngoại lai), Trung vị (Median — vững chắc khi phân phối lệch) và Yếu vị (Mode — giá trị phổ biến nhất).
2. **Độ phân tán (Spread):** *Dữ liệu trải rộng hay tụ chặt?* Gồm Phương sai (Variance), Độ lệch chuẩn (Standard Deviation — cùng đơn vị dữ liệu), và Khoảng tứ phân vị ($$\mathrm{IQR} = Q_3 - Q_1$$ — đo 50% dữ liệu ở giữa).
3. **Vị trí tương đối & Hình dạng (Position & Shape):** Phân vị (Percentiles), điểm chuẩn hóa $$Z$$-score, độ lệch (Skewness) và độ nhọn (Kurtosis).
4. **Tính vững (Robustness):** Biết cặp đôi nào nhạy cảm với ngoại lai (**Mean + SD**, tốt cho phân phối chuẩn đối xứng) và cặp đôi nào miễn nhiễm với ngoại lai (**Median + IQR**, bắt buộc dùng cho thu nhập, giá nhà hay dữ liệu đuôi nặng).

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

### Giải thích: tự tay tính Mean, Median, SD và IQR trước và sau khi có 1 ngoại lai

Hãy tự kiểm tra sức tàn phá của một điểm ngoại lai duy nhất trên mẫu lương của 5 nhân viên (triệu đồng/tháng):

1. **Mẫu gốc (sạch, đối xứng):** $$D_1 = \{10,\, 12,\, 13,\, 15,\, 15\}$$.
   - **Mean:** $$\bar x = \frac{10 + 12 + 13 + 15 + 15}{5} = 13.0$$.
   - **Median:** phần tử đứng giữa (thứ 3) $$= 13.0$$.
   - **Độ lệch chuẩn mẫu (chia $$n-1 = 4$$):**
     $$
     s = \sqrt{\frac{(-3)^2 + (-1)^2 + 0^2 + 2^2 + 2^2}{4}} = \sqrt{\frac{18}{4}} = \sqrt{4.5} \approx 2.12.
     $$
   - **IQR:** nửa dưới $$\{10, 12\} \Rightarrow Q_1 = 11$$; nửa trên $$\{15, 15\} \Rightarrow Q_3 = 15 \;\Longrightarrow\; \mathrm{IQR} = 15 - 11 = 4.0$$.

2. **Thay người thứ 5 bằng một giám đốc lương 115 triệu:** $$D_2 = \{10,\, 12,\, 13,\, 15,\, 115\}$$.
   - **Mean:** $$\bar x_{\text{mới}} = \frac{165}{5} = 33.0$$ (**tăng vọt +154%**, cao hơn gấp đôi lương của cả 4 nhân viên đầu!).
   - **Độ lệch chuẩn mẫu:**
     $$
     s_{\text{mới}} = \sqrt{\frac{(-23)^2 + (-21)^2 + (-20)^2 + (-18)^2 + 82^2}{4}} = \sqrt{\frac{8418}{4}} \approx 45.87
     $$
     (**tăng hơn 21 lần!**).
   - **Median mới:** phần tử thứ 3 vẫn là **$$13.0$$** (không nhúc nhích 1 xu!).
   - **IQR mới:** $$Q_1 = 11,\ Q_3 = 15$$ (với quy ước tính trên nửa dưới/nửa trên không gồm trung vị, hoặc đổi rất nhẹ tùy quy ước) → độ phân tán lõi hầu như miễn nhiễm!

**Kết luận dễ nhớ:** Khi gặp dữ liệu lệch phải (thu nhập, thời gian phản hồi web, giá bất động sản), hãy luôn báo cáo **Median + IQR** song hành cùng **Mean + SD**.

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
