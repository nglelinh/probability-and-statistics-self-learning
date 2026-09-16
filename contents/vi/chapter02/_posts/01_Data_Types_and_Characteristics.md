---
layout: post
title: "Bài 1: Các Loại Dữ Liệu và Đặc Trưng"
chapter: "02"
order: 1
owner: "nglelinh"
---

## 1. Mục Tiêu Bài Học

Sau khi hoàn thành bài học này, sinh viên sẽ có thể:
- Phân biệt rõ ràng giữa dữ liệu **định tính** (qualitative) và **định lượng** (quantitative).
- Nhận diện các thang đo dữ liệu: **Định danh** (Nominal), **Thứ bậc** (Ordinal), **Khoảng** (Interval), và **Tỷ lệ** (Ratio).
- Hiểu sự khác biệt giữa dữ liệu **rời rạc** (discrete) và **liên tục** (continuous).
- Sử dụng **Python (Pandas)** để kiểm tra và chuyển đổi các kiểu dữ liệu thực tế.

## 2. Kiến Thức Tiền Đề

- Kiến thức cơ bản về lập trình Python.
- Thư viện `pandas` cơ bản.

## 3. Động Lực và Giới Thiệu

Hãy tưởng tượng bạn là một Data Scientist tại một công ty thương mại điện tử. Bạn nhận được một tập dữ liệu chứa thông tin khách hàng: độ tuổi, thu nhập, xếp hạng sản phẩm (1-5 sao), và loại thẻ thành viên (Vàng, Bạc, Đồng). 

Nếu bạn tính trung bình cộng của "loại thẻ thành viên", kết quả sẽ vô nghĩa. Tại sao? Vì đó là dữ liệu định tính có thứ bậc, không phải con số thực sự. Ngược lại, tính trung bình của "thu nhập" lại hoàn toàn hợp lý. Việc hiểu rõ **loại dữ liệu** là bước đầu tiên và quan trọng nhất để chọn đúng phương pháp thống kê và giải thuật Machine Learning sau này. Sai lầm ngay từ bước này sẽ dẫn đến kết quả phân tích vô giá trị ("Garbage In, Garbage Out").

## 4. Các Khái Niệm Cốt Lõi và Lý Thuyết

### 4.1 Phân Loại Dữ Liệu (Data Types)

Dữ liệu thống kê thường được chia thành hai nhóm chính:

1.  **Dữ liệu Định tính (Qualitative / Categorical Data)**:
    -   Mô tả đặc tính, tính chất, không đo lường bằng con số cụ thể theo nghĩa toán học.
    -   Ví dụ: Giới tính (Nam/Nữ), Màu sắc (Đỏ/Xanh/Vàng), Mã bưu điện.

2.  **Dữ liệu Định lượng (Quantitative / Numerical Data)**:
    -   Đo lường bằng con số, có thể thực hiện các phép toán số học (+, -, *, /).
    -   Được chia nhỏ thành 2 loại:
        -   **Rời rạc (Discrete)**: Giá trị đếm được, hữu hạn hoặc vô hạn đếm được (ví dụ: số con cái trong gia đình, số lỗi trong một đoạn code).
        -   **Liên tục (Continuous)**: Có thể nhận bất kỳ giá trị nào trong một khoảng (ví dụ: chiều cao, cân nặng, nhiệt độ).

### 4.2 Thang Đo Dữ Liệu (Levels of Measurement)

Hiểu thang đo giúp xác định phép toán nào là hợp lệ.

| Thang Đo (Level) | Đặc Điểm | Ví Dụ | Phép Toán Hợp Lệ |
| :--- | :--- | :--- | :--- |
| **Định danh (Nominal)** | Chỉ dùng để gọi tên, phân loại. Thứ tự không quan trọng. | Giới tính, Tên thành phố, Màu mắt. | Đếm (Frequency), Mode. Không thể so sánh >, <. |
| **Thứ bậc (Ordinal)** | Có thứ tự rõ ràng, nhưng khoảng cách giữa các bậc không xác định hoặc không đều. | Xếp hạng phim (1-5 sao), Trình độ học vấn (Cử nhân, Thạc sĩ, Tiến sĩ). | So sánh thứ tự (Rank), Median. Không cộng trừ. |
| **Khoảng (Interval)** | Có thứ tự, khoảng cách đều nhau và có ý nghĩa. Không có điểm 0 tuyệt đối (0 không có nghĩa là "không có gì"). | Nhiệt độ (Độ C, Độ F), Năm lịch. | Cộng, Trừ, Mean. Không thể nói "gấp đôi" (40°C không nóng gấp đôi 20°C). |
| **Tỷ lệ (Ratio)** | Giống thang đo Khoảng nhưng có điểm 0 tuyệt đối. Tỷ lệ có ý nghĩa. | Chiều cao, Cân nặng, Thu nhập, Thời gian. | Tất cả phép toán (+, -, *, /). 100g nặng gấp đôi 50g. |

## 5. Ví Dụ Minh Họa

**Ví dụ 1: Phân loại biến số**

Xét các biến số sau trong một khảo sát y tế:
1.  **Nhóm máu (A, B, AB, O)**: Định tính, Thang đo Định danh.
2.  **Mức độ đau (Nhẹ, Vừa, Nặng)**: Định tính, Thang đo Thứ bậc.
3.  **Nhiệt độ cơ thể (°C)**: Định lượng, Liên tục, Thang đo Khoảng.
4.  **Số nhịp tim mỗi phút**: Định lượng, Rời rạc, Thang đo Tỷ lệ.

## 6. Triển Khai trên Python

Chúng ta sẽ sử dụng thư viện `pandas` để làm việc với các kiểu dữ liệu này. Trong Pandas:
-   `object` hoặc `category`: Thường dùng cho dữ liệu Định tính.
-   `int64`: Dữ liệu Định lượng Rời rạc.
-   `float64`: Dữ liệu Định lượng Liên tục.

```python
import pandas as pd
import numpy as np

# Tạo một tập dữ liệu giả lập
data = {
    'KhachHang_ID': [1, 2, 3, 4, 5],
    'GioiTinh': ['Nam', 'Nu', 'Nu', 'Nam', 'Nu'],
    'TrinhDo_HocVan': ['CuNhan', 'TienSi', 'ThacSi', 'CuNhan', 'ThacSi'],
    'Diem_DanhGia': [4, 5, 3, 4, 1], # Thang đo 1-5
    'ThuNhap_HangThang_Trieu': [15.5, 30.0, 22.5, 18.0, 25.0],
    'So_Con': [0, 2, 1, 0, 3]
}

df = pd.DataFrame(data)

print("--- Xem dữ liệu thô ---")
print(df)
print("\n--- Kiểu dữ liệu trong Pandas ---")
print(df.dtypes)

# 1. Chuyển đổi GioiTinh sang kiểu Category (Nominal)
df['GioiTinh'] = df['GioiTinh'].astype('category')

# 2. Chuyển đổi TrinhDo_HocVan sang kiểu Category có thứ tự (Ordinal)
trinh_do_order = ['CuNhan', 'ThacSi', 'TienSi']
df['TrinhDo_HocVan'] = pd.Categorical(df['TrinhDo_HocVan'], categories=trinh_do_order, ordered=True)

print("\n--- Sau khi chuyển đổi kiểu dữ liệu ---")
print(df.dtypes)

# Kiểm tra thứ tự của biến Ordinal
print("\n--- So sánh biến Ordinal ---")
print(f"Thạc Sĩ > Cử Nhân: {df.iloc[2]['TrinhDo_HocVan'] > df.iloc[0]['TrinhDo_HocVan']}")

# 3. Phân biệt Rời rạc và Liên tục
# So_Con là rời rạc, ThuNhap là liên tục.
# Pandas thường lưu cả hai dưới dạng số (int64/float64), 
# nhưng về mặt thống kê chúng ta xử lý khác nhau (ví dụ khi vẽ biểu đồ).
```

## 7. Giải Thích và Các Lỗi Thường Gặp

### Lỗi 1: Xử lý dữ liệu Ordinal như dữ liệu Ratio
-   **Sai lầm**: Tính trung bình cộng của mã bưu điện hoặc xếp hạng sao (trong một số trường hợp khắt khe).
-   **Giải thích**: Khoảng cách giữa "1 sao" và "2 sao" có thể không giống khoảng cách giữa "4 sao" và "5 sao" trong cảm nhận người dùng. Tuy nhiên, trong thực tế Data Science, đôi khi ta vẫn tạm chấp nhận tính trung bình cho thang đo Likert (1-5) để đơn giản hóa, nhưng cần cẩn trọng khi diễn giải.

### Lỗi 2: Nhầm lẫn giữa Rời rạc và Liên tục
-   **Sai lầm**: Cố gắng vẽ biểu đồ đường (line plot) cho dữ liệu danh mục.
-   **Hậu quả**: Biểu đồ vô nghĩa.

## 8. Bài Tập

### Bài tập lý thuyết
1.  Phân loại các biến sau theo **Loại dữ liệu** (Định tính/Định lượng) và **Thang đo**:
    a. Số điện thoại của khách hàng.
    b. Khoảng cách từ nhà đến công ty (km).
    c. IQ score.
    d. Hạng ghế máy bay (Economy, Business, First Class).

### Bài tập Python
1.  Tạo một DataFrame chứa thông tin về 10 bộ phim bao gồm: Tên phim, Thể loại, Năm sản xuất, Doanh thu, Rating (Rotten Tomatoes).
2.  Xác định kiểu dữ liệu của từng cột.
3.  Chuyển đổi cột "Rating" (nếu đang ở dạng chuỗi, ví dụ "85%") sang dạng số.
4.  Chuyển đổi cột "Thể loại" sang dạng biến phân loại (`category`).

## 9. Tài Liệu Tham Khảo

-   [1] Chapter 1, Sheldon Ross, *A First Course in Probability*, Pearson, 2012.
-   [2] Chapter 1, David Diez et al., *OpenIntro Statistics*.
-   [3] Pandas Documentation: [Categorical Data](https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html).

Bài tùy chọn sau, [02-04-00 Ứng dụng và phát triển gần đây]({% multilang_post_url contents/chapter02/04_Modern_Applications %}), nối loại dữ liệu, đặc trưng số và biểu đồ với thực hành ML lấy dữ liệu làm trung tâm (2021–2025).
