---
layout: post
title: "Bài 1: Các Loại Dữ Liệu và Đặc Trưng"
chapter: "02"
order: 1
owner: "nglelinh"
---

Nếu ai đó cộng mã bưu điện của TP. Hồ Chí Minh (`70000`) với Hà Nội (`10000`) rồi chia đôi để kết luận "mã bưu điện trung bình của hai thành phố là `40000`", hẳn bạn sẽ bật cười vì sự vô nghĩa đó. Thế nhưng trong phân tích dữ liệu thực tế, rất nhiều lỗi ngớ ngẩn lại xảy ra theo đúng kiểu ấy — chỉ tinh vi hơn một chút: lấy trung bình số sao đánh giá 1–5 rồi xem khoảng cách từ 1 lên 2 sao bằng hệt khoảng cách từ 4 lên 5 sao, hay tuyên bố hôm nay $$40^\circ\mathrm{C}$$ "nóng gấp đôi" hôm qua $$20^\circ\mathrm{C}$$!

Mọi công cụ thống kê — từ trung bình, trung vị, biểu đồ cột đến mô hình học máy — đều chỉ có ý nghĩa khi ta tôn trọng **bản chất và thang đo của dữ liệu**. Bài học mở đầu Chương 2 giúp bạn phân loại rạch ròi từng kiểu dữ liệu trước khi thực hiện bất kỳ phép tính nào.

---

## Kiến thức cần nhớ: Thang đo dữ liệu (NOIR) là gì và nghiên cứu những gì?

Trong thống kê mô tả, dữ liệu được tổ chức theo hai trục: **Định tính vs. Định lượng** (rời rạc/liên tục) và **4 thang đo kinh điển của Stevens (NOIR)** — mỗi nấc mở khóa thêm một nhóm phép toán hợp lệ:

1. **Định danh (Nominal):** Chỉ phân loại nhãn không có thứ tự (màu mắt, tỉnh thành, mã sản phẩm) → chỉ hợp lệ phép **đếm tần suất và Mode**.
2. **Thứ bậc (Ordinal):** Có thứ tự rõ ràng nhưng khoảng cách giữa các bậc không nhất thiết bằng nhau (xếp hạng 1–5 sao, mức độ hài lòng) → thêm được **Trung vị (Median) và phân vị**.
3. **Khoảng (Interval):** Hiệu số giữa hai giá trị có ý nghĩa vật lý đều nhau, nhưng **số 0 là quy ước tùy ý** (nhiệt độ $$^\circ\mathrm{C}$$, năm dương lịch) → cộng/trừ và **Trung bình** hợp lệ, nhưng **phép chia tỷ số vô nghĩa**.
4. **Tỷ lệ (Ratio):** Có **điểm 0 tuyệt đối** (thu nhập, chiều cao, thời gian, nhiệt độ Kelvin) → mọi phép toán cộng, trừ, nhân, chia tỷ số đều hợp lệ.

---

## 1. Mục tiêu bài học

Sau khi hoàn thành bài học này, người học có thể:
- Phân biệt rõ giữa dữ liệu **định tính** (qualitative) và **định lượng** (quantitative).
- Nhận diện bốn thang đo dữ liệu: **định danh** (Nominal), **thứ bậc** (Ordinal), **khoảng** (Interval) và **tỷ lệ** (Ratio).
- Phân biệt dữ liệu **rời rạc** (discrete) với dữ liệu **liên tục** (continuous).
- Sử dụng **Python (Pandas)** để kiểm tra và chuyển đổi kiểu dữ liệu trong thực tế.

## 2. Kiến thức tiền đề

- Kiến thức cơ bản về lập trình Python.
- Thao tác cơ bản với thư viện `pandas`.

## 3. Động lực và giới thiệu

Hãy xét một tình huống quen thuộc trong thương mại điện tử: bạn nhận được tập dữ liệu khách hàng gồm độ tuổi, thu nhập, điểm đánh giá sản phẩm (1–5 sao) và hạng thẻ thành viên (Vàng, Bạc, Đồng).

Nếu tính trung bình cộng của "hạng thẻ thành viên", kết quả sẽ không có ý nghĩa vì đây là dữ liệu định tính có thứ bậc, không phải đại lượng số học. Ngược lại, tính trung bình của "thu nhập" lại hoàn toàn hợp lý. Xác định đúng **loại dữ liệu** là bước nền tảng để chọn phương pháp thống kê và thuật toán học máy phù hợp; sai sót ngay từ bước này sẽ dẫn đến kết quả phân tích sai lệch ("Garbage In, Garbage Out").

## 4. Các khái niệm cốt lõi và lý thuyết

### 4.1 Phân loại dữ liệu (Data Types)

Dữ liệu thống kê thường được chia thành hai nhóm chính:

1. **Dữ liệu định tính (Qualitative / Categorical Data)**:
   - Mô tả thuộc tính hoặc nhãn phân loại, không biểu diễn đại lượng đo lường theo nghĩa số học.
   - Ví dụ: giới tính (Nam/Nữ), màu sắc (Đỏ/Xanh/Vàng), mã bưu điện.

2. **Dữ liệu định lượng (Quantitative / Numerical Data)**:
   - Biểu diễn bằng giá trị số, cho phép thực hiện các phép toán số học (+, -, *, /).
   - Gồm hai loại:
     - **Rời rạc (Discrete)**: Nhận các giá trị đếm được, hữu hạn hoặc vô hạn đếm được (ví dụ: số con trong gia đình, số lỗi trong một đoạn mã).
     - **Liên tục (Continuous)**: Có thể nhận bất kỳ giá trị nào trong một khoảng số thực (ví dụ: chiều cao, cân nặng, nhiệt độ).

### 4.2 Thang đo dữ liệu (Levels of Measurement)

Xác định đúng thang đo giúp biết phép toán và thước đo thống kê nào hợp lệ:

| Thang đo (Level) | Đặc điểm | Ví dụ | Phép toán hợp lệ |
| :--- | :--- | :--- | :--- |
| **Định danh (Nominal)** | Dùng để gọi tên và phân loại; thứ tự giữa các nhãn không có ý nghĩa. | Giới tính, tên thành phố, màu mắt. | Đếm tần số (Frequency), yếu vị (Mode). Không so sánh lớn hơn/nhỏ hơn. |
| **Thứ bậc (Ordinal)** | Có thứ tự rõ ràng, nhưng khoảng cách giữa các bậc không xác định hoặc không đều nhau. | Xếp hạng phim (1–5 sao), trình độ học vấn (Cử nhân, Thạc sĩ, Tiến sĩ). | So sánh thứ tự (Rank), trung vị (Median). Không cộng trừ trực tiếp. |
| **Khoảng (Interval)** | Có thứ tự, khoảng cách giữa các giá trị đều nhau và có ý nghĩa, nhưng không có điểm 0 tuyệt đối (giá trị 0 không có nghĩa là "không có gì"). | Nhiệt độ (°C, °F), năm lịch. | Cộng, trừ, tính trung bình (Mean). Không lấy tỷ số (40°C không nóng gấp đôi 20°C). |
| **Tỷ lệ (Ratio)** | Có đầy đủ tính chất của thang đo khoảng và có điểm 0 tuyệt đối, nên tỷ số có ý nghĩa. | Chiều cao, cân nặng, thu nhập, thời gian. | Mọi phép toán (+, -, *, /); ví dụ 100g nặng gấp đôi 50g. |

## 5. Ví dụ minh họa

**Ví dụ 1: Phân loại biến số**

Xét các biến số sau trong một khảo sát y tế:
1. **Nhóm máu (A, B, AB, O)**: Định tính, thang đo định danh.
2. **Mức độ đau (Nhẹ, Vừa, Nặng)**: Định tính, thang đo thứ bậc.
3. **Nhiệt độ cơ thể (°C)**: Định lượng, liên tục, thang đo khoảng.
4. **Số nhịp tim mỗi phút**: Định lượng, rời rạc, thang đo tỷ lệ.

## 6. Triển khai trên Python

Thư viện `pandas` hỗ trợ quản lý các kiểu dữ liệu tương ứng:
- `object` hoặc `category`: thường dùng cho dữ liệu định tính.
- `int64`: thường dùng cho dữ liệu định lượng rời rạc.
- `float64`: thường dùng cho dữ liệu định lượng liên tục.

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

## 7. Giải thích và các lỗi thường gặp

### Lỗi 1: Xử lý dữ liệu định danh hoặc thứ bậc như dữ liệu tỷ lệ
- **Sai lầm**: Tính trung bình cộng cho mã bưu điện (dữ liệu định danh) hoặc xếp hạng sao (dữ liệu thứ bậc) mà không xét đến bản chất thang đo.
- **Giải thích**: Mã bưu điện chỉ là nhãn định danh nên giá trị trung bình hoàn toàn vô nghĩa. Với xếp hạng sao, khoảng cách cảm nhận giữa "1 sao" và "2 sao" có thể không tương đương khoảng cách giữa "4 sao" và "5 sao". Trong thực tế phân tích dữ liệu, đôi khi thang đo Likert (1–5) vẫn được tính trung bình để tóm tắt nhanh, nhưng cần diễn giải thận trọng.

### Lỗi 2: Chọn sai biểu đồ cho kiểu dữ liệu
- **Sai lầm**: Vẽ biểu đồ đường (line plot) cho dữ liệu danh mục không có thứ tự.
- **Hậu quả**: Đường nối giữa các nhãn danh mục tạo cảm giác về một xu hướng liên tục không hề tồn tại.

### Giải thích: tự tay kiểm tra vì sao $$40^\circ\mathrm{C}$$ KHÔNG "nóng gấp đôi" $$20^\circ\mathrm{C}$$

Hãy kiểm chứng sự khác biệt giữa thang **Khoảng (Interval)** và thang **Tỷ lệ (Ratio)** bằng phép đổi đơn vị đơn giản:

1. **Kiểm tra nhiệt độ (Thang Khoảng):** So sánh hai ngày $$20^\circ\mathrm{C}$$ và $$40^\circ\mathrm{C}$$.
   - Tỷ số ngây thơ theo độ C: $$\frac{40}{20} = 2.0$$ ("nóng gấp đôi"?).
   - Đổi sang thang đo vật lý tuyệt đối **Kelvin** ($$K = {}^\circ\mathrm{C} + 273.15$$):
     $$
     20^\circ\mathrm{C} = 293.15\,\mathrm{K}, \qquad 40^\circ\mathrm{C} = 313.15\,\mathrm{K}
     \;\Longrightarrow\;
     \frac{313.15}{293.15} \approx 1.068!
     $$
   Năng lượng nhiệt thực tế chỉ tăng có **6.8%**, hoàn toàn không phải 100%! Nguyên nhân: điểm $$0^\circ\mathrm{C}$$ chỉ là nhiệt độ đóng băng của nước (quy ước), không phải trạng thái "không có nhiệt". Trong khi đó, *hiệu số* $$40 - 20 = 20^\circ\mathrm{C} = 313.15 - 293.15 = 20\,\mathrm{K}$$ lại hoàn toàn bất biến!

2. **Đối chiếu với thu nhập (Thang Tỷ lệ):** Thu nhập 20 triệu và 40 triệu VND. Đổi sang USD (chia cho $$25{,}000$$) ta được $$800$$ USD và $$1{,}600$$ USD — tỷ số $$\frac{1600}{800} = \frac{40}{20} = 2.0$$ **giữ nguyên bất biến** vì 0 đồng thực sự có nghĩa là không có tiền!

**Quy tắc vàng trước khi code:** Khi đưa biến định danh (như mã vùng `1, 2, 3`) vào mô hình học máy, hãy dùng **One-Hot Encoding** thay vì để nguyên số nguyên — nếu không, mô hình tuyến tính sẽ hiểu nhầm vùng `3` "lớn gấp ba lần" vùng `1`!

## Video tham khảo

- [Levels of Measurement - Nominal, Ordinal, Interval and Ratio](https://www.youtube.com/watch?v=8N5IMF-T3pc) — numiqo, tiếng Anh. **Xem trước.** Trình bày bốn thang đo định danh, thứ bậc, khoảng và tỷ lệ, khớp với bảng phân loại trước phần chuyển kiểu cột trong Pandas.
- [Scales of Measurement - Nominal, Ordinal, Interval, & Ratio Scale Data](https://www.youtube.com/watch?v=LuBD49SFpWs) — The Organic Chemistry Tutor, tiếng Anh. Bổ sung ví dụ cho bốn thang đo và chỉ rõ các trường hợp dữ liệu dạng số nhưng thực chất chỉ thuộc thang đo thứ bậc chứ không phải tỷ lệ.

## 8. Bài tập

### Bài tập lý thuyết
1. Phân loại các biến sau theo **loại dữ liệu** (định tính/định lượng) và **thang đo**:
   a. Số điện thoại của khách hàng.
   b. Khoảng cách từ nhà đến công ty (km).
   c. Chỉ số IQ (IQ score).
   d. Hạng ghế máy bay (Economy, Business, First Class).

### Bài tập Python
1. Tạo một DataFrame chứa thông tin về 10 bộ phim gồm: Tên phim, Thể loại, Năm sản xuất, Doanh thu, Điểm đánh giá (Rotten Tomatoes).
2. Xác định kiểu dữ liệu của từng cột.
3. Chuyển đổi cột điểm đánh giá (nếu đang ở dạng chuỗi, ví dụ `"85%"`) sang dạng số.
4. Chuyển đổi cột "Thể loại" sang kiểu biến phân loại (`category`).

## 9. Tài liệu tham khảo

- [1] Chapter 1, Sheldon Ross, *A First Course in Probability*, Pearson, 2012.
- [2] Chapter 1, David Diez et al., *OpenIntro Statistics*.
- [3] Pandas Documentation: [Categorical Data](https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html).

Bài tùy chọn tiếp theo, [02-04-00 Ứng dụng và phát triển gần đây]({% multilang_post_url contents/chapter02/04_Modern_Applications %}), kết nối phân loại dữ liệu, đặc trưng số và trực quan hóa với thực hành học máy lấy dữ liệu làm trung tâm (2021–2025).
