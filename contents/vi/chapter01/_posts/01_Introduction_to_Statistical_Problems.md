---
layout: post
title: 01-01-00 Nhập môn Thống kê Tính toán và Tư duy Dữ liệu
chapter: "01"
order: 1
owner: nglelinh
lang: vi
categories:
- chapter01
lesson_type: required
---

Mục tiêu chính của bài học mở đầu này không phải là giới thiệu ngay các công thức phức tạp, mà là xây dựng cách tiếp cận bài toán thống kê theo hướng tính toán. Thay vì chỉ giải bài tập bằng biến đổi giải tích trên giấy, chúng ta sẽ làm quen với cách mô phỏng trên máy tính. Sau bài học này, bạn sẽ phân biệt được thống kê cổ điển (dựa trên giải tích) với thống kê tính toán (dựa trên thuật toán), đồng thời thấy rõ vai trò của Luật số lớn thông qua thực nghiệm thay vì chứng minh hình thức.

---

## Giới thiệu: Vì sao cần thống kê tính toán?

Hãy xét một bài toán cụ thể: ước lượng xác suất để tổng số chấm của 10 lần gieo xúc xắc vượt quá 40. Nếu dùng thống kê cổ điển, ta cần áp dụng tổ hợp, hoán vị hoặc xấp xỉ bằng phân phối chuẩn (theo Định lý Giới hạn Trung tâm — CLT). Với bài toán nhỏ, cách làm này khả thi; nhưng với các bài toán phức tạp hơn — chẳng hạn tính xác suất rủi ro của một danh mục đầu tư gồm 50 mã cổ phiếu có tương quan với nhau — việc tìm ra công thức giải tích dạng đóng (closed-form solution) gần như không thể thực hiện được.

Trong những tình huống như vậy, **Thống kê Tính toán (Computational Statistics)** cung cấp một hướng tiếp cận khác. Thay vì tìm một phương trình giải tích mô tả chính xác hệ thống, ta dùng máy tính để mô phỏng quá trình ngẫu nhiên đó: tạo ra các kịch bản giả lập, lặp lại thí nghiệm hàng triệu lần trong vài giây và đếm tần suất kết quả. Cách làm này chuyển các bài toán xác suất trừu tượng thành bài toán lập trình cụ thể, giúp xử lý những bài toán mà phương pháp giải tích cổ điển khó giải trực tiếp.

## Thống kê giải tích và thống kê tính toán

Sự khác biệt cơ bản giữa hai cách tiếp cận nằm ở công cụ giải quyết vấn đề. Thống kê giải tích dựa vào các định lý toán học và các giả định đơn giản hóa (chẳng hạn giả định dữ liệu tuân theo phân phối chuẩn) để thu được kết quả lý thuyết. Ví dụ, để tính khoảng tin cậy 95% cho trung bình tổng thể khi biết độ lệch chuẩn $$\sigma$$, ta dùng công thức:

$$\bar{x} \pm 1.96 \frac{\sigma}{\sqrt{n}}$$

Ngược lại, thống kê tính toán sử dụng năng lực xử lý của máy tính để thực hiện các phép thử ngẫu nhiên lặp lại. Thay vì giả định phân phối chuẩn, ta có thể dùng phương pháp Bootstrap (lấy mẫu lại) để xây dựng phân phối thực nghiệm trực tiếp từ dữ liệu. Cách tiếp cận này linh hoạt, ít phụ thuộc vào các giả định phân phối chặt chẽ và trực quan hơn khi làm việc với dữ liệu thực tế.

## Mô phỏng và Luật số lớn

Trọng tâm của thống kê tính toán là **Mô phỏng Monte Carlo**. Nguyên lý của phương pháp này rất rõ ràng: để ước lượng xác suất của một biến cố, ta lặp lại phép thử ngẫu nhiên đó nhiều lần và tính tỷ lệ số lần biến cố xảy ra.

Cơ sở toán học đảm bảo tính hợp lý cho cách làm này là **Luật Số Lớn (Law of Large Numbers)**. Định lý phát biểu rằng khi số lần thử $$n$$ tiến tới vô cùng, trung bình mẫu quan sát được hội tụ theo xác suất về giá trị kỳ vọng thực $$\mu$$ của tổng thể:

$$\lim_{n \to \infty} P\left( \left| \bar{X}_n - \mu \right| < \varepsilon \right) = 1$$

Trong môi trường lập trình, $$n$$ tương ứng với số vòng lặp mô phỏng mà máy tính thực hiện.

## Ví dụ: Bài toán Sinh nhật (The Birthday Problem)

Một ví dụ kinh điển minh họa cho mô phỏng Monte Carlo là Bài toán Sinh nhật: *Trong một phòng có k người, xác suất để có ít nhất hai người trùng ngày sinh là bao nhiêu?*

Theo hướng lý thuyết, ta cần tính xác suất của biến cố phần bù bằng phép đếm tổ hợp. Với tư duy tính toán, ta có thể giải trực tiếp bằng Python như sau:

```python
import numpy as np

def simulate_birthday_problem(n_people, n_simulations=10000):
    """
    Mô phỏng bài toán sinh nhật để ước lượng xác suất có ít nhất
    2 người trùng ngày sinh trong nhóm n_people người.
    
    Args:
        n_people (int): Số người trong phòng.
        n_simulations (int): Số lần thực hiện mô phỏng.
        
    Returns:
        float: Xác suất ước lượng (tỷ lệ các lần có trùng lặp).
    """
    count_duplicate = 0
    
    for _ in range(n_simulations):
        # Sinh ngẫu nhiên ngày sinh cho n người (từ 1 đến 365)
        birthdays = np.random.randint(1, 366, size=n_people)
        
        # Kiểm tra xem có ngày sinh nào bị trùng không
        if len(np.unique(birthdays)) < n_people:
            count_duplicate += 1
            
    probability = count_duplicate / n_simulations
    return probability

# Thử nghiệm với nhóm 23 người
n_people = 23
prob = simulate_birthday_problem(n_people)
print(f"Xác suất trùng ngày sinh với {n_people} người: {prob:.4f}")
```

Đoạn mã trên không cần dùng công thức giai thừa hay lũy thừa, mà trực tiếp mô phỏng quá trình chọn ngày sinh ngẫu nhiên và đếm số lần có trùng lặp. Kết quả thu được xấp xỉ 0,507 (khoảng 50,7%), thường cao hơn đáng kể so với ước đoán trực giác ban đầu của nhiều người.

## Video tham khảo

- [Law of Large Numbers - Explained and Visualized](https://www.youtube.com/watch?v=MntX3zWNWec) — Jeremy Blitz-Jones, tiếng Anh. **Xem trước.** Trực quan hóa cách trung bình mẫu hội tụ về kỳ vọng khi số lần thử tăng, đúng với Luật số lớn làm cơ sở cho mô phỏng Monte Carlo trong bài.
- [6. Monte Carlo Simulation](https://www.youtube.com/watch?v=OgO1gpXSUzU) — MIT OpenCourseWare, tiếng Anh. John Guttag giải thích lý do Ulam sử dụng mô phỏng thay vì đếm tổ hợp và cách ước lượng đại lượng bằng mẫu ngẫu nhiên, cùng tư duy với bài toán xúc xắc và ước lượng $$\pi$$.
- [Check your intuition: The birthday problem - David Knuffke](https://www.youtube.com/watch?v=KtT_cgMzHx8) — TED-Ed, tiếng Anh. Phân tích bài toán sinh nhật trong thân bài: vì sao trực giác thường đánh giá thấp xác suất trùng lặp và vì sao với 23 người xác suất này đã vượt quá một nửa.

## Bài tập thực hành

**Bài 1: Mô phỏng đồng xu**
Viết hàm mô phỏng việc tung một đồng xu không đồng chất (xác suất ra mặt ngửa là $$p=0.6$$) 100 lần. Lặp lại thí nghiệm này 1.000 lần và vẽ biểu đồ histogram biểu diễn phân phối số lần xuất hiện mặt ngửa.

**Bài 2: Ước lượng số Pi**
Sử dụng phương pháp Monte Carlo để ước lượng số Pi ($$\pi$$). Gợi ý: sinh ngẫu nhiên các điểm trong hình vuông đơn vị và đếm tỷ lệ các điểm rơi vào hình tròn nội tiếp.

**Bài 3: Bài toán Monty Hall**
Lập trình mô phỏng bài toán Monty Hall để kiểm chứng rằng việc đổi lựa chọn làm tăng gấp đôi xác suất chiến thắng so với giữ nguyên lựa chọn ban đầu.

Bài tùy chọn tiếp theo, [01-11-00 Ứng dụng và phát triển gần đây]({% multilang_post_url contents/chapter01/11_Modern_Applications %}), kết nối mô phỏng Monte Carlo, cross-validation và lựa chọn mô hình với conformal prediction và đánh giá mô hình ngôn ngữ lớn (LLM) giai đoạn 2023–2024.
