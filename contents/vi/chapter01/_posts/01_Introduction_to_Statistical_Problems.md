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

Hãy thử đặt một câu hỏi vui giữa một lớp học chỉ có đúng 23 người: *xác suất để có ít nhất hai người trong phòng trùng ngày sinh nhật là bao nhiêu?* Trực giác của hầu hết chúng ta đều thốt lên "chắc khoảng 5–10% thôi, một năm có tận 365 ngày cơ mà!". Thế nhưng đáp án toán học lại vượt quá **50%** — ngang ngửa tung một đồng xu công bằng!

Câu đố kinh điển ấy hé lộ hai sự thật quan trọng nhất mở đầu toàn bộ khóa học này: thứ nhất, trực giác con người rất dễ bị đánh lừa trước sự ngẫu nhiên; và thứ hai, nhiều bài toán xác suất – thống kê ngoài đời thực còn phức tạp hơn bài toán sinh nhật gấp nghìn lần, đến mức không có công thức giấy bút nào giải nổi. Đó là lúc **thống kê tính toán** (computational statistics) bước lên sân khấu: kết hợp tư duy xác suất với sức mạnh mô phỏng máy tính để biến những bài toán bất khả thi thành thí nghiệm số kiểm chứng được.

---

## Kiến thức cần nhớ: Thống kê tính toán là gì và nghiên cứu những gì?

Nếu bạn đã tốt nghiệp đại học và từng học qua xác suất cơ bản, bạn hẳn nhớ cảm giác ngồi tính tay tích phân hàm mật độ hay tra bảng $$z$$, bảng $$t$$. **Thống kê tính toán** không vứt bỏ lý thuyết đó, mà mở rộng nó bằng **ba trụ cột lớn**:

1. **Thống kê giải tích (Analytical Statistics):** Dùng công thức đóng, định lý giới hạn (Luật số lớn, Định lý giới hạn trung tâm) để tìm đáp án chính xác dưới các giả định lý tưởng (như dữ liệu tuân theo phân phối chuẩn).
2. **Mô phỏng Monte Carlo (Simulation):** Thay vì giải tích phân nhiều chiều bằng tay, ta phát sinh hàng chục nghìn kịch bản ngẫu nhiên trên máy tính và lấy trung bình thực nghiệm — nhờ Luật số lớn, tần suất mô phỏng hội tụ về xác suất thật.
3. **Lấy mẫu lại & Kiểm định thực nghiệm (Resampling — Bootstrap, Cross-Validation):** Khi không biết phân phối gốc của dữ liệu, ta dùng chính tập dữ liệu đang có để tạo ra hàng nghìn mẫu giả lập, đo độ bất định và đánh giá mô hình học máy một cách trung thực.

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

### Giải thích: tự tay kiểm chứng Bài toán Sinh nhật và ước lượng $$\pi$$ bằng Monte Carlo

Để cảm nhận rõ sự khác biệt giữa **giải tích chính xác** và **mô phỏng Monte Carlo**, hãy cầm bút tự kiểm tra hai phép tính gọn gàng sau:

1. **Tính tay Bài toán Sinh nhật với $$k = 4$$ người (rồi hiểu vì sao $$k = 23$$ vượt 50%):**
   Thay vì đếm trực tiếp vô số cách trùng nhau, ta tính xác suất phần bù (*không có hai người nào trùng sinh nhật*):
   - Người 1 chọn tự do: $$\frac{365}{365}$$.
   - Người 2 khác người 1: $$\frac{364}{365}$$.
   - Người 3 khác 2 người đầu: $$\frac{363}{365}$$.
   - Người 4 khác 3 người đầu: $$\frac{362}{365}$$.
   Suy ra với $$k = 4$$ người:
   $$
   P(\text{có ít nhất 1 cặp trùng}) = 1 - \frac{365}{365}\cdot\frac{364}{365}\cdot\frac{363}{365}\cdot\frac{362}{365} \approx 1 - 0.9836 = 1.64\%.
   $$
   **Tại sao với $$k = 23$$ người xác suất lại vọt lên trên 50%?** Vì số *cặp* người tăng theo bậc hai: $$\binom{23}{2} = \frac{23 \times 22}{2} = 253$$ cặp! Mỗi cặp không trùng nhau với xác suất $$\frac{364}{365}$$, nên xấp xỉ nhanh cho $$1 - \bigl(\frac{364}{365}\bigr)^{253} \approx 1 - e^{-253/365} \approx 50.0\%$$ (giá trị chính xác là $$50.73\%$$).

2. **Ước lượng $$\pi$$ bằng mô phỏng Monte Carlo trên 8 điểm mẫu:**
   Ném ngẫu nhiên các điểm $$(x, y)$$ vào hình vuông đơn vị $$[0, 1]^2$$ (diện tích $$1$$) và đếm số điểm rơi vào góc phần tư hình tròn $$x^2 + y^2 \le 1$$ (diện tích $$\pi/4$$). Giả sử trong 8 lần ném đầu tiên, có 6 điểm thỏa $$x^2 + y^2 \le 1$$:
   $$
   \hat\pi = 4 \times \frac{6}{8} = 3.00.
   $$
   Với $$N = 8$$ điểm, sai số chuẩn còn lớn (cỡ $$O(1/\sqrt{N})$$). Khi máy tính tăng lên $$N = 100{,}000$$ điểm, sai số giảm $$\sqrt{100{,}000 / 8} \approx 112$$ lần, đẩy $$\hat\pi$$ sát $$3.1416$$ mà không hề dùng một công thức lượng giác nào!

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
