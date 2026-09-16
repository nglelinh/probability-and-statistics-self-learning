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

Trong bài học mở đầu này, mục tiêu quan trọng nhất không phải là cung cấp ngay lập tức các công thức phức tạp, mà là định hình lại tư duy của bạn về cách tiếp cận các vấn đề thống kê. Chúng ta sẽ cùng nhau tháo gỡ tư duy "giải bài tập trên giấy" để chuyển sang tư duy "mô phỏng trên máy tính". Sau khi hoàn thành bài học này, bạn sẽ hiểu được sự khác biệt cốt lõi giữa thống kê cổ điển (dựa trên giải tích) và thống kê tính toán (dựa trên thuật toán), đồng thời nắm bắt được sức mạnh của Luật số lớn thông qua thực nghiệm thay vì chứng minh toán học.

---

## Giới Thiệu: Tại Sao Phải Là Thống Kê Tính Toán?

Hãy tưởng tượng bạn đang đứng trước một bài toán thực tế: Bạn muốn ước lượng xác suất để tổng số chấm của 10 lần gieo xúc xắc vượt quá 40. Nếu sử dụng phương pháp thống kê cổ điển, bạn sẽ phải vận dụng tổ hợp, hoán vị, hoặc sử dụng xấp xỉ phân phối chuẩn (CLT). Với những người có nền tảng toán học vững chắc, điều này có thể thú vị, nhưng với các bài toán phức tạp hơn—ví dụ như tính xác suất rủi ro của một danh mục đầu tư gồm 50 mã cổ phiếu tương quan với nhau—việc tìm ra một công thức giải tích chính xác (closed-form solution) là điều gần như bất khả thi.

Đây chính là lúc **Thống kê Tính toán (Computational Statistics)** tỏa sáng. Thay vì cố gắng tìm ra một phương trình toán học hoàn hảo mô tả thế giới, chúng ta sử dụng sức mạnh tính toán của máy tính để "bắt chước" thế giới đó. Chúng ta tạo ra các thế giới giả lập, thực hiện thí nghiệm hàng triệu lần trong vài giây, và đếm kết quả. Cách tiếp cận này biến các bài toán thống kê trừu tượng thành các bài toán kỹ thuật phần mềm cụ thể, cho phép chúng ta giải quyết các vấn đề mà lý thuyết cổ điển phải bó tay.

## Thống Kê Giải Tích vs. Thống Kê Tính Toán

Sự khác biệt cơ bản nằm ở công cụ giải quyết vấn đề. Thống kê giải tích dựa vào các định lý toán học và các giả định đơn giản hóa (như giả định dữ liệu tuân theo phân phối Chuẩn) để đưa ra các kết quả chính xác về mặt lý thuyết. Ví dụ, để tính khoảng tin cậy cho trung bình, ta dùng công thức:

$$\bar{x} \pm 1.96 \frac{\sigma}{\sqrt{n}}$$

Ngược lại, thống kê tính toán chấp nhận sử dụng sức mạnh xử lý của CPU để thực hiện các phép thử ngẫu nhiên lặp đi lặp lại. Thay vì giả định phân phối chuẩn, chúng ta có thể dùng phương pháp Bootstrap (lấy mẫu lại) để xây dựng phân phối thực tế từ dữ liệu. Phương pháp tính toán thường linh hoạt hơn, ít phụ thuộc vào giả định hơn, và đặc biệt trực quan hơn đối với những người làm khoa học dữ liệu.

## Mô Phỏng và Luật Số Lớn

Trái tim của thống kê tính toán là **Mô phỏng Monte Carlo**. Ý tưởng cốt lõi cực kỳ đơn giản: Nếu bạn muốn biết xác suất của một biến cố, hãy thực hiện thí nghiệm đó thật nhiều lần và tính tần suất xuất hiện của nó.

Cơ sở toán học đảm bảo cho phương pháp này là **Luật Số Lớn (Law of Large Numbers)**. Định lý này phát biểu rằng khi số lượng thử nghiệm $$n$$ tiến tới vô cùng, trung bình mẫu quan sát được sẽ hội tụ về giá trị kỳ vọng thực của tổng thể:

$$\lim_{n \to \infty} P\left( \left| \bar{X}_n - \mu \right| < \varepsilon \right) = 1$$

Trong môi trường tính toán, $$n$$ chính là số vòng lặp mà máy tính thực hiện.

## Ví Dụ: Bài Toán Sinh Nhật (The Birthday Problem)

Một ví dụ kinh điển để minh họa sức mạnh của mô phỏng là Bài toán Sinh nhật. Câu hỏi đặt ra là: *Trong một phòng có k người, xác suất để có ít nhất hai người có cùng ngày sinh nhật là bao nhiêu?*

Về mặt lý thuyết, việc tính toán xác suất này đòi hỏi tư duy về xác suất phần bù và tổ hợp. Hãy xem cách chúng ta giải quyết nó bằng tư duy tính toán với Python:

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

Đoạn code trên không hề sử dụng công thức giai thừa hay lũy thừa phức tạp nào. Nó chỉ đơn giản là mô phỏng lại thực tế một cách chân thực. Kết quả sẽ xấp xỉ 0.507 (hoặc 50.7%), điều này thường gây ngạc nhiên vì trực giác của chúng ta thường đánh giá thấp xác suất trùng lặp.

## Bài Tập Thực Hành

**Bài 1: Mô phỏng Đồng Xu**
Viết hàm mô phỏng việc tung một đồng xu không đồng chất (xác suất ngửa là $$p=0.6$$) 100 lần. Lặp lại thí nghiệm này 1000 lần và vẽ biểu đồ histogram phân phối số lần mật ngửa thu được.

**Bài 2: Ước lượng Pi**
Sử dụng phương pháp Monte Carlo để ước lượng số Pi ($$\pi$$). Gợi ý: Hãy ném ngẫu nhiên các điểm vào một hình vuông đơn vị và đếm xem có bao nhiêu điểm rơi vào hình tròn nội tiếp.

**Bài 3: Monty Hall**
Lập trình mô phỏng bài toán Monty Hall nổi tiếng để chứng minh rằng việc thay đổi lựa chọn sẽ làm tăng gấp đôi cơ hội chiến thắng.

Bài tùy chọn sau, [01-11-00 Ứng dụng và phát triển gần đây]({% multilang_post_url contents/chapter01/11_Modern_Applications %}), nối Monte Carlo, cross-validation và chọn mô hình với conformal prediction và đánh giá LLM (2023–2024).
