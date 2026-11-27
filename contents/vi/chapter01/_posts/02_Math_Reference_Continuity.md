---
layout: post
title: 00-01-01 Tính liên tục và Tính liên tục đều
chapter: "01"
order: 3
owner: GitHub Copilot
lang: vi
categories:
- chapter00
lesson_type: required
---

Hãy hình dung một cảm biến nhiệt độ công nghiệp đo sai lệch chỉ $$0.001^\circ\mathrm{C}$$. Liệu sai số nhỏ xíu đó sẽ chỉ làm kết quả dự đoán lệch đi một chút không đáng kể, hay nó có thể khiến toàn bộ hệ thống điều khiển nhảy vọt sang một trạng thái hoàn toàn khác? Câu trả lời không nằm ở phần cứng, mà nằm ở một khái niệm giải tích quen thuộc nhưng thường bị dạy quá khô khan: **tính liên tục** và các cấp độ chặt chẽ hơn của nó (**liên tục đều**, **liên tục Lipschitz**).

Trong xác suất, thống kê và học máy, tính liên tục là "hợp đồng ổn định" của mọi mô hình: nó đảm bảo rằng khi dữ liệu đầu vào nhiễu đi một chút, xác suất hay giá trị hàm mất mát không bị vỡ vụn, và các thuật toán tối ưu như Gradient Descent có thể hội tụ an toàn.

---

## Kiến thức cần nhớ: Tính liên tục, Liên tục đều và Lipschitz là gì?

![Tính Liên tục và Giới hạn trong Giải tích Thống kê]({{ site.baseurl }}/img/course/ch01_02_continuity.svg)

Ba khái niệm trong bài này tạo thành một **chiếc thang độ mượt tăng dần** — mỗi bậc chặt hơn cho ta những bảo đảm mạnh hơn trong thống kê và học máy:

1. **Liên tục tại từng điểm (Pointwise Continuity):** Đầu vào thay đổi đủ nhỏ quanh điểm $$x_0$$ thì đầu ra thay đổi nhỏ tùy ý — nhưng mức "đủ nhỏ" $$\delta$$ có thể phải thu nhỏ thê thảm tùy theo vị trí $$x_0$$.
2. **Liên tục đều (Uniform Continuity):** Một thước đo $$\delta(\varepsilon)$$ duy nhất hoạt động *đồng thời trên toàn bộ miền xác định*, không phụ thuộc vào vị trí $$x$$ — cực kỳ cần thiết để đổi thứ tự giới hạn và tích phân trong lý thuyết xác suất.
3. **Liên tục Lipschitz ($$\lvert f(x) - f(y)\rvert \le L\lvert x - y\rvert$$):** Độ dốc của hàm bị chặn toàn cục bởi hằng số $$L$$. Đây là điều kiện vàng trong **bất đẳng thức tập trung độ đo**, **đối ngẫu Wasserstein-1**, và **chứng minh hội tụ của Gradient Descent**.

---

## Tính liên tục và Tính liên tục đều

**Tính liên tục** và **tính liên tục đều** đều mô tả mức độ ổn định của giá trị hàm số khi biến đầu vào thay đổi nhỏ. Tuy có liên hệ chặt chẽ, hai khái niệm này khác nhau về phạm vi áp dụng, trong đó tính liên tục đều là điều kiện chặt hơn so với tính liên tục thông thường.

### Định nghĩa tính liên tục

Một hàm số $$f: A \to \mathbb{R}$$ được gọi là **liên tục tại một điểm** $$c \in A$$ nếu với mọi số thực dương $$\varepsilon > 0$$, tồn tại một số thực dương $$\delta > 0$$ sao cho với mọi $$x \in A$$, nếu

$$\lvert x - c \rvert < \delta$$

thì

$$\lvert f(x) - f(c) \rvert < \varepsilon$$

Nói cách khác, với bất kỳ sai số cho phép $$\varepsilon > 0$$ nào ở đầu ra $$f(x)$$, ta luôn tìm được một lân cận đủ nhỏ quanh $$c$$ (có độ rộng $$2\delta$$) sao cho mọi giá trị $$x$$ thuộc lân cận đó đều cho đầu ra $$f(x)$$ sai khác không quá $$\varepsilon$$ so với $$f(c)$$. Điểm cần lưu ý là giá trị $$\delta$$ nhìn chung phụ thuộc vào cả $$\varepsilon$$ lẫn điểm $$c$$ đang xét.

Một hàm số được gọi là **liên tục trên tập hợp** $$A$$ nếu nó liên tục tại mọi điểm $$c \in A$$.

**Ví dụ về các hàm liên tục** gồm mọi đa thức (chẳng hạn $$f(x) = x^2 + 3x - 1$$), các hàm lượng giác như $$\sin(x)$$ và $$\cos(x)$$, và hàm mũ $$e^x$$ trên miền xác định tương ứng của chúng.

### Định nghĩa tính liên tục đều

**Tính liên tục đều** đặt ra một yêu cầu chặt hơn. Một hàm số $$f: A \to \mathbb{R}$$ được gọi là **liên tục đều trên tập hợp** $$A$$ nếu với mọi số thực dương $$\varepsilon > 0$$, tồn tại một số thực dương $$\delta > 0$$ sao cho với mọi $$x, y \in A$$, nếu

$$\lvert x - y \rvert < \delta$$

thì

$$\lvert f(x) - f(y) \rvert < \varepsilon$$

**Điểm khác biệt cốt lõi** so với tính liên tục theo điểm nằm ở thứ tự lượng từ: với tính liên tục **đều**, $$\delta$$ chỉ phụ thuộc vào $$\varepsilon$$ và dùng chung cho mọi cặp điểm $$x, y$$ trên toàn miền xác định $$A$$.

### Định nghĩa tính liên tục Lipschitz

**Tính liên tục Lipschitz** đưa ra một điều kiện cụ thể và mang tính định lượng hơn về tốc độ biến thiên của hàm số. Một hàm số $$f: A \to \mathbb{R}$$ được gọi là **liên tục Lipschitz** (hay **$$L$$-Lipschitz**) trên tập hợp $$A$$ nếu tồn tại hằng số $$L \geq 0$$ sao cho với mọi $$x, y \in A$$:

$$\lvert f(x) - f(y) \rvert \leq L \lvert x - y \rvert$$

Hằng số $$L$$ nhỏ nhất thỏa mãn bất đẳng thức trên được gọi là **hằng số Lipschitz** (hoặc **mô-đun Lipschitz**) của $$f$$.

**Các tính chất chính của hàm liên tục Lipschitz:**

1. **Tốc độ thay đổi bị chặn**: Điều kiện Lipschitz đảm bảo hàm số không thể biến thiên nhanh hơn một tốc độ tuyến tính xác định bởi $$L$$.
2. **Suy ra tính liên tục đều**: Mọi hàm liên tục Lipschitz đều liên tục đều (với $$L > 0$$, chỉ cần chọn $$\delta = \varepsilon/L$$).
3. **Khả vi hầu khắp nơi**: Hàm liên tục Lipschitz khả vi hầu khắp nơi, và tại mọi điểm có đạo hàm thì $$\lvert f'(x) \rvert \leq L$$.

**Ví dụ:**
- $$f(x) = \lvert x \rvert$$ là hàm 1-Lipschitz trên $$\mathbb{R}$$.
- $$f(x) = \sin(x)$$ là hàm 1-Lipschitz trên $$\mathbb{R}$$ (do $$\lvert \cos(x) \rvert \leq 1$$).
- $$f(x) = x^2$$ không liên tục Lipschitz trên $$\mathbb{R}$$, nhưng liên tục Lipschitz trên mọi khoảng bị chặn.

### Sự khác biệt chính và quan hệ bao hàm

Ba lớp hàm tương ứng với ba điều kiện trên tạo thành một quan hệ bao hàm theo mức độ chặt giảm dần:

**Tính liên tục Lipschitz ⊆ Tính liên tục đều ⊆ Tính liên tục**

1. **Tính chất cục bộ so với toàn cục**:
   - **Tính liên tục**: Tính chất cục bộ, kiểm tra riêng tại từng điểm.
   - **Tính liên tục đều**: Tính chất toàn cục trên cả miền xác định.
   - **Tính liên tục Lipschitz**: Tính chất toàn cục kèm chặn trên định lượng cho mức độ biến thiên.

2. **Sự phụ thuộc của $$\delta$$**:
   - **Tính liên tục**: $$\delta$$ có thể phụ thuộc vào cả $$\varepsilon$$ và điểm $$c$$ đang xét.
   - **Tính liên tục đều**: $$\delta$$ chỉ phụ thuộc vào $$\varepsilon$$, áp dụng đồng thời cho mọi cặp điểm trên miền xác định.
   - **Tính liên tục Lipschitz**: Có thể chọn tường minh $$\delta = \varepsilon/L$$ (khi $$L > 0$$).

3. **Kiểm soát tốc độ thay đổi**:
   - **Tính liên tục**: Không ràng buộc tốc độ thay đổi của hàm số.
   - **Tính liên tục đều**: Đảm bảo biến thiên của hàm nhỏ đều trên mọi đoạn đủ ngắn.
   - **Tính liên tục Lipschitz**: Đặt ra chặn trên tuyến tính rõ ràng cho tốc độ thay đổi.

4. **Quan hệ kéo theo**:
   - Mọi hàm liên tục Lipschitz đều liên tục đều.
   - Mọi hàm liên tục đều đều liên tục.
   - Các chiều ngược lại nói chung không đúng.

### Ví dụ chi tiết và so sánh

**Ví dụ 1: $$f(x) = x^2$$**
- **Trên $$\mathbb{R}$$**: Liên tục nhưng không liên tục đều (tốc độ thay đổi $$\lvert f'(x) \rvert = 2\lvert x \rvert$$ không bị chặn).
- **Trên $$[0,1]$$**: Liên tục, liên tục đều và liên tục Lipschitz với $$L = 2$$.

**Ví dụ 2: $$f(x) = \sin(x)$$**
- **Trên $$\mathbb{R}$$**: Liên tục, liên tục đều và là hàm 1-Lipschitz (vì $$\lvert \cos(x) \rvert \leq 1$$).

**Ví dụ 3: $$f(x) = \lvert x \rvert$$**
- **Trên $$\mathbb{R}$$**: Liên tục, liên tục đều và là hàm 1-Lipschitz.
- **Lưu ý**: Hàm không khả vi tại $$x = 0$$ nhưng vẫn liên tục Lipschitz.

**Ví dụ 4: $$f(x) = \sqrt{x}$$**
- **Trên $$[0,1]$$**: Liên tục và liên tục đều, nhưng không liên tục Lipschitz (đạo hàm không bị chặn gần $$x = 0$$).
- **Trên $$[a,1]$$ với $$a > 0$$**: Liên tục Lipschitz với $$L = 1/(2\sqrt{a})$$.

### Giải thích: tự tay kiểm chứng $$f(x) = x^2$$ liên tục nhưng KHÔNG liên tục đều trên $$\mathbb{R}$$

Nhiều sinh viên học thuộc câu *"$$f(x) = x^2$$ không liên tục đều trên $$\mathbb{R}$$"* mà chưa từng thử thay số thật để thấy nó hỏng ở đâu. Hãy tự tay kiểm tra ngay trên giấy:

1. **Cố định ngưỡng sai số đầu ra $$\varepsilon = 1$$.** Nếu $$f(x) = x^2$$ liên tục đều trên toàn bộ $$\mathbb{R}$$, thì phải tồn tại một khoảng cách $$\delta > 0$$ dùng chung cho mọi nơi sao cho hễ $$\lvert y - x\rvert < \delta$$ thì $$\lvert y^2 - x^2\rvert < 1$$.
2. **Chọn hai điểm rất gần nhau ở xa gốc tọa độ:** Đặt $$x_n = n$$ và $$y_n = n + \frac{1}{n}$$.
   - Khoảng cách đầu vào: $$\lvert y_n - x_n \rvert = \frac{1}{n}$$. Chọn $$n = 1000$$, hai điểm chỉ cách nhau $$0.001$$ (nhỏ hơn bất kỳ $$\delta > 0$$ nào bạn chọn trước!).
   - Khoảng cách đầu ra:
     $$
     \lvert f(y_n) - f(x_n)\rvert = \Bigl(n + \frac{1}{n}\Bigr)^2 - n^2 = 2 + \frac{1}{n^2} > 2 > \varepsilon = 1!
     $$
   Dù ta ép hai điểm đầu vào gần sát nhau cỡ $$0.000001$$, độ chênh đầu ra vẫn vượt quá $$2$$ vì độ dốc $$f'(x) = 2x$$ tăng ra vô cực!
3. **Đối chiếu với hàm 1-Lipschitz $$g(x) = \sin(x)$$:** Vì $$\lvert g'(x)\rvert = \lvert\cos(x)\rvert \le 1$$ tại mọi $$x \in \mathbb{R}$$, theo định lý giá trị trung bình ta luôn có $$\lvert \sin(y) - \sin(x)\rvert \le 1\cdot\lvert y - x\rvert$$ ở **mọi nơi trên $$\mathbb{R}$$** — chỉ cần chọn $$\delta = \varepsilon$$ là xong!

**Bài học thống kê:** Khi hàm mất mát của mô hình học máy có hằng số Lipschitz $$L$$ hữu hạn (như mất mát Huber hay logistic, thay vì bình phương không bị chặn), mô hình ít bị một vài điểm ngoại lai (outlier) ở xa gốc kéo lệch nghiêm trọng.

## Video tham khảo

- [Limits, L'Hôpital's rule, and epsilon delta definitions | Chapter 7, Essence of calculus](https://www.youtube.com/watch?v=kfF40MiS7zA) — 3Blue1Brown, tiếng Anh. **Xem trước.** Trình bày trực quan định nghĩa $$\varepsilon\text{–}\delta$$ của giới hạn và tính liên tục trước khi chuyển sang điều kiện liên tục đều.
- [Lecture 17: Uniform Continuity and the Definition of the Derivative](https://www.youtube.com/watch?v=f_sNWn7zujU) — MIT OpenCourseWare, tiếng Anh. Bài giảng giải tích của MIT đặt tính liên tục đều cạnh định nghĩa đạo hàm, làm rõ sự khác biệt giữa $$\delta$$ phụ thuộc điểm và $$\delta$$ dùng chung cho cả miền.
- [Epsilon-Delta Continuity | This Is What Continuity Actually Means | Visually Explained](https://www.youtube.com/watch?v=2SYsl3SomQc) — Demathify, tiếng Anh. Minh họa trực quan lý do $$\delta$$ có thể phụ thuộc vào điểm $$c$$ — ranh giới phân biệt tính liên tục thường với tính liên tục đều.
