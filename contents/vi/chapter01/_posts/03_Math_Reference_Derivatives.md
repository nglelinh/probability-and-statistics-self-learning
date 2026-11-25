---
layout: post
title: 00-01-02 Đạo hàm và Giải tích đa biến
chapter: "01"
order: 4
owner: GitHub Copilot
lang: vi
categories:
- chapter00
lesson_type: required
---

Hãy tưởng tượng bạn đang đứng giữa một dãy núi dày đặc sương mù trong không gian nhiều chiều và cần tìm đường xuống đáy thung lũng thấp nhất. Bạn không nhìn thấy bản đồ toàn cảnh, nhưng chỉ cần cảm nhận độ dốc ngay dưới chân mình, bạn biết ngay hướng nào dốc xuống nhanh nhất để bước đi.

Cảm giác "độ dốc cục bộ" đó chính là **đạo hàm** và phiên bản nhiều chiều của nó — **gradient**. Trong thống kê toán học và học máy, đạo hàm không phải để khảo sát hàm số cho vui: nó là động cơ đứng sau **Ước lượng hợp lý cực đại (MLE)**, **hồi quy bình phương tối thiểu**, và **thuật toán lan truyền ngược (backpropagation)** huấn luyện mọi mạng nơ-ron hiện đại.

---

## Kiến thức cần nhớ: Đạo hàm đa biến, Gradient, Jacobian và Hessian là gì?

Khi chuyển từ hàm một biến $$f(x)$$ sang hàm mất mát nhiều triệu tham số $$L(\theta_1, \dots, \theta_d)$$, giải tích đa biến cung cấp đúng **bốn công cụ cốt lõi**:

1. **Đạo hàm riêng & Vectơ Gradient ($$\nabla f$$):** Gom toàn bộ tốc độ thay đổi theo từng trục tọa độ thành một vectơ. Điểm kỳ diệu là $$\nabla f(x)$$ luôn chỉ đúng hướng làm $$f$$ **tăng nhanh nhất**, nên đi ngược hướng $$-\nabla f(x)$$ là cách xuống dốc nhanh nhất cục bộ.
2. **Ma trận Jacobian ($$J_f$$):** Khi đầu ra cũng là một vectơ (ví dụ một lớp mạng nơ-ron biến $$\mathbb{R}^n \to \mathbb{R}^m$$), tập hợp mọi đạo hàm riêng tạo thành ma trận $$m\times n$$ — cho phép áp dụng **quy tắc dây chuyền** bằng phép nhân ma trận.
3. **Ma trận Hessian ($$H_f$$):** Ma trận đối xứng gồm các đạo hàm riêng bậc hai, đo **độ cong** của mặt hàm số. Nếu Hessian xác định dương khắp nơi, mặt mất mát có dạng cái bát lồi — mọi điểm dừng $$\nabla f = 0$$ đều là cực tiểu toàn cục!
4. **Quy tắc dây chuyền (Chain Rule):** Chìa khóa phân rã đạo hàm của hàm hợp nhiều tầng thành tích các đạo hàm đơn giản từng bước.

---

## Đạo hàm và tốc độ thay đổi

Đạo hàm của một hàm một biến đo tốc độ thay đổi tức thời của hàm số, qua đó cho biết hành vi cục bộ của hàm quanh một điểm.

### Các khái niệm đạo hàm cơ bản

**Độ dốc giữa hai điểm:**

$$\text{Độ dốc} = \frac{y_2 - y_1}{x_2 - x_1}$$

**Đạo hàm (tốc độ thay đổi tức thời):**

$$f'(x_0) = \lim_{x_1 \to x_0} \frac{f(x_1) - f(x_0)}{x_1 - x_0} = \lim_{\Delta x \to 0} \frac{f(x_0 + \Delta x) - f(x_0)}{\Delta x}$$

Đạo hàm cho biết hàm số tăng hoặc giảm nhanh đến mức nào tại từng điểm — thông tin then chốt để tìm các điểm tối ưu, nơi tốc độ thay đổi bằng 0.

### Đường mức của hàm số

Đường mức là công cụ cơ bản trong giải tích đa biến dùng để trực quan hóa hàm hai biến $$f(x, y)$$, cho phép biểu diễn một mặt cong 3D trên mặt phẳng 2D.

Một **đường mức** của hàm số $$f(x, y)$$ là tập hợp tất cả các điểm $$(x, y)$$ thuộc miền xác định của $$f$$ tại đó hàm số nhận cùng một giá trị hằng số $$c$$:

$$f(x, y) = c$$

**Ví dụ:**
- Với $$f(x, y) = x^2 + y^2$$, các đường mức là các đường tròn đồng tâm: $$x^2 + y^2 = c$$ (với $$c > 0$$).
- Với $$f(x, y) = x + y$$, các đường mức là các đường thẳng song song: $$x + y = c$$.

Đường mức giúp ta nhận diện:
1. Hình dạng bề mặt của hàm số.
2. Hướng tăng nhanh nhất và giảm nhanh nhất.
3. Vị trí các điểm tối ưu tiềm năng.

---

## Các khái niệm chính của giải tích đa biến

### Đạo hàm riêng

Với hàm nhiều biến $$f(x_1, x_2, \ldots, x_n)$$, **đạo hàm riêng** theo biến $$x_i$$ được định nghĩa là:

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}$$

Đạo hàm riêng đo tốc độ thay đổi của $$f$$ khi chỉ riêng $$x_i$$ biến thiên còn tất cả các biến khác được giữ cố định.

### Vector Gradient

**Gradient** là vector tập hợp toàn bộ các đạo hàm riêng bậc nhất của hàm số:

$$\nabla f(\mathbf{x}) = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix}$$

Vector gradient chỉ theo hướng tăng dốc nhất của hàm số và luôn vuông góc với các đường mức.

### Ma trận Hessian

**Ma trận Hessian** là ma trận chứa tất cả các đạo hàm riêng bậc hai của hàm số:

$$\nabla^2 f(\mathbf{x}) = \mathbf{H} = \begin{pmatrix} 
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{pmatrix}$$

Ma trận Hessian phản ánh độ cong của hàm số và được dùng để:
- Phân loại các điểm tới hạn (cực tiểu, cực đại hay điểm yên ngựa).
- Xây dựng các thuật toán tối ưu hóa bậc hai như phương pháp Newton.

---

## Quy tắc dây chuyền cho hàm đa biến

Quy tắc dây chuyền (chain rule) là công cụ cơ bản để tính đạo hàm của hàm hợp, vốn xuất hiện thường xuyên trong các bài toán tối ưu hóa.

### Quy tắc dây chuyền cơ bản

Với hàm số $$z = f(x, y)$$, trong đó $$x = g(t)$$ và $$y = h(t)$$:

$$ \frac{dz}{dt} = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt} $$

### Quy tắc dây chuyền tổng quát

Với $$z = f(x_1, x_2, \ldots, x_n)$$, trong đó mỗi $$x_i = x_i(t_1, t_2, \ldots, t_m)$$:

$$ \frac{\partial z}{\partial t_j} = \sum_{i=1}^{n} \frac{\partial f}{\partial x_i} \frac{\partial x_i}{\partial t_j} $$

### Ứng dụng trong tối ưu hóa

Quy tắc dây chuyền được sử dụng trong nhiều bài toán:

1. **Tính gradient**: Tính gradient của các hàm mục tiêu có cấu trúc hàm hợp.
2. **Xử lý ràng buộc**: Tính đạo hàm khi các biến chịu ràng buộc dưới dạng hàm của các biến khác.
3. **Triển khai thuật toán**: Lan truyền ngược (backpropagation) trong mạng nơ-ron và vi phân tự động (automatic differentiation).
4. **Phân tích độ nhạy**: Đánh giá mức độ ảnh hưởng của sự thay đổi tham số lên nghiệm tối ưu.

### Ví dụ: Tối ưu hóa có ràng buộc

Xét bài toán tìm cực tiểu của $$f(x, y) = x^2 + y^2$$ với điều kiện ràng buộc $$g(x, y) = x + y - 1 = 0$$.

Từ ràng buộc ta rút ra $$y = 1 - x$$, đưa bài toán về việc tìm cực tiểu của hàm một biến:
$$h(x) = f(x, 1-x) = x^2 + (1-x)^2$$

Áp dụng quy tắc dây chuyền:
$$h'(x) = \frac{\partial f}{\partial x} \cdot 1 + \frac{\partial f}{\partial y} \cdot \frac{d(1-x)}{dx} = 2x + 2(1-x)(-1) = 4x - 2$$

Giải phương trình $$h'(x) = 0$$ ta được $$x = 1/2$$, suy ra điểm tối ưu là $$(1/2, 1/2)$$.

Ví dụ này cho thấy các khái niệm giải tích đa biến phối hợp với nhau như thế nào để giải bài toán tối ưu hóa một cách có hệ thống.

### Giải thích: tự tay tính Gradient, Hessian và một bước hạ độ dốc trên hàm 2 biến

Để thấy Gradient và Hessian phối hợp với nhau rõ như ban ngày, hãy xét hàm mất mát 2 tham số cực kỳ quen thuộc:

$$
L(w_1, w_2) = (w_1 - 2)^2 + 3(w_2 + 1)^2.
$$

1. **Tính Gradient $$\nabla L$$ và tìm cực tiểu chính xác:**
   $$
   \nabla L(w_1, w_2) = \begin{pmatrix} \frac{\partial L}{\partial w_1} \\ \frac{\partial L}{\partial w_2} \end{pmatrix}
   = \begin{pmatrix} 2(w_1 - 2) \\ 6(w_2 + 1) \end{pmatrix}.
   $$
   Đặt $$\nabla L = 0$$ ta có ngay điểm dừng duy nhất $$w^* = (2, -1)^\top$$ với $$L(w^*) = 0$$.

2. **Tính ma trận Hessian để kiểm tra độ cong:**
   $$
   H_L = \begin{pmatrix} \frac{\partial^2 L}{\partial w_1^2} & \frac{\partial^2 L}{\partial w_1 \partial w_2} \\ \frac{\partial^2 L}{\partial w_2 \partial w_1} & \frac{\partial^2 L}{\partial w_2^2} \end{pmatrix}
   = \begin{pmatrix} 2 & 0 \\ 0 & 6 \end{pmatrix}.
   $$
   Hai trị riêng là $$2 > 0$$ và $$6 > 0$$ (ma trận xác định dương), khẳng định $$L$$ **lồi chặt** và $$w^* = (2, -1)^\top$$ chắc chắn là cực tiểu toàn cục. (Tỷ số $$6/2 = 3$$ gọi là *số điều kiện* — cho biết thung lũng dốc theo chiều $$w_2$$ gấp 3 lần chiều $$w_1$$.)

3. **Chạy tay 1 bước Gradient Descent** xuất phát từ $$w^{(0)} = (0, 0)^\top$$ với bước học $$\eta = 0.1$$:
   - Mất mát ban đầu: $$L(0, 0) = (-2)^2 + 3(1)^2 = 7$$.
   - Gradient tại $$(0, 0)$$: $$\nabla L(0, 0) = (-4, 6)^\top$$.
   - Bước cập nhật: $$w^{(1)} = \begin{pmatrix}0\\0\end{pmatrix} - 0.1\begin{pmatrix}-4\\6\end{pmatrix} = \begin{pmatrix}0.4\\-0.6\end{pmatrix}$$.
   - Mất mát mới: $$L(0.4, -0.6) = (-1.6)^2 + 3(0.4)^2 = 2.56 + 0.48 = 3.04 < 7$$ — **giảm hơn một nửa chỉ sau 1 bước!**

## Video tham khảo

- [The paradox of the derivative | Chapter 2, Essence of calculus](https://www.youtube.com/watch?v=9vKqVkMQHKk) — 3Blue1Brown, tiếng Anh. **Xem trước.** Giải thích đạo hàm như tốc độ thay đổi tức thời, tương ứng với định nghĩa giới hạn ở đầu bài.
- [Visualizing the chain rule and product rule | Chapter 4, Essence of calculus](https://www.youtube.com/watch?v=YG15m2VwSjA) — 3Blue1Brown, tiếng Anh. Minh họa trực quan quy tắc dây chuyền cho phần đạo hàm của hàm hợp nhiều biến.
- [Partial Derivatives and the Gradient of a Function](https://www.youtube.com/watch?v=AXH9Xm6Rbfc) — Professor Dave Explains, tiếng Anh. Giới thiệu đạo hàm riêng và vector gradient — các công cụ gắn với đường mức và điểm tới hạn trong bài.
