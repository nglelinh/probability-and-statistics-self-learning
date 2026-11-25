---
layout: post
title: "Bài 1: Giới Thiệu về ANOVA"
chapter: "08"
order: 1
owner: nglelinh
lang: vi
categories:
- chapter08
lesson_type: required
---

Một công ty thương mại điện tử muốn so sánh tốc độ chuyển đổi của **5 thiết kế trang sản phẩm** (A, B, C, D, E). Phản xạ tự nhiên của nhiều người là lấy từng cặp ra chạy kiểm định $$t$$ hai mẫu: A vs. B, A vs. C, ..., D vs. E — tổng cộng $$\binom{5}{2} = 10$$ kiểm định. Nhưng hãy làm một phép tính nhỏ: ngay cả khi cả 5 thiết kế có hiệu quả *y hệt nhau*, xác suất ít nhất 1 trong 10 cặp tình cờ cho $$p < 0{,}05$$ lên tới $$1 - 0{,}95^{10} \approx 40{,}1\%$$! Bạn gần như tung đồng xu để chọn giao diện!

Đầu thế kỷ 20, nhà thống kê thiên tài Ronald Fisher giải quyết triệt để bài toán so sánh nhiều nhóm bằng một ý tưởng nghe thoạt đầu rất ngược đời: muốn so sánh **trung bình** của nhiều nhóm, ta lại đi **phân tích phương sai (Analysis of Variance — ANOVA)**! Bằng cách tách tổng biến thiên dữ liệu thành hai thành phần rõ ràng — *biến thiên giữa các nhóm (tín hiệu)* và *biến thiên bên trong từng nhóm (nhiễu)* — ANOVA cho phép kiểm định cả $$k$$ nhóm chỉ trong một phép thử duy nhất mà vẫn giữ chặt sai lầm Loại I ở mức $$5\%$$.

---

## Kiến thức cần nhớ: Logic phân rã phương sai của ANOVA là gì?

ANOVA là viên gạch nền tảng của thiết kế thí nghiệm hiện đại, kết nối ba khái niệm cốt lõi bạn đã học ở các chương trước:

1. **Chặn lạm phát sai lầm Loại I (Family-Wise Error Rate):** So sánh $$k$$ nhóm theo từng cặp tạo ra $$\binom{k}{2} = \frac{k(k-1)}{2}$$ phép thử (với $$k=6$$ đã là 15 cặp, FWER vượt 53%!). ANOVA gom tất cả về một giả thuyết không duy nhất: $$H_0: \mu_1 = \mu_2 = \cdots = \mu_k$$.
2. **Định lý phân rã tổng bình phương (Orthogonal Decomposition):** Mọi độ lệch của quan sát $$x_{ij}$$ so với trung bình chung $$\bar{x}$$ tách thành hai vector trực giao: độ lệch *của trung bình nhóm* so với trung bình chung ($$\mathrm{SS}_{\text{Between}}$$) cộng độ lệch *của từng cá thể* so với trung bình nhóm mình ($$\mathrm{SS}_{\text{Within}}$$). Nhờ định lý Pythagoras trong không gian dữ liệu, $$\mathrm{SS}_{\text{Total}} = \mathrm{SS}_{\text{Between}} + \mathrm{SS}_{\text{Within}}$$ luôn đúng tuyệt đối!
3. **Tỷ số Tín hiệu / Nhiễu và Ba giả định trụ cột:** Khi các nhóm thực sự khác nhau, biến thiên *giữa* các nhóm sẽ trội hơn hẳn biến thiên *trong* từng nhóm. Để phân phối $$F$$ chính xác, dữ liệu cần thỏa mãn ba giả định: quan sát độc lập, phần dư xấp xỉ chuẩn (kiểm tra bằng Shapiro–Wilk), và phương sai các nhóm đồng nhất (kiểm tra bằng kiểm định Levene).

---

## 1. Mục Tiêu Bài Học

Sau khi hoàn thành bài học này, sinh viên có thể:
- Hiểu khái niệm **Phân tích phương sai (Analysis of Variance - ANOVA)** và lý do kỹ thuật này có tên gọi như vậy.
- Giải thích vì sao **không nên dùng nhiều kiểm định $t$ (t-tests)** để so sánh đồng thời nhiều nhóm (hiện tượng lạm phát sai lầm loại I).
- Nắm vững nguyên lý **phân rã tổng bình phương (Sum of Squares / Variance Decomposition)**: $SS_{Total} = SS_{Between} + SS_{Within}$.
- Kiểm tra các **giả định (assumptions)** của ANOVA bằng Python.

## 2. Kiến Thức Tiền Đề

- Kiểm định giả thuyết (Chương 6).
- Phương sai và độ lệch chuẩn (Chương 2).
- Phân phối chuẩn (Chương 4).

## 3. Động Lực và Giới Thiệu

Giả sử bạn là một kỹ sư nông nghiệp muốn so sánh hiệu quả của 3 loại phân bón: A, B và C. Nếu chỉ có 2 nhóm A và B, bạn có thể dùng **kiểm định $t$** (Chương 6). Tuy nhiên, với 3 nhóm A, B và C, bạn phải thực hiện 3 phép so sánh cặp (A-B, B-C và A-C); nếu có 5 loại phân bón, số cặp so sánh tăng lên thành 10.

Vì sao ta không đơn giản chạy 10 kiểm định $t$ riêng lẻ? Ở mỗi phép kiểm định, ta chấp nhận một mức rủi ro sai lầm loại I nhất định (thường là $\alpha = 0.05$). Khi lặp lại quá nhiều kiểm định trên cùng bộ dữ liệu, xác suất mắc ít nhất một sai lầm loại I sẽ tích lũy và tăng lên đáng kể (tương tự như khi tung đồng xu nhiều lần thì xác suất xuất hiện ít nhất một mặt sấp sẽ tăng dần). ANOVA ra đời để giải quyết vấn đề này: **so sánh đồng thời nhiều trung bình tổng thể chỉ bằng một kiểm định duy nhất**.

Tên gọi "Phân tích phương sai" nghe có vẻ đối lập với mục tiêu "so sánh trung bình", và phần lý thuyết dưới đây sẽ làm rõ mối liên hệ đó.

## 4. Các Khái Niệm Cốt Lõi và Lý Thuyết

### 4.1 Vì Sao "Phân Tích Phương Sai" Lại Dùng Để So Sánh Trung Bình?

Ý tưởng cốt lõi của ANOVA là đối chiếu **hai nguồn biến động**:
1.  **Biến động GIỮA các nhóm (Between-Group Variability):** Phản ánh sự khác biệt giữa trung bình các loại phân bón (tín hiệu — Signal).
2.  **Biến động TRONG các nhóm (Within-Group Variability):** Phản ánh nhiễu ngẫu nhiên, sai số đo đạc và sự khác biệt giữa các cá thể trong cùng một nhóm (nhiễu — Noise).

Nếu tín hiệu lớn hơn nhiều so với nhiễu, ta có bằng chứng để kết luận trung bình các nhóm khác nhau. Tỷ số giữa hai nguồn biến động này được đo bằng thống kê $F$:

$$F = \frac{\text{Variability Between Groups}}{\text{Variability Within Groups}}$$

### 4.2 Phân Rã Tổng Bình Phương (Sum of Squares Decomposition)

Tổng biến động của toàn bộ dữ liệu được phân tách thành hai thành phần cộng tính:

$$SS_{Total} = SS_{Between} + SS_{Within}$$

-   $SS_{Total}$: Tổng bình phương độ lệch của mọi điểm dữ liệu so với trung bình chung (Grand Mean).
-   $SS_{Between}$ (hay $SS_{Tr}$ — Treatment): Tổng bình phương độ lệch giữa trung bình từng nhóm và trung bình chung, có trọng số theo cỡ mẫu từng nhóm (đại diện cho hiệu ứng xử lý).
-   $SS_{Within}$ (hay $SS_{E}$ — Error): Tổng bình phương độ lệch của từng điểm dữ liệu so với trung bình của chính nhóm chứa nó.

### 4.3 Các Giả Định Của ANOVA

Để kết quả kiểm định ANOVA đáng tin cậy, dữ liệu cần thỏa mãn 3 giả định:
1.  **Tính chuẩn (Normality):** Dữ liệu trong mỗi nhóm tuân theo phân phối chuẩn (kiểm tra bằng kiểm định Shapiro-Wilk hoặc biểu đồ Q-Q plot).
2.  **Phương sai đồng nhất (Homogeneity of Variance / Homoscedasticity):** Phương sai giữa các nhóm tương đương nhau (kiểm tra bằng kiểm định Levene hoặc Bartlett).
3.  **Tính độc lập (Independence):** Các quan sát độc lập với nhau.

## 5. Ví Dụ Minh Họa — Trực Giác Qua Hình Ảnh

Xét 3 nhóm học sinh học theo 3 phương pháp khác nhau:
-   **Trường hợp 1:** Phân bố của các nhóm tách biệt rõ ràng. Khi đó $SS_{Between}$ lớn và $SS_{Within}$ nhỏ, dẫn đến giá trị $F$ lớn và ta bác bỏ $H_0$ (có bằng chứng cho thấy phương pháp dạy tạo ra sự khác biệt).
-   **Trường hợp 2:** Phân bố của các nhóm chồng lấn nhiều lên nhau. Khi đó $SS_{Between}$ nhỏ và $SS_{Within}$ lớn, dẫn đến giá trị $F$ nhỏ và ta không đủ bằng chứng để bác bỏ $H_0$ (khác biệt quan sát được có thể chỉ do biến động ngẫu nhiên).

## 6. Triển Khai Trên Python

Trước khi thực hiện kiểm định ANOVA, ta nên trực quan hóa dữ liệu và kiểm tra các giả định cơ bản.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# Tạo dữ liệu giả lập
# Nhóm A: Hiệu quả trung bình, Phương sai nhỏ
group_a = stats.norm.rvs(loc=50, scale=5, size=30, random_state=1)
# Nhóm B: Hiệu quả cao hơn chút, Phương sai nhỏ
group_b = stats.norm.rvs(loc=55, scale=5, size=30, random_state=1)
# Nhóm C: Hiệu quả rất cao, Phương sai lớn hơn
group_c = stats.norm.rvs(loc=65, scale=10, size=30, random_state=1)

data = pd.DataFrame({
    'Value': list(group_a) + list(group_b) + list(group_c),
    'Group': ['A']*30 + ['B']*30 + ['C']*30
})

# 1. Trực quan hóa bằng Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x='Group', y='Value', data=data)
plt.title('So sánh phân phối 3 nhóm')
plt.show()
# Nhận xét: Nhóm C có vẻ cao hơn hẳn A và B. Boxplot giúp ta có cảm nhận ban đầu.

# 2. Kiểm tra giả định Homogeneity of Variance (Levene's Test)
# H0: Phương sai các nhóm bằng nhau
stat, p_value = stats.levene(group_a, group_b, group_c)
print(f"Levene's Test: Statistic={stat:.3f}, p-value={p_value:.3f}")

if p_value > 0.05:
    print("Không đủ bằng chứng bác bỏ H0 -> Phương sai đồng nhất.")
else:
    print("Bác bỏ H0 -> Phương sai không đồng nhất (Vi phạm giả định ANOVA).")

# 3. Kiểm tra giả định Normality (Shapiro-Wilk) cho một nhóm (ví dụ nhóm A)
# H0: Dữ liệu tuân theo phân phối chuẩn
stat, p_value = stats.shapiro(group_a)
print(f"Shapiro-Wilk for Group A: p-value={p_value:.3f}")
```

## 7. Giải Thích và Các Lỗi Thường Gặp

-   **Hiểu nhầm phổ biến:** "ANOVA cho biết nhóm nào lớn nhất." Thực tế, ANOVA chỉ cho biết *có ít nhất một nhóm khác biệt so với các nhóm còn lại*, chứ **không** chỉ ra cụ thể cặp nhóm nào khác nhau. Để xác định chi tiết, ta cần bước phân tích hậu kiểm (Post-hoc analysis) ở bài sau.
-   **Bỏ qua việc kiểm tra giả định:** Nhiều phân tích chạy ANOVA ngay mà không kiểm tra tính đồng nhất phương sai. Nếu phương sai giữa các nhóm chênh lệch lớn, kết quả kiểm định $F$ cổ điển có thể bị sai lệch; trong trường hợp đó nên sử dụng **Welch's ANOVA**.

---

### Ví dụ tính tay làm từng bước: Tự kiểm chứng đẳng thức $$\mathrm{SS}_{\text{Total}} = \mathrm{SS}_{\text{Between}} + \mathrm{SS}_{\text{Within}}$$

Xét 3 nhóm phương pháp học tập ($$k = 3$$), mỗi nhóm có $$n = 3$$ sinh viên (tổng $$N = 9$$) với điểm số:
- **Nhóm A:** `[2, 4, 6]` $$\implies \bar{x}_A = 4$$
- **Nhóm B:** `[5, 7, 9]` $$\implies \bar{x}_B = 7$$
- **Nhóm C:** `[8, 10, 12]` $$\implies \bar{x}_C = 10$$

Trung bình tổng thể (Grand Mean): $$\bar{x} = \frac{4 + 7 + 10}{3} = 7$$. Bây giờ tính tay từng thành phần:

1. **Biến thiên giữa các nhóm ($$\mathrm{SS}_{\text{Between}}$$ — Tín hiệu):**
   $$\mathrm{SS}_{\text{Between}} = \sum_{j=1}^3 n_j(\bar{x}_j - \bar{x})^2 = 3(4-7)^2 + 3(7-7)^2 + 3(10-7)^2 = 27 + 0 + 27 = 54.$$
2. **Biến thiên bên trong từng nhóm ($$\mathrm{SS}_{\text{Within}}$$ — Nhiễu nội tại):**
   - Nhóm A: $$(2-4)^2 + (4-4)^2 + (6-4)^2 = 4 + 0 + 4 = 8.$$
   - Nhóm B: $$(5-7)^2 + (7-7)^2 + (9-7)^2 = 4 + 0 + 4 = 8.$$
   - Nhóm C: $$(8-10)^2 + (10-10)^2 + (12-10)^2 = 4 + 0 + 4 = 8.$$
   Tổng: $$\mathrm{SS}_{\text{Within}} = 8 + 8 + 8 = 24.$$
3. **Tổng biến thiên toàn bộ ($$\mathrm{SS}_{\text{Total}}$$):**
   $$\mathrm{SS}_{\text{Total}} = \sum_{i,j}(x_{ij} - 7)^2 = (25+9+1) + (4+0+4) + (1+9+25) = 35 + 8 + 35 = 78.$$

Kiểm tra cộng lại: **$$54 + 24 = 78$$ — chính xác tuyệt đối!** Tỷ lệ phương sai giải thích được bởi sự khác biệt giữa các nhóm (còn gọi là $$\eta^2$$, hệt như $$R^2$$ trong hồi quy) là $$54/78 \approx 69{,}2\%$$!

### Giải thích: Bốn điều cốt lõi để hiểu sâu bản chất ANOVA

1. **Tại sao "phân tích phương sai" lại kiểm định được "trung bình"?** Vì nếu trung bình các nhóm tách xa nhau, các điểm trung bình nhóm $$\bar{x}_j$$ sẽ trải rộng ra quanh $$\bar{x}$$, làm $$\mathrm{SS}_{\text{Between}}$$ phình to vượt trội so với nhiễu nội bộ $$\mathrm{SS}_{\text{Within}}$$.
2. **ANOVA thực chất là trường hợp đặc biệt của Hồi quy Tuyến tính:** Nếu bạn mã hóa tên nhóm (A, B, C) thành biến giả (dummy variables) rồi chạy OLS, kiểm định $$F$$ tổng thể của hồi quy cho ra **chính xác cùng một con số $$F$$ và $$p$$-value** như ANOVA một yếu tố!
3. **Nhìn biểu đồ Boxplot trước khi bấm chạy kiểm định:** Boxplot cho bạn thấy ngay độ tách giữa các nhóm, độ rộng phương sai từng nhóm và các điểm ngoại lệ cực đoan.
4. **Khi giả định phương sai bằng nhau bị vi phạm:** Nếu kiểm định Levene bác bỏ đồng phương sai (ví dụ nhóm C biến động gấp 5 lần nhóm A), hãy chuyển sang **Welch’s ANOVA** hoặc kiểm định phi tham số **Kruskal–Wallis** thay vì ép dùng ANOVA cổ điển.

**Kiểm tra nhanh.** Trong ví dụ 3 nhóm trên, nếu ta cộng thêm cùng một độ nhiễu nội bộ khiến Nhóm A thành `[0, 4, 8]`, Nhóm B thành `[3, 7, 11]`, Nhóm C thành `[6, 10, 14]` (giữ nguyên trung bình 4, 7, 10!), thì $$\mathrm{SS}_{\text{Between}}$$ có thay đổi không, còn $$\mathrm{SS}_{\text{Within}}$$ tăng lên gấp mấy lần?

---

## Video tham khảo

- [Bài giảng 23: Giới thiệu phân tích phương sai (ANOVA)](https://www.youtube.com/watch?v=I19vEdaM1eY) — Nguyễn Văn Tuấn, tiếng Việt. **Xem trước.** Giới thiệu ANOVA bằng tiếng Việt: so sánh đồng thời nhiều trung bình thay vì chạy hàng loạt kiểm định $t$.
- [One-Way ANOVA [Analysis of Variance\] simply explained](https://www.youtube.com/watch?v=mOdYddj5IG8) — numiqo, tiếng Anh. Giải thích ANOVA một yếu tố qua biến động giữa nhóm và trong nhóm, tương ứng với phân rã tổng bình phương trong bài.
- [Levene's test [Test for variance equality\]](https://www.youtube.com/watch?v=x51GDTiPIfI) — numiqo, tiếng Anh. Trình bày kiểm định Levene cho giả định phương sai đồng nhất trước khi diễn giải thống kê $F$.

## 8. Bài Tập

1.  Vì sao khi so sánh 3 nhóm thuốc, ta không nên dùng 3 kiểm định $t$ riêng lẻ (A với B, B với C, A với C)?
2.  Biến động trong nhóm ("Within-Group") đại diện cho yếu tố nào trong một thí nghiệm khoa học?
3.  (Python) Tải tập dữ liệu `Iris`:
    -   Vẽ boxplot so sánh `sepal_width` giữa 3 loài hoa (`species`).
    -   Sử dụng kiểm định Levene để kiểm tra xem phương sai của `sepal_width` giữa 3 loài có đồng nhất hay không.

## 9. Tài Liệu Tham Khảo

-   [1] Chapter 12, *OpenIntro Statistics*.
-   [2] Chapter 12, Mario F. Triola, *Elementary Statistics*.
