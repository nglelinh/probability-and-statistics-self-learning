---
layout: post
title: "Bài 2: Tính Toán và Phân Tích ANOVA Một Yếu Tố"
chapter: "08"
order: 2
owner: nglelinh
lang: vi
categories:
- chapter08
lesson_type: required
---

Ở Bài 1, ta đã tách được tổng biến thiên của dữ liệu thành phần Tín hiệu ($$\mathrm{SS}_{\text{Between}} = 54$$) và phần Nhiễu ($$\mathrm{SS}_{\text{Within}} = 24$$). Nhưng khoan vội lấy 54 chia cho 24! Hai tổng bình phương đó không đứng trên cùng một mặt bằng so sánh vì chúng gom từ số lượng độ lệch độc lập (bậc tự do) khác nhau: một bên chỉ có $$k - 1 = 2$$ bậc tự do giữa 3 nhóm, còn bên kia có tới $$N - k = 6$$ bậc tự do trong nội bộ các nhóm.

Bài học này hoàn thiện trọn vẹn quy trình **ANOVA Một Yếu Tố (One-Way ANOVA)**: cách chuẩn hóa tổng bình phương thành **Trung bình bình phương (Mean Squares)** để tính **thống kê $$F$$**, đọc hiểu đầy đủ **Bảng ANOVA**, và thực hiện bước cuối cùng không thể thiếu — **Phân tích Hậu kiểm (Post-hoc Analysis) bằng Tukey’s HSD** để chỉ đích danh *cặp nhóm nào* thực sự khác nhau sau khi kiểm định $$F$$ báo động.

---

## Bản đồ kiến thức nền: Bốn mảnh ghép trong quy trình kiểm định ANOVA Một Yếu Tố

Một quy trình phân tích ANOVA hoàn chỉnh trên thực tế luôn đi qua bốn cột mốc không thể đảo thứ tự:

1. **Chuẩn hóa theo Bậc tự do (Degrees of Freedom):** Chia $$\mathrm{SS}_{\text{Between}}$$ cho $$df_1 = k - 1$$ được $$\mathrm{MS}_{\text{Between}}$$; chia $$\mathrm{SS}_{\text{Within}}$$ cho $$df_2 = N - k$$ được $$\mathrm{MS}_{\text{Within}}$$. Dưới $$H_0$$ (mọi nhóm cùng trung bình), cả hai đại lượng đều là ước lượng không chệch của cùng một phương sai nhiễu $$\sigma^2$$, nên tỷ số $$F = \mathrm{MS}_{\text{Between}} / \mathrm{MS}_{\text{Within}}$$ dao động quanh 1!
2. **Phân phối $$F(df_1, df_2)$$ một phía phải:** Vì $$\mathrm{MS}_{\text{Between}}$$ chỉ tăng lên khi trung bình các nhóm khác nhau, kiểm định ANOVA luôn là kiểm định **đuôi phải** — giá trị $$F$$ càng lớn hơn 1, p-value càng nhỏ.
3. **Tính chất "tổng quát" (Omnibus) của kiểm định $$F$$:** Bác bỏ $$H_0$$ chỉ khẳng định *có ít nhất một cặp nhóm khác nhau* (ví dụ A khác C), chứ hoàn toàn không nói tất cả các cặp đều khác nhau.
4. **Phân tích hậu kiểm Tukey’s HSD (Honestly Significant Difference):** Sau khi $$F$$ có ý nghĩa, Tukey dùng phân phối tầm Student hóa (*Studentized range* $$q_{\alpha;\,k,\,N-k}$$) để tạo ra một ngưỡng chênh lệch chung kiểm soát chặt sai lầm Loại I cho *toàn bộ* $$\binom{k}{2}$$ phép so sánh cặp cùng lúc.

---

## 1. Mục Tiêu Bài Học

Sau khi hoàn thành bài học này, sinh viên có thể:
- Thực hiện đầy đủ các bước **kiểm định giả thuyết ANOVA một yếu tố (One-Way ANOVA)**.
- Tính toán và diễn giải **bảng ANOVA** gồm bậc tự do (Degrees of Freedom), trung bình bình phương (Mean Squares), thống kê $F$ và giá trị $p$.
- Thực hiện **phân tích hậu kiểm (Post-hoc Analysis)** bằng kiểm định **Tukey's HSD** để xác định cụ thể các cặp nhóm khác biệt.
- Sử dụng thư viện `scipy.stats` và `statsmodels` cho quy trình phân tích hoàn chỉnh.

## 2. Kiến Thức Tiền Đề

- Bài 1: Giới thiệu về ANOVA.
- Thư viện `statsmodels` (cài đặt bằng lệnh `pip install statsmodels`).

## 3. Động Lực và Giới Thiệu

Ở bài trước, ta đã biết ANOVA sử dụng thống kê $F$ (tỷ số giữa biến động tín hiệu và biến động nhiễu) để kiểm tra xem có sự khác biệt giữa các nhóm hay không. Trong bài này, ta sẽ đi vào chi tiết **cách tính toán** bảng ANOVA và câu hỏi quan trọng tiếp theo: **cần làm gì sau khi phát hiện có sự khác biệt?**

Chẳng hạn, khi cần trả lời câu hỏi chiến dịch marketing nào hiệu quả hơn giữa Email, Facebook và TikTok, kết luận dừng ở mức "có sự khác biệt giữa các kênh" (kết quả ANOVA tổng quát) là chưa đủ để ra quyết định. Ta cần kết luận cụ thể hơn, ví dụ "Facebook hiệu quả hơn Email nhưng không khác biệt đáng kể so với TikTok" — và đó là vai trò của phân tích hậu kiểm (post-hoc).

## 4. Các Khái Niệm Cốt Lõi và Quy Trình

### 4.1 Giả Thuyết Của ANOVA Một Yếu Tố

-   $H_0$: $\mu_1 = \mu_2 = ... = \mu_k$ (trung bình tổng thể của tất cả các nhóm bằng nhau).
-   $H_a$: Có ít nhất một cặp trung bình ($\mu_i, \mu_j$) khác nhau.

### 4.2 Bảng ANOVA (The ANOVA Table)

Kết quả ANOVA thường được trình bày theo bảng chuẩn sau:

| Nguồn Biến Động (Source) | Bậc Tự Do (DF) | Tổng Bình Phương (SS) | Trung Bình Bình Phương (MS) | F-statistic | p-value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Between Groups** | $k - 1$ | $SS_{Between}$ | $MS_{B} = SS_{B} / DF_{B}$ | $F = MS_{B} / MS_{W}$ | Prob(>F) |
| **Within Groups (Error)** | $N - k$ | $SS_{Within}$ | $MS_{W} = SS_{W} / DF_{W}$ | | |
| **Total** | $N - 1$ | $SS_{Total}$ | | | |

Trong đó:
-   $k$: Số lượng nhóm.
-   $N$: Tổng số quan sát.
-   Nếu $p\_value < \alpha$ (thường chọn $\alpha = 0.05$), ta bác bỏ $H_0$.

### 4.3 Phân Tích Hậu Kiểm (Post-hoc Analysis)

Khi bác bỏ $H_0$, ta mới biết có sự khác biệt giữa các nhóm nhưng chưa biết cụ thể cặp nhóm nào khác nhau. Các kiểm định hậu kiểm phổ biến gồm:
-   **Tukey's HSD (Honest Significant Difference):** Kiểm định phổ biến nhất cho so sánh mọi cặp (pairwise comparisons), kiểm soát tốt xác suất sai lầm loại I tổng thể.
-   **Hiệu chỉnh Bonferroni (Bonferroni Correction):** Cách tính đơn giản nhưng mang tính bảo thủ cao, nên lực kiểm định thấp hơn (khó phát hiện khác biệt thực sự hơn).

## 5. Ví Dụ Minh Họa

Quay lại ví dụ so sánh 3 loại phân bón A, B và C. Giả sử kết quả ANOVA cho $F = 5.2$ và $p\_value = 0.01 < 0.05$, nên ta bác bỏ $H_0$ và kết luận hiệu quả của các loại phân bón có sự khác biệt. Tiếp theo, ta chạy kiểm định Tukey's HSD:
-   A với B: $p = 0.5$ (không có khác biệt có ý nghĩa thống kê).
-   B với C: $p = 0.04$ (có khác biệt có ý nghĩa thống kê).
-   A với C: $p = 0.02$ (có khác biệt có ý nghĩa thống kê).

Nếu trung bình của nhóm C cao hơn A và B, kết luận cuối cùng là phân bón C cho hiệu quả tốt hơn A và B, trong khi A và B không khác biệt đáng kể.

## 6. Triển Khai Trên Python

Ta sử dụng lại bộ dữ liệu giả lập gồm 3 nhóm A, B, C từ bài trước.

```python
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Tạo lại dữ liệu (để code độc lập)
group_a = stats.norm.rvs(loc=50, scale=5, size=30, random_state=1)
group_b = stats.norm.rvs(loc=55, scale=5, size=30, random_state=1)
group_c = stats.norm.rvs(loc=65, scale=10, size=30, random_state=1)

df = pd.DataFrame({
    'Value': list(group_a) + list(group_b) + list(group_c),
    'Group': ['A']*30 + ['B']*30 + ['C']*30
})

# CÁCH 1: Dùng Scipy (Nhanh gọn, chỉ ra F và p)
f_stat, p_val = stats.f_oneway(group_a, group_b, group_c)
print(f"Scipy One-Way ANOVA: F={f_stat:.3f}, p={p_val:.5f}")

# CÁCH 2: Dùng Statsmodels (Xuất Bảng ANOVA chi tiết - Khuyên dùng)
# Cú pháp giống R: Value phụ thuộc vào Group
model = ols('Value ~ C(Group)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("\n--- Bảng ANOVA (Statsmodels) ---")
print(anova_table)

# Kiểm tra kết luận
if p_val < 0.05:
    print("\n -> Có sự khác biệt có ý nghĩa thống kê giữa các nhóm.")
    print(" -> Tiến hành Post-hoc Test (Tukey's HSD)...")
    
    # Post-hoc Test
    tukey = pairwise_tukeyhsd(endog=df['Value'],     # Dữ liệu đo lường
                              groups=df['Group'],    # Nhóm
                              alpha=0.05)            # Mức ý nghĩa
    
    print("\n--- Kết quả Tukey HSD ---")
    print(tukey)
    
    # Vẽ biểu đồ khoảng tin cậy cho sự khác biệt
    # tukey.plot_simultaneous() 
else:
    print("\n -> Không đủ bằng chứng để bác bỏ H0. Các nhóm coi như ngang nhau.")

```

## 7. Giải Thích và Các Lỗi Thường Gặp

-   **Dùng kiểm định $t$ lặp lại thay cho hậu kiểm:** Như đã phân tích, việc chạy nhiều kiểm định $t$ không hiệu chỉnh sẽ làm tăng xác suất sai lầm loại I tổng thể. Tukey's HSD tự động hiệu chỉnh giá trị $p$ để giữ mức sai lầm loại I tổng thể (family-wise error rate) ở ngưỡng $\alpha$ (thường là 5%).
-   **Đọc nhầm bảng kết quả Tukey:**
    -   Cột `reject` (`True`/`False`) cho biết có bác bỏ giả thuyết hai nhóm bằng nhau hay không (`True` nghĩa là có khác biệt có ý nghĩa thống kê).
    -   Cột `meandiff` thể hiện hiệu trung bình giữa hai nhóm đang so sánh.

---

### Ví dụ tính tay cầm tay chỉ việc: Lập trọn vẹn Bảng ANOVA và chạy hậu kiểm Tukey HSD

Tiếp nối chính bộ số liệu 3 nhóm ($$k = 3$$, $$n = 3$$ mỗi nhóm, $$N = 9$$) từ Bài 1 với $$\bar{x}_A = 4,\; \bar{x}_B = 7,\; \bar{x}_C = 10$$ và $$\mathrm{SS}_{\text{Between}} = 54,\; \mathrm{SS}_{\text{Within}} = 24$$:

**Bước 1 — Hoàn thiện Bảng ANOVA:**

| Nguồn biến thiên | Tổng bình phương (SS) | Bậc tự do (df) | Trung bình bình phương (MS) | Thống kê $$F$$ |
| :--- | :--- | :--- | :--- | :--- |
| **Giữa các nhóm (Between)** | $$54{,}0$$ | $$k - 1 = 2$$ | $$54{,}0 / 2 = 27{,}0$$ | $$F = \frac{27{,}0}{4{,}0} = 6{,}75$$ |
| **Trong nhóm (Within)** | $$24{,}0$$ | $$N - k = 6$$ | $$24{,}0 / 6 = 4{,}0$$ | — |
| **Tổng cộng (Total)** | $$78{,}0$$ | $$N - 1 = 8$$ | — | — |

Tra phân phối $$F(2, 6)$$ ở mức $$\alpha = 0{,}05$$: giá trị tới hạn là $$F_{0{,}05;\,2,\,6} \approx 5{,}143$$ (và p-value chính xác là $$p \approx 0{,}0292 < 0{,}05$$). Vì $$6{,}75 > 5{,}143$$, ta **bác bỏ $$H_0$$**: ba phương pháp học tập *không* có cùng điểm trung bình!

**Bước 2 — Hậu kiểm Tukey’s HSD để tìm cặp khác biệt:**
Với $$k = 3, df_{\text{Within}} = 6, \alpha = 0{,}05$$, tra bảng Studentized range được $$q_{0{,}05;\,3,\,6} \approx 4{,}339$$. Tính ngưỡng khác biệt có ý nghĩa thực sự (HSD):

$$\mathrm{HSD} = q_{0{,}05;\,3,\,6}\sqrt{\frac{\mathrm{MS}_{\text{Within}}}{n}} = 4{,}339\sqrt{\frac{4{,}0}{3}} \approx 4{,}339 \times 1{,}155 \approx 5{,}01.$$

So sánh chênh lệch tuyệt đối của từng cặp trung bình với ngưỡng $$5{,}01$$:
- **Cặp C vs. A:** $$\lvert 10 - 4\rvert = 6{,}0 > 5{,}01$$ $$\implies$$ **Khác biệt có ý nghĩa thống kê!**
- **Cặp B vs. A:** $$\lvert 7 - 4\rvert = 3{,}0 < 5{,}01$$ $$\implies$$ Chưa đủ bằng chứng kết luận khác nhau.
- **Cặp C vs. B:** $$\lvert 10 - 7\rvert = 3{,}0 < 5{,}01$$ $$\implies$$ Chưa đủ bằng chứng kết luận khác nhau.

Thấy ngay giá trị của Tukey HSD: mặc dù điểm trung bình xếp đều $$4 < 7 < 10$$, với cỡ mẫu $$n = 3$$ mỗi nhóm, dữ liệu chỉ đủ sức khẳng định chắc chắn **Nhóm C vượt trội Nhóm A**!

### Khung giải thích: Bốn kỷ luật khi đọc Bảng ANOVA và Post-hoc (nên đọc chậm)

1. **Nếu kiểm định $$F$$ không bác bỏ $$H_0$$ ($$p \ge 0{,}05$$), DỪNG LẠI NGAY:** Đừng cố chạy hậu kiểm để "moi" xem có cặp nào tình cờ $$p < 0{,}05$$ không — đó là một dạng *p-hacking*.
2. **Đừng quên báo cáo độ lớn hiệu ứng ($$\eta^2$$ hoặc $$\omega^2$$):** Trong ví dụ trên, $$\eta^2 = \mathrm{SS}_{\text{Between}} / \mathrm{SS}_{\text{Total}} = 54/78 \approx 0{,}692$$ cho thấy yếu tố nhóm giải thích gần 70% biến thiên điểm số.
3. **Hiểu vì sao $$\mathrm{MS}_{\text{Within}}$$ lại nằm trong công thức Tukey:** Tukey dùng ước lượng phương sai gộp từ *tất cả* các nhóm ($$\mathrm{MS}_{\text{Within}}$$) nên có bậc tự do lớn hơn ($$N-k = 6$$) so với tách từng cặp chạy $$t$$-test (chỉ có $$2n-2 = 4$$ bậc tự do).
4. **Kiểm định $$F$$ của ANOVA 2 nhóm ($$k = 2$$) chính là bình phương của $$t$$-test hai mẫu:** Luôn nhớ đẳng thức đẹp mắt $$F = t^2$$ khi chỉ có 2 nhóm — ANOVA thực sự là sự tổng quát hóa liền mạch của $$t$$-test!

**Kiểm tra nhanh.** Trong bảng ANOVA trên, giả sử cỡ mẫu mỗi nhóm tăng từ $$n = 3$$ lên $$n = 10$$ (vẫn giữ nguyên trung bình 4, 7, 10 và phương sai trong nhóm $$\mathrm{MS}_{\text{Within}} = 4{,}0$$). Khi đó $$\mathrm{SS}_{\text{Between}}$$ và thống kê $$F$$ sẽ tăng lên gấp bao nhiêu lần?

---

## Video tham khảo

- [Phân tích phương sai một yếu tố (One-Way ANOVA) | Phân Tích Thống Kê 23 | Learn to do SCIENCE](https://www.youtube.com/watch?v=JWiznZZEjBw) — Learn to do SCIENCE, tiếng Việt. **Xem trước.** Hướng dẫn tính ANOVA một yếu tố bằng tiếng Việt, tương ứng với bảng kết quả `statsmodels` trong bài.
- [Using Linear Models for t-tests and ANOVA, Clearly Explained!!!](https://www.youtube.com/watch?v=NF5_btOaCig) — StatQuest with Josh Starmer, tiếng Anh. Diễn giải kiểm định $F$ như một trường hợp của mô hình tuyến tính với ma trận thiết kế, kết nối ANOVA với hồi quy.
- [Analysis of Variance (ANOVA) and F statistics .... MADE EASY!!!](https://www.youtube.com/watch?v=YllG5Cy6OFw) — Stats with Brian, tiếng Anh. Trình bày thống kê $F$ dưới dạng tỷ số giữa phương sai giữa nhóm và phương sai trong nhóm.

## 8. Bài Tập

1.  (Python) Tải tập dữ liệu `Diet` (hoặc tạo dữ liệu giả lập cho 3 chế độ ăn kiêng khác nhau):
    -   Biến phụ thuộc: Số cân nặng giảm được (`weight_loss`).
    -   Biến phân loại: Loại chế độ ăn (`diet_type`: 1, 2, 3).
    -   Thực hiện kiểm định ANOVA để kiểm tra xem có sự khác biệt giữa các chế độ ăn hay không.
    -   Nếu có khác biệt, sử dụng kiểm định Tukey's HSD để xác định cặp chế độ ăn nào khác nhau.

## 9. Tài Liệu Tham Khảo

-   [1] Chapter 12, *OpenIntro Statistics*.
-   [2] Statsmodels Documentation: [ANOVA](https://www.statsmodels.org/stable/anova.html).
