---
layout: post
title: "Bài 3: Tùy chọn: ANOVA hiện đại trong thí nghiệm sản phẩm"
chapter: "08"
order: 3
owner: nglelinh
lang: vi
categories:
- chapter08
lesson_type: optional
---

Bài tùy chọn này không trình bày lại định nghĩa cơ bản của ANOVA một yếu tố, kiểm định $$F$$ hay Tukey HSD. Thay vào đó, bài cho thấy các công cụ này chính là nền tảng phân tích cho thí nghiệm sản phẩm nhiều biến thể (A/B/n), giải thích vì sao thiết kế hai yếu tố là mô hình phù hợp cho một đợt triển khai giai thừa (factorial launch), và trình bày cách bảng ANOVA mở rộng thành mô hình hỗn hợp khi quan sát bị lồng theo người dùng hoặc truy vấn — góc nhìn mà Gelman (2005) đề xuất và các tài liệu ứng dụng giai đoạn 2022–2026 vẫn khuyến nghị.

Sau bài học này, bạn có thể biểu diễn thí nghiệm A/B/n dưới dạng ANOVA một yếu tố, bổ sung yếu tố thứ hai mà không phải chạy hàng loạt kiểm định $$t$$ chưa hiệu chỉnh, lựa chọn hợp lý giữa Tukey và FDR ở bước hậu kiểm, cũng như nhận biết khi nào cần thay bảng $$F$$ cổ điển bằng thành phần phương sai hiệu ứng ngẫu nhiên.

Kiến thức tiền đề bao gồm hai bài bắt buộc của Chương 08 và khái niệm đa kiểm định ở Chương 06, trong đó phép phân rã:

$$SS_{\mathrm{Total}} = SS_{\mathrm{Between}} + SS_{\mathrm{Within}}$$

được xem là đã nắm vững.

---

## Thí Nghiệm A/B/n Dưới Góc Nhìn ANOVA Một Yếu Tố

Một nền tảng thử nghiệm ba mẫu nút thanh toán không phải đang chạy hai kiểm định A/B rời rạc, mà đang so sánh đồng thời $$k \ge 3$$ trung bình. Mô hình chuẩn cho bài toán này là thiết kế ANOVA một yếu tố:

$$Y_{ij} = \mu + \tau_i + \varepsilon_{ij}, \qquad \varepsilon_{ij} \sim \mathcal{N}(0,\sigma^2),$$

với giả thuyết kiểm định tổng quát (omnibus) là $$H_0: \tau_1 = \cdots = \tau_k = 0$$. Thống kê $$F$$:

$$F = \frac{MS_{\mathrm{Between}}}{MS_{\mathrm{Within}}}$$

đóng vai trò bước sàng lọc đầu tiên. Việc chạy các kiểm định $$t$$ từng cặp không hiệu chỉnh lặp lại đúng sai lầm mà bài bắt buộc đã cảnh báo: xác suất sai lầm loại I gia tăng nhanh theo số cặp $$\binom{k}{2}$$.

Trong môi trường công nghiệp, phần thay đổi nằm ở *bản chất chỉ số* và *đơn vị gán xử lý*, chứ không phải cấu trúc đại số. Chẳng hạn, tỷ lệ chuyển đổi tuân theo phân phối Bernoulli nên ANOVA giả định chuẩn chỉ là xấp xỉ dựa trên định lý giới hạn trung tâm (CLT); một lựa chọn khác là chuyển sang GLM (Chương 07) và sử dụng kiểm định tỷ số hợp lý (likelihood-ratio test), vốn là phiên bản mở rộng của ANOVA trong họ phân phối mũ. Ngoài ra, khi xử lý được gán ở cấp người dùng, việc phân tích từng sự kiện như các quan sát i.i.d. sẽ đánh giá thấp $$MS_{\mathrm{Within}}$$ và tạo ra ý nghĩa thống kê giả tạo.

```python
import numpy as np
from scipy import stats

def oneway_anova(groups):
    """
    ANOVA một yếu tố cổ điển cho một list các mảng 1 chiều.

    Returns:
        F, p_value, ss_between, ss_within
    """
    all_y = np.concatenate(groups)
    grand = all_y.mean()
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_w = sum(((g - g.mean()) ** 2).sum() for g in groups)
    df_b = len(groups) - 1
    df_w = len(all_y) - len(groups)
    F = (ss_b / df_b) / (ss_w / df_w)
    p = stats.f.sf(F, df_b, df_w)
    return F, p, ss_b, ss_w
```

Kiểm định Tukey HSD (đã tính trong bài bắt buộc thứ hai) vẫn là lựa chọn hậu kiểm mặc định khi mọi cặp so sánh đều có ý nghĩa và thiết kế cân bằng. Tuy nhiên, nếu nhóm phân tích khảo sát 20 phân khúc *sau khi* đã xem dữ liệu, bài toán không còn thuộc phạm vi của Tukey mà chuyển sang kiểm soát tỷ lệ phát hiện sai (FDR) như ở Chương 06.

## Triển Khai Giai Thừa Là ANOVA Hai Chiều

Trong thực tế, các nhóm sản phẩm hiếm khi chỉ thay đổi một yếu tố duy nhất. Một đợt triển khai có thể kết hợp chéo giữa mô hình xếp hạng (cũ / mới) và bố cục hiển thị snippet (đối chứng / biến thể). Mô hình phù hợp là thiết kế ANOVA hai chiều (two-way ANOVA):

$$Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \varepsilon_{ijk}.$$

Số hạng tương tác $$(\alpha\beta)_{ij}$$ thường mang thông tin khoa học quan trọng nhất: mô hình xếp hạng mới có thể chỉ phát huy hiệu quả khi đi kèm bố cục snippet mới. Chạy hai thí nghiệm A/B tách rời sẽ che khuất hiệu ứng tương tác này và lãng phí cỡ mẫu. Bảng ANOVA tách tổng bình phương $$SS$$ thành các hiệu ứng chính và hiệu ứng tương tác, với mỗi hàng tương ứng một kiểm định $$F$$ và bậc tự do riêng.

Trong nghiên cứu về hiệu chỉnh hồi quy, Reluga, Ye và Zhao (2022) nhắc lại rằng hiệu trung bình một chiều *chính là* ước lượng ANOVA, còn việc bổ sung hiệp biến (ANCOVA) hoặc số hạng tương tác (ANHECOVA) làm thay đổi phương sai chứ không thay đổi khả năng nhận diện nhân quả dưới ngẫu nhiên hóa. Thí nghiệm giai thừa thể hiện cùng nguyên lý đó, chỉ khác là có hai yếu tố xử lý thay vì một yếu tố xử lý và một hiệp biến.

```python
import statsmodels.api as sm
from statsmodels.formula.api import ols

def two_way_anova_table(df):
    """
    ANOVA hai chiều có tương tác qua OLS.

    Các cột df: y (float), model (A/B), layout (A/B)
    """
    fit = ols("y ~ C(model) * C(layout)", data=df).fit()
    return sm.stats.anova_lm(fit, typ=2)
```

Sự khác biệt giữa tổng bình phương Type II và Type III trở nên đáng kể khi thiết kế mất cân bằng — tình huống thường gặp ở các thí nghiệm trực tuyến khi các ô không đạt cỡ mẫu đều nhau. Cần báo cáo đúng loại đã đăng ký trước, tránh lựa chọn kiểu tính sau khi đã xem kết quả.

## Từ Bảng ANOVA Đến Mô Hình Hỗn Hợp

Gelman (2005) lập luận rằng ANOVA trở nên *hữu ích hơn* khi mỗi hàng của bảng được nhìn nhận như một thành phần phương sai (variance component), kể cả các yếu tố theo truyền thống được gọi là cố định. Mô hình phân cấp:

$$Y_i = \mu + \tau_{a[i]} + u_{b[i]} + \varepsilon_i$$

có thể xem là một dạng ANOVA trong đó nhóm $$b$$ (người dùng, truy vấn, thị trường hoặc ngày) là hiệu ứng ngẫu nhiên. Khi đó, bảng ANOVA đóng vai trò tóm tắt độ lớn các thành phần phương sai $$\sigma^2$$ ước lượng được, thay vì chỉ là công cụ tìm kiếm các mức ý nghĩa thống kê.

Cách tiếp cận này được vận dụng rộng rãi trong thống kê ứng dụng hiện đại. Yu et al. (2022) chỉ ra rằng ANOVA truyền thống trong khoa học thí nghiệm thường bỏ qua cấu trúc phân cụm theo cá thể hoặc tế bào — nhận định hoàn toàn tương tự với trường hợp mỗi người dùng tạo ra 40 phiên truy cập trong dữ liệu sản phẩm. Bates et al. (2015) là tài liệu chuẩn về mặt tính toán cho mô hình hỗn hợp. Trong thí nghiệm sản phẩm, quy tắc thực hành gồm ba điểm:

1. Nếu mỗi đơn vị được gán xử lý chỉ có một giá trị kết cục tổng hợp, ANOVA một chiều hoặc hai chiều cổ điển (hay OLS tương đương) là đủ.
2. Nếu cần giữ lại các quan sát lặp, hãy đặt hệ số chặn ngẫu nhiên (random intercept) cho đơn vị gán và đọc kết quả kiểm định hiệu ứng xử lý từ mô hình hỗn hợp, thay vì dùng `scipy.stats.f_oneway` trên dữ liệu sự kiện thô.
3. Dùng *phép phân rã phương sai* của ANOVA để diễn giải kết quả mô hình hỗn hợp: bao nhiêu phần phương sai đến từ sự khác biệt giữa các người dùng so với biến động trong cùng một người dùng.

```python
import numpy as np
import statsmodels.formula.api as smf

def mixed_treatment_test(df):
    """
    Intercept ngẫu nhiên theo user; xử lý cố định.

    Các cột df: y, treatment, user
    """
    model = smf.mixedlm("y ~ treatment", df, groups=df["user"])
    return model.fit()
```

Giá trị $$p$$ của hiệu ứng cố định cho `treatment` khi đó tương đương về mục tiêu với kiểm định $$F$$ của ANOVA, nhưng sai số chuẩn ở mẫu số đã phản ánh đúng cấu trúc phân cụm.

## Chọn Phương Pháp Hậu Kiểm Theo Mục Tiêu Quyết Định

| Câu hỏi sau khi kiểm định omnibus có ý nghĩa | Công cụ | Lý do |
|---|---|---|
| Biến thể nào thắng trong số vài phương án đã đăng ký? | Tukey HSD hoặc đối lập hoạch định (planned contrasts) | Họ giả thuyết nhỏ và xác định trước |
| Phân khúc nào thay đổi trong số rất nhiều phân khúc? | Benjamini–Hochberg (Chương 06) | Họ giả thuyết lớn và mang tính thăm dò |
| Có sự thay đổi tổng thể nào để quyết định triển khai hay không? | Kiểm định $$F$$ omnibus hoặc OEC đăng ký trước | Tránh dò tìm ngẫu nhiên theo từng cặp |
| Người dùng hoặc truy vấn có phải nguồn nhiễu phân cụm không? | Mô hình hỗn hợp / thành phần phương sai | $$MS_{\mathrm{Within}}$$ cổ điển ước lượng sai |

Một quy trình phổ biến giai đoạn 2022–2026 trên các nền tảng thí nghiệm là: dùng kiểm định omnibus cho tiêu chí đánh giá tổng thể (OEC), dùng Tukey hoặc đối lập đăng ký trước để so sánh các biến thể, và áp dụng kiểm soát FDR cho hàng chục chỉ số chẩn đoán không đăng ký trước. Gộp cả ba loại kiểm định này vào cùng một bảng không hiệu chỉnh là nguyên nhân phổ biến dẫn đến các kết quả dương tính giả trong A/B/n.

## Thách Thức và Hướng Mở Rộng

ANOVA cổ điển giả định phương sai đồng nhất và sai số tuân theo phân phối chuẩn. Tuy nhiên, tỷ lệ chuyển đổi gần 0 hoặc 1, cũng như doanh thu có xác suất tập trung tại 0, thường vi phạm cả hai giả định này; khi đó GLM hoặc một phép biến đổi phù hợp (Chương 07) là công cụ bổ trợ cần thiết chứ không phủ nhận ý nghĩa của phép phân rã phương sai. Bên cạnh đó, thiết kế mất cân bằng, dữ liệu thiếu ở một số biến thể và việc xem sớm kết quả liên tục (peeking — Chương 06) đều làm sai lệch mức sai lầm loại I hữu hạn mẫu của kiểm định $$F$$. Ngoài ra, mô hình hỗn hợp khi số lượng nhóm quá ít thường ước lượng kém $$\sigma_u^2$$, nên phương án tổng hợp dữ liệu về cấp đơn vị gán có thể an toàn hơn.

Các hướng mở rộng gồm ANOVA hoán vị (permutation ANOVA) khi giả định chuẩn không phù hợp, phép biến đổi hạng căn chỉnh (aligned rank transform) trong các nghiên cứu tương tác người–máy, và ANOVA phân cấp Bayes khi nhiều yếu tố yếu chia sẻ chung một phân phối tiên nghiệm — đúng bối cảnh mà Gelman (2005) hướng tới.

## Video tham khảo

- [Full Factorial Design (DoE - Design of Experiments) Simply explained](https://www.youtube.com/watch?v=cHoYAyYHzB0) — numiqo, tiếng Anh. **Xem trước.** Giải thích thiết kế giai thừa hai yếu tố — mô hình phù hợp cho đợt triển khai nhiều yếu tố thay vì chạy hàng loạt kiểm định $t$.
- [Follow Up ANOVA – Tukey’s HSD Post Hoc Test (12-5)](https://www.youtube.com/watch?v=P1j2CkOjWtM) — Research By Design, tiếng Anh. Trình bày hậu kiểm Tukey HSD sau khi kiểm định $F$ có ý nghĩa, một trong hai nhánh hậu kiểm được đối chiếu với FDR trong bài.
- [The power of mixed-effects models | Longitudinal 3](https://www.youtube.com/watch?v=MOyjGnjAols) — Very Normal, tiếng Anh. Minh họa vai trò của thành phần phương sai ngẫu nhiên khi quan sát lồng trong người dùng hoặc truy vấn.

## Bài tập

**Bài 1 (khái niệm).** Một nhóm thử nghiệm bốn màu nút khác nhau, chạy sáu kiểm định $$t$$ từng cặp và công bố cặp có giá trị $$p$$ nhỏ nhất. Họ đã mắc đồng thời hai sai lầm nào từ Chương 08 và Chương 06? Cách thiết lập đối lập đăng ký trước (planned contrast) nào sẽ hợp lý hơn?

**Bài 2 (tính toán).** Mô phỏng thiết kế một yếu tố cân bằng với $$k = 4$$ nhóm, mỗi nhóm $$n = 80$$ quan sát, trong đó ba nhóm có trung bình bằng nhau và một nhóm cao hơn $$0.3\sigma$$. So sánh tỷ lệ bác bỏ của kiểm định omnibus và số cặp kết luận sai giữa kiểm định $$t$$ chưa hiệu chỉnh với Tukey HSD. Lặp lại thí nghiệm khi cỡ mẫu $$n$$ giữa các nhóm không đều.

**Bài 3 (tính toán).** Xây dựng dữ liệu theo thiết kế giai thừa hai chiều sao cho mô hình xếp hạng mới chỉ mang lại lợi ích khi kết hợp với bố cục mới (tương tác thuần). Chứng minh rằng hai kiểm định A/B tách rời đều có thể không có ý nghĩa thống kê, trong khi `two_way_anova_table` phát hiện được hiệu ứng tương tác.

**Bài 4 (mở).** Chọn một bộ dữ liệu thí nghiệm đa nhóm công khai hoặc nghiên cứu UX có đo lặp. Thực hiện: (i) ANOVA một yếu tố trên dữ liệu tổng hợp cấp người dùng, (ii) ANOVA trên dữ liệu sự kiện thô, và (iii) mô hình hỗn hợp có hệ số chặn ngẫu nhiên theo người dùng. Đối chiếu và giải thích sự khác biệt giữa ba giá trị $$p$$ dựa trên góc nhìn thành phần phương sai.

## Tài liệu tham khảo

Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015). Fitting linear mixed-effects models using lme4. *Journal of Statistical Software, 67*(1), 1–48. [https://doi.org/10.18637/jss.v067.i01](https://doi.org/10.18637/jss.v067.i01)

Gelman, A. (2005). Analysis of variance—why it is more important than ever. *Statistical Science, 20*(1), 1–31. [https://doi.org/10.1214/088342305000000016](https://doi.org/10.1214/088342305000000016)

Reluga, K., Ye, T., & Zhao, Q. (2022). A unified analysis of regression adjustment in randomized experiments. [https://arxiv.org/abs/2210.04360](https://arxiv.org/abs/2210.04360)

Tukey, J. W. (1949). Comparing individual means in the analysis of variance. *Biometrics, 5*(2), 99–114. Thủ tục hậu kiểm kinh điển đã trình bày trong bài bắt buộc.

Yu, Z., Grieco, S. F., Holmes, T. C., Xu, X., et al. (2022). Beyond t-test and ANOVA: Applications of mixed-effects models for more rigorous statistical analysis in neuroscience research. *Neuron, 110*(1), 21–35. [https://doi.org/10.1016/j.neuron.2021.10.030](https://doi.org/10.1016/j.neuron.2021.10.030)
