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

Bài tùy chọn này không viết lại ANOVA một yếu tố, kiểm định $$F$$, hay Tukey HSD. Nó cho thấy các đối tượng đó chính là phân tích thí nghiệm sản phẩm nhiều biến thể, vì sao bố trí hai chiều là mô hình đúng cho một lần ra mắt giai thừa, và cách bảng ANOVA trở thành mô hình hỗn hợp khi người dùng hoặc truy vấn bị lồng—dạng Gelman biện hộ năm 2005 và các bài ứng dụng 2022–2026 vẫn khuyến nghị.

Sau bài, bạn viết được thí nghiệm A/B/n như ANOVA một yếu tố, thêm yếu tố thứ hai mà không chạy một nắm $$t$$-test chưa hiệu chỉnh, chọn giữa Tukey và FDR cho bước hậu kiểm, và nói khi nào bảng $$F$$ nên được thay bằng một thành phần phương sai hiệu ứng ngẫu nhiên.

Hai bài bắt buộc Chương 08, cộng ý đa kiểm định từ Chương 06, là tiền đề. Phân rã

$$SS_{\mathrm{Total}} = SS_{\mathrm{Between}} + SS_{\mathrm{Within}}$$

được giả định đã biết.

---

## A/B/n là ANOVA một yếu tố mặc áo sản phẩm

Một nền tảng đưa ba nút thanh toán không phải đang chạy hai A/B test. Nó đang so sánh $$k \ge 3$$ trung bình. Mô hình đúng là bố trí một chiều bắt buộc

$$Y_{ij} = \mu + \tau_i + \varepsilon_{ij}, \qquad \varepsilon_{ij} \sim \mathcal{N}(0,\sigma^2),$$

và câu hỏi omnibus là $$H_0: \tau_1 = \cdots = \tau_k = 0$$. Statistic $$F$$

$$F = \frac{MS_{\mathrm{Between}}}{MS_{\mathrm{Within}}}$$

là cổng đầu đúng. Các $$t$$-test từng cặp không hiệu chỉnh họ là sai lầm mà bài bắt buộc đã gọi tên: sai lầm loại I tăng theo $$\binom{k}{2}$$.

Điều đổi trong dùng công nghiệp là *metric* và *đơn vị gán*, không phải đại số. Chuyển đổi là Bernoulli, nên ANOVA Gauss là xấp xỉ được CLT của chương khoảng biện minh, hoặc người ta chuyển sang GLM (Chương 07) và đọc một likelihood-ratio test vốn là ANOVA trong họ mũ. Gán ở cấp người dùng; phân tích sự kiện như các hàng i.i.d. làm thấp $$MS_{\mathrm{Within}}$$ và sản xuất ra ý nghĩa thống kê.

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

Tukey HSD, đã tính trong bài bắt buộc thứ hai, vẫn là hậu kiểm mặc định khi mọi cặp đều đáng quan tâm và thiết kế cân. Khi một nhóm soi hai mươi phân khúc *sau* khi xem dữ liệu, họ không còn ở thế giới Tukey; họ ở thế giới FDR Chương 06.

## Ra mắt giai thừa là ANOVA hai chiều

Nhóm sản phẩm hiếm khi đổi một yếu tố. Một lần ra mắt có thể chéo mô hình xếp hạng mới (cũ / mới) với bố cục snippet mới (đối chứng / biến thể). Đó là bố trí hai chiều

$$Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \varepsilon_{ijk}.$$

Tương tác $$(\alpha\beta)_{ij}$$ là số hạng thú vị về khoa học: mô hình mới có thể chỉ giúp khi snippet đổi. Chạy hai A/B test tách rời giấu số hạng đó và phí mẫu. Bảng ANOVA tách $$SS$$ thành hiệu ứng chính và tương tác; mỗi hàng là một kiểm định $$F$$ với bậc tự do riêng.

Reluga, Ye và Zhao (2022), viết về hiệu chỉnh hồi quy, nhắc rằng hiệu trung bình một chiều *chính là* estimator ANOVA, và việc thêm hiệp biến (ANCOVA) hoặc tương tác (ANHECOVA) đổi phương sai, không đổi xác định nhân quả, khi xử lý được ngẫu nhiên hóa. Thí nghiệm giai thừa là cùng lời nhắc với hai yếu tố xử lý thay vì một xử lý và một hiệp biến.

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

Tổng bình phương Type II so với Type III có ý nghĩa khi thiết kế lệch—trạng thái thường của thí nghiệm trực tuyến không đạt cỡ mẫu trên mọi ô. Báo cáo đúng type đã đăng ký trước; đừng chọn mua.

## Từ bảng ANOVA đến mô hình hỗn hợp

Gelman (2005) lập luận rằng ANOVA *quan trọng hơn* khi ta coi mọi hàng của bảng như một thành phần phương sai, kể cả những hàng cổ điển gọi là cố định. Mô hình phân cấp

$$Y_{ij} = \mu + \tau_{a[i]} + u_{b[i]} + \varepsilon_i$$

là một ANOVA trong đó batch $$b$$ (người dùng, truy vấn, thị trường, ngày) là hiệu ứng ngẫu nhiên. Bảng $$F$$ khi đó là tóm tắt các $$\sigma^2$$ ước lượng, không phải nghi lễ gắn sao.

Góc nhìn đó là thứ các bài ứng dụng hiện đại vận hành hóa. Yu et al. (2022) chỉ ra ANOVA mặc định trong khoa học phòng thí nghiệm bỏ qua động vật hoặc tế bào như một cụm; cùng đoạn văn áp dụng cho người dùng đóng góp 40 phiên. Bates et al. (2015) là tham chiếu tính toán. Với thí nghiệm sản phẩm, bản dịch thực hành là:

1. Nếu bạn có một kết cục gộp mỗi đơn vị được gán, ANOVA một hoặc hai chiều cổ điển (hoặc OLS tương đương) là đủ.
2. Nếu phải giữ các quan sát lặp, đặt intercept ngẫu nhiên trên đơn vị gán và đọc kiểm định xử lý từ mô hình hỗn hợp, không từ `scipy.stats.f_oneway` trên sự kiện thô.
3. Dùng *phân rã* ANOVA để giải thích output mô hình hỗn hợp: bao nhiêu phương sai nằm giữa người dùng so với trong người dùng.

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

$$p$$-value hiệu ứng cố định trên `treatment` khi đó so sánh được với kiểm định $$F$$ của ANOVA, nhưng mẫu số dùng đúng phân cụm.

## Các lựa chọn hậu kiểm khớp quyết định

| Quyết định sau omnibus có ý nghĩa | Công cụ | Vì sao |
|---|---|---|
| Biến thể nào trong vài phương án đã đăng ký thắng? | Tukey HSD hoặc một đối lập hoạch định | Họ nhỏ và đóng |
| Lát nào trong rất nhiều lát đã dịch? | Benjamini–Hochberg (Chương 06) | Họ lớn và thăm dò |
| Có gì dịch không, cho cổng ship/không ship? | $$F$$ omnibus hoặc OEC đăng ký trước | Tránh câu cá từng cặp |
| Người dùng hoặc truy vấn có phải nguồn nhiễu? | Mô hình hỗn hợp / thành phần phương sai | $$MS_{\mathrm{Within}}$$ cổ điển sai |

Một mẫu 2022–2026 phổ biến trên nền tảng thí nghiệm là: kiểm định omnibus trên tiêu chí đánh giá tổng thể, Tukey hoặc đối lập đăng ký trước trên các biến thể, và FDR trên năm mươi metric chẩn đoán không ai đăng ký. Trộn ba việc đó vào một bảng chưa hiệu chỉnh là cách A/B/n mang tiếng thắng giả.

## Thách thức và mở rộng

ANOVA giả định phương sai không đổi và, trong dẫn xuất cổ điển, sai số Gauss. Tỷ lệ chuyển đổi gần 0 hoặc 1, và doanh thu có khối tại không, phá cả hai; GLM hoặc một phép biến đổi khi đó là bạn đồng hành Chương 07, không phải lý do bỏ phân rã. Các ô lệch, biến thể thiếu, và peeking tuần tự (Chương 06) đều đổi sai lầm loại I hữu hạn mẫu của kiểm định $$F$$. Mô hình hỗn hợp với ít nhóm ước lượng $$\sigma_u^2$$ kém; gộp có thể an toàn hơn.

Mở rộng gồm ANOVA hoán vị khi giả định Gauss khó tin, biến đổi hạng căn chỉnh trong các nghiên cứu tương tác người–máy, và ANOVA phân cấp Bayes khi nhiều yếu tố yếu chia một prior—bối cảnh Gelman vốn nhằm tới.

## Bài tập

**Bài 1 (khái niệm).** Một nhóm chạy bốn màu nút và sáu $$t$$-test từng cặp, rồi đưa cặp có $$p$$ nhỏ nhất. Họ đã kết hợp hai sai lầm Chương 08/06 nào? Đối lập đăng ký trước nào sẽ sạch hơn?

**Bài 2 (tính toán).** Mô phỏng thiết kế một chiều cân với $$k = 4$$ nhóm, $$n = 80$$, ba trung bình bằng nhau và một nhóm nâng $$0.3\sigma$$. So sánh tỷ lệ bác bỏ omnibus và số thắng từng cặp sai dưới $$t$$-test chưa hiệu chỉnh so với Tukey. Lặp với $$n$$ không đều.

**Bài 3 (tính toán).** Dựng giai thừa hai chiều trong đó mô hình xếp hạng mới chỉ giúp với bố cục mới (tương tác thuần). Chỉ ra rằng hai A/B test tách có thể cùng không có ý nghĩa trong khi `two_way_anova_table` bắt được tương tác.

**Bài 4 (mở).** Lấy một thí nghiệm đa nhóm công khai hoặc nghiên cứu UX có đo lặp. Khớp (i) ANOVA một yếu tố trên các tổng hợp cấp người dùng, (ii) ANOVA trên sự kiện thô, và (iii) mô hình hỗn hợp với intercept người dùng. Hòa giải ba $$p$$-value bằng góc nhìn thành phần phương sai.

## Tài liệu tham khảo

Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015). Fitting linear mixed-effects models using lme4. *Journal of Statistical Software, 67*(1), 1–48. [https://doi.org/10.18637/jss.v067.i01](https://doi.org/10.18637/jss.v067.i01)

Gelman, A. (2005). Analysis of variance—why it is more important than ever. *Statistical Science, 20*(1), 1–31. [https://doi.org/10.1214/088342305000000016](https://doi.org/10.1214/088342305000000016)

Reluga, K., Ye, T., & Zhao, Q. (2022). A unified analysis of regression adjustment in randomized experiments. [https://arxiv.org/abs/2210.04360](https://arxiv.org/abs/2210.04360)

Tukey, J. W. (1949). Comparing individual means in the analysis of variance. *Biometrics, 5*(2), 99–114. Thủ tục hậu kiểm vẫn được trích dẫn, đã dạy trong bài bắt buộc.

Yu, Z., Grieco, S. F., Holmes, T. C., Xu, X., et al. (2022). Beyond t-test and ANOVA: Applications of mixed-effects models for more rigorous statistical analysis in neuroscience research. *Neuron, 110*(1), 21–35. [https://doi.org/10.1016/j.neuron.2021.10.030](https://doi.org/10.1016/j.neuron.2021.10.030)
