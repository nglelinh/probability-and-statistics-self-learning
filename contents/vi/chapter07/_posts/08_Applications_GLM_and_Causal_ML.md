---
layout: post
title: "07-08-00 Tùy chọn: GLM, mô hình hỗn hợp và hồi quy nhân quả"
chapter: "07"
order: 8
owner: nglelinh
lang: vi
categories:
- chapter07
lesson_type: optional
---

Bài tùy chọn này không dạy lại OLS, ridge, lasso hay logistic regression. Nó đưa các công cụ tuyến tính đó vào dạng mà công nghiệp và học máy nhân quả thực sự triển khai: mô hình tuyến tính tổng quát và mô hình hỗn hợp cho kết cục không Gauss và dữ liệu phân cụm, hiệu chỉnh hồi quy (CUPED và ANCOVA kiểu Lin) cho thí nghiệm, và double/debiased machine learning khi cơ chế gán không phải một đồng xu sạch.

Sau bài, bạn viết được log-likelihood của GLM, nói được khi nào cần hiệu ứng ngẫu nhiên, cài CUPED như hiệu chỉnh biến kiểm soát, và phác bước residual-on-residual khiến double ML an toàn với regularisation.

Tiền đề là các bài bắt buộc Chương 07 (hồi quy đơn, OLS, regularisation, logistic) và bài suy diễn hệ số Chương 06. Đại số ma trận ở mức $$(X^\top X)^{-1}$$ là đủ.

---

## Từ một đường thẳng đến GLM

OLS giả định

$$Y = X\beta + \varepsilon, \qquad E[\varepsilon \mid X] = 0.$$

Mô hình tuyến tính tổng quát (McCullagh và Nelder, 1989) giữ bộ dự đoán tuyến tính $$\eta = X\beta$$ và đổi hai mảnh: một link $$g(E[Y \mid X]) = \eta$$, và một hàm phương sai họ mũ. Mô hình Poisson log-tuyến tính cho đếm, mô hình Tweedie cho bồi thường bảo hiểm có khối tại không, và mô hình logistic bạn đã biết là cùng một đối tượng. Logistic regression là GLM Bernoulli với link logit; MLE của nó là bộ cực tiểu cross-entropy từ Chương 05.

Công nghiệp vẫn khớp GLM ở quy mô lớn vì chúng được hiệu chỉnh, rẻ, và kiểm toán được. Các stack gợi ý và quảng cáo thường giữ một GLM (hoặc mảnh tuyến tính “wide”) cạnh mạng sâu: Cheng et al. (2016) đặt tên mẫu đó là Wide & Deep. Mảnh tuyến tính ghi nhớ các ID thưa; mảnh sâu khái quát. Điểm thống kê là GLM không phải bản hạ cấp của mạng. Đó là likelihood mà mạng đang xấp xỉ, cộng một bảng hệ số diễn giải được khi nhà quản lý hoặc định phí bảo hiểm yêu cầu.

```python
import numpy as np
from scipy.optimize import minimize

def poisson_glm_mle(X, y):
    """
    MLE cho GLM Poisson link log, không offset.

    Args:
        X: thiết kế (n, p), thêm cột 1 nếu muốn hệ số chặn
        y: đếm không âm

    Returns:
        beta_hat
    """
    def nll(beta):
        eta = X @ beta
        # y*eta - exp(eta) là log-likelihood Poisson sai một hằng
        return np.mean(np.exp(eta) - y * eta)

    start = np.zeros(X.shape[1])
    out = minimize(nll, start, method="L-BFGS-B")
    return out.x
```

## Mô hình hỗn hợp khi người dùng không i.i.d.

Nhật ký A/B và dữ liệu sản phẩm theo thời gian vi phạm độc lập. Cùng một người dùng đóng góp nhiều phiên; cùng một truy vấn xuất hiện nhiều ngày. Mô hình tuyến tính hỗn hợp viết

$$Y_{ij} = x_{ij}^\top \beta + u_i + \varepsilon_{ij}, \qquad u_i \sim \mathcal{N}(0, \sigma_u^2),$$

nên hiệu ứng người dùng $$u_i$$ là một intercept ngẫu nhiên. Mô hình tuyến tính hỗn hợp tổng quát (GLMM) đặt hiệu ứng ngẫu nhiên đó trong một GLM. Bates, Mächler, Bolker và Walker (2015) làm stack tính toán (`lme4`) mà hầu hết các lĩnh vực ứng dụng vẫn gọi. Yu, Grieco, Holmes, Xu và đồng nghiệp (2022) lập luận, trong một primer thần kinh học chuyển nguyên văn sang dữ liệu sản phẩm theo phiên, rằng coi đo lặp như các hàng OLS độc lập là lỗi thống kê mặc định; mô hình hỗn hợp là bản sửa.

Với một nền tảng thí nghiệm, quy tắc thực hành là: nếu xử lý được gán ở cấp người dùng, hãy phân tích ở cấp người dùng (một kết cục gộp mỗi người) *hoặc* khớp mô hình hỗn hợp với hiệu ứng ngẫu nhiên người dùng. Phân tích sự kiện thô bằng OLS làm thấp sai số chuẩn và là thất bại hợp lệ Chương 06 mặc áo hồi quy Chương 07.

## Hiệu chỉnh hồi quy và CUPED

Ngẫu nhiên hóa đã xác định hiệu ứng xử lý trung bình (ATE). Hồi quy được dùng để *giảm phương sai*, không để sửa nhiễu. Deng, Xu, Kohavi và Walker (2013) đưa CUPED (Controlled-experiment Using Pre-Experiment Data): thay kết cục $$Y$$ bằng

$$Y^{\mathrm{cv}} = Y - \theta X, \qquad \theta = \frac{\mathrm{Cov}(Y,X)}{\mathrm{Var}(X)},$$

trong đó $$X$$ là hiệp biến trước thí nghiệm, thường là cùng metric tuần trước. ATE trên $$Y^{\mathrm{cv}}$$ bằng ATE trên $$Y$$, và

$$\mathrm{Var}(Y^{\mathrm{cv}}) = \mathrm{Var}(Y)\,(1-\rho_{YX}^2).$$

Tương quan 0,7 giảm một nửa phương sai. Đó là biến kiểm soát từ Monte Carlo, không phải thủ thuật nhân quả mới.

Lin (2013) làm chính xác khuyến nghị OLS tương tác đầy đủ: gồm xử lý, hiệp biến, *và* tương tác xử lý-hiệp biến, rồi dùng sai số chuẩn dị phương sai (sandwich). Reluga, Ye và Zhao (2022) so sánh thống nhất các estimator ANOVA (hiệu trung bình), ANCOVA, và tương tác đầy đủ (“ANHECOVA”), lấy lại sự thật rằng ANCOVA *không* trội đều so với hiệu trung bình thô, trong khi estimator có tương tác thì có, về mặt tiệm cận, dưới ngẫu nhiên hóa hoàn toàn.

Deng và đồng tác giả (2023) xem lại CUPED sau mười năm và biện hộ cho góc nhìn *augmentation* phủ metric tỷ lệ và, thận trọng, hiệp biến trong thí nghiệm. Các bài 2022–2024 (CUPAC, CUPED chỉnh bằng ML, hiệu chỉnh phi tuyến cross-fit) là cùng ý tưởng Chương 07 với hồi quy linh hoạt hơn cho $$E[Y \mid X]$$.

```python
import numpy as np

def cuped_ate(y, t, x):
    """
    Hiệu trung bình trên kết cục đã chỉnh CUPED.

    Args:
        y: kết cục
        t: xử lý 0/1
        x: hiệp biến trước thí nghiệm (cùng độ dài)

    Returns:
        (ate_raw, ate_cuped, theta, var_reduction)
    """
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    y_cv = y - theta * (x - x.mean())
    ate_raw = y[t == 1].mean() - y[t == 0].mean()
    ate_cv = y_cv[t == 1].mean() - y_cv[t == 0].mean()
    reduction = 1 - np.var(y_cv, ddof=1) / np.var(y, ddof=1)
    return ate_raw, ate_cv, theta, reduction
```

Căn giữa $$X$$ giữ intercept đọc được; nó không đổi ATE. Trong thí nghiệm ngẫu nhiên bạn **không** cần double ML. CUPED thường hoặc hồi quy Lin là đủ, và đó là thứ hầu hết nền tảng chạy.

## Double machine learning khi gán là quan sát

Nếu xử lý $$T$$ không ngẫu nhiên, hồi quy

$$Y = \alpha + \tau T + X^\top \beta + \varepsilon$$

ước lượng $$\tau$$ nhân quả chỉ khi không còn nhiễu và $$E[Y \mid T,X]$$ được đặc tả đúng. ML có regularise dùng ngây thơ làm plug-in cho các hàm nhiễu $$E[Y \mid X]$$ và $$E[T \mid X]$$ để lại một chệch regularisation không biến mất ở cấp $$n^{-1/2}$$.

Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey và Robins (2018) sửa bằng *double/debiased ML*: ước lượng hai hàm nhiễu với cross-fitting, lập phần dư

$$\tilde{Y} = Y - \hat{E}[Y \mid X], \qquad \tilde{T} = T - \hat{E}[T \mid X],$$

và hồi quy $$\tilde{Y}$$ theo $$\tilde{T}$$. Tính trực giao Neyman khiến moment mục tiêu không nhạy với sai nhiễu bậc một; cross-fitting giết chệch overfitting. Thư viện EconML của Microsoft là bản triển khai sản xuất mà các nhóm ứng dụng thực sự gọi.

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

def dml_ate(y, t, X, n_splits=5, random_state=0):
    """
    DML tuyến tính riêng phần, cross-fit, cho ATE vô hướng.
    """
    y, t = np.asarray(y, float), np.asarray(t, float)
    n = len(y)
    y_res = np.zeros(n)
    t_res = np.zeros(n)
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train, test in folds.split(X):
        my = GradientBoostingRegressor(random_state=random_state)
        mt = GradientBoostingRegressor(random_state=random_state)
        my.fit(X[train], y[train])
        mt.fit(X[train], t[train])
        y_res[test] = y[test] - my.predict(X[test])
        t_res[test] = t[test] - mt.predict(X[test])
    tau = np.dot(t_res, y_res) / np.dot(t_res, t_res)
    return tau
```

Athey và Imbens (2019) khảo sát vì sao các nhà kinh tế—và, cùng toán đó, nhà khoa học dữ liệu trong marketplace—cần bộ công cụ này. Rừng nhân quả (Wager và Athey, 2018) mở từ ATE vô hướng sang hiệu ứng dị biệt. Chương hồi quy là nhà đúng: bước cuối vẫn là một hồi quy, trên phần dư.

## Dùng gì khi nào

| Bối cảnh | Công cụ | Vì sao |
|---|---|---|
| Thí nghiệm ngẫu nhiên, metric Gauss | Hiệu trung bình hoặc CUPED / Lin | Xác định nhờ thiết kế; hồi quy chỉ mua phương sai |
| Thí nghiệm ngẫu nhiên, đếm hoặc nhị phân | GLM hoặc biến đổi, rồi CUPED trên phần dư GLM | Khớp likelihood với kết cục |
| Đo lặp / người dùng lồng nhau | Gộp về đơn vị gán, hoặc GLMM | Độc lập sai ở cấp sự kiện |
| Xử lý quan sát, nhiều hiệp biến | Double ML hoặc rừng nhân quả | Nếu không, chệch regularisation |

## Thách thức và mở rộng

CUPED với hiệp biến *sau* xử lý có thể trừ một phần hiệu ứng; bài 2013 nói rõ $$X$$ phải trước thí nghiệm. Mô hình hỗn hợp với ít cụm cho sai số chuẩn lạc quan. Double ML giả định không còn nhiễu chưa đo, điều không thư viện nào kiểm tra hộ. GLM lệch hiệu chỉnh khi thiếu tương tác; mô hình Tweedie bỏ offset mùa sẽ trông chính xác và sai.

Mở rộng gồm targeted learning, học chính sách từ bandit đã ghi, và kết hợp CUPED với kiểm định tuần tự Chương 06.

## Bài tập

**Bài 1 (khái niệm).** Một nhóm hồi quy doanh thu tuần theo dummy xử lý và hai mươi hiệp biến sau nhấp. Họ trích hệ số xử lý như “lift nhân quả.” Giả định nào của CUPED/Lin họ đã phá, và họ có thể đã ước lượng estimand nào thay thế?

**Bài 2 (tính toán).** Mô phỏng thí nghiệm ngẫu nhiên $$Y = 2 + 0.05 T + 0.8 X + \varepsilon$$, $$X \sim N(0,1)$$ quan sát trước gán. So sánh RMSE của ATE thô và của `cuped_ate` trên 500 lần lặp. Rồi thay $$X$$ bằng biến sau xử lý $$X_{\mathrm{post}} = X + 0.5 T$$. Ước lượng điểm CUPED đi đâu?

**Bài 3 (tính toán).** Dùng `dml_ate`, sinh dữ liệu quan sát trong đó $$T$$ phụ thuộc $$X$$ và $$Y$$ phụ thuộc cả hai. So sánh OLS của $$Y$$ theo $$(T,X)$$, OLS của $$Y$$ theo chỉ $$T$$, và DML. Lặp lại với $$X$$ sai đặc tả (bỏ một confounder). Phương pháp nào thất bại thầm?

**Bài 4 (mở).** Khớp GLM Poisson hoặc Tweedie trên một tập đếm hoặc bảo hiểm công khai. So sánh kỳ vọng dự đoán và đồ thị phần dư với mô hình tuyến tính trên đếm thô. Rồi thêm một nhân tố nhóm (vùng, người dùng, hoặc năm hợp đồng) và thảo luận xem mô hình hỗn hợp có đáng dùng không.

## Tài liệu tham khảo

Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015). Fitting linear mixed-effects models using lme4. *Journal of Statistical Software, 67*(1), 1–48. [https://doi.org/10.18637/jss.v067.i01](https://doi.org/10.18637/jss.v067.i01)

Cheng, H.-T., Koc, L., Harmsen, J., Shaked, T., Chandra, T., Aradhye, H., … Shah, H. (2016). Wide & Deep learning for recommender systems. Trong *Proceedings of the 1st Workshop on Deep Learning for Recommender Systems* (tr. 7–10). [https://doi.org/10.1145/2988450.2988454](https://doi.org/10.1145/2988450.2988454)

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal, 21*(1), C1–C68. [https://doi.org/10.1111/ectj.12097](https://doi.org/10.1111/ectj.12097)

Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). Improving the sensitivity of online controlled experiments by utilizing pre-experiment data. Trong *WSDM 2013* (tr. 123–132). [https://doi.org/10.1145/2433396.2433413](https://doi.org/10.1145/2433396.2433413)

Deng, A., et al. (2023). From augmentation to decomposition: A new look at CUPED in 2023. [https://arxiv.org/abs/2312.02935](https://arxiv.org/abs/2312.02935)

Lin, W. (2013). Agnostic notes on regression adjustments to experimental data: Reexamining Freedman’s critique. *The Annals of Applied Statistics, 7*(1), 295–318.

McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (tái bản 2). Chapman & Hall. Chuyên khảo GLM vẫn được trích dẫn.

Reluga, K., Ye, T., & Zhao, Q. (2022). A unified analysis of regression adjustment in randomized experiments. [https://arxiv.org/abs/2210.04360](https://arxiv.org/abs/2210.04360)

Yu, Z., Grieco, S. F., Holmes, T. C., Xu, X., et al. (2022). Beyond t-test and ANOVA: Applications of mixed-effects models for more rigorous statistical analysis in neuroscience research. *Neuron, 110*(1), 21–35. [https://doi.org/10.1016/j.neuron.2021.10.030](https://doi.org/10.1016/j.neuron.2021.10.030)
