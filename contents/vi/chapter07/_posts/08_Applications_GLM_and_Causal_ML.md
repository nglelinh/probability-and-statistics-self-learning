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

Một công ty công nghệ muốn kiểm chứng liệu thuật toán tìm kiếm mới có làm tăng doanh thu trung bình trên mỗi người dùng thêm **1%** hay không. Nếu dùng kiểm định hai mẫu thông thường, độ nhiễu tự nhiên giữa các người dùng (người chi 50.000 đ, người chi 50 triệu đ) lớn đến mức thí nghiệm phải kéo dài suốt **8 tuần** mới đủ lực kiểm định. Nhưng chỉ bằng một dòng công thức tinh tế mang tên **CUPED** — tận dụng lịch sử chi tiêu *trước thí nghiệm* $$X$$ của chính người dùng đó làm biến kiểm soát tuyến tính — phương sai giảm ngay gần một nửa, rút thời gian thí nghiệm từ 8 tuần xuống còn **4 tuần** mà vẫn giữ nguyên độ tin cậy!

Bài đọc tùy chọn này nối thẳng mô hình tuyến tính cổ điển của Chương 07 tới ba vũ khí thực chiến của khoa học dữ liệu và kinh tế lượng hiện đại: **Mô hình Tuyến tính Tổng quát (GLM)** cho dữ liệu đếm/lệch, **Mô hình Hỗn hợp (Mixed Effects)** cho dữ liệu lặp theo người dùng, và **Hiệu chỉnh hồi quy (CUPED) cùng Double/Debiased Machine Learning (DML)** cho bài toán ước lượng tác động nhân quả.

---

## Kiến thức cần nhớ: Bốn bước tiến hóa từ OLS cổ điển sang Suy diễn Nhân quả hiện đại

Khi dữ liệu thực tế phá vỡ các giả định đẹp đẽ của OLS, thống kê hiện đại mở rộng theo bốn nhánh chính:

1. **Mô hình Tuyến tính Tổng quát (GLM):** Thống nhất hồi quy tuyến tính (Gauss), hồi quy logistic (Bernoulli–Logit) và hồi quy đếm (Poisson–Log) dưới một khung duy nhất: phân phối thuộc họ mũ + hàm liên kết $$g(\mathbb{E}[Y\mid X]) = X\beta$$.
2. **Mô hình Hỗn hợp (Mixed-Effects / Hierarchical Models):** Khi một người dùng phát sinh 50 phiên truy cập, 50 quan sát đó *không độc lập* nhau. Mô hình hỗn hợp tách biến thiên thành hiệu ứng cố định (chung cho toàn hệ thống) và hiệu ứng ngẫu nhiên (riêng từng người dùng/cửa hàng) để tránh sai số chuẩn bị đánh giá quá lạc quan.
3. **Giảm phương sai trong thí nghiệm A/B (CUPED & Hồi quy Lin):** Dùng biến hiệp phương sai đo *trước thời điểm phân nhóm ngẫu nhiên* $$X$$ để trừ bớt phần biến thiên có thể dự đoán trước của $$Y$$ — giảm phương sai đúng theo hệ số $$(1 - \rho^2)$$ mà tuyệt đối không gây độ chệch.
4. **Double / Debiased Machine Learning (DML):** Khi không thể chạy A/B (dữ liệu quan sát có nhiều biến gây nhiễu phi tuyến $$W$$), ta dùng học máy dự đoán phần dư của cả kết cục $$\tilde{Y} = Y - \hat{m}(W)$$ lẫn biến xử lý $$\tilde{T} = T - \hat{e}(W)$$ (kết hợp *cross-fitting*), rồi hồi quy $$\tilde{Y}$$ theo $$\tilde{T}$$ theo đúng Định lý Frisch–Waugh–Lovell để khử độ chệch điều chuẩn.

---

## Từ Đường Thẳng Đến Mô Hình Tuyến Tính Tổng Quát (GLM)

Mô hình OLS giả định:

$$Y = X\beta + \varepsilon, \qquad E[\varepsilon \mid X] = 0.$$

Mô hình tuyến tính tổng quát (GLM — McCullagh và Nelder, 1989) giữ nguyên bộ dự đoán tuyến tính $$\eta = X\beta$$ nhưng mở rộng hai thành phần: một hàm liên kết (link function) $$g(E[Y \mid X]) = \eta$$ và một phân phối thuộc họ mũ kèm hàm phương sai tương ứng. Nhờ đó, mô hình Poisson với liên kết log cho dữ liệu đếm, mô hình Tweedie cho dữ liệu bồi thường bảo hiểm có xác suất tập trung tại 0, và hồi quy logistic đều nằm trong cùng một khung thống nhất. Cụ thể, hồi quy logistic là GLM với phân phối Bernoulli và hàm liên kết logit, trong đó ước lượng hợp lý cực đại (MLE) trùng với nghiệm cực tiểu hóa hàm mất mát cross-entropy đã học ở Chương 05.

Trong thực tế công nghiệp, GLM vẫn được sử dụng rộng rãi ở quy mô lớn vì xác suất dự đoán được hiệu chỉnh tốt, chi phí tính toán thấp và dễ kiểm toán. Nhiều hệ thống gợi ý và quảng cáo kết hợp một GLM (hoặc thành phần tuyến tính "wide") song song với mạng nơ-ron sâu — kiến trúc mà Cheng et al. (2016) gọi là *Wide & Deep*: phần tuyến tính ghi nhớ các tương tác đặc trưng thưa, còn phần sâu đảm nhiệm khả năng khái quát hóa. Dưới góc độ thống kê, GLM không phải phiên bản thu gọn của mạng nơ-ron; nó xác định trực tiếp hàm hợp lý mà mạng đang xấp xỉ, đồng thời cung cấp bảng hệ số có thể diễn giải khi cơ quan quản lý hoặc chuyên gia định phí bảo hiểm yêu cầu.

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

## Mô Hình Hỗn Hợp Khi Dữ Liệu Người Dùng Không Độc Lập

Nhật ký thí nghiệm A/B và dữ liệu sản phẩm theo thời gian thường vi phạm giả định độc lập giữa các quan sát: cùng một người dùng tạo ra nhiều phiên truy cập, hoặc cùng một truy vấn xuất hiện lặp lại qua nhiều ngày. Mô hình tuyến tính hỗn hợp (linear mixed-effects model) biểu diễn cấu trúc này dưới dạng:

$$Y_{ij} = x_{ij}^\top \beta + u_i + \varepsilon_{ij}, \qquad u_i \sim \mathcal{N}(0, \sigma_u^2),$$

trong đó hiệu ứng riêng của người dùng $$u_i$$ là một hệ số chặn ngẫu nhiên (random intercept). Khi đặt hiệu ứng ngẫu nhiên đó vào trong khuôn khổ GLM, ta thu được mô hình tuyến tính hỗn hợp tổng quát (GLMM). Bates, Mächler, Bolker và Walker (2015) đã phát triển bộ công cụ tính toán (`lme4`) được dùng phổ biến trong nhiều lĩnh vực ứng dụng. Yu, Grieco, Holmes, Xu và đồng nghiệp (2022) cũng chỉ ra (trong bối cảnh khoa học thần kinh, nhưng áp dụng trực tiếp sang dữ liệu sản phẩm theo phiên) rằng việc xem các phép đo lặp như các quan sát OLS độc lập là một sai sót thống kê phổ biến, và mô hình hỗn hợp là cách khắc phục phù hợp.

Trên các nền tảng thí nghiệm, quy tắc thực hành rõ ràng là: nếu xử lý (treatment) được gán ở cấp người dùng, hãy phân tích ở cấp người dùng (tổng hợp thành một giá trị kết cục cho mỗi người) *hoặc* sử dụng mô hình hỗn hợp có hiệu ứng ngẫu nhiên theo người dùng. Áp dụng OLS trực tiếp lên dữ liệu sự kiện thô sẽ đánh giá thấp sai số chuẩn — thực chất là lỗi về tính hợp lệ của suy diễn thống kê từ Chương 06, chỉ khác là xuất hiện trong khuôn khổ hồi quy của Chương 07.

## Hiệu Chỉnh Hồi Quy và CUPED

Trong thí nghiệm ngẫu nhiên, chính quá trình ngẫu nhiên hóa đã đảm bảo nhận diện đúng hiệu ứng xử lý trung bình (ATE). Hồi quy trong bối cảnh này được dùng để *giảm phương sai*, chứ không phải để khử nhiễu gây nhiễu (confounding). Deng, Xu, Kohavi và Walker (2013) đề xuất phương pháp CUPED (*Controlled-experiment Using Pre-Experiment Data*), thay thế biến kết cục $$Y$$ bằng biến đã hiệu chỉnh:

$$Y^{\mathrm{cv}} = Y - \theta X, \qquad \theta = \frac{\mathrm{Cov}(Y,X)}{\mathrm{Var}(X)},$$

trong đó $$X$$ là hiệp biến đo trước thí nghiệm (thường là chính chỉ số mục tiêu ở tuần trước đó). ATE tính trên $$Y^{\mathrm{cv}}$$ vẫn bằng ATE tính trên $$Y$$, trong khi phương sai giảm theo tỷ lệ:

$$\mathrm{Var}(Y^{\mathrm{cv}}) = \mathrm{Var}(Y)\,(1-\rho_{YX}^2).$$

Chẳng hạn, mức tương quan 0,7 giúp giảm gần một nửa phương sai (49%). Về bản chất, đây là kỹ thuật biến kiểm soát (control variate) trong phương pháp Monte Carlo, chứ không phải một giả định nhân quả mới.

Lin (2013) chuẩn hóa khuyến nghị về mô hình OLS có tương tác đầy đủ: đưa vào mô hình cả biến xử lý, hiệp biến đã căn giữa *và* số hạng tương tác giữa xử lý với hiệp biến, kết hợp sai số chuẩn bền vững với phương sai thay đổi (sandwich standard errors). Reluga, Ye và Zhao (2022) phân tích thống nhất các ước lượng ANOVA (hiệu trung bình), ANCOVA và mô hình tương tác đầy đủ ("ANHECOVA"), khẳng định lại kết quả rằng ANCOVA *không* luôn có phương sai nhỏ hơn hiệu trung bình thô, trong khi ước lượng có tương tác đảm bảo cải thiện (hoặc ít nhất không kém hơn) về mặt tiệm cận dưới ngẫu nhiên hóa hoàn toàn.

Mười năm sau bài báo gốc, Deng và đồng tác giả (2023) nhìn lại CUPED và đề xuất góc nhìn bổ trợ (*augmentation*) bao quát cả các chỉ số dạng tỷ lệ (ratio metrics) và — với điều kiện thận trọng — các hiệp biến thu thập trong thí nghiệm. Các nghiên cứu giai đoạn 2022–2024 (như CUPAC, CUPED kết hợp học máy, hay hiệu chỉnh phi tuyến dùng cross-fitting) đều kế thừa cùng nguyên lý của Chương 07 bằng cách dùng các mô hình linh hoạt hơn để ước lượng $$E[Y \mid X]$$.

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

Việc căn giữa $$X$$ giúp hệ số chặn dễ diễn giải hơn mà không làm thay đổi ATE. Trong thí nghiệm ngẫu nhiên, bạn **không** cần đến double ML; CUPED chuẩn hoặc hồi quy tương tác theo Lin là đủ và cũng là phương pháp mà phần lớn các nền tảng thí nghiệm đang vận hành.

## Double Machine Learning Khi Dữ Liệu Là Quan Sát

Khi biến xử lý $$T$$ không được phân bổ ngẫu nhiên, mô hình hồi quy:

$$Y = \alpha + \tau T + X^\top \beta + \varepsilon$$

chỉ ước lượng đúng hiệu ứng nhân quả $$\tau$$ nếu không còn biến gây nhiễu chưa quan sát và dạng hàm $$E[Y \mid T,X]$$ được đặc tả chính xác. Nếu đưa trực tiếp các mô hình học máy có điều chuẩn vào làm ước lượng thay thế (plug-in) cho hai hàm nhiễu $$E[Y \mid X]$$ và $$E[T \mid X]$$, độ chệch do điều chuẩn sẽ không triệt tiêu đủ nhanh ở cấp $$n^{-1/2}$$.

Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey và Robins (2018) giải quyết vấn đề này bằng phương pháp *double/debiased ML*: ước lượng hai hàm nhiễu bằng kỹ thuật chia mẫu chéo (*cross-fitting*), tạo hai phần dư:

$$\tilde{Y} = Y - \hat{E}[Y \mid X], \qquad \tilde{T} = T - \hat{E}[T \mid X],$$

rồi hồi quy $$\tilde{Y}$$ theo $$\tilde{T}$$. Tính trực giao Neyman giúp điều kiện moment mục tiêu không nhạy cảm với sai số ước lượng bậc một của các hàm nhiễu, còn cross-fitting loại bỏ độ chệch do quá khớp. Trong thực tế, thư viện EconML của Microsoft là một bản cài đặt phổ biến được nhiều nhóm ứng dụng sử dụng.

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

Athey và Imbens (2019) giải thích vì sao các nhà kinh tế lượng — và cùng cơ sở toán học đó, các nhà khoa học dữ liệu làm việc trên các nền tảng thị trường (marketplace) — cần đến nhóm phương pháp này. Rừng nhân quả (causal forests — Wager và Athey, 2018) tiếp tục mở rộng từ ATE vô hướng sang ước lượng hiệu ứng xử lý dị biệt (heterogeneous treatment effects). Chương về hồi quy là vị trí phù hợp cho nội dung này, bởi bước ước lượng cuối cùng vẫn là một phép hồi quy thực hiện trên các phần dư.

## Lựa Chọn Công Cụ Theo Bối Cảnh

| Bối cảnh | Công cụ | Lý do |
|---|---|---|
| Thí nghiệm ngẫu nhiên, chỉ số gần chuẩn | Hiệu trung bình hoặc CUPED / Lin | Nhận diện nhân quả nhờ thiết kế; hồi quy chỉ dùng để giảm phương sai |
| Thí nghiệm ngẫu nhiên, dữ liệu đếm hoặc nhị phân | GLM hoặc phép biến đổi, sau đó CUPED trên phần dư GLM | Hàm hợp lý phù hợp với bản chất biến kết cục |
| Đo lặp / quan sát lồng trong người dùng | Tổng hợp về đơn vị gán, hoặc dùng GLMM | Giả định độc lập bị vi phạm ở cấp sự kiện |
| Dữ liệu quan sát, nhiều hiệp biến | Double ML hoặc rừng nhân quả | Tránh độ chệch do điều chuẩn khi dùng ML trực tiếp |

## Thách Thức và Hướng Mở Rộng

Dùng CUPED với hiệp biến đo *sau* khi xử lý có thể hấp thụ một phần hiệu ứng nhân quả và làm chệch ước lượng; Deng et al. (2013) nêu rõ $$X$$ phải là dữ liệu trước thí nghiệm. Mô hình hỗn hợp khi số lượng cụm quá ít thường cho sai số chuẩn thấp hơn thực tế. Double ML vẫn đòi hỏi giả định không còn biến gây nhiễu chưa đo lường — điều mà không thư viện phần mềm nào có thể tự kiểm tra. Ngoài ra, GLM có thể lệch hiệu chỉnh nếu thiếu các số hạng tương tác quan trọng; chẳng hạn mô hình Tweedie bỏ qua yếu tố mùa vụ (offset) có thể cho chỉ số khớp bề ngoài tốt nhưng dự báo sai lệch.

Các hướng mở rộng tiếp theo bao gồm học có mục tiêu (targeted learning), học chính sách từ dữ liệu bandit đã ghi nhận (off-policy learning), và kết hợp CUPED với kiểm định tuần tự ở Chương 06.

---

### Ví dụ tính tay làm từng bước: Tính mức tiết kiệm thời gian thử nghiệm A/B nhờ CUPED

Giả sử trong một thử nghiệm A/B, doanh thu người dùng trong tuần thí nghiệm $$Y$$ có độ lệch chuẩn $$\sigma_Y = 20$$ (phương sai $$\mathrm{Var}(Y) = 400$$). Ta có thêm dữ liệu doanh thu tuần *trước* thí nghiệm $$X$$ của chính những người dùng đó, với cùng độ lệch chuẩn $$\sigma_X = 20$$ và hệ số tương quan $$\rho = \mathrm{Corr}(X, Y) = 0{,}70$$:

1. **Tính hệ số điều chỉnh tối ưu CUPED:**
   $$\theta^* = \frac{\mathrm{Cov}(Y, X)}{\mathrm{Var}(X)} = \rho\frac{\sigma_Y}{\sigma_X} = 0{,}70 \times \frac{20}{20} = 0{,}70.$$
2. **Xây dựng biến kết cục đã giảm phương sai:**
   $$Y_{\text{CUPED}} = Y - 0{,}70\,(X - \bar{X}).$$
   Vì $$X$$ được đo *trước* khi chia nhóm ngẫu nhiên nên $$\mathbb{E}[X \mid T=1] = \mathbb{E}[X \mid T=0]$$, đảm bảo hiệu trung bình $$\mathbb{E}[Y_{\text{CUPED}} \mid T=1] - \mathbb{E}[Y_{\text{CUPED}} \mid T=0]$$ bằng **đúng** tác động nhân quả thật (hoàn toàn không chệch!).
3. **Tính phương sai mới của $$Y_{\text{CUPED}}$$:**
   $$\mathrm{Var}(Y_{\text{CUPED}}) = \mathrm{Var}(Y)(1 - \rho^2) = 400\,(1 - 0{,}70^2) = 400 \times 0{,}51 = 204.$$
4. **Quy đổi ra thời gian chạy thí nghiệm:**
   Phương sai giảm **49%** đồng nghĩa ta chỉ cần **51% cỡ mẫu** để đạt cùng sai số chuẩn và cùng lực kiểm định — một thí nghiệm vốn mất **8 tuần** nay chỉ cần khoảng **4,1 tuần**!

### Giải thích: Bốn nguyên tắc quan trọng khi dùng hồi quy cho nhân quả và sản phẩm

1. **Biến kiểm soát trong CUPED bắt buộc phải đo TRƯỚC khi can thiệp:** Tuyệt đối không đưa vào CUPED một biến bị ảnh hưởng bởi chính tính năng đang thử nghiệm (*post-treatment bias*), nếu không ước lượng tác động nhân quả sẽ bị bóp méo.
2. **Nhận diện đúng cấu trúc phân cụm dữ liệu:** Nếu chia nhóm A/B theo *người dùng* nhưng lại phân tích ở cấp độ *lượt nhấp chuột* mà không dùng mô hình hỗn hợp hoặc sai số chuẩn theo cụm (cluster-robust SE), p-value sẽ nhỏ giả tạo hàng chục lần.
3. **Hiểu vì sao DML cần cả hai bước "khử phần dư" lẫn *cross-fitting*:** Nếu chỉ nhét biến xử lý $$T$$ vào chung một mô hình Lasso/Random Forest với $$W$$, điều chuẩn sẽ co nhầm hệ số nhân quả về 0. Hồi quy phần dư $$\tilde{Y}$$ theo phần dư $$\tilde{T}$$ trên các fold tách biệt khử sạch sai số bậc nhất này.
4. **Mọi phương pháp dữ liệu quan sát đều mang giả định không kiểm chứng được:** DML xử lý xuất sắc các biến gây nhiễu *quan sát được* $$W$$, nhưng không thể tự cứu bạn nếu còn tồn tại biến gây nhiễu *ẩn* chưa được đo lường.

**Kiểm tra nhanh.** Trong ví dụ CUPED trên, nếu tương quan giữa doanh thu trước và trong thí nghiệm tăng lên tới $$\rho = 0{,}80$$, phương sai giảm bao nhiêu phần trăm và thí nghiệm 8 tuần có thể rút ngắn xuống còn khoảng mấy tuần?

---

## Video tham khảo

- [Explaining generalized linear models (GLMs) | VNT #15](https://www.youtube.com/watch?v=1fZeBafD1nw) — Very Normal, tiếng Anh. **Xem trước.** Giới thiệu mô hình tuyến tính tổng quát giữ nguyên bộ dự đoán tuyến tính nhưng mở rộng hàm liên kết và phân phối họ mũ, bao gồm hai trường hợp logistic và Poisson.
- [Simple Explanation of Mixed Models (Hierarchical Linear Models, Multilevel Models)](https://www.youtube.com/watch?v=5tOifM51ZOk) — Simplistics (QuantPsych), tiếng Anh. Giải thích hiệu ứng ngẫu nhiên khi nhiều quan sát thuộc cùng một cá thể không độc lập với nhau.
- [Double Machine Learning for Causal and Treatment Effects](https://www.youtube.com/watch?v=eHOjmyoPCFU) — Becker Friedman Institute University of Chicago, tiếng Anh. Bài giảng của Chernozhukov về double/debiased machine learning với điều kiện moment trực giao và cross-fitting.

## Bài tập

**Bài 1 (khái niệm).** Một nhóm hồi quy doanh thu tuần theo biến giả xử lý và 20 hiệp biến đo sau thời điểm nhấp chuột, rồi xem hệ số của biến xử lý là "tác động nhân quả". Họ đã vi phạm giả định nào của CUPED/Lin, và đại lượng thực sự họ đang ước lượng có thể lệch theo hướng nào?

**Bài 2 (tính toán).** Mô phỏng thí nghiệm ngẫu nhiên $$Y = 2 + 0.05 T + 0.8 X + \varepsilon$$ với $$X \sim N(0,1)$$ quan sát trước khi gán xử lý. So sánh RMSE của ATE thô và của `cuped_ate` qua 500 lần lặp. Sau đó thay $$X$$ bằng biến sau xử lý $$X_{\mathrm{post}} = X + 0.5 T$$ và quan sát sự thay đổi của ước lượng điểm CUPED.

**Bài 3 (tính toán).** Sử dụng `dml_ate`, sinh dữ liệu quan sát trong đó $$T$$ phụ thuộc vào $$X$$ và $$Y$$ phụ thuộc vào cả $$T$$ lẫn $$X$$. So sánh OLS của $$Y$$ theo $$(T,X)$$, OLS của $$Y$$ chỉ theo $$T$$, và DML. Lặp lại thí nghiệm khi $$X$$ bị đặc tả thiếu (bỏ sót một biến gây nhiễu) và nhận xét phương pháp nào cho kết quả sai lệch mà không có cảnh báo.

**Bài 4 (mở).** Khớp một GLM Poisson hoặc Tweedie trên một bộ dữ liệu đếm hoặc bảo hiểm công khai. So sánh giá trị kỳ vọng dự đoán và đồ thị phần dư với mô hình hồi quy tuyến tính trên dữ liệu đếm thô. Sau đó bổ sung một nhân tố nhóm (vùng, người dùng hoặc năm hợp đồng) và thảo luận liệu mô hình hỗn hợp có cần thiết hay không.

## Tài liệu tham khảo

Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015). Fitting linear mixed-effects models using lme4. *Journal of Statistical Software, 67*(1), 1–48. [https://doi.org/10.18637/jss.v067.i01](https://doi.org/10.18637/jss.v067.i01)

Cheng, H.-T., Koc, L., Harmsen, J., Shaked, T., Chandra, T., Aradhye, H., … Shah, H. (2016). Wide & Deep learning for recommender systems. Trong *Proceedings of the 1st Workshop on Deep Learning for Recommender Systems* (tr. 7–10). [https://doi.org/10.1145/2988450.2988454](https://doi.org/10.1145/2988450.2988454)

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal, 21*(1), C1–C68. [https://doi.org/10.1111/ectj.12097](https://doi.org/10.1111/ectj.12097)

Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). Improving the sensitivity of online controlled experiments by utilizing pre-experiment data. Trong *WSDM 2013* (tr. 123–132). [https://doi.org/10.1145/2433396.2433413](https://doi.org/10.1145/2433396.2433413)

Deng, A., et al. (2023). From augmentation to decomposition: A new look at CUPED in 2023. [https://arxiv.org/abs/2312.02935](https://arxiv.org/abs/2312.02935)

Lin, W. (2013). Agnostic notes on regression adjustments to experimental data: Reexamining Freedman’s critique. *The Annals of Applied Statistics, 7*(1), 295–318.

McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (tái bản lần 2). Chapman & Hall. Chuyên khảo kinh điển về GLM.

Reluga, K., Ye, T., & Zhao, Q. (2022). A unified analysis of regression adjustment in randomized experiments. [https://arxiv.org/abs/2210.04360](https://arxiv.org/abs/2210.04360)

Yu, Z., Grieco, S. F., Holmes, T. C., Xu, X., et al. (2022). Beyond t-test and ANOVA: Applications of mixed-effects models for more rigorous statistical analysis in neuroscience research. *Neuron, 110*(1), 21–35. [https://doi.org/10.1016/j.neuron.2021.10.030](https://doi.org/10.1016/j.neuron.2021.10.030)
