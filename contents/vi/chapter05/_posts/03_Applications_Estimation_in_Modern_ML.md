---
layout: post
title: "05-03-00 Tùy chọn: Ước lượng trong học máy hiện đại"
chapter: "05"
order: 3
owner: nglelinh
lang: vi
categories:
- chapter05
lesson_type: optional
---

Bài tùy chọn này không thay thế lý thuyết ước lượng điểm và ước lượng khoảng của chương. Nó cho thấy các estimator đó chính là động cơ thống kê bên trong các hệ thống khoa học máy tính và khoa học dữ liệu đương đại: maximum likelihood trong mạng sâu, method of moments cho mô hình hỗn hợp, empirical Bayes trong xếp hạng sản xuất, và các khoảng hiện đại khi không có sai số chuẩn dạng đóng.

Sau bài học, bạn nhận ra huấn luyện cross-entropy chính là maximum likelihood, giải thích được vì sao khớp moment vẫn được dùng cho hỗn hợp Gauss, cài được bộ co Beta–Binomial kiểu empirical Bayes như trên tìm kiếm thương mại điện tử, và phân biệt khoảng bootstrap với tập conformal.

Tiền đề là các bài bắt buộc về ước lượng điểm (chệch, phương sai, MLE, method of moments) và ước lượng khoảng (khoảng tin cậy, bootstrap), cùng Python/NumPy cơ bản. Không thêm tiên đề xác suất mới.

---

## Vì sao ước lượng rời giáo trình

Một dịch vụ xếp hạng, một mô hình ngôn ngữ, và một pipeline phân cụm đều giải đúng bài bạn đã học: từ mẫu $$X_1,\ldots,X_n$$, tạo $$\hat{\theta}$$ và, khi quyết định đắt, một khoảng quanh nó. Điều đổi giữa 2022 và 2026 không phải định nghĩa estimator. Đó là quy mô thực thi, và các ràng buộc kỹ thuật—độ trễ, khởi động lạnh, ngân sách tính toán—quyết định *estimator nào* được chọn.

Ba mẫu sản xuất chiếm ưu thế. Mô hình sâu cực tiểu hóa âm log-likelihood, tức chúng *là* MLE. Mô hình hỗn hợp trong nhận dạng người nói, phân cụm đơn bào và phát hiện bất thường vẫn khớp moment khi likelihood đa cực trị. Hệ thống xếp hạng và quảng cáo co các tỷ lệ nhiễu về một prior ước lượng từ dữ liệu—đúng bước empirical Bayes mà Efron làm rõ và các nền tảng lớn vẫn đưa vào sản xuất.

## Maximum likelihood là cách mạng hiện đại học

Với mẫu i.i.d., MLE cực đại hóa

$$\hat{\theta}_{\mathrm{MLE}} = \arg\max_\theta \sum_{i=1}^n \log p(x_i \mid \theta).$$

Trong bộ phân loại, quan sát là cặp $$(x,y)$$ và $$p(y \mid x,\theta)$$ là softmax. Âm log-likelihood trung bình chính là hàm mất mát cross-entropy trong mọi thư viện deep learning lớn. Huấn luyện mạng bằng stochastic gradient descent vì thế là MLE tính toán: score function $$\nabla_\theta \log p$$ được lan truyền ngược, chứ không được đặt bằng không dưới dạng đóng.

Mô hình ngôn ngữ tự hồi quy lặp lại cùng một đồng nhất theo từng token. Huấn luyện next-token cực đại hóa

$$\sum_t \log p_\theta(w_t \mid w_{<t}),$$

tức MLE của họ categorical có điều kiện. Hoffmann et al. (2022) coi likelihood đó như một đối tượng *thực nghiệm*: họ huấn luyện hơn 400 transformer và chỉ ra rằng, với ngân sách tính toán cố định, mô hình tối ưu theo compute scale số tham số và số token huấn luyện gần như theo cùng tỷ lệ. Nội dung thống kê của kết quả “Chinchilla” là chất lượng của một MLE được quyết định đồng thời bởi chiều của $$\theta$$ và cỡ mẫu. Đó là câu chuyện bias–variance cổ điển, đọc trên một frontier tối ưu theo compute thay vì bảng MSE giáo trình.

Hai lưu ý giữ cho đồng nhất này trung thực. Thứ nhất, regularisation, early stopping và tăng cường dữ liệu khiến trọng số đã huấn luyện là MLE *có phạt*, không phải maximizer thô. Thứ hai, mạng quá tham số hóa có thể nội suy mẫu huấn luyện, nên định lý “MLE tiệm cận chuẩn cấp $$n^{-1/2}$$” không áp dụng theo nghĩa đen. Điều còn lại là *hàm mục tiêu*: thực hành vẫn báo cáo âm log-likelihood trên tập giữ lại, và các luật scale vẫn là luật về likelihood đó.

```python
import numpy as np

def softmax_nll(logits, labels):
    """
    Âm log-likelihood trung bình của mô hình categorical.

    Args:
        logits: mảng (n, k), điểm số lớp chưa chuẩn hóa
        labels: mảng nguyên (n,), giá trị trong {0, ..., k-1}

    Returns:
        NLL trung bình, trùng cross-entropy đa lớp.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return -np.mean(log_probs[np.arange(len(labels)), labels])
```

## Method of moments cho hỗn hợp Gauss

Hỗn hợp Gauss $$K$$ thành phần có mật độ

$$p(x) = \sum_{k=1}^K \pi_k \, \mathcal{N}(x; \mu_k, \Sigma_k).$$

Likelihood không lõm. Expectation–maximisation tìm MLE địa phương và là mặc định của `sklearn.mixture.GaussianMixture`. Method of moments, Pearson đưa vào năm 1894 cho hỗn hợp một chiều hai thành phần, khớp moment thực nghiệm với moment suy ra từ $$(\pi,\mu,\Sigma)$$.

Ở chiều cao, các moment liên quan là tensor. Hsu và Kakade (2013) cùng Anandkumar, Ge, Hsu, Kakade và Telgarsky (2014) chứng minh ba moment đầu của GMM cầu xác định tham số qua phân rã tensor đối xứng, với bảo đảm thời gian đa thức khi các kỳ vọng độc lập tuyến tính. Những bài đó vẫn là trích dẫn mà các giáo trình lý thuyết học dùng.

Điểm mới là tính khả thi. Lập một tensor moment cấp $$d$$ tốn $$O(n^d)$$ bộ nhớ. Pereira, Kileel và Kolda (2022) đưa biểu thức ẩn cho tensor moment của GMM để khớp moment chạy với chi phí $$O(n^2)$$ hoặc $$O(n^3)$$ cho hiệp phương sai tổng quát, và $$O(n)$$ cho hiệp phương sai đường chéo, mà không vật chất hóa tensor. Cùng ý tưởng tiếp tục trong các công trình moment ẩn 2024–2025. Thông điệp thống kê cho chương này không đổi: khi mặt likelihood khó chịu, hãy đẳng thức hóa moment.

Minh họa một chiều lấy lại bước gốc của Pearson.

```python
import numpy as np

def mom_two_gaussians(x):
    """
    Phác thảo method of moments cho hỗn hợp trọng số đều
    0.5 N(-m, 1) + 0.5 N(+m, 1), dùng moment bậc hai.

    Với hỗn hợp hai thành phần tổng quát, khớp từ ba moment
    trở lên và giải đa thức Pearson.
    """
    second = np.mean(x ** 2)
    # Với họ đối xứng này, E[X^2] = m^2 + 1
    m_hat = np.sqrt(max(second - 1.0, 0.0))
    return m_hat
```

Trong sản xuất, ước lượng moment thường *khởi tạo* EM chứ không thay thế nó—tổ hợp thừa hưởng tính xác định toàn cục của MoM và hiệu quả địa phương của EM.

## Empirical Bayes trong xếp hạng sản xuất

Empirical Bayes ước lượng prior từ chính dữ liệu sẽ được cập nhật bằng prior đó. Với tỷ lệ nhấp, mô hình Beta–Binomial là công cụ chính. Nếu mục $$i$$ có $$c_i$$ nhấp trên $$n_i$$ lần hiển thị và prior là $$\mathrm{Beta}(\alpha,\beta)$$,

$$\hat{p}_i = \frac{\alpha + c_i}{\alpha + \beta + n_i}.$$

Kỳ vọng prior $$m = \alpha/(\alpha+\beta)$$ và độ mạnh $$s = \alpha+\beta$$ không phải đoán. Chúng được ước lượng từ tập hợp các mục—bằng khớp kỳ vọng và phương sai của các tỷ lệ quan sát (prior method of moments) hoặc cực đại hóa likelihood biên. Mục mới với $$n_i = 0$$ được hiện ở kỳ vọng prior; mục có $$n_i$$ rất lớn gần như MLE thô $$c_i/n_i$$. Đó là co James–Stein trong lớp sản phẩm.

Han, Castells, Gupta, Xu và Salaka (2022) triển khai mẫu này cho khởi động lạnh trên tìm kiếm sản phẩm Amazon: một mô hình không hành vi cung cấp prior, còn nhấp và mua quan sát cập nhật posterior. Thí nghiệm trực tuyến trên 50 triệu truy vấn tăng impression sản phẩm mới 13,53% và mua sản phẩm mới 11,14%. Công trình liên quan 2023 (EBRank) dùng cùng tách prior/posterior để giảm thiên lệch khai thác trong learning-to-rank.

```python
import numpy as np

def fit_beta_mom(clicks, imps, eps=1e-6):
    """
    Prior Beta bằng method of moments từ một bảng các mục.

    Args:
        clicks, imps: mảng đếm 1 chiều, cùng độ dài

    Returns:
        (alpha, beta) của prior Beta đã khớp
    """
    rates = clicks / np.maximum(imps, 1)
    # Giảm trọng số các mục gần như chưa có impression
    w = imps / imps.sum()
    m = np.average(rates, weights=w)
    v = np.average((rates - m) ** 2, weights=w)
    # Beta kỳ vọng m, phương sai m(1-m)/(s+1) => s = m(1-m)/v - 1
    s = m * (1 - m) / max(v, eps) - 1
    s = max(s, 1.0)
    return s * m, s * (1 - m)

def posterior_ctr(clicks, imps, alpha, beta):
    """Kỳ vọng posterior CTR theo empirical Bayes."""
    return (alpha + clicks) / (alpha + beta + imps)
```

Liên hệ ngược với chương này là theo nghĩa đen. Tham số prior là một ước lượng điểm. Kỳ vọng posterior là một ước lượng điểm khác. Khoảng tin cậy từ posterior Beta, hoặc bootstrap của $$(\hat{\alpha},\hat{\beta})$$, là ước lượng khoảng.

## Khoảng khi không có sai số chuẩn giải tích

Bài ước lượng khoảng bắt buộc xây

$$\bar{x} \pm z_{1-\alpha/2} \frac{s}{\sqrt{n}}$$

rồi khoảng percentile bootstrap. Đó vẫn là công cụ đúng cho trung bình hoặc một phiếm hàm đơn giản. Mô hình sâu phá dẫn xuất: tham số là hàng triệu trọng số, mất mát không lồi, và sai số chuẩn từ Hessian không tính được.

Hai câu trả lời được dùng trong các hệ 2022–2026. Bootstrap (và họ hàng deep ensembles) lấy mẫu lại quá trình huấn luyện và đọc khoảng percentile từ phân phối thực nghiệm của statistic—cùng thuật toán bạn đã cài, với chi phí cao hơn. Conformal prediction, Angelopoulos và Bates (2023) trình bày cho độc giả học máy, bọc *bất kỳ* mô hình đã khớp trong một tập dự đoán $$C(x)$$ thỏa

$$P\bigl(Y_{n+1} \in C(X_{n+1})\bigr) \ge 1-\alpha$$

dưới tính trao đổi được, không giả định mô hình đúng. Split conformal dùng một fold hiệu chỉnh để tính phân vị của điểm không phù hợp; bảo đảm là hữu hạn mẫu. Đó không phải khoảng tin cậy cho $$\theta$$. Đó là khoảng dự đoán cho một đáp số mới, thường đúng thứ một hệ thống triển khai cần.

```python
import numpy as np

def split_conformal_interval(calib_scores, q=0.9):
    """
    Phân vị split-conformal cho phần dư tuyệt đối.

    Args:
        calib_scores: |y - f(x)| trên fold hiệu chỉnh giữ lại
        q: độ phủ mục tiêu, ví dụ 0.9

    Returns:
        Nửa độ rộng phần dư để cộng/trừ vào f(x_new)
    """
    n = len(calib_scores)
    level = np.ceil((n + 1) * q) / n
    return np.quantile(calib_scores, min(level, 1.0), method="higher")
```

## Điều một nhà khoa học dữ liệu nên giữ lại

Chọn estimator khớp *cấu trúc* bài toán, không phải uy tín của lớp mô hình. Nếu mô hình quan sát là họ mũ xác định tốt và bạn tối ưu được, dùng MLE (hoặc bản có regularise). Nếu likelihood đa cực trị hoặc bạn cần bộ khởi tạo được xác định về lý thuyết, khớp moment. Nếu bạn có một bảng lớn các đơn vị tương tự và nhiều đơn vị thưa, hãy co với prior ước lượng từ bảng. Nếu bạn cần bảo đảm độ phủ quanh một dự đoán hộp đen, ưu tiên conformal hoặc bootstrap hơn khoảng chuẩn ngây thơ.

## Thách thức và mở rộng

MLE trong mạng sâu thành công về tính toán và mong manh về suy diễn: sai số chuẩn, mô hình sai đặc tả, và lệch train–serve đều phá tiệm cận giáo trình. Estimator method of moments có thể có phương sai mẫu hữu hạn lớn dù nhất quán; tensor ẩn giảm chi phí, không giảm nhiễu thống kê. Empirical Bayes giả định tính trao đổi giữa các mục; một cú dịch chuyển danh mục đột ngột ước lượng sai $$(\alpha,\beta)$$. Độ phủ conformal là biên, không điều kiện trên một lát người dùng hiếm.

Hướng mở nằm trên chương này gồm ước lượng không cần likelihood (suy diễn dựa trên mô phỏng, score matching), empirical Bayes phân cấp cho nhiều thị trường xếp hạng liên quan, và conformal dưới dịch chuyển phân phối.

## Bài tập

**Bài 1 (khái niệm).** Một bài báo mô hình ngôn ngữ báo cáo training loss theo compute. Đối tượng nào trong chương này là training loss? Vì sao NLL huấn luyện nhỏ hơn, tự nó, không phải bằng chứng $$\hat{\theta}$$ gần hơn một tham số sinh dữ liệu “thật”?

**Bài 2 (tính toán).** Mô phỏng GMM một chiều hai thành phần với trọng số không đều. Ước lượng tham số bằng (i) method of moments qua ba moment đầu và (ii) EM. Lặp 200 lần, so sánh chệch và RMSE. Khi nào EM thắng, và khi nào khởi tạo xấu làm EM tệ hơn MoM?

**Bài 3 (tính toán).** Dùng các hàm Beta–Binomial ở trên, sinh 5.000 mục với prior thật $$\mathrm{Beta}(2,20)$$ và quan sát nhị thức. Khôi phục $$(\hat{\alpha},\hat{\beta})$$ bằng method of moments. Vẽ tỷ lệ MLE thô đối chiếu kỳ vọng posterior cho các mục có $$n \in \{10, 100, 1000\}$$ impression. Giải thích độ co.

**Bài 4 (mở).** Chọn một tập learning-to-rank hoặc tìm kiếm sản phẩm công khai. Định nghĩa lát khởi động lạnh (mục dưới một ngưỡng nhấp). So sánh CTR thô, bộ co về trung bình toàn cục, và bộ co empirical Bayes trên một tuần giữ lại. Báo cáo cả chất lượng xếp hạng lẫn hiệu chỉnh. Sai đặc tả prior nào sẽ làm EB tệ hơn trung bình toàn cục?

## Tài liệu tham khảo

Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning, 16*(4), 494–591. [https://doi.org/10.1561/2200000101](https://doi.org/10.1561/2200000101)

Anandkumar, A., Ge, R., Hsu, D., Kakade, S. M., & Telgarsky, M. (2014). Tensor decompositions for learning latent variable models. *Journal of Machine Learning Research, 15*(80), 2773–2832. Vẫn là tham chiếu lý thuyết chuẩn cho ước lượng biến ẩn bằng moment.

Efron, B. (2010). *Large-Scale Inference: Empirical Bayes Methods for Estimation, Testing, and Prediction*. Cambridge University Press. Chuyên khảo hiện đại đứng sau co trong sản xuất.

Han, C., Castells, P., Gupta, P., Xu, X., & Salaka, V. (2022). Addressing cold start in product search via empirical Bayes. Trong *Proceedings of the 31st ACM International Conference on Information and Knowledge Management* (tr. 3141–3151). [https://doi.org/10.1145/3511808.3557066](https://doi.org/10.1145/3511808.3557066)

Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., … Sifre, L. (2022). Training compute-optimal large language models. Trong *Advances in Neural Information Processing Systems, 35*. [https://arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556)

Hsu, D., & Kakade, S. M. (2013). Learning mixtures of spherical Gaussians: Moment methods and spectral decompositions. Trong *ITCS 2013*. Method of moments phổ cho GMM; vẫn được các bài tensor ẩn 2022–2025 trích dẫn.

Pereira, J. M., Kileel, J., & Kolda, T. G. (2022). Tensor moments of Gaussian mixture models: Theory and applications. [https://arxiv.org/abs/2202.06930](https://arxiv.org/abs/2202.06930)
