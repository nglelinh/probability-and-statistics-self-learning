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

Bài đọc tùy chọn này không thay thế lý thuyết ước lượng điểm và ước lượng khoảng trong chương, mà minh họa cách các bộ ước lượng cổ điển vận hành bên trong các hệ thống khoa học máy tính và khoa học dữ liệu hiện đại: ước lượng hợp lý cực đại (MLE) trong mạng học sâu, phương pháp moment cho mô hình hỗn hợp, Empirical Bayes trong hệ thống xếp hạng thực tế, và các phương pháp ước lượng khoảng khi không tồn tại công thức sai số chuẩn dạng đóng.

Sau bài học này, bạn có thể nhận diện huấn luyện bằng hàm mất mát cross-entropy chính là MLE, giải thích vì sao phương pháp khớp moment vẫn hữu ích cho mô hình hỗn hợp Gauss, cài đặt bộ ước lượng co (shrinkage estimator) Beta–Binomial theo Empirical Bayes như trong hệ thống tìm kiếm thương mại điện tử, và phân biệt khoảng tin cậy Bootstrap với tập dự đoán conformal.

Kiến thức tiên quyết gồm các bài bắt buộc về ước lượng điểm (độ chệch, phương sai, MLE, phương pháp moment) và ước lượng khoảng (khoảng tin cậy, Bootstrap), cùng kỹ năng Python/NumPy cơ bản. Bài học không đưa thêm tiên đề xác suất mới.

---

## Từ Giáo Trình Đến Hệ Thống Thực Tế

Một hệ thống xếp hạng, một mô hình ngôn ngữ và một quy trình phân cụm đều giải quyết cùng một bài toán cốt lõi: từ mẫu dữ liệu $$X_1,\ldots,X_n$$, tìm ước lượng $$\hat{\theta}$$ và — khi sai lầm kéo theo chi phí lớn — xây dựng thêm một khoảng bất định quanh ước lượng đó. Điều thay đổi trong giai đoạn 2022–2026 không nằm ở định nghĩa bộ ước lượng, mà ở quy mô triển khai và các ràng buộc kỹ thuật như độ trễ, bài toán khởi động lạnh (cold start) và ngân sách tính toán — những yếu tố quyết định *bộ ước lượng nào* được lựa chọn.

Ba hướng tiếp cận phổ biến nhất trong thực tế triển khai gồm: các mô hình học sâu cực tiểu hóa âm log-likelihood (tức thực chất là MLE); các mô hình hỗn hợp trong nhận dạng người nói, phân cụm dữ liệu đơn bào và phát hiện bất thường vẫn sử dụng phương pháp khớp moment khi hàm hợp lý có nhiều cực trị địa phương; và các hệ thống xếp hạng, quảng cáo thực hiện co (shrinkage) các tỷ lệ nhiễu về một phân phối tiên nghiệm (prior) ước lượng từ dữ liệu — đúng với phương pháp Empirical Bayes mà Efron hệ thống hóa và nhiều nền tảng lớn vẫn áp dụng.

## Ước Lượng Hợp Lý Cực Đại (MLE): Cách Các Mạng Nơ-ron Hiện Đại Học

Với mẫu độc lập và có cùng phân phối (i.i.d.), MLE cực đại hóa hàm log-hợp lý:

$$\hat{\theta}_{\mathrm{MLE}} = \arg\max_\theta \sum_{i=1}^n \log p(x_i \mid \theta).$$

Trong bài toán phân loại, mỗi quan sát là một cặp $$(x,y)$$ và xác suất có điều kiện $$p(y \mid x,\theta)$$ được biểu diễn qua hàm softmax. Khi đó, âm log-likelihood trung bình trùng khớp với hàm mất mát cross-entropy trong mọi thư viện học sâu phổ biến. Việc huấn luyện mạng bằng thuật toán hạ gradient ngẫu nhiên (stochastic gradient descent) thực chất là giải bài toán MLE bằng tối ưu số: hàm score $$\nabla_\theta \log p$$ được tính qua lan truyền ngược thay vì giải phương trình đạo hàm bằng 0 ở dạng đóng.

Các mô hình ngôn ngữ tự hồi quy cũng áp dụng đúng nguyên lý này trên từng token. Quá trình huấn luyện dự đoán token tiếp theo cực đại hóa:

$$\sum_t \log p_\theta(w_t \mid w_{<t}),$$

tức là bài toán MLE cho họ phân phối phân loại có điều kiện (conditional categorical). Hoffmann và cộng sự (2022) khảo sát hàm hợp lý này dưới góc độ *thực nghiệm*: nhóm tác giả huấn luyện hơn 400 mô hình transformer và chỉ ra rằng, với một ngân sách tính toán cố định, mô hình tối ưu theo tài nguyên tính toán cần mở rộng số tham số và số token huấn luyện gần như theo cùng tỷ lệ. Về bản chất thống kê, kết quả "Chinchilla" cho thấy chất lượng của MLE phụ thuộc đồng thời vào số chiều của $$\theta$$ và cỡ mẫu — chính là câu chuyện đánh đổi giữa độ chệch và phương sai cổ điển, được nhìn trên đường biên tối ưu theo tài nguyên tính toán thay vì bảng MSE trong giáo trình.

Có hai lưu ý quan trọng để hiểu đúng mối tương đương này. Thứ nhất, các kỹ thuật điều chuẩn (regularisation), dừng sớm (early stopping) và tăng cường dữ liệu khiến bộ trọng số sau huấn luyện là MLE *có phạt (penalized MLE)* chứ không phải cực đại toàn cục không ràng buộc. Thứ hai, các mạng quá tham số hóa có thể khớp hoàn toàn dữ liệu huấn luyện (nội suy), nên các kết quả cổ điển như "MLE có phân phối tiệm cận chuẩn với tốc độ $$n^{-1/2}$$" không áp dụng theo nghĩa đen. Điểm giữ nguyên là *hàm mục tiêu*: trong thực hành, mô hình vẫn được đánh giá bằng âm log-likelihood trên tập kiểm tra giữ lại (held-out set), và các quy luật mở rộng (scaling laws) vẫn mô tả hành vi của chính hàm hợp lý đó.

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

## Phương Pháp Moment Cho Mô Hình Hỗn Hợp Gauss

Một mô hình hỗn hợp Gauss (GMM) gồm $$K$$ thành phần có hàm mật độ:

$$p(x) = \sum_{k=1}^K \pi_k \, \mathcal{N}(x; \mu_k, \Sigma_k).$$

Hàm hợp lý của mô hình này không lõm. Thuật toán Expectation–Maximisation (EM) tìm nghiệm MLE địa phương và là phương pháp mặc định trong `sklearn.mixture.GaussianMixture`. Trong khi đó, phương pháp moment — do Pearson đề xuất từ năm 1894 cho hỗn hợp một chiều gồm hai thành phần — khớp các moment thực nghiệm với các moment lý thuyết biểu diễn theo $$(\pi,\mu,\Sigma)$$.

Trong không gian nhiều chiều, các moment bậc cao trở thành các tensor. Hsu và Kakade (2013) cùng Anandkumar, Ge, Hsu, Kakade và Telgarsky (2014) chứng minh rằng ba moment đầu tiên của GMM cầu xác định duy nhất tham số thông qua phân rã tensor đối xứng, với bảo đảm thời gian chạy đa thức khi các vector kỳ vọng độc lập tuyến tính. Đây vẫn là các tài liệu nền tảng trong nhiều giáo trình lý thuyết học máy.

Bước tiến gần đây nằm ở tính khả thi tính toán. Việc lưu trữ tường minh một tensor moment bậc $$d$$ đòi hỏi bộ nhớ $$O(n^d)$$. Pereira, Kileel và Kolda (2022) xây dựng biểu thức ẩn (implicit) cho tensor moment của GMM, giúp việc khớp moment chạy với chi phí $$O(n^2)$$ hoặc $$O(n^3)$$ đối với ma trận hiệp phương sai tổng quát và $$O(n)$$ đối với ma trận hiệp phương sai đường chéo mà không cần khởi tạo tường minh toàn bộ tensor. Hướng tiếp cận moment ẩn này tiếp tục được phát triển trong các nghiên cứu giai đoạn 2024–2025. Thông điệp thống kê cốt lõi vẫn không đổi: khi bề mặt hàm hợp lý phức tạp và nhiều cực trị, ta có thể giải quyết bằng phương pháp cân bằng moment.

Đoạn mã dưới đây minh họa ý tưởng gốc của Pearson trong trường hợp một chiều đối xứng:

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

Trong triển khai thực tế, ước lượng moment thường được dùng để *khởi tạo* cho thuật toán EM thay vì thay thế hoàn toàn — sự kết hợp này tận dụng khả năng định vị toàn cục của phương pháp moment cùng độ hiệu quả địa phương của EM.

## Empirical Bayes Trong Hệ Thống Xếp Hạng Thực Tế

Empirical Bayes ước lượng phân phối tiên nghiệm (prior) trực tiếp từ chính tập dữ liệu sẽ được cập nhật bằng prior đó. Với bài toán ước lượng tỷ lệ nhấp chuột (CTR), mô hình Beta–Binomial là lựa chọn quen thuộc. Nếu mục $$i$$ có $$c_i$$ lượt nhấp trên $$n_i$$ lượt hiển thị và phân phối tiên nghiệm là $$\mathrm{Beta}(\alpha,\beta)$$, kỳ vọng hậu nghiệm là:

$$\hat{p}_i = \frac{\alpha + c_i}{\alpha + \beta + n_i}.$$

Kỳ vọng tiên nghiệm $$m = \alpha/(\alpha+\beta)$$ và độ mạnh $$s = \alpha+\beta$$ không cần chọn cảm tính, mà được ước lượng từ toàn bộ danh sách các mục — bằng cách khớp trung bình và phương sai của các tỷ lệ quan sát được (phương pháp moment cho prior) hoặc cực đại hóa hàm hợp lý biên. Một mục mới chưa có dữ liệu ($$n_i = 0$$) sẽ nhận ước lượng bằng kỳ vọng tiên nghiệm, còn mục có $$n_i$$ rất lớn sẽ tiến sát về ước lượng MLE thô $$c_i/n_i$$. Đây chính là nguyên lý ước lượng co (shrinkage) kiểu James–Stein được áp dụng ở tầng sản phẩm.

Han, Castells, Gupta, Xu và Salaka (2022) triển khai mô hình này cho bài toán khởi động lạnh (cold start) trên hệ thống tìm kiếm sản phẩm của Amazon: một mô hình không dùng dữ liệu hành vi cung cấp prior, còn lượt nhấp và lượt mua thực tế cập nhật posterior. Thử nghiệm trực tuyến trên 50 triệu truy vấn ghi nhận lượt hiển thị sản phẩm mới tăng 13,53% và lượt mua sản phẩm mới tăng 11,14%. Một nghiên cứu liên quan năm 2023 (EBRank) áp dụng cùng cấu trúc tách prior/posterior này để giảm thiên lệch khai thác (exposure bias) trong bài toán học xếp hạng (learning-to-rank).

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

Mối liên hệ với nội dung chương này rất trực tiếp: tham số của phân phối tiên nghiệm là một ước lượng điểm, kỳ vọng hậu nghiệm là một ước lượng điểm khác, còn khoảng tin cậy xây từ phân phối hậu nghiệm Beta hoặc từ Bootstrap cho $$(\hat{\alpha},\hat{\beta})$$ chính là ước lượng khoảng.

## Ước Lượng Khoảng Khi Không Có Sai Số Chuẩn Giải Tích

Trong bài ước lượng khoảng bắt buộc, ta đã xây dựng khoảng xấp xỉ chuẩn:

$$\bar{x} \pm z_{1-\alpha/2} \frac{s}{\sqrt{n}}$$

và khoảng phân vị Bootstrap. Các công cụ này vẫn phát huy hiệu quả với trung bình mẫu hoặc các thống kê đơn giản. Tuy nhiên, với mô hình học sâu, các giả định dẫn xuất cổ điển không còn phù hợp: mô hình có hàng triệu tham số, hàm mất mát không lồi, và sai số chuẩn từ ma trận Hessian gần như không thể tính trực tiếp.

Hai giải pháp được sử dụng rộng rãi trong giai đoạn 2022–2026 gồm Bootstrap (cùng họ phương pháp deep ensembles) và dự đoán conformal (conformal prediction). Bootstrap tái lấy mẫu quá trình huấn luyện và đọc khoảng phân vị từ phân phối thực nghiệm của thống kê — vẫn là thuật toán đã học nhưng với chi phí tính toán lớn hơn. Trong khi đó, conformal prediction — được Angelopoulos và Bates (2023) hệ thống hóa cho cộng đồng học máy — bọc quanh *bất kỳ* mô hình đã huấn luyện nào để tạo ra một tập dự đoán $$C(x)$$ thỏa mãn:

$$P\bigl(Y_{n+1} \in C(X_{n+1})\bigr) \ge 1-\alpha$$

dưới giả định tính hoán vị được (exchangeability), mà không cần giả định mô hình gốc được đặc tả đúng. Phương pháp split conformal dùng một tập hiệu chỉnh (calibration fold) riêng để tính phân vị của điểm không phù hợp (nonconformity score), mang lại bảo đảm độ phủ ngay cả với cỡ mẫu hữu hạn. Lưu ý rằng đây không phải là khoảng tin cậy cho tham số $$\theta$$, mà là khoảng dự đoán cho một giá trị đầu ra mới — thường chính là đại lượng mà các hệ thống triển khai thực tế cần.

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

## Nguyên Tắc Lựa Chọn Trong Thực Hành

Hãy chọn bộ ước lượng phù hợp với *cấu trúc* bài toán thay vì chạy theo độ phổ biến của lớp mô hình. Nếu mô hình quan sát thuộc họ phân phối xác định rõ và có thể tối ưu trực tiếp, hãy dùng MLE (hoặc MLE có điều chuẩn). Nếu hàm hợp lý có nhiều cực trị hoặc cần bước khởi tạo có cơ sở lý thuyết, hãy áp dụng phương pháp khớp moment. Khi làm việc với bảng dữ liệu lớn gồm nhiều thực thể tương tự nhưng phần lớn có ít quan sát, hãy áp dụng ước lượng co với prior ước lượng từ chính tập dữ liệu đó. Cuối cùng, khi cần bảo đảm độ phủ quanh dự đoán của một mô hình hộp đen, hãy ưu tiên conformal prediction hoặc Bootstrap thay vì dùng khoảng xấp xỉ chuẩn một cách máy móc.

## Thách Thức và Hướng Mở Rộng

MLE trong mạng học sâu rất hiệu quả về mặt tối ưu tính toán nhưng lại mong manh về mặt suy diễn thống kê: việc tính sai số chuẩn, nguy cơ sai đặc tả mô hình và độ lệch giữa huấn luyện và triển khai (train–serve skew) đều làm phá vỡ các kết quả tiệm cận cổ điển. Bộ ước lượng theo phương pháp moment dù nhất quán nhưng có thể có phương sai lớn trên mẫu hữu hạn; kỹ thuật tensor ẩn giúp giảm chi phí tính toán chứ không làm giảm nhiễu thống kê. Empirical Bayes dựa trên giả định các mục có thể hoán vị với nhau, nên một biến động đột ngột trong danh mục sản phẩm có thể làm ước lượng $$(\alpha,\beta)$$ bị lệch. Trong khi đó, bảo đảm độ phủ của conformal prediction là độ phủ biên (marginal coverage), không phải độ phủ có điều kiện trên từng nhóm người dùng hiếm gặp.

Các hướng nghiên cứu mở tiếp nối chương này bao gồm ước lượng không cần hàm hợp lý tường minh (suy diễn dựa trên mô phỏng, score matching), Empirical Bayes phân cấp cho nhiều thị trường xếp hạng liên quan, và conformal prediction dưới điều kiện dịch chuyển phân phối.

## Video Tham Khảo

- [Neural Networks Part 6: Cross Entropy](https://www.youtube.com/watch?v=6ArSys5qHAU) — StatQuest with Josh Starmer, tiếng Anh. **Nên xem trước.** Giải thích cross-entropy chính là âm log-likelihood trung bình của mô hình phân loại — tương ứng với mối liên hệ giữa huấn luyện mạng nơ-ron và MLE trong bài.
- [Gaussian Mixture Model](https://www.youtube.com/watch?v=EWd1xRkyEog) — ritvikmath, tiếng Anh. Giới thiệu mô hình hỗn hợp Gauss, lớp mô hình được khớp bằng phương pháp moment trong bài khi hàm hợp lý có nhiều cực trị.
- [Bayesian Updates and Conjugate Priors](https://www.youtube.com/watch?v=GqSX-8AQL90) — Steve Brunton, tiếng Anh. Minh họa cập nhật Bayes với phân phối tiên nghiệm liên hợp — cơ chế nền tảng cho kỹ thuật Empirical Bayes trong bài toán xếp hạng.

## Bài Tập

**Bài 1 (khái niệm).** Một bài báo về mô hình ngôn ngữ báo cáo giá trị training loss theo ngân sách tính toán. Training loss tương ứng với đối tượng thống kê nào trong chương này? Vì sao training NLL nhỏ hơn tự nó chưa phải là bằng chứng cho thấy $$\hat{\theta}$$ tiến gần hơn đến tham số sinh dữ liệu "thật"?

**Bài 2 (tính toán).** Mô phỏng mô hình hỗn hợp Gauss một chiều gồm hai thành phần với trọng số không đều. Ước lượng tham số bằng (i) phương pháp moment qua ba moment đầu tiên và (ii) thuật toán EM. Lặp lại 200 lần rồi so sánh độ chệch và RMSE. Trong trường hợp nào EM cho kết quả tốt hơn, và khi nào khởi tạo kém khiến EM tệ hơn phương pháp moment?

**Bài 3 (tính toán).** Sử dụng các hàm Beta–Binomial ở trên, sinh 5.000 mục với prior thực $$\mathrm{Beta}(2,20)$$ và quan sát theo phân phối nhị thức. Ước lượng lại $$(\hat{\alpha},\hat{\beta})$$ bằng phương pháp moment. Vẽ biểu đồ so sánh tỷ lệ MLE thô với kỳ vọng hậu nghiệm cho các mục có $$n \in \{10, 100, 1000\}$$ lượt hiển thị và giải thích hiện tượng co ước lượng.

**Bài 4 (mở).** Chọn một tập dữ liệu công khai về học xếp hạng (learning-to-rank) hoặc tìm kiếm sản phẩm. Định nghĩa nhóm khởi động lạnh (các mục có số lượt nhấp dưới một ngưỡng xác định). So sánh CTR thô, ước lượng co về trung bình toàn cục và ước lượng co Empirical Bayes trên tập dữ liệu giữ lại trong một tuần; báo cáo cả chất lượng xếp hạng lẫn độ hiệu chỉnh xác suất. Dạng sai đặc tả prior nào có thể khiến Empirical Bayes hoạt động kém hơn trung bình toàn cục?

## Tài Liệu Tham Khảo

Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning, 16*(4), 494–591. [https://doi.org/10.1561/2200000101](https://doi.org/10.1561/2200000101)

Anandkumar, A., Ge, R., Hsu, D., Kakade, S. M., & Telgarsky, M. (2014). Tensor decompositions for learning latent variable models. *Journal of Machine Learning Research, 15*(80), 2773–2832. Tài liệu tham chiếu lý thuyết tiêu chuẩn cho ước lượng mô hình biến ẩn bằng phương pháp moment.

Efron, B. (2010). *Large-Scale Inference: Empirical Bayes Methods for Estimation, Testing, and Prediction*. Cambridge University Press. Chuyên khảo hiện đại làm nền tảng cho các phương pháp ước lượng co trong thực tế triển khai.

Han, C., Castells, P., Gupta, P., Xu, X., & Salaka, V. (2022). Addressing cold start in product search via empirical Bayes. Trong *Proceedings of the 31st ACM International Conference on Information and Knowledge Management* (tr. 3141–3151). [https://doi.org/10.1145/3511808.3557066](https://doi.org/10.1145/3511808.3557066)

Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., … Sifre, L. (2022). Training compute-optimal large language models. Trong *Advances in Neural Information Processing Systems, 35*. [https://arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556)

Hsu, D., & Kakade, S. M. (2013). Learning mixtures of spherical Gaussians: Moment methods and spectral decompositions. Trong *ITCS 2013*. Phương pháp moment phổ cho GMM; vẫn được các nghiên cứu về tensor ẩn giai đoạn 2022–2025 trích dẫn.

Pereira, J. M., Kileel, J., & Kolda, T. G. (2022). Tensor moments of Gaussian mixture models: Theory and applications. [https://arxiv.org/abs/2202.06930](https://arxiv.org/abs/2202.06930)
