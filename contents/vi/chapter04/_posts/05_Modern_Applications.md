---
layout: post
title: 04-05-00 Ứng dụng và phát triển gần đây
chapter: "04"
order: 5
owner: nglelinh
lang: vi
categories:
- chapter04
lesson_type: optional
---

Các họ phân phối cơ bản trong chương này — Bernoulli, Poisson, nhị thức âm, Gauss và phân phối lấy mẫu — vẫn là nền tảng của nhiều bài toán thực tế như dự báo nhu cầu, phân tích nhật ký máy chủ và mô hình sinh rời rạc. Bài đọc tùy chọn này không chứng minh lại công thức PMF, mà chỉ ra ba hướng ứng dụng thực tiễn của các họ phân phối này (ở dạng nguyên bản hoặc mở rộng nhẹ) trong giai đoạn 2020–2024.

## Mô Hình Đếm: Khi Kỳ Vọng Chưa Đủ

Lượt nhấp chuột (click), phiếu hỗ trợ (ticket) và số lượng hàng bán ra đều là các số nguyên không âm. Với biến ngẫu nhiên Poisson $$N\sim\mathrm{Poisson}(\lambda)$$, ta có:

$$
P(N=k)=\frac{\lambda^k e^{-\lambda}}{k!},\qquad \mathbb{E}[N]=\mathrm{Var}(N)=\lambda.
$$

Tuy nhiên, dữ liệu đếm ngoài thực tế thường gặp hiện tượng **phân tán quá mức (overdispersion)**, tức $$\mathrm{Var}>\mathbb{E}$$. Phân phối nhị thức âm (Negative Binomial) giữ nguyên cấu trúc kỳ vọng của Poisson nhưng bổ sung thêm một tham số phân tán — vẫn là công thức PMF quen thuộc của phân phối rời rạc, nay đóng vai trò hàm hợp lý trong mô hình tuyến tính tổng quát (GLM). Trong thực tế triển khai, [`PoissonRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html) và `TweedieRegressor` của scikit-learn là những mô hình tuyến tính phổ biến. Ở hướng học sâu, Salinas, Flunkert, Gasthaus và Januschowski (*DeepAR*, [*International Journal of Forecasting* 36, 2020](https://doi.org/10.1016/j.ijforecast.2019.07.001); [arXiv:1704.04110](https://arxiv.org/abs/1704.04110)) gắn đầu ra (output head) phân phối nhị thức âm (hoặc Gauss) lên một mạng hồi tiếp dùng chung để chia sẻ thông tin giữa hàng nghìn chuỗi thời gian liên quan. Đầu ra của mô hình không chỉ là một giá trị điểm mà là toàn bộ PMF dự báo, cho phép lấy mẫu phục vụ bài toán quản lý tồn kho. Như vậy, kiến thức Chương 04 nằm ở chính lớp đầu ra xác suất; mạng nơ-ron chỉ đóng vai trò ánh xạ linh hoạt sang cặp tham số $$(\lambda,\alpha)$$.

```python
import numpy as np
from sklearn.linear_model import PoissonRegressor

rng = np.random.default_rng(0)
x = rng.normal(size=(400, 2))
lam = np.exp(0.4 * x[:, 0] - 0.2 * x[:, 1])
y = rng.poisson(lam)
model = PoissonRegressor(alpha=1e-4).fit(x, y)
print("mean predicted λ:", model.predict(x).mean(), "mean y:", y.mean())
```

Nếu hai giá trị trung bình in ra lệch nhau nhiều, vấn đề nằm ở giả định mô hình Poisson với hàm liên kết log (log-link) chưa phù hợp với dữ liệu, chứ không phải do "mạng nơ-ron thất bại".

## Quá Trình Poisson Trong Các Hệ Thống Thực Tế

Một biến ngẫu nhiên Poisson đếm số sự kiện xảy ra trong một cửa sổ thời gian cố định. **Quá trình Poisson (Poisson process)** mở rộng ý tưởng này cho thời gian liên tục: số sự kiện trên các khoảng thời gian rời nhau là độc lập, và số sự kiện trên một khoảng có độ dài $$t$$ tuân theo $$\mathrm{Poisson}(\lambda t)$$. Lưu lượng truy cập (request), bản ghi lỗi (crash log) và lệnh giao dịch thường được mô hình hóa bằng quá trình này trước khi cân nhắc đến mạng nơ-ron. Khi sự kiện trong quá khứ có xu hướng *tự kích thích* các sự kiện tương lai (chẳng hạn một bài đăng nổi bật kéo theo nhiều bài đăng tiếp theo), cường độ $$\lambda(t)$$ phụ thuộc vào lịch sử: đó là quá trình Hawkes và các phiên bản mở rộng như quá trình điểm theo thời gian dùng mạng nơ-ron (neural temporal point process). Mei và Eisner (*The Neural Hawkes Process*, [NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/6463c88460bd63bbe256e495c63aa40b-Abstract.html)) thay thế hàm nhân cộng tính cổ điển bằng mạng LSTM theo thời gian liên tục. Bài tổng quan của Shchur, Türkmen, Januschowski và Günnemann (*Neural Temporal Point Processes: A Review*, [IJCAI 2021](https://www.ijcai.org/proceedings/2021/600); [arXiv:2104.03528](https://arxiv.org/abs/2104.03528)) vẫn thường xuyên được trích dẫn trong các nghiên cứu giai đoạn 2022–2025 (bao gồm kiến trúc transformer và các mô hình không cần cường độ tường minh). Từ Chương 04, khái niệm cốt lõi cần nắm là biến đếm Poisson và việc tần suất $$\lambda$$ có thể biến thiên theo thời gian; phần phát triển sau đó là cách tham số hóa hàm cường độ này.

## Mô Hình Sinh Rời Rạc Dưới Góc Nhìn PMF

Một mô hình ngôn ngữ tự hồi quy thực chất là một biến ngẫu nhiên phân loại (categorical) có không gian giá trị rất lớn tại mỗi vị trí token: $$X_t\mid X_{<t}\sim\mathrm{Categorical}(\pi_\theta(X_{<t}))$$. Trong khi đó, các mô hình khuếch tán (diffusion models) vốn rất hiệu quả trên *ảnh* lại dựa vào hàm score Gauss $$\nabla_x\log p_t(x)$$, còn dữ liệu token rời rạc không tồn tại gradient theo nghĩa liên tục này. Lou, Meng và Ermon (*Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution*, [ICML 2024](https://proceedings.mlr.press/v235/lou24a.html); [arXiv:2310.16834](https://arxiv.org/abs/2310.16834); mã nguồn [Score-Entropy-Discrete-Diffusion](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion)) đề xuất khái niệm **score entropy** để khớp *tỷ số* xác suất $$p(y)/p(x)$$ trên không gian trạng thái rời rạc và xây dựng mô hình SEDD. Trên các bộ chuẩn đánh giá ngôn ngữ, SEDD đạt kết quả cạnh tranh với các mô hình tự hồi quy cỡ GPT-2, đồng thời hỗ trợ điền khuyết linh hoạt thay vì chỉ sinh văn bản từ trái sang phải. Ngay cả khi chưa học sâu về lý thuyết quá trình ngẫu nhiên, ta vẫn thấy rõ bản chất từ Chương 04: dữ liệu là một biến ngẫu nhiên rời rạc, mô hình biểu diễn một PMF (hoặc tỷ số các PMF), và quá trình sinh mẫu là thao tác tính toán tương ứng với việc rút mẫu từ phân phối $$p$$. Công trình D3PM của Austin và cộng sự ([NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/958c530554f2820e0d14e14d964d48d9-Abstract.html)) là một cách tiếp cận khuếch tán rời rạc sớm hơn, thường được trích dẫn song song với SEDD.

Ngoài ra, kiến thức về phân phối lấy mẫu trong bài trước cũng giải thích vì sao các ước lượng Monte Carlo cho perplexity hay tỷ lệ bao phủ cần nêu rõ cỡ mẫu $$n$$: giá trị ta quan sát được là trung bình mẫu $$\bar{X}_n$$, chứ không nên dựa vào một lần sinh ngẫu nhiên đơn lẻ.

## Điểm Cốt Lõi Cần Ghi Nhớ

Khi một bài báo nghiên cứu đề cập đến "hàm hợp lý nhị thức âm", "cường độ Hawkes nơ-ron" hay "score entropy trên token", chúng đều sử dụng trực tiếp ngôn ngữ xác suất của chương này. Các bước ước lượng tham số và kiểm định giả thuyết ở các chương sau cũng sẽ vận hành trên chính những họ phân phối này.

## Video Tham Khảo

- [Regression with Count Data: Poisson and Negative Binomial](https://www.youtube.com/watch?v=uGKnoAw-PFQ) — Matthew E. Clapham, tiếng Anh. **Nên xem trước.** Trình bày hồi quy dữ liệu đếm với phân phối Poisson và nhị thức âm khi phương sai lớn hơn kỳ vọng, tương ứng với mục mô hình đếm trong bài.
- [L22.2 Definition of the Poisson Process](https://www.youtube.com/watch?v=D_EGYzqmapc) — MIT OpenCourseWare, tiếng Anh. Định nghĩa quá trình Poisson của MIT với tính chất số gia độc lập và phân phối Poisson trên từng khoảng, liên hệ trực tiếp với mục nhật ký hệ thống và lượt request.
- [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution – Paper Explained](https://www.youtube.com/watch?v=K_9wQ6LZNpI) — AI Coffee Break with Letitia, tiếng Anh. Giải thích bài báo SEDD được trích dẫn trong bài: mô hình sinh rời rạc khớp tỷ số xác suất — tức làm việc với một PMF trên không gian token thay vì dự báo điểm.

## Nguồn

1. D. Salinas, V. Flunkert, J. Gasthaus và T. Januschowski, “DeepAR: Probabilistic forecasting with autoregressive recurrent networks,” *Int. J. Forecasting* 36(3), 2020. [DOI](https://doi.org/10.1016/j.ijforecast.2019.07.001) · [arXiv:1704.04110](https://arxiv.org/abs/1704.04110)
2. scikit-learn, `PoissonRegressor`. [tài liệu](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html)
3. H. Mei và J. Eisner, “The Neural Hawkes Process,” NeurIPS 2017. [tóm tắt](https://proceedings.neurips.cc/paper/2017/hash/6463c88460bd63bbe256e495c63aa40b-Abstract.html)
4. O. Shchur, A. C. Türkmen, T. Januschowski và S. Günnemann, “Neural Temporal Point Processes: A Review,” IJCAI 2021. [arXiv:2104.03528](https://arxiv.org/abs/2104.03528)
5. A. Lou, C. Meng và S. Ermon, “Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution,” ICML 2024. [PMLR](https://proceedings.mlr.press/v235/lou24a.html) · [arXiv:2310.16834](https://arxiv.org/abs/2310.16834)
6. J. Austin và cộng sự, “Structured Denoising Diffusion Models in Discrete State-Spaces,” NeurIPS 2021. [tóm tắt](https://proceedings.neurips.cc/paper/2021/hash/958c530554f2820e0d14e14d964d48d9-Abstract.html)
