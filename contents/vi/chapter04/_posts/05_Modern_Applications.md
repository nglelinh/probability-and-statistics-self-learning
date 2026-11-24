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

Một kỹ sư dữ liệu tại sàn thương mại điện tử xây dựng hệ thống dự báo số đơn hàng bán ra mỗi ngày cho 10.000 mã sản phẩm. Anh chọn mô hình Poisson quen thuộc với trung bình $$\lambda = 10$$ đơn/ngày. Kết quả? Kho hàng liên tục rơi vào hai thái cực: tuần thì tồn kho chất đống, tuần lại cháy hàng giữa chừng. Thủ phạm không phải mạng nơ-ron hay thuật toán tối ưu, mà là một giả định âm thầm của phân phối Poisson: kỳ vọng và phương sai buộc phải bằng nhau ($$\mathrm{Var}(N) = \mathbb{E}[N] = \lambda$$), trong khi nhu cầu thực tế luôn có hiện tượng **phân tán quá mức (overdispersion)** — những ngày bình thường bán 2 đơn, ngày khuyến mãi vọt lên 25 đơn.

Bài đọc tùy chọn này nối thẳng các họ phân phối kinh điển của Chương 04 — Bernoulli, Poisson, nhị thức âm (Negative Binomial), Gauss — với ba bài toán kỹ thuật sống động giai đoạn 2020–2024: dự báo chuỗi thời gian xác suất (DeepAR), mô hình hóa sự kiện đến trong hệ thống phân tán, và mô hình sinh rời rạc. Công thức PMF trên bảng phấn không hề cũ đi; nó đang nằm ngay ở lớp đầu ra (output head) của các hệ thống học máy sản xuất.

---

## Bản đồ kiến thức nền: Các họ phân phối xác suất đóng vai trò gì trong hệ thống hiện đại?

Với người đã tốt nghiệp đại học, giá trị lớn nhất của việc học các họ phân phối không nằm ở việc thuộc lòng bảng công thức, mà ở việc hiểu **ba trụ cột mô hình hóa** sau:

1. **Họ phân phối đếm & Hiện tượng phân tán quá mức (Overdispersion):** Poisson là điểm khởi đầu lý tưởng cho các sự kiện độc lập xảy ra với tốc độ không đổi. Khi tốc độ thay đổi ngẫu nhiên theo ngữ cảnh (nhu cầu mua sắm, lưu lượng máy chủ), trộn Poisson với phân phối Gamma sinh ra **phân phối nhị thức âm (Negative Binomial)** — giữ nguyên kỳ vọng nhưng mở rộng phương sai theo tham số phân tán $$\alpha$$.
2. **Mô hình tuyến tính tổng quát (GLM) & Dự báo xác suất (Probabilistic Forecasting):** Thay vì dự báo một con số duy nhất (như hồi quy bình phương tối thiểu), các hệ thống hiện đại (từ `PoissonRegressor` đến *DeepAR*) dùng mô hình học máy để dự báo **tham số của cả một phân phối xác suất**, từ đó tính trực tiếp phân vị tồn kho hay rủi ro đuôi.
3. **Quá trình điểm (Point Processes) & Mô hình sinh rời rạc (Discrete Generative Models):** Quá trình Poisson không đồng nhất và quá trình Hawkes mô tả các dòng sự kiện có tính "tự kích thích" (lệnh giao dịch, lan truyền tin tức); trong khi đó, các mô hình ngôn ngữ và sinh ảnh rời rạc thực chất đang học phân phối xác suất có điều kiện (PMF) trên tập từ vựng hữu hạn.

---

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

---

### Ví dụ tính tay cầm tay chỉ việc: Vì sao Poisson đánh giá quá thấp rủi ro "ngày 0 đơn"?

Xét một mặt hàng có số đơn trung bình mỗi ngày là $$\mu = 10$$. Ta so sánh hai giả định phân phối cho số đơn hàng $$N$$ trong ngày:

- **Mô hình 1 — Poisson($$\lambda = 10$$):**
  - Kỳ vọng $$\mathbb{E}[N] = 10$$, phương sai $$\mathrm{Var}(N) = 10$$, độ lệch chuẩn $$\sigma = \sqrt{10} \approx 3{,}16$$ đơn.
  - Xác suất xảy ra một ngày không bán được đơn nào ($$N = 0$$):
    $$P_{\text{Pois}}(N = 0) = e^{-10} \approx 0{,}0000454 \approx 0{,}0045\%.$$
- **Mô hình 2 — Negative Binomial** với cùng kỳ vọng $$\mu = 10$$ nhưng hệ số phân tán $$\alpha = 0{,}5$$ (tức tham số hình dạng $$r = 1/\alpha = 2$$ và $$p = \frac{r}{r+\mu} = \frac{2}{12} = \frac{1}{6}$$):
  - Kỳ vọng vẫn là $$\mathbb{E}[N] = 10$$, nhưng phương sai tăng vọt lên
    $$\mathrm{Var}(N) = \mu + \alpha \mu^2 = 10 + 0{,}5 \times 10^2 = 60 \implies \sigma = \sqrt{60} \approx 7{,}75\text{ đơn}$$
    (độ lệch chuẩn rộng gấp gần **2,5 lần** Poisson!).
  - Xác suất xảy ra ngày 0 đơn:
    $$P_{\text{NB}}(N = 0) = p^r = \left(\frac{1}{6}\right)^2 = \frac{1}{36} \approx 2{,}78\%!$$

Cùng một giá trị trung bình 10 đơn/ngày, nhưng mô hình nhị thức âm dự báo xác suất bán 0 đơn cao gấp hơn **600 lần** Poisson (và tương tự, xác suất ngày bùng nổ trên 25 đơn cũng cao hơn hẳn).

### Khung giải thích: Từ công thức PMF đến hệ thống sản xuất (nên đọc chậm)

1. **Nhận diện bản chất biến mục tiêu:** Dữ liệu đếm luôn không âm và rời rạc; áp hồi quy Gauss trực tiếp có thể dự báo ra số đơn âm và bỏ qua phương sai phụ thuộc vào mức độ lớn của kỳ vọng.
2. **Kiểm tra tỷ số phương sai / kỳ vọng:** Nếu phương sai thực nghiệm vượt xa trung bình ($$\mathrm{Var} \gg \mathbb{E}$$), Poisson sẽ bóp nghẹt đuôi phân phối — cần nâng cấp lên Negative Binomial hoặc Tweedie.
3. **Tách kiến trúc mạng khỏi lớp xác suất:** Trong DeepAR hay GLM, mạng nơ-ron chỉ làm nhiệm vụ nén lịch sử thành các tham số $$(\mu_t, \alpha_t)$$; mọi phép tính rủi ro, khoảng dự báo và tối ưu tồn kho đều dựa trên chính PMF đã học ở Chương 04.
4. **Đánh giá bằng toàn bộ phân phối thay vì chỉ sai số điểm:** Hai mô hình có cùng sai số trung bình tuyệt đối (MAE) vẫn có thể chênh nhau hàng trăm lần ở xác suất đuôi — nơi quyết định chi phí cháy hàng thực tế.

**Kiểm tra nhanh.** Nếu hệ số phân tán $$\alpha \to 0$$ trong công thức phương sai $$\mathrm{Var}(N) = \mu + \alpha\mu^2$$ của phân phối nhị thức âm, phương sai hội tụ về giá trị nào và phân phối tiến về họ nào?

---

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
