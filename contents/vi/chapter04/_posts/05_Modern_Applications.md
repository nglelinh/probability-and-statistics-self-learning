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

Bernoulli, Poisson, nhị thức âm, Gauss và phân phối lấy mẫu trong chương này vẫn là các họ có tên nằm dưới dự báo nhu cầu, nhật ký máy chủ, và mô hình sinh rời rạc. Bài tùy chọn này không suy lại PMF. Nó chỉ ba chỗ các họ đó được dùng, nguyên bản hoặc mở rộng nhẹ, trong hệ thống 2020–2024.

## Mô hình đếm: khi kỳ vọng chưa đủ

Click, ticket và hàng bán là số nguyên không âm. Biến Poisson $$N\sim\mathrm{Poisson}(\lambda)$$ có

$$
P(N=k)=\frac{\lambda^k e^{-\lambda}}{k!},\qquad \mathbb{E}[N]=\mathrm{Var}(N)=\lambda.
$$

Số đếm thật thường **phân tán quá mức** ($$\mathrm{Var}>\mathbb{E}$$). Nhị thức âm giữ kỳ vọng Poisson và thêm tham số phân tán—cùng PMF như bài phân phối rời rạc, nay là hàm hợp lý GLM. [`PoissonRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html) và `TweedieRegressor` của scikit-learn là dạng tuyến tính trong sản xuất. Salinas, Flunkert, Gasthaus và Januschowski (*DeepAR*, [*International Journal of Forecasting* 36, 2020](https://doi.org/10.1016/j.ijforecast.2019.07.001); [arXiv:1704.04110](https://arxiv.org/abs/1704.04110)) gắn đầu nhị thức âm (hoặc Gauss) lên một mạng hồi tiếp dùng chung để hàng nghìn chuỗi liên quan vay sức. Đầu ra không phải một điểm; đó là cả PMF dự đoán bạn có thể lấy mẫu cho tồn kho. Chương 04 là cái đầu; mạng chỉ là ánh xạ linh hoạt vào $$(\lambda,\alpha)$$.

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

Nếu hai trung bình in ra lệch nặng, họ Poisson log-link mới sai—không phải “mạng nơ-ron thất bại.”

## Quá trình Poisson trong hệ thống

Một biến ngẫu nhiên Poisson đếm sự kiện trong cửa sổ cố định. Một **quá trình Poisson** cho cửa sổ trượt: số gia trên các khoảng rời nhau độc lập, và số đếm trên khoảng độ dài $$t$$ là $$\mathrm{Poisson}(\lambda t)$$. Lượt request, bản ghi crash và lệnh giao dịch thường được mô hình như vậy trước khi ai nhắc tới mạng nơ-ron. Khi sự kiện quá khứ *kích thích* tương lai (một bài nổi làm tăng bài tiếp), cường độ $$\lambda(t)$$ phụ thuộc lịch sử: quá trình Hawkes, rồi temporal point process nơ-ron. Mei và Eisner (*The Neural Hawkes Process*, [NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/6463c88460bd63bbe256e495c63aa40b-Abstract.html)) thay hạt nhân cộng tính bằng LSTM thời gian liên tục. Shchur, Türkmen, Januschowski và Günnemann (*Neural Temporal Point Processes: A Review*, [IJCAI 2021](https://www.ijcai.org/proceedings/2021/600); [arXiv:2104.03528](https://arxiv.org/abs/2104.03528)) là bài tổng quan vẫn được trích dẫn cho các công trình 2022–2025 (transformer, mô hình không cường độ). Đối tượng Chương 04 bạn cần là biến đếm Poisson và ý rằng một suất $$\lambda$$ có thể phụ thuộc thời gian; phần còn lại là cường độ được tham số hóa.

## Mô hình sinh rời rạc như PMF

Mô hình ngôn ngữ tự hồi quy là một biến ngẫu nhiên categorical khổng lồ tại mỗi vị trí: $$X_t\mid X_{<t}\sim\mathrm{Categorical}(\pi_\theta(X_{<t}))$$. Mô hình khuếch tán thắng trên *ảnh* dùng score Gauss $$\nabla_x\log p_t(x)$$. Token rời rạc không có gradient đó. Lou, Meng và Ermon (*Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution*, [ICML 2024](https://proceedings.mlr.press/v235/lou24a.html); [arXiv:2310.16834](https://arxiv.org/abs/2310.16834); mã [Score-Entropy-Discrete-Diffusion](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion)) định nghĩa **score entropy** khớp *tỷ số* xác suất $$p(y)/p(x)$$ trên không gian trạng thái rời rạc và xây SEDD. Trên benchmark ngôn ngữ, SEDD cạnh tranh với mô hình tự hồi quy cỡ GPT-2 và cho phép điền khuyết, không chỉ lấy mẫu trái-sang-phải. Bạn không cần lý thuyết quá trình ngẫu nhiên để thấy khẳng định Chương 04: dữ liệu là một RV rời rạc; mô hình là một PMF (hoặc tỷ số PMF); lấy mẫu là bản tính toán của “rút từ $$p$$.” D3PM của Austin và cộng sự ([NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/958c530554f2820e0d14e14d964d48d9-Abstract.html)) là công thức khuếch tán rời rạc sớm hơn vẫn được trích dẫn cạnh SEDD.

Bài phân phối lấy mẫu sau đó giải thích vì sao ước lượng Monte Carlo của perplexity hay tỷ lệ phủ cần một $$n$$ được nêu: bạn đang nhìn $$\bar{X}_n$$, không phải một lần sinh may mắn.

## Điều cần mang đi

Khi một bài báo nói “hàm hợp lý nhị thức âm,” “cường độ Hawkes nơ-ron,” hoặc “score entropy trên token,” đó là ngôn ngữ chương này. Ước lượng và kiểm định đến sau; chúng tác động trên cùng các họ có tên này.

## Nguồn

1. D. Salinas, V. Flunkert, J. Gasthaus và T. Januschowski, “DeepAR: Probabilistic forecasting with autoregressive recurrent networks,” *Int. J. Forecasting* 36(3), 2020. [DOI](https://doi.org/10.1016/j.ijforecast.2019.07.001) · [arXiv:1704.04110](https://arxiv.org/abs/1704.04110)
2. scikit-learn, `PoissonRegressor`. [tài liệu](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html)
3. H. Mei và J. Eisner, “The Neural Hawkes Process,” NeurIPS 2017. [tóm tắt](https://proceedings.neurips.cc/paper/2017/hash/6463c88460bd63bbe256e495c63aa40b-Abstract.html)
4. O. Shchur, A. C. Türkmen, T. Januschowski và S. Günnemann, “Neural Temporal Point Processes: A Review,” IJCAI 2021. [arXiv:2104.03528](https://arxiv.org/abs/2104.03528)
5. A. Lou, C. Meng và S. Ermon, “Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution,” ICML 2024. [PMLR](https://proceedings.mlr.press/v235/lou24a.html) · [arXiv:2310.16834](https://arxiv.org/abs/2310.16834)
6. J. Austin và cộng sự, “Structured Denoising Diffusion Models in Discrete State-Spaces,” NeurIPS 2021. [tóm tắt](https://proceedings.neurips.cc/paper/2021/hash/958c530554f2820e0d14e14d964d48d9-Abstract.html)
