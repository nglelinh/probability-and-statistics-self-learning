---
layout: post
title: 01-11-00 Ứng dụng và phát triển gần đây
chapter: "01"
order: 11
owner: nglelinh
lang: vi
categories:
- chapter01
lesson_type: optional
---

Monte Carlo, cross-validation và các tiêu chí thông tin trong chương này không phải bài tập lớp rồi bỏ lại. Đó là ngôn ngữ làm việc của cách hệ thống học máy hiện đại *tự đánh giá* khi không có công thức đóng. Bài tùy chọn này nối những công cụ ấy với ba phát triển khoảng 2022–2024. Các bài bắt buộc không đổi; mục tiêu là thấy *chỗ nào* mô phỏng, lấy mẫu lại và chọn mô hình xuất hiện trong bài báo và thư viện thật.

## Conformal prediction như Monte Carlo hold-out

Bộ phân loại hoặc hồi quy $$f$$ thường trả một điểm $$\hat{y}$$. Các bối cảnh rủi ro cao (chẩn đoán, xếp hạng, kiểm duyệt) cần một *tập* $$C(x)$$ phủ nhãn thật với xác suất do người dùng chọn. **Split conformal prediction** làm việc đó bằng đúng ý tưởng tính toán của tập validation. Khớp $$f$$ trên fold huấn luyện; trên fold hiệu chỉnh rời kích thước $$n$$ tính điểm không phù hợp $$s_i = s(x_i, y_i)$$ (ví dụ phần dư tuyệt đối); lấy phân vị thực nghiệm

$$
\hat{q} = \text{Quantile}_{1-\alpha}\bigl(s_1,\ldots,s_n,+\infty\bigr).
$$

Tập dự đoán $$C(x)=\{y : s(x,y)\le \hat{q}\}$$ khi đó thỏa, dưới giả thiết trao đổi được (exchangeability),

$$
P\bigl(Y_{n+1}\in C(X_{n+1})\bigr) \ge 1-\alpha.
$$

Không cần nhiễu Gauss, không cần mô hình đúng đặc tả—chỉ cần tính trao đổi cũng biện minh cho cross-validation thông thường. Chuyên khảo của Angelopoulos và Bates (*Foundations and Trends in Machine Learning*, 2023; [arXiv:2107.07511](https://arxiv.org/abs/2107.07511)) là công thức mà phần lớn phần mềm 2023–2026 trích dẫn. Thư viện scikit-learn-contrib [MAPIE](https://mapie.readthedocs.io/) (Cordier, Blot, Lacombe, Morzadec, Capitaine, Brunel, COPA 2023) bọc cùng phân vị quanh bất kỳ ước lượng nào. Khi gặp khoảng bootstrap ở các chương sau, hãy đọc lại đoạn này: conformal thay “giả định một phân phối lấy mẫu” bằng “tái sử dụng mẫu hiệu chỉnh”—đúng tư duy Chương 01.

## Mô phỏng để đánh giá LLM

Không có độ chính xác dạng đóng cho mô hình hội thoại. Liang, Bommasani, Lee và cộng sự (*Holistic Evaluation of Language Models*, [TMLR 2023](https://openreview.net/forum?id=iO4LZibEqW); [arXiv:2211.09110](https://arxiv.org/abs/2211.09110); bộ công cụ [stanford-crfm/helm](https://github.com/stanford-crfm/helm)) coi đánh giá như một thí nghiệm được thiết kế: 16 kịch bản cốt lõi, bảy metric (accuracy, **calibration**, robustness, fairness, bias, toxicity, efficiency), và 30 mô hình chạy dưới một giao thức chung. Luật số lớn ở bài mở đầu chính là thứ khiến trung bình đa kịch bản có nghĩa; một con số headline là Monte Carlo một lần rút.

Sở thích của người dùng còn kém giải tích hơn. Chiang, Zheng, Sheng, Angelopoulos, Li, Li, Zhu, Zhang, Jordan, Gonzalez và Stoica (*Chatbot Arena*, [ICML 2024](https://proceedings.mlr.press/v235/chiang24b.html); [arXiv:2403.04132](https://arxiv.org/abs/2403.04132)) thu phiếu so cặp ẩn danh và ước lượng sức mạnh Bradley–Terry $$\theta_i$$ cho mỗi mô hình. Nếu $$i$$ thắng $$j$$ với xác suất

$$
P(i \succ j) = \frac{e^{\theta_i}}{e^{\theta_i}+e^{\theta_j}},
$$

thì xếp hạng là bài toán ước lượng thống kê, không phải ảnh chụp bảng xếp hạng. Khoảng tin cậy và lấy mẫu cặp thích nghi trong bài đó là cùng trực giác Monte Carlo / resampling như mô phỏng bài toán sinh nhật: bạn không suy ra $$P(i\succ j)$$ từ một likelihood của “trí tuệ”; bạn *đếm các lần so sánh* và lượng hóa bất định.

## Cross-validation thực sự ước lượng cái gì

Bài CV Chương 01 coi điểm CV như ước lượng test error của “cái” mô hình đã khớp. Bates, Hastie và Tibshirani (*Journal of the American Statistical Association*, 2023; [PMC11412612](https://pmc.ncbi.nlm.nih.gov/articles/PMC11412612/)) chứng minh rằng, ngay với bình phương tối thiểu, CV $$K$$-fold ước lượng lỗi dự đoán *trung bình* của các mô hình huấn luyện trên những lần rút tập huấn luyện khác—không phải lỗi của đúng $$\hat{f}$$ bạn sẽ đưa vào sản xuất. Lỗi theo fold tương quan (mỗi điểm vừa dùng để train vừa dùng để test), nên sai số chuẩn ngây thơ quá nhỏ và khoảng tin cậy thông thường thiếu phủ. Cách sửa của họ là CV **lồng** chỉ để ước lượng phương sai đó, đúng cảnh báo “dùng dữ liệu hai lần, hãy cẩn thận” trong bài resampling bắt buộc. Trong sklearn đó là `cross_val_score` trong vòng ngoài, hoặc `GridSearchCV` lồng trong một splitter khác—không phải một `best_score_` lạc quan.

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score

X, y = make_regression(n_samples=200, n_features=20, noise=8.0, random_state=0)
inner = KFold(n_splits=5, shuffle=True, random_state=0)
outer = KFold(n_splits=5, shuffle=True, random_state=1)
search = GridSearchCV(Ridge(), {"alpha": np.logspace(-2, 2, 9)}, cv=inner)
# CV ngoài ước lượng lỗi của *quy trình chọn mô hình*, không của một alpha.
scores = cross_val_score(search, X, y, cv=outer)
print("nested CV mean:", scores.mean(), "sd:", scores.std(ddof=1))
```

Đoạn mã không tuyên bố định lý mới. Đó là sự phân biệt Bates–Hastie–Tibshirani bằng code: đại lượng bạn báo cáo là trung bình trên các mô hình được khớp lại và tinh chỉnh lại.

## Điều cần mang đi

Khi một bài báo nói “chúng tôi conformal hóa hộp đen,” “chúng tôi báo điểm Arena kèm khoảng,” hoặc “chúng tôi lồng bộ tinh chỉnh,” đó là ngôn ngữ chương này. Các bài sau thêm biến ngẫu nhiên và kiểm định; chúng không thay thế mô phỏng, resampling, hay thói quen hỏi *loại lỗi nào* mà một tiêu chí đang ước lượng.

## Nguồn

1. A. N. Angelopoulos và S. Bates, “Conformal Prediction: A Gentle Introduction,” *Foundations and Trends in Machine Learning* 16(4), 2023. [DOI](https://doi.org/10.1561/2200000101) · [arXiv:2107.07511](https://arxiv.org/abs/2107.07511)
2. T. Cordier và cộng sự, “Flexible and Systematic Uncertainty Estimation with Conformal Prediction via the MAPIE Library,” COPA 2023. [tài liệu](https://mapie.readthedocs.io/)
3. P. Liang và cộng sự, “Holistic Evaluation of Language Models,” *TMLR*, 2023. [OpenReview](https://openreview.net/forum?id=iO4LZibEqW) · [arXiv:2211.09110](https://arxiv.org/abs/2211.09110)
4. W.-L. Chiang và cộng sự, “Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference,” ICML 2024. [PMLR](https://proceedings.mlr.press/v235/chiang24b.html) · [arXiv:2403.04132](https://arxiv.org/abs/2403.04132)
5. S. Bates, T. Hastie và R. Tibshirani, “Cross-Validation: What Does It Estimate and How Well Does It Do It?,” *JASA*, 2023. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11412612/)
