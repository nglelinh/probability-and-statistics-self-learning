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

Mô phỏng Monte Carlo, kiểm định chéo (cross-validation) và các tiêu chí thông tin trong chương này không chỉ dừng ở bài tập minh họa, mà là công cụ thực tế giúp các hệ thống học máy hiện đại tự đánh giá khi không có công thức dạng đóng. Bài đọc tùy chọn này kết nối các khái niệm đó với ba hướng phát triển giai đoạn 2022–2024, giúp bạn thấy rõ nơi mô phỏng, lấy mẫu lại và lựa chọn mô hình xuất hiện trong các công trình nghiên cứu và thư viện thực tế.

## Conformal prediction như một quy trình kiểm định tách mẫu

Một mô hình phân loại hoặc hồi quy $$f$$ thông thường chỉ trả về một giá trị dự đoán điểm $$\hat{y}$$. Trong các bài toán đòi hỏi kiểm soát rủi ro cao (chẩn đoán y khoa, xếp hạng, kiểm duyệt nội dung), người dùng cần một *tập dự đoán* $$C(x)$$ chứa nhãn thật với mức xác suất đảm bảo cho trước. **Split conformal prediction** thực hiện điều này bằng chính nguyên lý của tập kiểm định tách riêng (hold-out): ta khớp mô hình $$f$$ trên tập huấn luyện; trên tập hiệu chỉnh (calibration fold) độc lập kích thước $$n$$, tính điểm không phù hợp (nonconformity score) $$s_i = s(x_i, y_i)$$ (ví dụ phần dư tuyệt đối); rồi lấy phân vị thực nghiệm

$$
\hat{q} = \text{Quantile}_{1-\alpha}\bigl(s_1,\ldots,s_n,+\infty\bigr).
$$

Khi đó, dưới giả thiết tính trao đổi được (exchangeability) của dữ liệu, tập dự đoán $$C(x)=\{y : s(x,y)\le \hat{q}\}$$ đảm bảo mức độ phủ hữu hạn mẫu:

$$
P\bigl(Y_{n+1}\in C(X_{n+1})\bigr) \ge 1-\alpha.
$$

Kết quả này không yêu cầu nhiễu tuân theo phân phối chuẩn và cũng không đòi hỏi mô hình phải đúng đặc tả — chỉ cần giả thiết trao đổi được, tương tự nền tảng của kiểm định chéo thông thường. Chuyên khảo của Angelopoulos và Bates (*Foundations and Trends in Machine Learning*, 2023; [arXiv:2107.07511](https://arxiv.org/abs/2107.07511)) trình bày công thức được phần lớn thư viện giai đoạn 2023–2026 tham chiếu, còn thư viện thuộc hệ sinh thái scikit-learn-contrib [MAPIE](https://mapie.readthedocs.io/) (Cordier, Blot, Lacombe, Morzadec, Capitaine, Brunel, COPA 2023) cho phép áp dụng quy trình hiệu chỉnh phân vị này lên nhiều bộ ước lượng khác nhau. Khi học đến khoảng tin cậy bootstrap ở các chương sau, có thể đối chiếu lại ý tưởng này: conformal prediction thay việc giả định một phân phối lấy mẫu bằng việc tái sử dụng trực tiếp tập hiệu chỉnh — đúng tinh thần tính toán của Chương 01.

## Mô phỏng trong đánh giá mô hình ngôn ngữ lớn (LLM)

Không có công thức dạng đóng nào để đo độ chính xác của một mô hình hội thoại. Liang, Bommasani, Lee và cộng sự (*Holistic Evaluation of Language Models*, [TMLR 2023](https://openreview.net/forum?id=iO4LZibEqW); [arXiv:2211.09110](https://arxiv.org/abs/2211.09110); bộ công cụ [stanford-crfm/helm](https://github.com/stanford-crfm/helm)) xem đánh giá LLM như một thí nghiệm thống kê có thiết kế chặt chẽ: 16 kịch bản cốt lõi, 7 nhóm thước đo (accuracy, **calibration**, robustness, fairness, bias, toxicity, efficiency) áp dụng thống nhất trên 30 mô hình theo cùng một giao thức. Chính Luật số lớn ở bài mở đầu đảm bảo cho giá trị trung bình trên nhiều kịch bản có ý nghĩa thống kê, thay vì chỉ dựa vào một con số đơn lẻ như một lần rút mẫu Monte Carlo.

Mức độ ưa thích của người dùng thậm chí còn khó mô hình hóa bằng giải tích hơn. Chiang, Zheng, Sheng, Angelopoulos, Li, Li, Zhu, Zhang, Jordan, Gonzalez và Stoica (*Chatbot Arena*, [ICML 2024](https://proceedings.mlr.press/v235/chiang24b.html); [arXiv:2403.04132](https://arxiv.org/abs/2403.04132)) thu thập phiếu so sánh cặp ẩn danh và ước lượng tham số năng lực Bradley–Terry $$\theta_i$$ cho từng mô hình. Với xác suất mô hình $$i$$ thắng mô hình $$j$$ là

$$
P(i \succ j) = \frac{e^{\theta_i}}{e^{\theta_i}+e^{\theta_j}},
$$

việc xếp hạng trở thành một bài toán ước lượng thống kê thay vì một bảng điểm cố định. Các khoảng tin cậy và chiến lược lấy mẫu cặp thích nghi trong nghiên cứu này chia sẻ cùng trực giác Monte Carlo và lấy mẫu lại như bài toán sinh nhật: thay vì suy diễn $$P(i\succ j)$$ từ một giả thuyết giải tích về năng lực mô hình, ta đếm kết quả các lượt so sánh lặp lại và lượng hóa độ bất định của ước lượng.

## Cross-validation thực sự ước lượng đại lượng nào?

Bài học về cross-validation trong Chương 01 thường xem điểm CV như ước lượng sai số kiểm tra của chính mô hình $$\hat{f}$$ vừa huấn luyện. Tuy nhiên, Bates, Hastie và Tibshirani (*Journal of the American Statistical Association*, 2023; [PMC11412612](https://pmc.ncbi.nlm.nih.gov/articles/PMC11412612/)) chứng minh rằng, ngay cả với hồi quy bình phương tối thiểu, kiểm định chéo $$K$$-fold thực chất ước lượng sai số dự đoán *trung bình* của các mô hình được huấn luyện trên những lần rút tập huấn luyện khác nhau — chứ không phải sai số riêng của mô hình $$\hat{f}$$ cụ thể sẽ được triển khai. Vì sai số giữa các fold có tương quan (mỗi điểm dữ liệu vừa tham gia huấn luyện ở fold này vừa dùng để kiểm định ở fold khác), sai số chuẩn tính theo công thức độc lập thông thường sẽ bị đánh giá thấp, khiến khoảng tin cậy có độ phủ thấp hơn mức danh nghĩa. Hướng khắc phục của nhóm tác giả là dùng kiểm định chéo **lồng nhau (nested CV)** để ước lượng phương sai này — phản ánh cùng nguyên tắc thận trọng khi tái sử dụng dữ liệu nhiều lần trong bài học về resampling. Trong scikit-learn, quy trình này tương ứng với việc đặt `GridSearchCV` bên trong một vòng `cross_val_score` ngoài, thay vì chỉ báo cáo giá trị `best_score_` vốn có xu hướng lạc quan.

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

Đoạn mã trên minh họa trực tiếp phân biệt của Bates, Hastie và Tibshirani: đại lượng được báo cáo là hiệu suất trung bình của cả quy trình huấn luyện và tinh chỉnh lại mô hình trên các tập dữ liệu khác nhau, không phải của một giá trị `alpha` cố định.

## Tóm lược

Khi một bài báo trình bày việc hiệu chỉnh tập dự đoán bằng conformal prediction cho mô hình hộp đen, báo cáo bảng xếp hạng Arena kèm khoảng tin cậy, hay dùng kiểm định chéo lồng nhau để đánh giá quy trình tinh chỉnh, đó đều là những ứng dụng trực tiếp của các ý tưởng trong chương này. Các chương tiếp theo sẽ bổ sung lý thuyết biến ngẫu nhiên và kiểm định giả thuyết, nhưng mô phỏng, lấy mẫu lại và thói quen xác định rõ *đại lượng sai số nào đang được ước lượng* vẫn luôn là nền tảng xuyên suốt.

## Video tham khảo

- [A Tutorial on Conformal Prediction](https://www.youtube.com/watch?v=nql000Lu_iE) — Anastasios Nikolas Angelopoulos, tiếng Anh. **Xem trước.** Trình bày split conformal prediction từ điểm không phù hợp trên tập hiệu chỉnh đến bước lấy phân vị, tương ứng chuyên khảo arXiv:2107.07511 trong bài.
- [Holistic Evaluation of Language Models (HELM) - Yifan Mai, Stanford University](https://www.youtube.com/watch?v=fLoYcXLG6ro) — PyTorch, tiếng Anh. Yifan Mai giới thiệu khung đánh giá đa kịch bản HELM, tương ứng góc nhìn thí nghiệm Monte Carlo cho mô hình ngôn ngữ trong bài.
- [11.5 Nested CV for Algorithm Selection (L11 Model Eval. Part 4)](https://www.youtube.com/watch?v=XXFLFWHP9Nc) — Sebastian Raschka, tiếng Anh. Giải thích kiểm định chéo lồng nhau để đánh giá toàn bộ quy trình chọn mô hình, gắn trực tiếp với kết quả của Bates, Hastie và Tibshirani.

## Nguồn

1. A. N. Angelopoulos và S. Bates, “Conformal Prediction: A Gentle Introduction,” *Foundations and Trends in Machine Learning* 16(4), 2023. [DOI](https://doi.org/10.1561/2200000101) · [arXiv:2107.07511](https://arxiv.org/abs/2107.07511)
2. T. Cordier và cộng sự, “Flexible and Systematic Uncertainty Estimation with Conformal Prediction via the MAPIE Library,” COPA 2023. [tài liệu](https://mapie.readthedocs.io/)
3. P. Liang và cộng sự, “Holistic Evaluation of Language Models,” *TMLR*, 2023. [OpenReview](https://openreview.net/forum?id=iO4LZibEqW) · [arXiv:2211.09110](https://arxiv.org/abs/2211.09110)
4. W.-L. Chiang và cộng sự, “Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference,” ICML 2024. [PMLR](https://proceedings.mlr.press/v235/chiang24b.html) · [arXiv:2403.04132](https://arxiv.org/abs/2403.04132)
5. S. Bates, T. Hastie và R. Tibshirani, “Cross-Validation: What Does It Estimate and How Well Does It Do It?,” *JASA*, 2023. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11412612/)
