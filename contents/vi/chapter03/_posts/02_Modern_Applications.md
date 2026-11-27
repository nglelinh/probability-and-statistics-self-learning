---
layout: post
title: 03-02-00 Ứng dụng và phát triển gần đây
chapter: "03"
order: 2
owner: nglelinh
lang: vi
categories:
- chapter03
lesson_type: optional
---

Nếu một ứng dụng thời tiết báo *"80% khả năng mưa"* vào 100 ngày khác nhau, và thực tế trời mưa đúng 80 ngày trong số đó, ta gọi ứng dụng ấy **được hiệu chỉnh tốt (well-calibrated)**. Nhưng nhiều mạng nơ-ron sâu và mô hình ngôn ngữ lớn (LLM) hiện đại lại mắc bệnh "tự tin thái quá": chúng gán xác suất 99% cho câu trả lời sai bét! Và còn một hố sâu lớn hơn nữa: nhìn thấy những người mang ô thường gặp trời mưa ($$P(\text{Mưa}\mid \text{Mang ô})$$ rất cao) hoàn toàn không có nghĩa là cất ô ở nhà sẽ làm trời nắng ($$P(\text{Mưa}\mid \mathrm{do}(\text{Không mang ô}))$$)!

Bài học này cho thấy những khái niệm sơ cấp của Chương 3 — xác suất có điều kiện, Bayes và tính độc lập — chính là chiếc chìa khóa mở cửa vào bốn chủ đề nóng nhất của AI hiện đại: **hiệu chỉnh xác suất**, **xác suất sinh token của LLM**, **Naive Bayes thực chiến**, và **suy diễn nhân quả**.

---

## Kiến thức cần nhớ: Xác suất cơ bản bước vào AI hiện đại qua bốn cánh cửa nào?

![Định lý Bayes trong Hệ thống Gợi ý và Chẩn đoán AI]({{ site.baseurl }}/img/course/ch03_02_modern_bayes.svg)

1. **Hiệu chỉnh xác suất (Calibration — ECE & Brier Score):** Đo xem con số xác suất mô hình in ra có khớp với *tần suất thực tế* hay không — quan trọng trong y tế, tín dụng và xe tự lái.
2. **Xác suất của LLM (Token Softmax vs. Độ tin cậy sự thật):** Xác suất softmax của LLM đo *độ hợp lý ngôn ngữ của token kế tiếp* theo ngữ cảnh, chứ **không** tự động là xác suất mệnh đề đó đúng sự thật ngoài đời.
3. **Naive Bayes trong thực chiến:** Giả định độc lập có điều kiện $$P(x_1,\dots,x_d\mid y) = \prod P(x_j\mid y)$$ dù "ngây thơ" nhưng chạy siêu nhanh, chống overfit tốt trên văn bản chiều cao và là baseline khó đánh bại.
4. **Đồ thị nhân quả (Causal DAG & toán tử $$\mathrm{do}$$):** Phân biệt rạch ròi giữa *quan sát thụ động* $$P(Y\mid X)$$ và *can thiệp chủ động* $$P(Y\mid \mathrm{do}(X))$$ khi có biến nhiễu ẩn (confounder).

---

## Hiệu chỉnh xác suất (Calibration): khi $$P(Y=1\mid X=x)$$ là một tần suất

Một mô hình được gọi là **hiệu chỉnh tốt (calibrated)** nếu trong số các quan sát $$x$$ được mô hình dự đoán xác suất dương bằng $$p$$, tỷ lệ thực sự thuộc lớp dương xấp xỉ đúng bằng $$p$$:

$$
P\bigl(Y=1 \mid \hat{p}(X)=p\bigr) \approx p.
$$

Sai số hiệu chỉnh kỳ vọng (Expected Calibration Error — ECE) chia các giá trị $$\hat{p}$$ thành các khoảng (bin) và so sánh tần suất dương thực tế trong từng khoảng với xác suất dự đoán trung bình của khoảng đó — cùng logic với ví dụ bệnh hiếm ở bài bắt buộc nhưng trình bày dưới dạng histogram. Guo, Pleiss, Sun và Weinberger (*On Calibration of Modern Neural Networks*, [ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)) chỉ ra rằng mạng nơ-ron sâu có thể đạt độ chính xác cao nhưng hiệu chỉnh kém; việc điều chỉnh một tham số **nhiệt độ** $$T$$ trên logit theo công thức $$\mathrm{softmax}(z/T)$$ thường cải thiện đáng kể biểu đồ độ tin cậy (reliability diagram). Minderer, Djolonga, Romijnders, Hubis, Zhai, Houlsby, Tran và Lucic (*Revisiting the Calibration of Modern Neural Networks*, [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/8420d359404024567b5aefda1231af24-Abstract.html); [arXiv:2106.07998](https://arxiv.org/abs/2106.07998)) bổ sung rằng kết quả này phụ thuộc vào kiến trúc: các mô hình Vision Transformer gần đây có thể hiệu chỉnh *tốt hơn* so với các kiến trúc CNN cũ. Với dữ liệu dạng bảng, [`CalibratedClassifierCV`](https://scikit-learn.org/stable/modules/calibration.html) trong scikit-learn (áp dụng hiệu chỉnh Platt hoặc hồi quy isotonic trên tập tách riêng) là công cụ phổ biến. Xác suất có điều kiện không phải điểm số cảm tính, mà là một phát biểu định lượng về tần suất dài hạn.

## Mô hình ngôn ngữ lớn (LLM) nói gì khi đưa ra "xác suất"?

Mô hình ngôn ngữ định nghĩa phân phối có điều kiện trên token kế tiếp, $$P(x_t\mid x_{<t})$$, tức là sự kết hợp của quy tắc nhân xác suất trên một không gian rời rạc rất lớn. Sau giai đoạn tinh chỉnh bằng RLHF, phân phối có điều kiện này thường hiệu chỉnh kém. Tian, Mitchell, Zhou, Sharma, Rafailov, Yao, Finn và Manning (*Just Ask for Calibration*, [EMNLP 2023](https://aclanthology.org/2023.emnlp-main.330/)) cho thấy mức độ tự tin được diễn đạt bằng lời (*verbalized confidence*, chẳng hạn "Tôi chắc 70%") từ ChatGPT, GPT-4 và Claude thường hiệu chỉnh tốt hơn xác suất token nội bộ của chính mô hình, giúp giảm ECE khoảng một nửa trên TriviaQA, SciQ và TruthfulQA. Sự phân biệt trong Chương 03 lý giải chính xác hiện tượng này: $$P(\text{token}\mid\text{prompt})$$ là hàm hợp lý (likelihood) trong mô hình sinh văn bản, còn $$P(\text{đúng}\mid\text{câu trả lời})$$ mới là xác suất hậu nghiệm (posterior) mà người dùng cần; hai đại lượng này chỉ trùng nhau khi mô hình xác suất được đặc tả đúng với bài toán.

## Naive Bayes vẫn được dùng rộng rãi trong thực tế

Ví dụ lọc thư rác trong bài học bắt buộc không chỉ là bài tập minh họa. [`MultinomialNB`](https://scikit-learn.org/stable/modules/naive_bayes.html) trong scikit-learn vẫn được sử dụng để lọc thư và gắn nhãn tài liệu vì dưới giả thiết **độc lập có điều kiện**

$$
P(x_1,\ldots,x_d\mid y) = \prod_{j=1}^d P(x_j\mid y),
$$

xác suất hậu nghiệm $$P(y\mid x)$$ có thể tính nhanh chỉ bằng vài phép đếm và phép nhân. Trong thực tế, người dùng hiểu rõ giới hạn của giả thiết đơn giản hóa này: các token *không* thực sự độc lập khi biết nhãn lớp, nhưng biên quyết định thu được vẫn đủ hiệu quả cho nhiều bài toán. Khi Transformer vượt trội hơn Naive Bayes, nguyên nhân cốt lõi là nó mô hình hóa được sự phụ thuộc có điều kiện $$P(x_j\mid x_{-j},y)$$ thay vì giả định $$P(x_j\mid y)$$ độc lập — tức vẫn nằm trong khung khái niệm về tính độc lập của chương này.

## Nhập môn đồ thị nhân quả: $$P(Y\mid X)$$ khác $$P(Y\mid \mathrm{do}(X))$$

Nghịch lý Simpson trong phần bài tập là lời nhắc rằng điều kiện hóa (conditioning) không đồng nghĩa với can thiệp (intervention). Sharma và Kiciman (*DoWhy*, [arXiv:2011.04216](https://arxiv.org/abs/2011.04216); [pywhy.org/dowhy](https://www.pywhy.org/dowhy/)) chuyển nguyên tắc này thành một quy trình phần mềm gồm bốn bước: mô hình hóa đồ thị nhân quả giả định, *nhận diện* (*identify*) xem hiệu ứng can thiệp $$P(Y\mid\mathrm{do}(T))$$ có biểu diễn được theo các xác suất quan sát hay không, *ước lượng* (*estimate*), và *kiểm chứng bác bỏ* (*refute*, bằng biến giả dược hoặc kiểm tra nhiễu chưa quan sát). Thư viện này không thay thế định lý Bayes, mà ngăn việc đồng nhất xác suất quan sát $$P(\text{hồi phục}\mid\text{thuốc})$$ từ dữ liệu bệnh viện với hiệu ứng nhân quả của việc chỉ định thuốc. Chỉ cần một đồ thị đơn giản gồm can thiệp $$T$$, kết cục $$Y$$ và biến nhiễu $$Z$$ cũng đủ thấy vì sao công thức xác suất có điều kiện thuần túy $$P(A\mid B)=P(A\cap B)/P(B)$$ không đủ để đưa ra quyết định chính sách.

## Tóm lược

Khi một công trình nghiên cứu báo cáo chỉ số ECE, yêu cầu mô hình ngôn ngữ nêu xác suất bằng lời, triển khai `MultinomialNB` trong hệ thống thực tế hay nhận diện hiệu ứng nhân quả bằng DoWhy, đó đều là những ứng dụng trực tiếp của các khái niệm trong chương này. Các chương tiếp theo về biến ngẫu nhiên sẽ cụ thể hóa *cách* gán giá trị số cho biến cố, nhưng không làm thay đổi ý nghĩa nền tảng của xác suất có điều kiện $$P(\,\cdot\mid\,\cdot\,)$$.

### Giải thích: tự tay tính Sai số Hiệu chỉnh (ECE) và phân biệt $$P(Y\mid X)$$ với $$P(Y\mid \mathrm{do}(X))$$

1. **Tính tay Expected Calibration Error (ECE) trên 100 dự đoán:**
   Chia 100 dự đoán của một mô hình AI thành 2 nhóm (bin) theo mức độ tự tin:
   - **Nhóm 1 (60 mẫu):** Mô hình báo tự tin trung bình $$\text{conf}_1 = 0.90$$, nhưng độ chính xác thực tế chỉ đạt $$\text{acc}_1 = 0.70$$ (tự tin thái quá 20%!).
   - **Nhóm 2 (40 mẫu):** Mô hình báo tự tin trung bình $$\text{conf}_2 = 0.60$$, độ chính xác thực tế đúng $$\text{acc}_2 = 0.60$$ (hiệu chỉnh hoàn hảo).
   Tính sai số hiệu chỉnh kỳ vọng (trung bình có trọng số theo số mẫu):
   $$
   \mathrm{ECE} = \frac{60}{100}\lvert 0.90 - 0.70\rvert + \frac{40}{100}\lvert 0.60 - 0.60\rvert = 0.6(0.20) + 0.4(0) = 0.12\;(12\%).
   $$
   Một mô hình có $$\mathrm{ECE} = 12\%$$ cần được hiệu chỉnh lại (ví dụ bằng *Temperature Scaling*) trước khi dùng cho các quyết định rủi ro cao!

2. **Tại sao $$P(Y\mid X) \neq P(Y\mid \mathrm{do}(X))$$ (Ví dụ phong vũ biểu và trời mưa):**
   Giả sử bão ($$Z$$) gây ra cả hai hiện tượng: làm kim phong vũ biểu tụt xuống ($$X=1$$) VÀ làm trời mưa ($$Y=1$$).
   - **Xác suất quan sát có điều kiện:** Thấy kim phong vũ biểu tụt tự nhiên, khả năng trời sắp mưa rất cao: $$P(Y=1 \mid X=1) = 0.85$$.
   - **Xác suất can thiệp:** Bạn thò tay vặn cưỡng bức kim phong vũ biểu xuống ($$\mathrm{do}(X=1)$$). Hành động này cắt đứt mũi tên từ bão $$Z \to X$$, chẳng ảnh hưởng gì tới bầu trời:
     $$P(Y=1 \mid \mathrm{do}(X=1)) = P(Y=1) = 0.20 \neq 0.85!$$
   Đây chính là ranh giới quan trọng giữa *mô hình dự đoán* và *mô hình ra quyết định nhân quả*.

## Video tham khảo

- [Probability Calibration : Data Science Concepts](https://www.youtube.com/watch?v=AunotauS5yI) — ritvikmath, tiếng Anh. **Xem trước.** Giải thích khái niệm hiệu chỉnh xác suất — tần suất thực tế trong từng khoảng phải khớp với xác suất dự đoán — tương ứng định nghĩa ECE và biểu đồ độ tin cậy trong bài.
- [Naive Bayes, Clearly Explained!!!](https://www.youtube.com/watch?v=O2L2Uv9pdDA) — StatQuest with Josh Starmer, tiếng Anh. Trình bày cách Naive Bayes nhân các xác suất có điều kiện dưới giả thiết độc lập, gắn với mục ứng dụng lọc thư rác.
- [Simpson's Paradox](https://www.youtube.com/watch?v=ebEkn-BiW5k) — minutephysics, tiếng Anh. Minh họa nghịch lý Simpson khi việc điều kiện hóa theo nhóm làm đảo chiều kết luận, giải thích lý do cần tách biệt $$P(Y\mid X)$$ và $$P(Y\mid \mathrm{do}(X))$$.

## Nguồn

1. C. Guo, G. Pleiss, Y. Sun và K. Q. Weinberger, “On Calibration of Modern Neural Networks,” ICML 2017. [PMLR](https://proceedings.mlr.press/v70/guo17a.html)
2. M. Minderer và cộng sự, “Revisiting the Calibration of Modern Neural Networks,” NeurIPS 2021. [tóm tắt](https://proceedings.neurips.cc/paper/2021/hash/8420d359404024567b5aefda1231af24-Abstract.html) · [arXiv:2106.07998](https://arxiv.org/abs/2106.07998)
3. K. Tian và cộng sự, “Just Ask for Calibration,” EMNLP 2023. [ACL Anthology](https://aclanthology.org/2023.emnlp-main.330/)
4. scikit-learn, “Probability calibration” và “Naive Bayes.” [calibration](https://scikit-learn.org/stable/modules/calibration.html) · [NB](https://scikit-learn.org/stable/modules/naive_bayes.html)
5. A. Sharma và E. Kiciman, “DoWhy: An End-to-End Library for Causal Inference,” 2020. [arXiv:2011.04216](https://arxiv.org/abs/2011.04216) · [tài liệu](https://www.pywhy.org/dowhy/)
