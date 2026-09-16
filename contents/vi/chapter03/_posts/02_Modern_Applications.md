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

Không gian mẫu, xác suất có điều kiện, Bayes và độc lập trong chương này chính là những đối tượng mà hệ thống sản xuất dùng sai khi coi điểm softmax như một khả năng, likelihood token như một niềm tin, hoặc tương quan như nguyên nhân. Bài tùy chọn này giữ nguyên các suy diễn bắt buộc và chỉ chỉ ra bốn chỗ các đối tượng đó xuất hiện trong thực hành khoa học máy tính 2021–2023.

## Calibration: khi $$P(Y=1\mid X=x)$$ là một tần suất

Một mô hình được gọi là **hiệu chỉnh (calibrated)** nếu, trong mọi $$x$$ có xác suất dự đoán $$p$$, khoảng một tỷ lệ $$p$$ là dương:

$$
P\bigl(Y=1 \mid \hat{p}(X)=p\bigr) \approx p.
$$

Expected calibration error (ECE) chia $$\hat{p}$$ thành các bin và so tần suất trong bin với điểm trung bình của bin—ví dụ bệnh hiếm ở bài bắt buộc dưới dạng histogram. Guo, Pleiss, Sun và Weinberger (*On Calibration of Modern Neural Networks*, [ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)) cho thấy mạng sâu có thể vừa chính xác vừa hiệu chỉnh kém; một **nhiệt độ** $$T$$ trên logit, $$\mathrm{softmax}(z/T)$$, thường sửa sơ đồ tin cậy. Minderer, Djolonga, Romijnders, Hubis, Zhai, Houlsby, Tran và Lucic (*Revisiting the Calibration of Modern Neural Networks*, [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/8420d359404024567b5aefda1231af24-Abstract.html); [arXiv:2106.07998](https://arxiv.org/abs/2106.07998)) thấy câu chuyện phụ thuộc kiến trúc: Vision Transformer gần đây có thể hiệu chỉnh *tốt hơn* CNN cũ. [`CalibratedClassifierCV`](https://scikit-learn.org/stable/modules/calibration.html) của sklearn (Platt hoặc hồi quy isotonic trên hold-out) là bản bảng. Xác suất có điều kiện không phải điểm cảm tính; đó là một khẳng định về tần suất dài hạn.

## LLM muốn nói gì khi nói “xác suất”

Mô hình ngôn ngữ định nghĩa phân phối có điều kiện trên token tiếp theo, $$P(x_t\mid x_{<t})$$, tức quy tắc Bayes cộng quy tắc chuỗi trên một không gian rời rạc khổng lồ. Sau RLHF, điều kiện đó là độ tin kém: Tian, Mitchell, Zhou, Sharma, Rafailov, Yao, Finn và Manning (*Just Ask for Calibration*, [EMNLP 2023](https://aclanthology.org/2023.emnlp-main.330/)) cho thấy độ tin *nói ra* (“Tôi chắc 70%”) từ ChatGPT / GPT-4 / Claude thường hiệu chỉnh tốt hơn xác suất token của chính mô hình, giảm ECE khoảng một nửa trên TriviaQA, SciQ và TruthfulQA. Phân biệt Chương 03 là chính xác: $$P(\text{token}\mid\text{prompt})$$ là likelihood trong một câu chuyện sinh; $$P(\text{đúng}\mid\text{câu trả lời})$$ là posterior bạn muốn. Chúng trùng nhau chỉ khi bạn đã đặc tả đúng thí nghiệm.

## Naive Bayes vẫn đang chạy

Phép tính lọc spam ở bài bắt buộc không phải đồ chơi. [`MultinomialNB`](https://scikit-learn.org/stable/modules/naive_bayes.html) của sklearn vẫn lọc thư và gắn thẻ tài liệu vì, dưới giả thiết **độc lập có điều kiện**

$$
P(x_1,\ldots,x_d\mid y) = \prod_{j=1}^d P(x_j\mid y),
$$

posterior $$P(y\mid x)$$ chỉ là vài lần đếm và một phép nhân. Cách dùng sản xuất thành thật về lời nói dối: các token *không* độc lập khi biết lớp, nhưng biên quyết định vẫn có thể chạy. Khi Transformer thắng Naive Bayes, đó là vì nó mô hình $$P(x_j\mid x_{-j},y)$$ thay vì $$P(x_j\mid y)$$. Đó là phần độc lập của chương này, không phải một nhánh toán mới.

## Đồ thị nhân quả nhẹ: $$P(Y\mid X)$$ không phải $$P(Y\mid \mathrm{do}(X))$$

Nghịch lý Simpson trong bài tập là lời cảnh báo rằng điều kiện hóa không phải can thiệp. Sharma và Kiciman (*DoWhy*, [arXiv:2011.04216](https://arxiv.org/abs/2011.04216); [pywhy.org/dowhy](https://www.pywhy.org/dowhy/)) biến cảnh báo đó thành phần mềm: bạn vẽ đồ thị cơ chế giả định, *identify* xem $$P(Y\mid\mathrm{do}(T))$$ có phải hàm của các điều kiện quan sát được không, *estimate*, rồi *refute* (placebo, kiểm tra nhiễu không quan sát). Thư viện không thay định lý Bayes. Nó từ chối giả vờ rằng $$P(\text{hồi phục}\mid\text{thuốc})$$ tính trên dữ liệu bệnh viện là hiệu ứng của việc chỉ định thuốc. Một hình một cạnh—can thiệp $$T$$, kết cục $$Y$$, nhiễu $$Z$$—đủ để thấy vì sao đồng nhất thức bắt buộc $$P(A\mid B)=P(A\cap B)/P(B)$$ là nút sai cho chính sách.

## Điều cần mang đi

Khi một bài báo nói “chúng tôi báo ECE,” “chúng tôi hỏi mô hình một xác suất nói ra,” “chúng tôi đưa MultinomialNB vào sản xuất,” hoặc “chúng tôi identify hiệu ứng trong DoWhy,” đó là ngôn ngữ chương này. Các bài biến ngẫu nhiên ở chương sau tinh chỉnh *cách* ta gán số cho biến cố, không thay ý nghĩa của $$P(\,\cdot\mid\,\cdot\,)$$.

## Nguồn

1. C. Guo, G. Pleiss, Y. Sun và K. Q. Weinberger, “On Calibration of Modern Neural Networks,” ICML 2017. [PMLR](https://proceedings.mlr.press/v70/guo17a.html)
2. M. Minderer và cộng sự, “Revisiting the Calibration of Modern Neural Networks,” NeurIPS 2021. [tóm tắt](https://proceedings.neurips.cc/paper/2021/hash/8420d359404024567b5aefda1231af24-Abstract.html) · [arXiv:2106.07998](https://arxiv.org/abs/2106.07998)
3. K. Tian và cộng sự, “Just Ask for Calibration,” EMNLP 2023. [ACL Anthology](https://aclanthology.org/2023.emnlp-main.330/)
4. scikit-learn, “Probability calibration” và “Naive Bayes.” [calibration](https://scikit-learn.org/stable/modules/calibration.html) · [NB](https://scikit-learn.org/stable/modules/naive_bayes.html)
5. A. Sharma và E. Kiciman, “DoWhy: An End-to-End Library for Causal Inference,” 2020. [arXiv:2011.04216](https://arxiv.org/abs/2011.04216) · [tài liệu](https://www.pywhy.org/dowhy/)
