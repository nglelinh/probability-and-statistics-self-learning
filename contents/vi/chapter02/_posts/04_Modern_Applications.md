---
layout: post
title: 02-04-00 Ứng dụng và phát triển gần đây
chapter: "02"
order: 4
owner: nglelinh
lang: vi
categories:
- chapter02
lesson_type: optional
---

Ở kỷ nguyên của những kho dữ liệu hàng tỷ dòng và các bản đồ embedding hàng nghìn chiều, có người cho rằng thống kê mô tả của Chương 2 đã trở nên "quá sơ đẳng". Thực tế tại các phòng thí nghiệm AI hàng đầu lại chứng minh điều ngược lại: ngay cả những bộ dữ liệu danh tiếng nhất như ImageNet hay các tập huấn luyện LLM cũng chứa tỷ lệ lỗi nhãn đáng kể, và hàng loạt hình ảnh t-SNE / UMAP đẹp mắt trên báo chí lại thường tạo ra những "cụm ảo" không hề tồn tại trong không gian gốc!

Bài học này nối thẳng thống kê mô tả cổ điển với thực hành khoa học dữ liệu hiện đại: từ kiểm toán tập dữ liệu quy mô lớn, các thước đo thống kê vững (robust statistics) chống lỗi nhãn, cho đến cạm bẫy hình học khi trực quan hóa không gian embedding.

---

## Kiến thức cần nhớ: Thống kê mô tả hiện đại ở quy mô dữ liệu lớn gồm những trụ cột nào?

![EDA Tự động và Phát hiện Trôi dạt Dữ liệu (Data Drift)]({{ site.baseurl }}/img/course/ch02_04_modern_eda.svg)

1. **Kiểm toán & Thiết kế tập dữ liệu (Data-Centric AI):** Dùng bảng chéo, phân phối nhóm con và kiểm tra trùng lặp để phát hiện thiên lệch và rò rỉ dữ liệu ngay trước khi huấn luyện mô hình.
2. **Thống kê vững trước lỗi nhãn (Robust Summary Statistics):** Thay Mean/SD bằng Median, Trimmed Mean, hàm mất mát Huber và **MAD (Median Absolute Deviation)** để điểm nhiễu cực đoan không phá hỏng kết quả đo.
3. **Giới hạn của trực quan hóa phi tuyến (t-SNE, UMAP):** Các thuật toán giảm chiều xuống 2D chỉ giữ *láng giềng cục bộ*, thường làm méo khoảng cách toàn cục và kích thước cụm — không bao giờ dùng hình UMAP để kết luận "cụm A xa cụm B gấp đôi cụm C".
4. **Dữ liệu bảng hỗn hợp (Mixed-Type Tabular Data):** Kết hợp biến liên tục, biến phân loại nhiều mức và cấu trúc phân phối tiên nghiệm trong các mô hình hiện đại (như TabPFN hay Gradient Boosted Trees).

---

## Thiết kế tập dữ liệu như thống kê mô tả ở quy mô lớn

Gadre, Ilharco, Fang và cộng sự (*DataComp*, [NeurIPS 2023 Datasets and Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2023/hash/56332d41d55ad7ad8024aac625881be7-Abstract-Datasets_and_Benchmarks.html); [arXiv:2304.14108](https://arxiv.org/abs/2304.14108); [datacomp.ai](https://www.datacomp.ai)) cố định mã huấn luyện của một mô hình CLIP và chỉ *thay đổi tập dữ liệu huấn luyện*. Từ kho 12,8 tỷ cặp ảnh–văn bản (CommonPool), nhóm tác giả áp dụng các quy tắc lọc mang đúng tinh thần thống kê mô tả của Chương 02: lọc ngôn ngữ tiếng Anh, ngưỡng điểm tương đồng CLIP, khử trùng lặp và các tiêu chí chất lượng liên quan. Tập con tốt nhất của họ, DataComp-1B (1,4 tỷ cặp), giúp ViT-L/14 đạt 79,2% độ chính xác zero-shot trên ImageNet — cao hơn 3,7 điểm phần trăm so với CLIP ViT-L/14 của OpenAI ở cùng ngân sách tính toán. Bài học rất rõ ràng: trung bình độ tương đồng, tần suất nhãn ngôn ngữ và histogram điểm số không chỉ là bước khám phá dữ liệu ban đầu, mà là can thiệp trực tiếp lên hiệu năng mô hình. Hướng tiếp cận tương tự cho mô hình ngôn ngữ được trình bày trong DataComp-LM (Jeffrey Li, Fang, Smyrnis, Ivgi và cộng sự, NeurIPS 2024; [arXiv:2406.11794](https://arxiv.org/abs/2406.11794)). Khi bạn tính trung vị hay lọc giá trị ngoại lai trong Pandas, bạn đang thực hiện phiên bản thu nhỏ của chính quy trình DataComp.

## Lỗi nhãn và các đặc trưng tóm tắt vững

Nhãn lớp định danh cũng là một thang đo. Nếu 5% nhãn trong ImageNet bị gán sai, trung bình mẫu của "độ chính xác" trở thành một ước lượng vị trí bị nhiễu — cùng hiện tượng nhạy cảm với ngoại lai đã thấy khi so sánh trung bình cộng và trung vị ở bài đặc trưng số. Northcutt, Jiang và Chuang (*Confident Learning*, [JAIR 2021](https://doi.org/10.1613/jair.1.12125)) ước lượng phân phối đồng thời giữa nhãn nhiễu $$\tilde{Y}$$ và nhãn sạch ẩn $$Y$$ từ xác suất dự đoán ngoài mẫu, qua đó phát hiện các mẫu có khả năng sai nhãn. Thư viện mã nguồn mở [cleanlab](https://docs.cleanlab.ai/) triển khai phương pháp này cho ứng dụng thực tế. Với biến liên tục, `RobustScaler` trong scikit-learn (dùng trung vị và IQR thay cho trung bình và độ lệch chuẩn) thể hiện cùng nguyên tắc: chọn đại lượng tóm tắt phù hợp với thang đo và mức độ nhiễu thực tế của dữ liệu.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

rng = np.random.default_rng(0)
x = rng.normal(size=(200, 1))
x[0] = 40.0  # một cảm biến hỏng
print("mean / sd:", StandardScaler().fit(x).mean_[0], StandardScaler().fit(x).scale_[0])
print("median / IQR-scale:", RobustScaler().fit(x).center_[0], RobustScaler().fit(x).scale_[0])
```

Giá trị trung bình bị kéo lệch mạnh, trong khi trung vị hầu như không thay đổi. Đó không phải công thức mới, mà chính là lý do các phương pháp lấy dữ liệu làm trung tâm nhấn mạnh việc *làm sạch dữ liệu* trước khi *khớp mô hình*.

## Khi biểu đồ gây hiểu lầm: embedding trong thực hành hiện đại

Bộ tứ Anscombe ở bài trực quan hóa cho thấy bốn tập dữ liệu có cùng trung bình, phương sai và tương quan nhưng mang bốn hình dạng hoàn toàn khác nhau. Phiên bản hiện đại của hiện tượng này là các biểu đồ t-SNE hoặc UMAP hai chiều dựng từ ma trận đếm hàng nghìn gene. Chari và Pachter (*PLOS Computational Biology*, 2023; [doi:10.1371/journal.pcbi.1011288](https://doi.org/10.1371/journal.pcbi.1011288)) chỉ ra rằng việc nén dữ liệu từ hàng trăm chiều xuống hai chiều *tất yếu* làm biến dạng khoảng cách và cấu trúc lân cận, song nhiều phân tích sinh học vẫn diễn giải đám mây điểm 2D đó như thể chính dữ liệu gốc. Bài học của Chương 02 vẫn giữ nguyên giá trị: mọi biểu đồ đều là một *thống kê tóm tắt*. Nếu thống kê đó là một phép chiếu phi tuyến, cần nêu rõ nó bảo toàn đặc trưng nào và làm méo đặc trưng nào, đồng thời ưu tiên các góc nhìn có mục tiêu cụ thể (một gene, một chỉ số cụm) thay vì chỉ dựa vào một bản đồ tổng hợp duy nhất.

## Dữ liệu bảng hỗn hợp kiểu và phân phối tiên nghiệm (prior)

Hollmann, Müller, Eggensperger và Hutter (*TabPFN*, [ICLR 2023](https://iclr.cc/virtual/2023/poster/12113); [arXiv:2207.01848](https://arxiv.org/abs/2207.01848)) huấn luyện một mô hình Transformer trên các tập dữ liệu bảng tổng hợp sinh từ phân phối tiên nghiệm gồm nhiều kiểu biến hỗn hợp và đồ thị nhân quả đơn giản, nhờ đó có thể phân loại một tập dữ liệu bảng mới cỡ nhỏ chỉ trong một lượt lan truyền xuôi. Công trình đăng trên *Nature* (2025) tiếp tục mở rộng ý tưởng này. Ở góc độ thống kê, điểm đáng chú ý là: làm việc với dữ liệu bảng luôn bắt đầu từ việc xác định rõ cột nào là định danh, thứ bậc hay tỷ lệ (nội dung Bài 1 của chương này); bỏ qua bước xác định kiểu dữ liệu không phải là "học đầu-cuối (end-to-end)", mà là đặc tả sai mô hình.

## Tóm lược

Khi các công trình hiện đại đề cập đến việc lọc Common Crawl theo điểm CLIP, phát hiện nhãn lỗi bằng cleanlab, hay thận trọng khi gom cụm trên UMAP, đó đều là những ứng dụng trực tiếp của tư duy Chương 02. Các chương xác suất tiếp theo sẽ bổ sung mô hình sinh dữ liệu và lấy mẫu, nhưng không thay thế kỷ luật nền tảng về kiểu dữ liệu, đặc trưng tóm tắt và trực quan hóa.

### Giải thích: tự tay tính MAD để phát hiện điểm ngoại lai mà Z-score cổ điển bỏ sót

Tại sao trong dữ liệu lớn có lỗi nhãn, quy tắc quen thuộc "gắn cờ ngoại lai khi $$\lvert Z\rvert > 3$$" lại hay thất bại (hiện tượng *masking*)? Hãy thử ngay trên 6 điểm dữ liệu có **1 điểm nhập sai nhãn cực lớn**:

$$
x = \{10,\; 11,\; 12,\; 12,\; 13,\; 100\}.
$$

1. **Thử bằng Z-score cổ điển (Mean & SD):**
   - Trung bình $$\bar x = \frac{158}{6} \approx 26.33$$; độ lệch chuẩn mẫu $$s \approx 36.12$$.
   - Tính Z-score của chính kẻ phá rối $$x = 100$$:
     $$
     Z = \frac{100 - 26.33}{36.12} \approx 2.04 < 3!
     $$
   **Bất ngờ chưa?** Quy tắc $$\lvert Z\rvert > 3$$ **bỏ lọt hoàn toàn** giá trị $$100$$ — bởi chính số $$100$$ đã kéo phình cả trung bình lẫn độ lệch chuẩn lên để tự "ngụy trang" cho mình!

2. **Thử bằng Modified Z-score dựa trên Median & MAD:**
   - Trung vị: $$\tilde x = \frac{12 + 12}{2} = 12$$.
   - Độ lệch tuyệt đối so với trung vị $$\lvert x_i - 12\rvert$$: $$\{2,\; 1,\; 0,\; 0,\; 1,\; 88\}$$. Sắp xếp lại: $$\{0, 0, 1, 1, 2, 88\} \;\Longrightarrow\; \mathrm{MAD} = \text{Median} = 1.0$$.
   - Tính Modified Z-score cho $$x = 100$$ (hệ số $$0.6745$$ chuẩn hóa khớp phân phối chuẩn):
     $$
     M = 0.6745 \times \frac{100 - 12}{1.0} \approx 59.4 \gg 3.5!
     $$
   **Bắt sống ngay lập tức!** Đây là lý do trong các pipeline kiểm toán dữ liệu hiện đại, **Median + MAD** luôn là tuyến phòng thủ đầu tiên trước lỗi nhãn và cảm biến.

## Video tham khảo

- [StatQuest: t-SNE, Clearly Explained](https://www.youtube.com/watch?v=NEaUSP4YerM) — StatQuest with Josh Starmer, tiếng Anh. **Xem trước.** Giải thích cách t-SNE bảo toàn lân cận cục bộ nhưng làm biến dạng khoảng cách toàn cục, tương ứng cảnh báo của Chari và Pachter trong bài.
- [UMAP Dimension Reduction, Main Ideas!!!](https://www.youtube.com/watch?v=eN0wFzBA4Sc) — StatQuest with Josh Starmer, tiếng Anh. Giới thiệu UMAP — một phương pháp chiếu phi tuyến khác — làm rõ lý do hình chiếu 2D không đồng nhất với dữ liệu gốc.
- [GenAI and Datacomp: Creating the Largest Public Multimodal Dataset in Academia](https://www.youtube.com/watch?v=ba-aqPF6xuw) — AI Council, tiếng Anh. Trình bày cách DataComp lọc kho dữ liệu ảnh–văn bản bằng điểm số và tần suất, minh họa thống kê mô tả như một công cụ cải thiện mô hình.

## Nguồn

1. S. Y. Gadre và cộng sự, “DataComp: In search of the next generation of multimodal datasets,” NeurIPS 2023. [tóm tắt](https://proceedings.neurips.cc/paper_files/paper/2023/hash/56332d41d55ad7ad8024aac625881be7-Abstract-Datasets_and_Benchmarks.html) · [arXiv:2304.14108](https://arxiv.org/abs/2304.14108)
2. J. Li và cộng sự, “DataComp-LM: In search of the next generation of training sets for language models,” NeurIPS 2024. [arXiv:2406.11794](https://arxiv.org/abs/2406.11794)
3. C. G. Northcutt, L. Jiang và I. L. Chuang, “Confident Learning: Estimating Uncertainty in Dataset Labels,” *JAIR* 70, 2021. [DOI](https://doi.org/10.1613/jair.1.12125) · [cleanlab](https://docs.cleanlab.ai/)
4. T. Chari và L. Pachter, “The specious art of single-cell genomics,” *PLOS Comput. Biol.* 19(8), 2023. [DOI](https://doi.org/10.1371/journal.pcbi.1011288)
5. N. Hollmann, S. Müller, K. Eggensperger và F. Hutter, “TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second,” ICLR 2023. [ICLR](https://iclr.cc/virtual/2023/poster/12113) · [arXiv:2207.01848](https://arxiv.org/abs/2207.01848)
6. scikit-learn, `RobustScaler`. [tài liệu](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)
