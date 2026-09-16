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

Loại dữ liệu, đặc trưng số và biểu đồ trong chương này trông sơ đẳng cho đến khi một pipeline coi chúng là phần phụ. Hệ thống thị giác máy tính và bảng biểu hiện đại hỏng trước hết ở *cái gì đã được đo*, *cách nó được tóm tắt*, và *điều bức tranh che*. Bài tùy chọn này ánh xạ ba kỹ năng đó sang thực hành lấy dữ liệu làm trung tâm khoảng 2021–2025. Lý thuyết bắt buộc không đổi; mục tiêu là thấy cùng thang đo và tóm tắt bên trong thiết kế tập dữ liệu, làm sạch nhãn, và biểu đồ embedding.

## Thiết kế tập dữ liệu như thống kê mô tả ở quy mô lớn

Gadre, Ilharco, Fang và cộng sự (*DataComp*, [NeurIPS 2023 Datasets and Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2023/hash/56332d41d55ad7ad8024aac625881be7-Abstract-Datasets_and_Benchmarks.html); [arXiv:2304.14108](https://arxiv.org/abs/2304.14108); [datacomp.ai](https://www.datacomp.ai)) đóng băng mã huấn luyện của một mô hình CLIP và *thay tập huấn luyện*. Một bể 12,8 tỷ cặp ảnh–văn bản (CommonPool) được lọc bằng các quy tắc kiểu Chương 02: văn bản tiếng Anh, điểm tương đồng CLIP, khử trùng lặp, và các ngưỡng chất lượng liên quan. Tập con tốt nhất của họ, DataComp-1B (1,4 tỷ cặp), huấn luyện ViT-L/14 đạt 79,2% ImageNet zero-shot—cao hơn CLIP ViT-L/14 của OpenAI 3,7 điểm ở cùng compute. Bài học thẳng: trung bình độ tương đồng, tần suất nhãn ngôn ngữ, và histogram điểm số không phải “bài EDA.” Chúng *là* can thiệp. Bản tương ứng cho mô hình ngôn ngữ là DataComp-LM (Jeffrey Li, Fang, Smyrnis, Ivgi và cộng sự, NeurIPS 2024; [arXiv:2406.11794](https://arxiv.org/abs/2406.11794)). Khi bạn tính trung vị hay loại outlier trong Pandas, bạn đang chạy một bộ lọc DataComp thu nhỏ.

## Lỗi nhãn và tóm tắt vững

Nhãn lớp định danh là một thang đo. Nếu 5% thẻ ImageNet sai, trung bình mẫu của “độ chính xác” là ước lượng vị trí bị nhiễm—cùng độ nhạy mà ví dụ mean-versus-median ở bài đặc trưng số đã chỉ ra. Northcutt, Jiang và Chuang (*Confident Learning*, [JAIR 2021](https://doi.org/10.1613/jair.1.12125)) ước lượng phân phối đồng thời của nhãn nhiễu $$\tilde{Y}$$ và nhãn sạch ẩn $$Y$$ từ xác suất dự đoán ngoài mẫu, rồi gắn cờ các lỗi khả dĩ. Gói mã nguồn mở [cleanlab](https://docs.cleanlab.ai/) là dạng sản xuất. `RobustScaler` của sklearn (trung vị và IQR thay cho trung bình và độ lệch chuẩn) là bản liên tục: chọn tóm tắt khớp thang đo và mức nhiễm bạn thực sự có.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

rng = np.random.default_rng(0)
x = rng.normal(size=(200, 1))
x[0] = 40.0  # một cảm biến hỏng
print("mean / sd:", StandardScaler().fit(x).mean_[0], StandardScaler().fit(x).scale_[0])
print("median / IQR-scale:", RobustScaler().fit(x).center_[0], RobustScaler().fit(x).scale_[0])
```

Trung bình in ra nhảy; trung vị gần như đứng yên. Đó không phải công thức mới. Đó là lý do các bài lấy dữ liệu làm trung tâm nói *làm sạch* trước *khớp mô hình*.

## Hình nói dối: embedding như biểu đồ gây hiểu lầm hiện đại

Bộ tứ Anscombe ở bài trực quan đã cho bốn tập có cùng mean, phương sai, tương quan nhưng bốn hình khác nhau. Phiên bản những năm 2020 là t-SNE hoặc UMAP hai chiều của một ma trận đếm 2.000 gene. Chari và Pachter (*PLOS Computational Biology*, 2023; [doi:10.1371/journal.pcbi.1011288](https://doi.org/10.1371/journal.pcbi.1011288)) chỉ ra rằng nén hàng trăm chiều xuống hai *nhất thiết* làm méo khoảng cách và lân cận, rồi nhà sinh học lại coi đám mây điểm như chính dữ liệu. Đạo đức Chương 02 vẫn đúng: một biểu đồ là một *thống kê*. Nếu thống kê đó là embedding phi tuyến, hãy nói rõ nó giữ gì và bịa gì. Ưu tiên các góc nhìn có mục tiêu (một gene, một điểm cụm) hơn một bản đồ “tất cả trong một.”

## Kiểu hỗn hợp và prior trên bảng

Hollmann, Müller, Eggensperger và Hutter (*TabPFN*, [ICLR 2023](https://iclr.cc/virtual/2023/poster/12113); [arXiv:2207.01848](https://arxiv.org/abs/2207.01848)) huấn luyện một Transformer một lần trên các bảng tổng hợp rút từ prior gồm biến số hỗn hợp và đồ thị nhân quả đơn giản, rồi phân loại một bảng *mới* nhỏ trong một lượt lan truyền. Bài Nature 2025 mở rộng cùng ý. Bạn không cần kiến trúc để dùng điểm thống kê: việc với bảng bắt đầu bằng việc tuyên bố cột nào là định danh, thứ bậc, hay tỷ lệ—bài đầu chương này—và một phương pháp bỏ qua tuyên bố đó không phải “end-to-end,” mà là đặc tả sai.

## Điều cần mang đi

Khi một bài báo nói “chúng tôi lọc Common Crawl theo điểm CLIP,” “chúng tôi dùng cleanlab trên nhãn,” hoặc “đừng gom cụm trên UMAP,” đó là ngôn ngữ Chương 02. Các chương xác suất sau thêm mô hình lấy mẫu; chúng không thay thế kỷ luật về kiểu, tóm tắt, và hình.

## Nguồn

1. S. Y. Gadre và cộng sự, “DataComp: In search of the next generation of multimodal datasets,” NeurIPS 2023. [tóm tắt](https://proceedings.neurips.cc/paper_files/paper/2023/hash/56332d41d55ad7ad8024aac625881be7-Abstract-Datasets_and_Benchmarks.html) · [arXiv:2304.14108](https://arxiv.org/abs/2304.14108)
2. J. Li và cộng sự, “DataComp-LM: In search of the next generation of training sets for language models,” NeurIPS 2024. [arXiv:2406.11794](https://arxiv.org/abs/2406.11794)
3. C. G. Northcutt, L. Jiang và I. L. Chuang, “Confident Learning: Estimating Uncertainty in Dataset Labels,” *JAIR* 70, 2021. [DOI](https://doi.org/10.1613/jair.1.12125) · [cleanlab](https://docs.cleanlab.ai/)
4. T. Chari và L. Pachter, “The specious art of single-cell genomics,” *PLOS Comput. Biol.* 19(8), 2023. [DOI](https://doi.org/10.1371/journal.pcbi.1011288)
5. N. Hollmann, S. Müller, K. Eggensperger và F. Hutter, “TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second,” ICLR 2023. [ICLR](https://iclr.cc/virtual/2023/poster/12113) · [arXiv:2207.01848](https://arxiv.org/abs/2207.01848)
6. scikit-learn, `RobustScaler`. [tài liệu](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)
