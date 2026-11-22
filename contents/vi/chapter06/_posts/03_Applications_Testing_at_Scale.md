---
layout: post
title: "06-03-00 Tùy chọn: Kiểm định giả thuyết ở quy mô lớn"
chapter: "06"
order: 3
owner: nglelinh
lang: vi
categories:
- chapter06
lesson_type: optional
---

Bài đọc tùy chọn này giữ nguyên các định nghĩa về $$H_0$$, $$p$$-value, sai lầm loại I/II và suy diễn hệ số hồi quy, đồng thời theo dõi cách các công cụ đó vận hành trong những hệ thống thực hiện hàng nghìn kiểm định mỗi ngày: nền tảng thử nghiệm A/B công nghiệp, nghiên cứu tương quan toàn hệ gene (GWAS), thí nghiệm quảng cáo — xếp hạng, và các kiểm định tuần tự vẫn giữ nguyên tính hợp lệ khi người phân tích theo dõi kết quả liên tục ("nhìn trộm" — peeking).

Sau bài học này, bạn có thể cài đặt kiểm định hai mẫu thực tế của một nền tảng thí nghiệm, giải thích vì sao việc nhìn trộm kết quả làm phình sai lầm loại I, áp dụng thủ tục Benjamini–Hochberg (đồng thời biết khi nào cần dùng Bonferroni hoặc ngưỡng toàn hệ gene), và trình bày được nguyên lý xây dựng một $$p$$-value luôn hợp lệ (always-valid $$p$$-value).

Kiến thức tiên quyết gồm hai bài bắt buộc của Chương 06 (khung kiểm định giả thuyết và suy diễn mô hình tuyến tính) cùng phân phối Bernoulli/Nhị thức. Phần mô phỏng đa kiểm định trong bài kiểm định giả thuyết sẽ được kế thừa trực tiếp mà không chứng minh lại.

---

## Thử Nghiệm A/B: Kiểm Định Giả Thuyết Trong Môi Trường Sản Phẩm

Các thử nghiệm có đối chứng trực tuyến (online controlled experiments) thực chất đặt ra một câu hỏi thống kê kinh điển. Gọi $$p_C$$ và $$p_T$$ lần lượt là xác suất chuyển đổi của nhóm đối chứng (control) và nhóm thử nghiệm (treatment). Nền tảng thí nghiệm kiểm định cặp giả thuyết:

$$H_0: p_T = p_C \qquad \text{versus} \qquad H_1: p_T \ne p_C$$

bằng một thống kê hai mẫu, thường là kiểm định $$z$$ cho hiệu hai tỷ lệ hoặc kiểm định $$t$$ cho các chỉ số liên tục. Kohavi, Tang và Xu (2020) mô tả các nền tảng thí nghiệm tại Microsoft, Google và LinkedIn, nơi mỗi hệ thống vận hành hàng chục nghìn kiểm định như vậy mỗi năm. Khung suy diễn cốt lõi chính là những gì bạn đã học; thách thức kỹ thuật nằm ở *độ tin cậy* của thí nghiệm: định nghĩa chỉ số đo lường (metric), hiện tượng can nhiễu giữa người dùng (interference), hiệu ứng kéo dài (carryover effect), và thực tế rằng một mức tăng (lift) "có ý nghĩa thống kê" 0,1% có thể mang lại giá trị hàng triệu đô-la nhưng cũng có thể chỉ bắt nguồn từ lỗi đo lường.

Kohavi, Deng và Vermeer (2022) tổng hợp các *hiểu lầm phản trực giác (intuition busters)* vẫn thường xuất hiện trong thực hành: diễn giải khoảng tin cậy 95% đã tính như xác suất 95% cho thí nghiệm hiện tại, dùng lực kiểm định hậu kiểm (post-hoc power) thay cho lực kiểm định thiết kế, loại bỏ giá trị ngoại lai sau khi đã xem kết quả, và chia lưu lượng không đều mà không hiệu chỉnh công thức phương sai. Mỗi lỗi nêu trên đều xuất phát từ việc áp dụng sai khung kiến thức Chương 06, chứ không phải do một khung lý thuyết mới.

```python
import numpy as np
from scipy import stats

def two_proportion_ztest(success_c, n_c, success_t, n_t):
    """
    z-test hai phía cho H0: p_T = p_C.

    Returns:
        (z, p_value, p_c, p_t, lift)
    """
    p_c = success_c / n_c
    p_t = success_t / n_t
    p_pool = (success_c + success_t) / (n_c + n_t)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    z = (p_t - p_c) / se
    p_value = 2 * stats.norm.sf(abs(z))
    return z, p_value, p_c, p_t, p_t - p_c
```

Ở quy mô nền tảng, phép kiểm định này luôn đi kèm các chỉ số bảo vệ (guardrail metrics), một tiêu chí đánh giá tổng thể được đăng ký trước và bước kiểm tra lệch tỷ lệ mẫu (Sample Ratio Mismatch — SRM); bản thân SRM là một kiểm định độ phù hợp (goodness-of-fit test) giúp phát hiện xem cơ chế phân nhóm ngẫu nhiên có bị lỗi hay không.

## Nhìn Trộm Kết Quả (Peeking), Dừng Tùy Ý và Kiểm Định Tuần Tự

Một $$p$$-value mẫu cố định chỉ hợp lệ tại *một* cỡ mẫu $$n$$ được ấn định trước. Nếu nhà phân tích theo dõi bảng điều khiển mỗi ngày và dừng thí nghiệm ngay lần đầu thấy $$p < 0.05$$, tỷ lệ sai lầm loại I thực tế sẽ cao hơn mức 5% rất nhiều. Hiện tượng này gọi là dừng tùy ý (optional stopping), và lại là hành vi phổ biến trên các giao diện hiển thị $$p$$-value theo thời gian thực.

Johari, Koomen, Pekelis và Walsh (2022) hình thức hóa khái niệm $$p$$-value *luôn hợp lệ (always-valid $$p$$-value)*: một chuỗi $$p_n$$ sao cho với mọi thời điểm dừng $$\tau$$, ta luôn có:

$$P_{H_0}(p_\tau \le \alpha) \le \alpha.$$

Nhóm tác giả xây dựng kiểm định này thông qua kiểm định tỷ số hợp lý tuần tự hỗn hợp (mixture sequential probability ratio test — mSPRT) và triển khai trên nền tảng thương mại phục vụ hàng trăm nghìn thí nghiệm. Cái giá phải trả về mặt thống kê là lực kiểm định (power): kiểm định luôn hợp lệ thường đòi hỏi cỡ mẫu lớn hơn so với kiểm định cỡ mẫu cố định ở cùng mức ý nghĩa.

Ramdas, Grünwald, Vovk và Shafer (2023) đặt các kiểm định luôn hợp lệ vào khung tổng quát *suy diễn an toàn bất kỳ lúc nào (safe anytime-valid inference)*. Đối tượng cơ sở là e-value hoặc test martingale $$M_n$$ — một quá trình không âm thỏa mãn $$E_{H_0}[M_{n+1} \mid \mathcal{F}_n] \le M_n$$. Theo bất đẳng thức Ville:

$$P_{H_0}\Bigl(\sup_n M_n \ge 1/\alpha\Bigr) \le \alpha,$$

suy ra $$p_n = \inf_{k \le n} 1/M_k$$ luôn hợp lệ tại mọi thời điểm dừng. Ngoài ra, các e-value từ những thí nghiệm độc lập có thể nhân trực tiếp với nhau, cho phép nền tảng quản lý ngân sách sai lầm trên cả một chuỗi kiểm định liên tiếp.

```python
import numpy as np

def peeking_inflation(n_looks=20, n_per_look=200, p=0.1, n_sims=4000, alpha=0.05):
    """
    Sai lầm loại I nếu dừng ở p < alpha đầu tiên khi null đúng.
    """
    rng = np.random.default_rng(0)
    false_reject = 0
    for _ in range(n_sims):
        c = t = 0
        nc = nt = 0
        rejected = False
        for _look in range(n_looks):
            c += rng.binomial(n_per_look, p)
            t += rng.binomial(n_per_look, p)
            nc += n_per_look
            nt += n_per_look
            _, pval, *_ = two_proportion_ztest(c, nc, t, nt)
            if pval < alpha:
                rejected = True
                break
        false_reject += rejected
    return false_reject / n_sims
```

Khi chạy hàm trên với các tham số mặc định, tỷ lệ sai lầm loại I thực tế vào khoảng 20–25% thay vì 5%. Con số này giải thích vì sao các phương pháp kiểm định tuần tự đã bước ra từ thử nghiệm lâm sàng để trở thành tiêu chuẩn trong các nền tảng phần mềm quy mô lớn.

## Đa Kiểm Định: Hệ Gene, Quảng Cáo và Kiểm Soát FDR Trực Tuyến

Khi kiểm định một giả thuyết đơn lẻ ở mức $$\alpha$$, xác suất sai lầm loại I là $$\alpha$$. Tuy nhiên, khi kiểm định $$m$$ giả thuyết không độc lập (và đều đúng), xác suất mắc ít nhất một bác bỏ sai tăng lên $$1-(1-\alpha)^m$$ — kết quả đã được mô phỏng trong bài bắt buộc. Hai hướng hiệu chỉnh phổ biến nhất trong thực hành phục vụ những mục tiêu khác nhau và không thể thay thế cho nhau.

**Kiểm soát sai lầm trên cả họ (FWER).** Phương pháp Bonferroni sử dụng ngưỡng $$\alpha/m$$. Trong các nghiên cứu tương quan toàn hệ gene (GWAS) — được Uffelmann và cộng sự (2021) tổng quan — ngưỡng bác bỏ tiêu chuẩn là $$5 \times 10^{-8}$$, tương ứng với hiệu chỉnh kiểu Bonferroni cho khoảng một triệu biến thể di truyền phổ biến độc lập. Lựa chọn nghiêm ngặt này ưu tiên khả năng tái lập hơn lực kiểm định, nên một phát hiện ("hit") trong GWAS được xem là sự kiện có độ tin cậy cao.

**Kiểm soát tỷ lệ phát hiện giả (FDR).** Benjamini và Hochberg (1995) đề xuất kiểm soát kỳ vọng của tỷ lệ bác bỏ sai trong tổng số các giả thuyết bị bác bỏ:

$$\mathrm{FDR} = E\Bigl[\frac{V}{\max(R,1)}\Bigr].$$

Ta sắp xếp các $$p$$-value theo thứ tự tăng dần $$p_{(1)} \le \cdots \le p_{(m)}$$, tìm chỉ số $$k$$ lớn nhất thỏa mãn:

$$p_{(k)} \le \frac{k}{m}\alpha,$$

và bác bỏ toàn bộ $$k$$ giả thuyết đầu tiên. Thủ tục Benjamini–Hochberg (BH) là lựa chọn mặc định trong phân tích RNA-seq, chẩn đoán hình ảnh y khoa và các bảng theo dõi kiểm định hàng trăm chỉ số hoặc phân khúc người dùng. Tuy vậy, BH *không* thay thế ngưỡng GWAS vì chi phí khoa học của một locus gene dương tính giả phù hợp hơn với tiêu chí kiểm soát sai lầm trên cả họ.

**Kiểm soát FDR trực tuyến (Online FDR).** Các nền tảng sản phẩm không nhận toàn bộ $$p$$-value cùng một lúc mà các thí nghiệm diễn ra liên tục theo dòng thời gian. Robertson, Wason và Ramdas (2023) tổng quan các thủ tục trực tuyến (như LORD, SAFFRON, ADDIS) phân bổ "ngân sách $$\alpha$$" ($$\alpha$$-wealth) theo thời gian và bổ sung lại một phần ngân sách mỗi khi có phát hiện mới. Đây là mô hình đa kiểm định phù hợp cho các nhóm quảng cáo hoặc xếp hạng triển khai thí nghiệm mới mỗi tuần và không bao giờ có một danh sách giả thuyết cố định.

```python
import numpy as np

def benjamini_hochberg(p_values, alpha=0.05):
    """
    Mặt nạ bác bỏ BH. Trả về mảng boolean cùng thứ tự với p_values.
    """
    p = np.asarray(p_values)
    m = len(p)
    order = np.argsort(p)
    thresh = alpha * (np.arange(1, m + 1) / m)
    below = p[order] <= thresh
    if not below.any():
        return np.zeros(m, dtype=bool)
    cutoff = np.max(np.where(below)[0])
    rejected = np.zeros(m, dtype=bool)
    rejected[order[: cutoff + 1]] = True
    return rejected
```

## Suy Diễn Cho Nhiều Hệ Số Cũng Là Bài Toán Đa Kiểm Định

Bài bắt buộc thứ hai đã xây dựng khoảng tin cậy và $$p$$-value cho từng hệ số hồi quy. Trong các mô hình xếp hạng hoặc quảng cáo, số lượng hệ số có thể lên đến hàng nghìn. Việc báo cáo mọi hệ số có $$p < 0.05$$ như một "đặc trưng có ý nghĩa" thực chất là mắc lỗi đa kiểm định. Trong thực tế triển khai, các nhóm kỹ thuật thường chọn một trong ba hướng: (i) coi hệ số chỉ là thành phần phục vụ dự đoán và không tiến hành kiểm định từng hệ số, (ii) áp dụng thủ tục BH hoặc phương pháp knockoffs để chọn biến có kiểm soát sai lầm, hoặc (iii) dùng điều chuẩn (regularisation) và đánh giá bằng các chỉ số dự đoán. Bản thân công thức suy diễn của mô hình tuyến tính không sai, nhưng chúng trở nên thiếu đầy đủ khi ta kiểm định đồng thời trên rất nhiều hệ số.

## Thách Thức và Hướng Mở Rộng

Nhìn trộm kết quả không phải mối đe dọa duy nhất đối với tính hợp lệ của thí nghiệm. Hiện tượng can nhiễu (interference — khi tác động lên một người dùng làm thay đổi kết quả của người dùng khác), hiệu ứng mới lạ (novelty effect) và lệch tỷ lệ mẫu (Sample Ratio Mismatch) đều có thể tạo ra kết quả "có ý nghĩa thống kê" dù không có hiệu ứng xử lý thực sự. Kiểm định luôn hợp lệ đánh đổi cỡ mẫu để đảm bảo tính hợp lệ ở mọi thời điểm dừng; vì thế, các nhóm cần kết quả sớm đôi khi ưu tiên giới hạn trước số lần xem (thiết kế tuần tự theo nhóm — group sequential design) thay vì dùng e-process tuần tự liên tục. Với FDR trực tuyến, quy trình đòi hỏi một dòng thí nghiệm được ghi nhận đầy đủ; việc âm thầm loại bỏ các kiểm định không thành công khỏi dòng sẽ làm mất hiệu lực của chứng minh toán học.

Các hướng mở rộng tiếp nối chương này gồm kỹ thuật giảm phương sai trong kiểm định (như CUPED, sẽ đề cập ở Chương 07), suy diễn nhân quả khi cơ chế gán nhóm không phải phân phối Bernoulli thuần túy, và kết hợp e-value cho phân tích tổng hợp liên tục (living meta-analysis).

## Video Tham Khảo

- [Trustworthy Online Controlled Experiments at Large Scale w/ Ronny Kohavi | DECISIONS/Replay by Split](https://www.youtube.com/watch?v=kTAFOCynWIg) — Split, tiếng Anh. **Nên xem trước.** Ronny Kohavi trình bày bản chất của thử nghiệm trực tuyến như kiểm định hai mẫu ở quy mô công nghiệp — tương ứng với phần A/B testing mở đầu bài.
- [Peeking at A/B Tests: Why it matters, and what to do about it](https://www.youtube.com/watch?v=6G8F22NSCxk) — KDD2017 video, tiếng Anh. Báo cáo tại hội nghị KDD về hiện tượng peeking: theo dõi p-value và dừng sớm làm tăng vọt sai lầm loại I — tương ứng với mục kiểm định tuần tự.
- [False Discovery Rates, FDR, clearly explained](https://www.youtube.com/watch?v=K8LQSvtjcEo) — StatQuest with Josh Starmer, tiếng Anh. Giải thích trực quan về FDR và thủ tục Benjamini–Hochberg khi kiểm định hàng nghìn giả thuyết, đối chiếu với phương pháp Bonferroni.

## Bài Tập

**Bài 1 (khái niệm).** Bảng theo dõi của một nhà cung cấp cập nhật $$p$$-value mỗi giờ và tô xanh khi giá trị xuống dưới 0,05. Giả định nào của kiểm định mẫu cố định đã bị vi phạm? Hãy nêu một giải pháp thay thế luôn hợp lệ về mặt thống kê và một giải pháp vận hành mà không cần thay đổi phép kiểm định.

**Bài 2 (tính toán).** Sử dụng hàm `peeking_inflation`, vẽ biểu đồ tỷ lệ sai lầm loại I theo số lần xem kết quả với $$L \in \{1, 2, 5, 10, 20, 50\}$$. Vẽ chồng đường hiệu chỉnh Bonferroni $$\alpha / L$$ áp dụng cho mỗi lần xem. Lực kiểm định giảm đi bao nhiêu nếu áp dụng hiệu chỉnh Bonferroni cho 20 lần xem với mức tăng thực (lift) 2% và cỡ mẫu $$n = 5{,}000$$ mỗi nhánh?

**Bài 3 (tính toán).** Mô phỏng $$m = 1{,}000$$ giá trị $$p$$-value, trong đó 100 giá trị sinh từ $$z$$-score theo phân phối $$N(3,1)$$ và 900 giá trị còn lại có phân phối đều trên $$[0, 1]$$. So sánh số phát hiện đúng và phát hiện sai ở mức $$\alpha = 0.05$$ khi không hiệu chỉnh, khi dùng Bonferroni và khi dùng BH. Liên hệ kết quả Bonferroni với tư duy GWAS và kết quả BH với bảng theo dõi chỉ số quảng cáo.

**Bài 4 (mở).** Đọc bài báo của Kohavi, Deng và Vermeer (2022), chọn một hiểu lầm phản trực giác (intuition buster) và tái hiện bằng một mô phỏng nhỏ. Từ đó, đề xuất một thay đổi giao diện trên nền tảng thí nghiệm (chẳng hạn văn bản cảnh báo, vô hiệu hóa nút dừng sớm, hoặc công cụ bắt buộc tính cỡ mẫu) để hạn chế sai lầm đó.

## Tài Liệu Tham Khảo

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B, 57*(1), 289–300. Thủ tục kiểm soát FDR kinh điển vẫn được trích dẫn rộng rãi.

Johari, R., Koomen, P., Pekelis, L., & Walsh, D. (2022). Always valid inference: Continuous monitoring of A/B tests. *Operations Research, 70*(3), 1806–1821. [https://doi.org/10.1287/opre.2021.2135](https://doi.org/10.1287/opre.2021.2135)

Kohavi, R., Deng, A., & Vermeer, L. (2022). A/B testing intuition busters: Common misunderstandings in online controlled experiments. Trong *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (tr. 3168–3177). [https://doi.org/10.1145/3534678.3539160](https://doi.org/10.1145/3534678.3539160)

Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*. Cambridge University Press. Tài liệu tham chiếu chuẩn mực trong công nghiệp cho các nền tảng thí nghiệm trực tuyến.

Ramdas, A., Grünwald, P., Vovk, V., & Shafer, G. (2023). Game-theoretic statistics and safe anytime-valid inference. *Statistical Science, 38*(4), 576–601. [https://doi.org/10.1214/23-STS894](https://doi.org/10.1214/23-STS894)

Robertson, D. S., Wason, J. M. S., & Ramdas, A. (2023). Online multiple hypothesis testing. *Statistical Science, 38*(4), 557–575. [https://doi.org/10.1214/23-STS901](https://doi.org/10.1214/23-STS901)

Uffelmann, E., Huang, Q. Q., Munung, N. S., de Vries, J., Okada, Y., Martin, A. R., … Posthuma, D. (2021). Genome-wide association studies. *Nature Reviews Methods Primers, 1*, 59. [https://doi.org/10.1038/s43586-021-00056-9](https://doi.org/10.1038/s43586-021-00056-9)
