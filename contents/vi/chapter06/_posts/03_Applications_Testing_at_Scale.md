---
layout: post
title: 06-03-00 Tùy chọn: Kiểm định giả thuyết ở quy mô lớn
chapter: "06"
order: 3
owner: nglelinh
lang: vi
categories:
- chapter06
lesson_type: optional
---

Bài tùy chọn này giữ nguyên định nghĩa $$H_0$$, $$p$$-value, sai lầm loại I/II, và suy diễn hệ số. Nó theo các đối tượng đó vào những hệ thống đang chạy chúng hàng nghìn lần mỗi ngày: nền tảng A/B công nghiệp, nghiên cứu tương quan toàn bộ hệ gene, thí nghiệm quảng cáo và xếp hạng, và các kiểm định tuần tự vẫn hợp lệ khi có người “nhìn trộm”.

Sau bài, bạn viết được kiểm định hai mẫu mà một nền tảng thí nghiệm thực sự tính, giải thích được vì sao peeking làm phình sai lầm loại I, áp dụng Benjamini–Hochberg và nói khi nào Bonferroni hoặc ngưỡng toàn bộ hệ gene được dùng thay thế, và phác được một $$p$$-value luôn hợp lệ.

Các bài bắt buộc Chương 06 (khung kiểm định và suy diễn mô hình tuyến tính) cùng mô hình nhị thức/Bernoulli là đủ. Phần mô phỏng đa kiểm định trong bài kiểm định giả thuyết sẽ hữu ích; ta tái sử dụng trực giác đó chứ không dẫn lại.

---

## A/B testing là kiểm định giả thuyết có product manager kèm theo

Thí nghiệm kiểm soát trực tuyến hỏi một câu giáo trình. Gọi $$p_C$$ và $$p_T$$ là xác suất chuyển đổi trên đối chứng và xử lý. Nền tảng kiểm định

$$H_0: p_T = p_C \qquad \text{versus} \qquad H_1: p_T \ne p_C$$

bằng một statistic hai mẫu, thường là $$z$$-test cho hiệu tỷ lệ hoặc $$t$$-test cho metric thực. Kohavi, Tang và Xu (2020) mô tả các nền tảng tại Microsoft, Google và LinkedIn, mỗi nơi chạy hàng chục nghìn kiểm định như vậy mỗi năm. Khung xương suy diễn là thứ bạn đã biết. Phần kỹ thuật là về *độ tin cậy*: định nghĩa metric, nhiễu giữa người dùng, hiệu ứng mang sang, và việc một lift “có ý nghĩa” 0,1% có thể đáng giá hàng triệu đồng thời là lỗi đo.

Kohavi, Deng và Vermeer (2022) liệt kê các *intuition buster* mà nhà cung cấp vẫn đưa ra: coi khoảng 95% đã tính như một phát biểu xác suất 95% về thí nghiệm hiện tại, dùng power hậu kiểm như power thiết kế, loại outlier sau khi xem kết quả, và chia traffic không đều mà không chỉnh công thức phương sai. Mỗi sai lầm đó là dùng sai khung Chương 06, không phải một khung mới.

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

Ở quy mô nền tảng, cùng kiểm định đó được bọc bằng metric bảo vệ, một tiêu chí đánh giá tổng thể đăng ký trước, và kiểm tra Sample Ratio Mismatch—bản thân SRM là một goodness-of-fit test xem cơ chế gán có bị gãy không.

## Peeking, dừng tùy ý, và kiểm định tuần tự

Một $$p$$-value mẫu cố định hợp lệ tại *một* $$n$$ chọn trước. Nếu nhà phân tích làm mới dashboard mỗi ngày và dừng lần đầu $$p < 0.05$$, sai lầm loại I cao hơn 5% rất nhiều. Đó là optional stopping, và đó là hành vi mặc định trên mọi UI thí nghiệm vẽ $$p$$-value trực tiếp.

Johari, Koomen, Pekelis và Walsh (2022) hình thức hóa $$p$$-value *luôn hợp lệ*: một quá trình $$p_n$$ sao cho với mọi thời điểm dừng $$\tau$$,

$$P_{H_0}(p_\tau \le \alpha) \le \alpha.$$

Họ xây dựng bằng mixture sequential probability ratio test (mSPRT). Phương pháp đã được triển khai trên một nền tảng thương mại phân tích hàng trăm nghìn thí nghiệm. Giá thống kê là power: kiểm định luôn hợp lệ thường cần nhiều mẫu hơn kiểm định $$n$$ cố định cùng mức.

Ramdas, Grünwald, Vovk và Shafer (2023) đặt các kiểm định luôn hợp lệ vào *suy diễn an toàn bất kỳ lúc nào*. Đối tượng nguyên thủy là e-value hoặc test martingale $$M_n$$, một quá trình không âm với $$E_{H_0}[M_{n+1} \mid \mathcal{F}_n] \le M_n$$. Bất đẳng thức Ville cho

$$P_{H_0}\Bigl(\sup_n M_n \ge 1/\alpha\Bigr) \le \alpha,$$

nên $$p_n = \inf_{k \le n} 1/M_k$$ luôn hợp lệ. E-value nhân được giữa các thí nghiệm độc lập, nhờ đó một nền tảng chi tiêu ngân sách sai lầm trên một dòng kiểm định.

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

Chạy hàm đó với mặc định cho sai lầm loại I khoảng 20–25%, không phải 5%. Con số đó là lý do phương pháp tuần tự đi từ sổ tay thử nghiệm lâm sàng vào phần mềm tiêu dùng.

## Đa kiểm định: hệ gene, quảng cáo, và FDR trực tuyến

Kiểm định một giả thuyết ở mức $$\alpha$$ thì sai lầm loại I là $$\alpha$$. Kiểm định $$m$$ null đúng độc lập thì xác suất có ít nhất một bác bỏ sai là $$1-(1-\alpha)^m$$. Bài bắt buộc đã mô phỏng sự thật đó. Hai hiệu chỉnh chiếm ưu thế trong thực hành, và chúng không thay thế được nhau.

**Sai lầm trên cả họ.** Bonferroni dùng $$\alpha/m$$. Các nghiên cứu tương quan toàn bộ hệ gene, Uffelmann et al. (2021) tổng quan, thông thường bác bỏ tại $$5 \times 10^{-8}$$, một ngưỡng kiểu Bonferroni cho khoảng một triệu biến thể phổ biến độc lập. Lựa chọn đó ưu tiên tái lập hơn power, và vì thế một “hit” GWAS được coi là sự kiện hiếm.

**Tỷ lệ phát hiện giả.** Benjamini và Hochberg (1995) kiểm soát kỳ vọng tỷ lệ bác bỏ sai trong tập đã bác bỏ,

$$\mathrm{FDR} = E\Bigl[\frac{V}{\max(R,1)}\Bigr].$$

Sắp $$p_{(1)} \le \cdots \le p_{(m)}$$ và bác bỏ $$k$$ giá trị đầu thỏa

$$p_{(k)} \le \frac{k}{m}\alpha.$$

BH là mặc định trong RNA-seq, ảnh y khoa, và mọi dashboard kiểm định hàng trăm metric hoặc phân khúc. Nó *không* thay thế ngưỡng GWAS: cái giá khoa học của một locus gene giả gần với sai lầm trên cả họ hơn.

**FDR trực tuyến.** Nền tảng không nhận mọi $$p$$-value cùng lúc. Thí nghiệm đến thành dòng. Robertson, Wason và Ramdas (2023) khảo sát các thủ tục trực tuyến (LORD, SAFFRON, ADDIS) chi tiêu $$\alpha$$-wealth theo thời gian và lấy lại wealth khi có phát hiện. Đó là mô hình đa tính đúng cho một tổ chức quảng cáo hoặc xếp hạng bắt đầu kiểm định mới mỗi tuần và sẽ không bao giờ có một danh sách giả thuyết đóng.

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

## Suy diễn cho nhiều hệ số là cùng một bài toán

Bài bắt buộc thứ hai xây khoảng tin cậy và $$p$$-value cho hệ số hồi quy. Trong mô hình xếp hạng hoặc quảng cáo, các hệ số đếm đến hàng nghìn. Báo cáo mọi $$p < 0.05$$ như một “feature có ý nghĩa” là lỗi đa kiểm định. Thực hành sản xuất hoặc (i) coi hệ số là công cụ dự đoán và không kiểm định chúng, (ii) áp dụng BH hoặc knockoffs để chọn biến, hoặc (iii) regularise và báo cáo metric dự đoán. Công thức mô hình tuyến tính không trở thành sai; chúng trở thành thiếu khi bạn tìm kiếm trên nhiều công thức.

## Thách thức và mở rộng

Peeking không phải mối đe dọa hợp lệ duy nhất. Nhiễu (xử lý của một người dùng đổi kết cục của người khác), hiệu ứng mới lạ, và Sample Ratio Mismatch đều tạo ra kết quả *có ý nghĩa* mà không phải hiệu ứng xử lý. Kiểm định luôn hợp lệ đổi validity bằng cỡ mẫu; các nhóm không chờ được đôi khi thích giới hạn cứng số lần nhìn (thiết kế tuần tự theo nhóm) hơn một e-process tuần tự đầy đủ. FDR trực tuyến cần một dòng xác định rõ; lặng lẽ bỏ các kiểm định thất bại khỏi dòng sẽ phá chứng minh.

Các mở rộng nằm trên chương này gồm kiểm định giảm phương sai (CUPED, Chương 07 sẽ lấy lại), suy diễn nhân quả khi gán không phải Bernoulli sạch, và gộp e-value cho meta-analysis sống.

## Bài tập

**Bài 1 (khái niệm).** Dashboard của một nhà cung cấp cập nhật $$p$$-value mỗi giờ và tô thẻ xanh ở 0,05. Giả định nào của kiểm định mẫu cố định bị vi phạm? Nêu một thay thế luôn hợp lệ và một thay thế vận hành không đổi kiểm định.

**Bài 2 (tính toán).** Dùng `peeking_inflation`, vẽ sai lầm loại I theo số lần nhìn với $$L \in \{1, 2, 5, 10, 20, 50\}$$. Chồng đường Bonferroni $$\alpha / L$$ áp dụng cho mỗi lần nhìn. Bạn mất bao nhiêu power nếu Bonferroni-hiệu chỉnh 20 lần nhìn trên lift thật 2% với $$n = 5{,}000$$ mỗi nhánh?

**Bài 3 (tính toán).** Mô phỏng $$m = 1{,}000$$ $$p$$-value trong đó 100 đến từ $$z$$-score $$N(3,1)$$ và phần còn lại đều. So sánh số phát hiện đúng và sai dưới $$\alpha = 0.05$$ không hiệu chỉnh, Bonferroni, và BH. Liên hệ số Bonferroni với tư duy GWAS và số BH với dashboard metric quảng cáo.

**Bài 4 (mở).** Đọc Kohavi, Deng và Vermeer (2022). Chọn một intuition buster và tái hiện bằng mô phỏng nhỏ. Rồi đề xuất một thay đổi UI nền tảng—copy, nút bị tắt, máy tính cỡ mẫu bắt buộc—khiến sai lầm đó khó hơn.

## Tài liệu tham khảo

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B, 57*(1), 289–300. Thủ tục FDR vẫn được trích dẫn.

Johari, R., Koomen, P., Pekelis, L., & Walsh, D. (2022). Always valid inference: Continuous monitoring of A/B tests. *Operations Research, 70*(3), 1806–1821. [https://doi.org/10.1287/opre.2021.2135](https://doi.org/10.1287/opre.2021.2135)

Kohavi, R., Deng, A., & Vermeer, L. (2022). A/B testing intuition busters: Common misunderstandings in online controlled experiments. Trong *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (tr. 3168–3177). [https://doi.org/10.1145/3534678.3539160](https://doi.org/10.1145/3534678.3539160)

Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*. Cambridge University Press. Tham chiếu công nghiệp cho nền tảng thí nghiệm.

Ramdas, A., Grünwald, P., Vovk, V., & Shafer, G. (2023). Game-theoretic statistics and safe anytime-valid inference. *Statistical Science, 38*(4), 576–601. [https://doi.org/10.1214/23-STS894](https://doi.org/10.1214/23-STS894)

Robertson, D. S., Wason, J. M. S., & Ramdas, A. (2023). Online multiple hypothesis testing. *Statistical Science, 38*(4), 557–575. [https://doi.org/10.1214/23-STS901](https://doi.org/10.1214/23-STS901)

Uffelmann, E., Huang, Q. Q., Munung, N. S., de Vries, J., Okada, Y., Martin, A. R., … Posthuma, D. (2021). Genome-wide association studies. *Nature Reviews Methods Primers, 1*, 59. [https://doi.org/10.1038/s43586-021-00056-9](https://doi.org/10.1038/s43586-021-00056-9)
