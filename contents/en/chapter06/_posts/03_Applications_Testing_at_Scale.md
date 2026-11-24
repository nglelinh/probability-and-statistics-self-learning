---
layout: post
title: "06-03-00 Optional: Hypothesis Testing at Scale"
chapter: "06"
order: 3
owner: nglelinh
lang: en
categories:
- chapter06
lesson_type: optional
---

Imagine a product manager watching a live A/B test dashboard every morning over coffee. On Day 1, $$p = 0.42$$. On Day 2, $$p = 0.19$$. On Day 4, the dashboard flashes green: **$$p = 0.038 < 0.05$$!** Thrilled, the team stops the test and ships the new checkout button—never realizing that by peeking 20 times and stopping at the first green flash, they quietly inflated their false-positive rate from **5%** to nearly **25%**!

This optional lesson leaves the definitions of $$H_0$$, $$p$$-values, and Type I/II errors untouched and follows them straight into the massive engines that run them tens of thousands of times a day: industrial **A/B platforms**, **sequential anytime-valid testing (e-values / mSPRT)**, **genome-wide association studies (Bonferroni)**, and **False Discovery Rate (Benjamini–Hochberg & online FDR)**.

You should finish able to compute a two-proportion $$z$$-test and Sample Ratio Mismatch check by hand, explain why peeking breaks fixed-sample $$p$$-values, run Benjamini–Hochberg versus Bonferroni by hand, and sketch how always-valid $$p$$-values restore rigor.

---

## Background Knowledge Map: How Hypothesis Testing Scales Up

Three statistical ideas govern testing when you move from one textbook experiment to 10,000 live experiments.

### 1. Industrial A/B Testing & Data Integrity (SRM)
Every online A/B test is a two-sample hypothesis test $$H_0: p_T = p_C$$ backed by strict randomization hygiene. Before trusting any lift, platforms run a **Sample Ratio Mismatch (SRM)** $$\chi^2$$ goodness-of-fit test checking whether the observed user split matches the intended 50/50 allocation—catching logging bugs and bot traffic before bad decisions ship.
- **Core questions:** Is the observed lift statistically significant, practically meaningful, and free of assignment bugs?
- **Real-world reach:** Product experimentation at Microsoft, Google, Meta, Netflix, and e-commerce platforms.

### 2. Sequential Testing & Anytime-Valid Inference (e-Values & Martingales)
Classical $$p$$-values assume a single pre-committed sample size $$n$$. **Safe anytime-valid inference** (mSPRT, **e-values**, test martingales $$M_n$$) uses Ville’s inequality $$P_{H_0}(\sup_n M_n \ge 1/\alpha) \le \alpha$$ so you can monitor continuously and stop whenever you want without ever breaking the $$\alpha$$ false-positive guarantee.
- **Core questions:** How much extra sample size does continuous monitoring cost in exchange for peeking safety?
- **Real-world reach:** Commercial A/B platforms (Optimizely, Statsig), adaptive clinical trials, and continuous safety monitoring.

### 3. Multiplicity Control (FWER vs. FDR vs. Online FDR)
Testing $$m$$ independent true nulls at $$\alpha = 0.05$$ gives a $$1 - 0.95^m$$ chance of at least one false alarm ($$64\%$$ at $$m = 20$$!). **Bonferroni** controls the **Family-Wise Error Rate (FWER)** (zero false alarms—vital in GWAS genetics), while **Benjamini–Hochberg** controls the **False Discovery Rate (FDR)** (expected share of false alarms among discoveries—vital in dashboards and genomics). **Online FDR** extends this to endless streams of experiments.
- **Core questions:** When do we pay for strict FWER versus more powerful FDR control?
- **Real-world reach:** Genome-wide association studies (GWAS), RNA-seq differential expression, and ad-ranking metric scorecards.

---

## A/B testing is hypothesis testing with a product manager attached

An online controlled experiment tests conversion probabilities $$p_C$$ (control) and $$p_T$$ (treatment):

$$H_0: p_T = p_C \qquad \text{versus} \qquad H_1: p_T \ne p_C$$

with a two-sample $$z$$-test for proportions or a $$t$$-test for continuous metrics. Kohavi, Tang, and Xu (2020) document platforms running tens of thousands of tests a year, while Kohavi, Deng, and Vermeer (2022) catalog common *intuition busters* (treating a 95% CI as a 95% probability for the current run, misusing post-hoc power, or trimming outliers after seeing the result).

---

### Worked Example 1: Two-Proportion $$z$$-Test & Sample Ratio Mismatch Check by Hand

Suppose a 50/50 checkout A/B test observes:
- **Control:** $$100$$ conversions out of $$n_C = 1{,}000$$ users ($$\hat{p}_C = 0.100$$).
- **Treatment:** $$125$$ conversions out of $$n_T = 1{,}000$$ users ($$\hat{p}_T = 0.125$$, a $$+2.5$$ percentage-point lift).

1. **Compute the two-proportion $$z$$-test:**
   - Pooled rate: $$\hat{p}_{\text{pool}} = \dfrac{100 + 125}{2{,}000} = 0.1125$$.
   - Standard error: $$\text{SE} = \sqrt{0.1125(1 - 0.1125)\left(\frac{1}{1000} + \frac{1}{1000}\right)} = \sqrt{0.0001997} \approx 0.01413$$.
   - Test statistic: $$z = \dfrac{0.125 - 0.100}{0.01413} \approx \mathbf{1.769} \implies p = 2\Phi(-1.769) \approx \mathbf{0.077}$$.
   At $$\alpha = 0.05$$, we do **not** reject $$H_0$$ yet—the 2.5% lift is not distinguishable from sampling noise at $$n = 1{,}000$$ per arm!
2. **Now run a Sample Ratio Mismatch (SRM) check** on a second 50/50 experiment that logged $$n_C = 4{,}850$$ and $$n_T = 5{,}150$$ users (total $$10{,}000$$, expected $$5{,}000$$ each):
   $$\chi^2 = \frac{(4850 - 5000)^2}{5000} + \frac{(5150 - 5000)^2}{5000} = \frac{22500 + 22500}{5000} = \mathbf{9.00} \implies p = \mathbf{0.0027}!$$
   Even though 48.5% vs. 51.5% looks “close enough” to the naked eye, $$p = 0.0027$$ screams that a redirect bug or bot filter broke randomization—invalidating *every* downstream metric!

---

```python
import numpy as np
from scipy import stats

def two_proportion_ztest(success_c, n_c, success_t, n_t):
    """
    Two-sided z-test for H0: p_T = p_C.

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

## Peeking, optional stopping, and sequential tests

A fixed-sample $$p$$-value is valid at *one* pre-chosen $$n$$. Refreshing a dashboard daily and stopping at the first $$p < 0.05$$ inflates Type I error dramatically.

Johari, Koomen, Pekelis, and Walsh (2022) formalize **always-valid $$p$$-values** satisfying $$P_{H_0}(p_\tau \le \alpha) \le \alpha$$ for *every* data-dependent stopping time $$\tau$$ using a mixture sequential probability ratio test (mSPRT). Ramdas, Grünwald, Vovk, and Shafer (2023) unify this inside **safe anytime-valid inference** using **e-values** / test martingales $$M_n \ge 0$$ with $$\mathbb{E}_{H_0}[M_{n+1} \mid \mathcal{F}_n] \le M_n$$: by **Ville’s inequality**,

$$P_{H_0}\Bigl(\sup_n M_n \ge 1/\alpha\Bigr) \le \alpha,$$

so $$p_n = \inf_{k \le n} 1/M_k$$ stays valid at every single refresh! The price is statistical power: always-valid tests require slightly wider rejection boundaries (and thus larger average $$n$$) than fixed-horizon tests.

```python
import numpy as np

def peeking_inflation(n_looks=20, n_per_look=200, p=0.1, n_sims=4000, alpha=0.05):
    """
    Type I error if you stop at the first p < alpha under a true null.
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

Running that function with the defaults yields a Type I error near **20–25%**, not 5%—which is why sequential methodology moved straight from clinical trials into consumer software.

## Multiple testing: genomes, ads, and online FDR

Test $$m$$ independent true nulls at level $$\alpha$$ and the chance of at least one false positive explodes to $$1 - (1-\alpha)^m$$. Two corrections anchor practice:

- **Family-wise error (Bonferroni):** reject at $$\alpha / m$$. In GWAS (Uffelmann et al., 2021), ~1 million independent genomic variants yield the iconic threshold $$5 \times 10^{-8}$$—prioritizing reproducibility over power.
- **False Discovery Rate (Benjamini–Hochberg, 1995):** controls $$\mathrm{FDR} = \mathbb{E}\bigl[V / \max(R, 1)\bigr]$$. Sort $$p_{(1)} \le \cdots \le p_{(m)}$$, find the **largest** $$k$$ satisfying $$p_{(k)} \le \frac{k}{m}\alpha$$, and reject $$1, \ldots, k$$.
- **Online FDR** (Robertson, Wason, & Ramdas, 2023): procedures like LORD and SAFFRON manage an $$\alpha$$-wealth budget across endless streams of arriving experiments.

---

### Worked Example 2: Bonferroni vs. Benjamini–Hochberg by Hand ($$m = 5$$ Metrics)

Suppose an A/B scorecard tests $$m = 5$$ product metrics at target level $$\alpha = 0.05$$, returning sorted $$p$$-values:

$$p_{(1)} = 0.004, \quad p_{(2)} = 0.015, \quad p_{(3)} = 0.028, \quad p_{(4)} = 0.045, \quad p_{(5)} = 0.300.$$

Compare all three decision rules side by side:

| Rank $$k$$ | $$p_{(k)}$$ | Unadjusted ($$\le 0.05$$) | Bonferroni ($$\le 0.01$$) | BH Threshold $$\frac{k}{5}(0.05)$$ | $$p_{(k)} \le \text{BH}$$? |
|------------|-------------|---------------------------|---------------------------|------------------------------------|----------------------------|
| 1 | $$0.004$$ | Reject | **Reject** | $$0.010$$ | Yes ✓ |
| 2 | $$0.015$$ | Reject | Keep | $$0.020$$ | Yes ✓ |
| 3 | $$0.028$$ | Reject | Keep | $$0.030$$ | **Yes ✓ (largest $$k = 3$$)** |
| 4 | $$0.045$$ | Reject | Keep | $$0.040$$ | No ✗ |
| 5 | $$0.300$$ | Keep | Keep | $$0.050$$ | No ✗ |

- **Unadjusted:** rejects 4 metrics (inflated false-positive risk!).
- **Bonferroni (FWER):** rejects only **1** metric ($$0.004$$)—ultra-conservative.
- **Benjamini–Hochberg (FDR):** rejects **3** metrics (ranks 1, 2, and 3), keeping the expected proportion of false discoveries among those 3 below 5%!

---

```python
import numpy as np

def benjamini_hochberg(p_values, alpha=0.05):
    """
    BH rejection mask. Returns a boolean array aligned with p_values.
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

## Inference for many coefficients is the same problem

In a ranking or ads regression with thousands of features, celebrating every coefficient with $$p < 0.05$$ is a classic multiple-testing trap. Production teams either treat coefficients purely as predictive weights, apply BH/knockoffs for controlled variable selection, or regularize and evaluate out-of-sample.

---

### Deep-Dive Explanation: Why Peeking and Multiple Testing Are the Exact Same Trap (Read Slowly)

At first glance, peeking at one A/B test 20 times and testing 20 different metrics once look like different problems. Walk through these four steps to see why they are mathematically twins.

#### Step 1: Under $$H_0$$, a $$p$$-value is a uniform random draw on $$[0,1]$$
On a single pre-registered look at a single metric, $$P(p \le 0.05) = 5\%$$.

#### Step 2: Taking the minimum over many looks or metrics shifts the distribution left
Whether you take $$\min(p_{\text{day 1}}, \ldots, p_{\text{day 20}})$$ across time OR $$\min(p_{\text{metric 1}}, \ldots, p_{\text{metric 20}})$$ across columns, you are asking for the **extreme minimum** of 20 random draws—which drops below $$0.05$$ far more than 5% of the time!

#### Step 3: Why e-values and martingales defeat peeking cleanly
Think of an **e-value** $$M_n$$ as a gambler’s wealth starting at $$1$$ dollar playing a fair game against $$H_0$$. Because $$\mathbb{E}_{H_0}[M_{n+1} \mid \mathcal{F}_n] \le M_n$$, no stopping strategy—not even peeking every millisecond—can give a gambler more than a 5% chance of *ever* multiplying $1 into $20 ($$1/\alpha = 20$$)!

#### Step 4: Match the guardrails to the real business cost
Use **SRM** first to verify clean data; use **always-valid $$p$$-values** (or fixed pre-committed horizons) to stop peeking; use **Bonferroni** when a single false alarm is catastrophic (GWAS, safety); and use **Benjamini–Hochberg** when screening dozens of exploratory segments or metrics.

---

## Challenges and extensions

Peeking is not the only validity threat. Interference (one user’s treatment changes another user’s outcome), novelty effects, and Sample Ratio Mismatch all produce *significant* results that are not treatment effects. Always-valid tests buy validity with sample size; teams that cannot wait sometimes prefer a hard cap on looks (group sequential designs) instead of a fully sequential e-process. Online FDR needs a well-defined stream; quietly dropping failed tests from the stream breaks the proof.

Extensions that sit on this chapter include variance-reduced tests (CUPED, taken up in Chapter 07), causal inference when assignment is not a clean Bernoulli, and e-value aggregation for living meta-analysis.

## Video tham khảo

- [Trustworthy Online Controlled Experiments at Large Scale w/ Ronny Kohavi | DECISIONS/Replay by Split](https://www.youtube.com/watch?v=kTAFOCynWIg) — Split, English. **xem trước.** Ronny Kohavi on online experiments as two-sample tests at industrial scale — the A/B frame that opens this note.
- [Peeking at A/B Tests: Why it matters, and what to do about it](https://www.youtube.com/watch?v=6G8F22NSCxk) — KDD2017 video, English. The KDD talk on peeking: watching a p-value and stopping early inflates type I error — the sequential-testing section.
- [False Discovery Rates, FDR, clearly explained](https://www.youtube.com/watch?v=K8LQSvtjcEo) — StatQuest with Josh Starmer, English. The false-discovery rate and Benjamini–Hochberg for thousands of hypotheses — the multiple-testing section set beside Bonferroni.

## Exercises

**Exercise 1 (conceptual).** A vendor’s dashboard updates a $$p$$-value every hour and paints the card green at 0.05. Which assumption of the fixed-sample test is violated? Name one always-valid alternative and one operational alternative that does not change the test.

**Exercise 2 (computational).** Using `peeking_inflation`, plot Type I error against the number of looks for $$L \in \{1, 2, 5, 10, 20, 50\}$$. Overlay the Bonferroni line $$\alpha / L$$ applied to each look. How much power do you lose if you Bonferroni-correct 20 looks on a true 2% lift with $$n = 5{,}000$$ per arm?

**Exercise 3 (computational).** Simulate $$m = 1{,}000$$ $$p$$-values of which 100 come from $$N(3,1)$$ $$z$$-scores and the rest are uniform. Compare the number of true and false discoveries under uncorrected $$\alpha = 0.05$$, Bonferroni, and BH. Relate the Bonferroni count to the GWAS mindset and the BH count to an ads-metrics dashboard.

**Exercise 4 (open).** Read Kohavi, Deng, and Vermeer (2022). Pick one intuition buster and reproduce it with a small simulation. Then propose a platform UI change—copy, a disabled button, a forced sample-size calculator—that would make that mistake harder.

## References

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B, 57*(1), 289–300. The still-cited FDR procedure.

Johari, R., Koomen, P., Pekelis, L., & Walsh, D. (2022). Always valid inference: Continuous monitoring of A/B tests. *Operations Research, 70*(3), 1806–1821. [https://doi.org/10.1287/opre.2021.2135](https://doi.org/10.1287/opre.2021.2135)

Kohavi, R., Deng, A., & Vermeer, L. (2022). A/B testing intuition busters: Common misunderstandings in online controlled experiments. In *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (pp. 3168–3177). [https://doi.org/10.1145/3534678.3539160](https://doi.org/10.1145/3534678.3539160)

Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*. Cambridge University Press. The industrial reference for experimentation platforms.

Ramdas, A., Grünwald, P., Vovk, V., & Shafer, G. (2023). Game-theoretic statistics and safe anytime-valid inference. *Statistical Science, 38*(4), 576–601. [https://doi.org/10.1214/23-STS894](https://doi.org/10.1214/23-STS894)

Robertson, D. S., Wason, J. M. S., & Ramdas, A. (2023). Online multiple hypothesis testing. *Statistical Science, 38*(4), 557–575. [https://doi.org/10.1214/23-STS901](https://doi.org/10.1214/23-STS901)

Uffelmann, E., Huang, Q. Q., Munung, N. S., de Vries, J., Okada, Y., Martin, A. R., … Posthuma, D. (2021). Genome-wide association studies. *Nature Reviews Methods Primers, 1*, 59. [https://doi.org/10.1038/s41586-021-00056-9](https://doi.org/10.1038/s41586-021-00056-9)
