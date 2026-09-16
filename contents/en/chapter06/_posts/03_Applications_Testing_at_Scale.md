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

This optional lesson leaves the definitions of $$H_0$$, $$p$$-values, Type I/II errors, and coefficient inference untouched. It follows those objects into the systems that now run them thousands of times a day: industrial A/B platforms, genome-wide association studies, advertising and ranking experiments, and sequential tests that must remain valid when someone peeks.

You should finish able to write the two-sample test that an experimentation platform actually computes, explain why “peeking” inflates Type I error, apply Benjamini–Hochberg and say when Bonferroni or a genome-wide threshold is used instead, and sketch an always-valid $$p$$-value.

The required Chapter 06 lessons (the testing framework and inference for linear models) plus a binomial or Bernoulli model are enough. Familiarity with the multiple-testing simulation already in the hypothesis-testing lesson will help; we reuse that intuition rather than re-deriving it.

---

## A/B testing is hypothesis testing with a product manager attached

An online controlled experiment asks a textbook question. Let $$p_C$$ and $$p_T$$ be conversion probabilities on control and treatment. The platform tests

$$H_0: p_T = p_C \qquad \text{versus} \qquad H_1: p_T \ne p_C$$

with a two-sample statistic, typically a $$z$$-test for a difference of proportions or a $$t$$-test for a real-valued metric. Kohavi, Tang, and Xu (2020) document platforms at Microsoft, Google, and LinkedIn that each run tens of thousands of such tests a year. The inferential skeleton is the one you already know. The engineering is about *trustworthiness*: metric definition, interference between users, carry-over, and the fact that a “significant” lift of 0.1% can be worth millions and also be a measurement bug.

Kohavi, Deng, and Vermeer (2022) catalogue *intuition busters* that vendors still ship: treating a computed 95% interval as a 95% probability statement about the current experiment, using post-hoc power as if it were design power, removing outliers after seeing the result, and allocating traffic unequally without adjusting the variance formula. Every one of those mistakes is a misuse of the Chapter 06 framework, not a new framework.

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

At platform scale the same test is wrapped in guardrail metrics, a pre-registered overall evaluation criterion, and a Sample Ratio Mismatch check—the last is itself a goodness-of-fit test that the assignment mechanism was not broken.

## Peeking, optional stopping, and sequential tests

A fixed-sample $$p$$-value is valid at *one* pre-chosen $$n$$. If an analyst refreshes a dashboard every day and stops at the first time $$p < 0.05$$, the Type I error is far above 5%. That is optional stopping, and it is the default user behaviour on every experimentation UI that draws a live $$p$$-value.

Johari, Koomen, Pekelis, and Walsh (2022) formalise *always-valid* $$p$$-values: a process $$p_n$$ such that for every stopping time $$\tau$$,

$$P_{H_0}(p_\tau \le \alpha) \le \alpha.$$

Their construction uses a mixture sequential probability ratio test (mSPRT). It has been deployed on a commercial platform that analysed hundreds of thousands of experiments. The statistical price is power: an always-valid test typically needs more samples than a fixed-$$n$$ test of the same level.

Ramdas, Grünwald, Vovk, and Shafer (2023) place always-valid tests inside *safe anytime-valid inference*. The primitive is an e-value or test martingale $$M_n$$, a non-negative process with $$E_{H_0}[M_{n+1} \mid \mathcal{F}_n] \le M_n$$. Ville’s inequality gives

$$P_{H_0}\Bigl(\sup_n M_n \ge 1/\alpha\Bigr) \le \alpha,$$

so $$p_n = \inf_{k \le n} 1/M_k$$ is always valid. E-values multiply across independent experiments, which is how a platform can spend an error budget over a stream of tests.

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

Running that function with the defaults yields a Type I error near 20–25%, not 5%. The number is the reason sequential methodology moved from clinical-trial manuals into consumer software.

## Multiple testing: genomes, ads, and online FDR

Test one hypothesis at level $$\alpha$$ and the Type I error is $$\alpha$$. Test $$m$$ independent true nulls and the probability of at least one false rejection is $$1-(1-\alpha)^m$$. The required lesson already simulates that fact. Two corrections dominate practice, and they are not interchangeable.

**Family-wise error.** Bonferroni uses $$\alpha/m$$. Genome-wide association studies, as reviewed by Uffelmann et al. (2021), conventionally reject at $$5 \times 10^{-8}$$, a Bonferroni-style threshold for about one million independent common variants. That choice privileges reproducibility over power and is why a GWAS hit is treated as a rare event.

**False discovery rate.** Benjamini and Hochberg (1995) control the expected fraction of false rejections among the rejected set,

$$\mathrm{FDR} = E\Bigl[\frac{V}{\max(R,1)}\Bigr].$$

Sort $$p_{(1)} \le \cdots \le p_{(m)}$$ and reject the first $$k$$ with

$$p_{(k)} \le \frac{k}{m}\alpha.$$

BH is the default in RNA-seq, imaging, and any dashboard that tests hundreds of metrics or segments. It is *not* a substitute for the GWAS threshold: the scientific cost of a false genomic locus is closer to a family-wise error.

**Online FDR.** Platforms do not receive all $$p$$-values at once. Experiments arrive as a stream. Robertson, Wason, and Ramdas (2023) survey online procedures (LORD, SAFFRON, ADDIS) that spend an $$\alpha$$-wealth over time and earn wealth back on discoveries. That is the right multiplicity model for an ads or ranking organisation that starts new tests every week and will never have a single closed list of hypotheses.

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

The second required lesson builds confidence intervals and $$p$$-values for regression coefficients. In a ranking or ads model those coefficients number in the thousands. Reporting every $$p < 0.05$$ as a “significant feature” is a multiple-testing error. Production practice either (i) treats coefficients as prediction tools and does not test them, (ii) applies BH or knockoffs for variable selection, or (iii) uses regularisation and reports predictive metrics. The linear-model formulas do not become false; they become incomplete once you search over many of them.

## Challenges and extensions

Peeking is not the only validity threat. Interference (one user’s treatment changes another user’s outcome), novelty effects, and Sample Ratio Mismatch all produce *significant* results that are not treatment effects. Always-valid tests buy validity with sample size; teams that cannot wait sometimes prefer a hard cap on looks (group sequential designs) instead of a fully sequential e-process. Online FDR needs a well-defined stream; quietly dropping failed tests from the stream breaks the proof.

Extensions that sit on this chapter include variance-reduced tests (CUPED, taken up in Chapter 07), causal inference when assignment is not a clean Bernoulli, and e-value aggregation for living meta-analysis.

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

Uffelmann, E., Huang, Q. Q., Munung, N. S., de Vries, J., Okada, Y., Martin, A. R., … Posthuma, D. (2021). Genome-wide association studies. *Nature Reviews Methods Primers, 1*, 59. [https://doi.org/10.1038/s43586-021-00056-9](https://doi.org/10.1038/s43586-021-00056-9)
