---
layout: post
title: "05-03-00 Optional: Estimation in Modern Machine Learning"
chapter: "05"
order: 3
owner: nglelinh
lang: en
categories:
- chapter05
lesson_type: optional
---

Suppose a brand-new backpack launches on an e-commerce site, receives 2 impressions, and 1 shopper clicks it—giving a raw click-through rate of $$1/2 = 50\%$$. Meanwhile, a bestselling backpack with 10,000 impressions has 1,800 clicks ($$18\%$$). If you rank products by raw Maximum Likelihood ($$c_i / n_i$$), every lucky 1-for-2 rookie shoots straight to the top of page one while proven bestsellers sink!

This optional lesson does not replace the point- and interval-estimation theory in Chapter 05. It reveals how those exact estimators run the engines of modern AI and data science: **Maximum Likelihood** inside deep networks and LLMs, **Method of Moments** for multimodal mixtures, **Empirical Bayes shrinkage** inside production search ranking, and **Bootstrap / Conformal intervals** when analytic standard errors vanish.

After working through it you should be able to recognize cross-entropy training as maximum likelihood, explain why moment matching is still used for Gaussian mixtures, compute and implement a Beta–Binomial empirical-Bayes shrinker, and contrast a bootstrap interval with a conformal prediction set.

The prerequisites are the required lessons on point estimation (bias, variance, MLE, method of moments) and interval estimation (confidence intervals and the bootstrap), plus elementary Python with NumPy.

---

## Background Knowledge Map: How Chapter 05 Estimators Scale to Modern ML

Four classical estimation pillars map directly onto 2022–2026 production ML systems.

### 1. Maximum Likelihood Estimation (MLE) as Deep Learning Loss
Minimizing cross-entropy in image classification or next-token loss in an LLM is *literally* maximizing the log-likelihood $$\sum \log p_\theta(y_i \mid x_i)$$. Scaling laws (like Chinchilla) are empirical tradeoffs between parameter dimension $$\dim(\theta)$$ and sample size $$n$$ for that exact likelihood.
- **Core questions:** When does penalized MLE work well despite non-convexity and over-parameterization?
- **Real-world reach:** LLM pretraining, computer vision classifiers, and probabilistic forecasting.

### 2. Method of Moments (MoM) & Tensor Decomposition
When a likelihood surface has many local traps (like Gaussian Mixture Models), EM can stall in a poor local optimum. **Method of Moments** equates empirical moments (means, covariances, third-order tensors) to theoretical moments—providing globally consistent estimators and smart initializers for EM.
- **Core questions:** How can implicit moment tensors avoid $$O(n^3)$$ memory blow-up in high dimensions?
- **Real-world reach:** Speaker recognition, single-cell clustering, topic modeling, and anomaly detection.

### 3. Empirical Bayes & James–Stein Shrinkage
When estimating thousands of parallel rates (product CTRs, hospital mortality rates, baseball batting averages) with wildly unequal sample sizes $$n_i$$, **Empirical Bayes** estimates a shared prior from the whole panel and shrinks noisy small-$$n$$ estimates toward that prior—drastically lowering total MSE.
- **Core questions:** How much weight should we give the population prior versus an item’s own clicks?
- **Real-world reach:** Amazon/e-commerce cold-start ranking, ad-click prediction, and genomics differential expression.

---

## Why estimation left the textbook

A ranking service, a language model, and a clustering pipeline all solve the same formal problem you have already studied: given a sample $$X_1,\ldots,X_n$$, produce $$\hat{\theta}$$ and, when a decision is costly, an uncertainty interval around it. What changed between 2022 and 2026 is not the definition of an estimator—it is the scale, latency, cold-start sparsity, and compute budget that dictate *which* estimator wins.

## Maximum likelihood is how modern networks learn

For i.i.d. observations the MLE maximizes

$$\hat{\theta}_{\mathrm{MLE}} = \arg\max_\theta \sum_{i=1}^n \log p(x_i \mid \theta).$$

In a classifier, $$p(y \mid x,\theta)$$ is a softmax, and the average negative log-likelihood is *identically* the **cross-entropy loss** used in PyTorch and TensorFlow. Training with SGD is computational MLE: the score function $$\nabla_\theta \log p$$ is backpropagated rather than set to zero in closed form.

Autoregressive language models make the same identification token by token, maximizing $$\sum_t \log p_\theta(w_t \mid w_{<t})$$. Hoffmann et al. (2022) (“Chinchilla”) trained over 400 transformers and showed that for a fixed compute budget, optimal MLE quality scales parameters and training tokens roughly equally—a modern compute-frontier version of the classical bias–variance tradeoff.

Two honest caveats: (1) regularization, weight decay, and early stopping produce a *penalized* MLE (MAP-like), and (2) over-parameterized nets break classical $$n^{-1/2}$$ asymptotic normality, even while held-out negative log-likelihood remains the gold-standard objective.

---

### Worked Example 1: Verifying Cross-Entropy = Negative Log-Likelihood by Hand

Suppose a 2-class neural network outputs raw logits for two training examples:
- **Example 1** (true label $$y_1 = 0$$): logits $$z^{(1)} = (2.0,\; 0.0)$$.
- **Example 2** (true label $$y_2 = 1$$): logits $$z^{(2)} = (0.0,\; 1.0)$$.

Compute the softmax probabilities and average negative log-likelihood by hand:
1. **Example 1:** $$p(y=0 \mid x_1) = \dfrac{e^2}{e^2 + e^0} = \dfrac{7.389}{8.389} \approx 0.8808 \implies -\log(0.8808) \approx \mathbf{0.1269}$$.
2. **Example 2:** $$p(y=1 \mid x_2) = \dfrac{e^1}{e^0 + e^1} = \dfrac{2.718}{3.718} \approx 0.7311 \implies -\log(0.7311) \approx \mathbf{0.3133}$$.
3. **Average NLL (= Cross-Entropy Loss):**
   $$\mathcal{L} = \frac{0.1269 + 0.3133}{2} = \mathbf{0.2201}.$$
Maximizing the joint likelihood $$0.8808 \times 0.7311 = 0.6439$$ is 100% mathematically identical to minimizing $$\mathcal{L} = 0.2201$$!

---

```python
import numpy as np

def softmax_nll(logits, labels):
    """
    Average negative log-likelihood of a categorical model.

    Args:
        logits: array of shape (n, k), unnormalised class scores
        labels: integer array of shape (n,), values in {0, ..., k-1}

    Returns:
        Scalar mean NLL, equal to multiclass cross-entropy.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return -np.mean(log_probs[np.arange(len(labels)), labels])
```

## Method of moments for Gaussian mixtures

A $$K$$-component Gaussian mixture has density

$$p(x) = \sum_{k=1}^K \pi_k \, \mathcal{N}(x; \mu_k, \Sigma_k).$$

Because the likelihood is non-concave, EM only guarantees a local MLE. **Method of Moments** (introduced by Karl Pearson in 1894) matches empirical moments to theoretical moments. In high dimensions, Hsu and Kakade (2013) and Anandkumar et al. (2014) proved that the first three moment tensors uniquely identify spherical GMM parameters in polynomial time; Pereira, Kileel, and Kolda (2022) made this practical via *implicit* moment tensors avoiding $$O(n^3)$$ storage. In production, MoM often *initializes* EM—combining MoM’s global identification with EM’s local statistical efficiency.

```python
import numpy as np

def mom_two_gaussians(x):
    """
    Method-of-moments sketch for an equal-weight mixture
    0.5 N(-m, 1) + 0.5 N(+m, 1), using the second moment.

    For a general two-component mixture one matches three
    or more moments and solves Pearson's polynomial.
    """
    second = np.mean(x ** 2)
    # For this symmetric family, E[X^2] = m^2 + 1
    m_hat = np.sqrt(max(second - 1.0, 0.0))
    return m_hat
```

## Empirical Bayes in production ranking

**Empirical Bayes** estimates a shared prior directly from the ensemble of items, then updates each item via Bayes’ rule. For click-through rates with a $$\mathrm{Beta}(\alpha, \beta)$$ prior and $$c_i$$ clicks in $$n_i$$ impressions, the posterior mean is

$$\hat{p}_i = \frac{\alpha + c_i}{\alpha + \beta + n_i} = \underbrace{\frac{\alpha+\beta}{\alpha+\beta+n_i}}_{\text{prior weight}}\cdot\frac{\alpha}{\alpha+\beta} + \underbrace{\frac{n_i}{\alpha+\beta+n_i}}_{\text{data weight}}\cdot\frac{c_i}{n_i}.$$

Han, Castells, Gupta, Xu, and Salaka (2022) deployed this cold-start pattern in Amazon product search across 50 million queries, lifting new-product impressions by **13.53%** and new-product purchases by **11.14%**.

---

### Worked Example 2: Beta–Binomial Empirical Bayes Shrinkage by Hand

Suppose across all backpacks on your site, the fitted Empirical Bayes prior is $$\mathrm{Beta}(\alpha = 2,\; \beta = 18)$$—representing a prior mean CTR of $$m = \frac{2}{20} = \mathbf{10\%}$$ with “pseudo-count strength” $$s = \alpha + \beta = 20$$ impressions. Watch how it automatically tames the cold-start paradox from our opening story:

| Product | Clicks / Imps | Raw MLE $$c_i / n_i$$ | Empirical Bayes $$\frac{2 + c_i}{20 + n_i}$$ | Behavior |
|---------|---------------|-----------------------|----------------------------------------------|----------|
| **Brand-new item** | $$0 / 0$$ | Undefined! | $$2 / 20 = \mathbf{10.0\%}$$ | Falls back smoothly to prior mean |
| **Lucky rookie** | $$1 / 2$$ | $$50.0\%$$ (noisy!) | $$(2+1)/(20+2) = \mathbf{13.6\%}$$ | Heavily shrunk toward 10% |
| **Proven bestseller** | $$100 / 500$$ | $$20.0\%$$ | $$(2+100)/(20+500) = \mathbf{19.6\%}$$ | Data dominates ($$500 \gg 20$$)! |

Notice the magic: the proven bestseller (19.6%) cleanly outranks the lucky 1-for-2 rookie (13.6%) without any ad-hoc `if impressions < 50` hack!

---

```python
import numpy as np

def fit_beta_mom(clicks, imps, eps=1e-6):
    """
    Method-of-moments Beta prior from a panel of items.

    Args:
        clicks, imps: 1-d arrays of counts, same length

    Returns:
        (alpha, beta) of the fitted Beta prior
    """
    rates = clicks / np.maximum(imps, 1)
    # Down-weight items with almost no impressions
    w = imps / imps.sum()
    m = np.average(rates, weights=w)
    v = np.average((rates - m) ** 2, weights=w)
    # Beta mean m, variance m(1-m)/(s+1) => s = m(1-m)/v - 1
    s = m * (1 - m) / max(v, eps) - 1
    s = max(s, 1.0)
    return s * m, s * (1 - m)

def posterior_ctr(clicks, imps, alpha, beta):
    """Empirical-Bayes posterior mean CTR."""
    return (alpha + clicks) / (alpha + beta + imps)
```

## Intervals when the analytic standard error is missing

The required interval-estimation lesson constructs $$\bar{x} \pm z_{1-\alpha/2}\frac{s}{\sqrt{n}}$$ and bootstrap percentile intervals. Deep networks with millions of weights break Hessian-based standard errors. Two modern tools fill the gap:
- **Bootstrap / Deep Ensembles:** resample training data or random seeds and read empirical percentiles.
- **Split Conformal Prediction** (Angelopoulos & Bates, 2023): use a held-out calibration fold to produce a finite-sample prediction set $$C(x)$$ satisfying $$P(Y_{n+1} \in C(X_{n+1})) \ge 1 - \alpha$$ under exchangeability alone—a *prediction* interval for $$Y_{\text{new}}$$, not a confidence interval for $$\theta$$.

```python
import numpy as np

def split_conformal_interval(calib_scores, q=0.9):
    """
    Split-conformal quantile for absolute residuals.

    Args:
        calib_scores: |y - f(x)| on a held-out calibration fold
        q: target coverage, e.g. 0.9

    Returns:
        Residual half-width to add/subtract from f(x_new)
    """
    n = len(calib_scores)
    level = np.ceil((n + 1) * q) / n
    return np.quantile(calib_scores, min(level, 1.0), method="higher")
```

---

### Deep-Dive Explanation: Matching the Estimator to the Problem’s Geometry (Read Slowly)

How does a senior statistician decide between MLE, Method of Moments, Empirical Bayes, and Conformal Prediction on a real project? Follow these four steps slowly.

#### Step 1: Check whether the likelihood is clean and well-identified
If your data follow a smooth exponential family (logistic, Poisson, Gaussian) and optimization converges reliably, **MLE** (or penalized MLE) achieves the lowest asymptotic variance (Cramér–Rao bound).

#### Step 2: Check for multimodal likelihood traps
In latent-variable models (mixtures, topic models, HMMs), pure likelihood optimization gets stuck in local peaks. Use **Method of Moments** to find a consistent global starting point, then refine with EM.

#### Step 3: Check whether you are estimating thousands of sparse units simultaneously
Whenever you have a panel of many items/users/stores with tiny sample sizes $$n_i$$, raw per-unit MLE has catastrophic variance. **Empirical Bayes** borrows strength across the whole panel via a data-estimated prior—trading tiny bias for massive variance reduction.

#### Step 4: Ask whether you need a parameter CI or a future-outcome prediction interval
If you want uncertainty around an average treatment effect $$\theta$$, use a confidence interval or bootstrap. If you want a guarantee that a *specific customer’s* future outcome $$Y_{n+1}$$ falls inside your interval, use **conformal prediction** on a clean calibration split.

---

## What a data scientist should take away

Choose the estimator that matches the *structure* of the problem, not the prestige of the model class. If the observation model is a well-specified exponential family and you can optimize it, use MLE (or a regularized version). If the likelihood is multimodal or you need a theoretically identified initializer, match moments. If you have a large panel of sparse units, shrink with an empirical-Bayes prior. If you need a finite-sample coverage guarantee around a black-box prediction, prefer conformal or the bootstrap to a naive normal interval.

## Challenges and extensions

MLE in deep nets is computationally successful and inferentially delicate: standard errors, model misspecification, and train–serve skew all break textbook asymptotics. Method-of-moments estimators can have large finite-sample variance even when consistent; implicit tensors reduce cost, not statistical noise. Empirical Bayes assumes exchangeability across items; a sudden assortment shift silently mis-estimates $$(\alpha,\beta)$$. Conformal coverage is marginal, not conditional on a rare slice of users.

Open directions that sit on top of this chapter include likelihood-free estimation (simulation-based inference, score matching), hierarchical empirical Bayes for many related ranking markets, and conformal methods under distribution shift.

## Video tham khảo

- [Neural Networks Part 6: Cross Entropy](https://www.youtube.com/watch?v=6ArSys5qHAU) — StatQuest with Josh Starmer, English. **xem trước.** Cross-entropy is the average negative log-likelihood of a categorical model — the identity this note draws between network training and maximum likelihood.
- [Gaussian Mixture Model](https://www.youtube.com/watch?v=EWd1xRkyEog) — ritvikmath, English. Gaussian mixture models, the family this note fits by moments when the likelihood is multimodal.
- [Bayesian Updates and Conjugate Priors](https://www.youtube.com/watch?v=GqSX-8AQL90) — Steve Brunton, English. Bayesian updates with conjugate priors, the shrinkage mechanism this note calls empirical Bayes in ranking.

## Exercises

**Exercise 1 (conceptual).** A language-model paper reports training loss versus compute. Which object in this chapter is the training loss? Why is a smaller training NLL not, by itself, evidence that $$\hat{\theta}$$ is closer to a “true” data-generating parameter?

**Exercise 2 (computational).** Simulate a two-component univariate GMM with unequal weights. Estimate the parameters by (i) method of moments via the first three moments and (ii) EM. Repeat 200 times and compare bias and RMSE. When does EM win, and when does a bad initialisation make it worse than MoM?

**Exercise 3 (computational).** Using the Beta–Binomial functions above, generate 5,000 items with a true $$\mathrm{Beta}(2,20)$$ prior and binomial observations. Recover $$(\hat{\alpha},\hat{\beta})$$ by method of moments. Plot raw MLE rates versus posterior means for items with $$n \in \{10, 100, 1000\}$$ impressions. Interpret the shrinkage.

**Exercise 4 (open).** Pick a public learning-to-rank or product-search dataset. Define a cold-start slice (items below a click threshold). Compare raw CTR, a global-mean shrinker, and an empirical-Bayes shrinker on a held-out week. Report both ranking quality and calibration. What prior misspecification would make EB worse than the global mean?

## References

Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning, 16*(4), 494–591. [https://doi.org/10.1561/2200000101](https://doi.org/10.1561/2200000101)

Anandkumar, A., Ge, R., Hsu, D., Kakade, S. M., & Telgarsky, M. (2014). Tensor decompositions for learning latent variable models. *Journal of Machine Learning Research, 15*(80), 2773–2832. Still the standard theoretical reference for moment-based latent-variable estimation.

Efron, B. (2010). *Large-Scale Inference: Empirical Bayes Methods for Estimation, Testing, and Prediction*. Cambridge University Press. The modern monograph behind production shrinkage.

Han, C., Castells, P., Gupta, P., Xu, X., & Salaka, V. (2022). Addressing cold start in product search via empirical Bayes. In *Proceedings of the 31st ACM International Conference on Information and Knowledge Management* (pp. 3141–3151). [https://doi.org/10.1145/3511808.3557066](https://doi.org/10.1145/3511808.3557066)

Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., … Sifre, L. (2022). Training compute-optimal large language models. In *Advances in Neural Information Processing Systems, 35*. [https://arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556)

Hsu, D., & Kakade, S. M. (2013). Learning mixtures of spherical Gaussians: Moment methods and spectral decompositions. In *ITCS 2013*. Classic spectral method of moments for GMMs; still cited by the 2022–2025 implicit-tensor papers.

Pereira, J. M., Kileel, J., & Kolda, T. G. (2022). Tensor moments of Gaussian mixture models: Theory and applications. [https://arxiv.org/abs/2202.06930](https://arxiv.org/abs/2202.06930)
