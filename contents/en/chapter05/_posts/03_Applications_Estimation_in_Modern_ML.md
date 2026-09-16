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

This optional lesson does not replace the point- and interval-estimation theory in this chapter. It shows how those same estimators are the statistical engine inside contemporary computer-science and data-science systems: maximum likelihood in deep networks, method of moments for mixture models, empirical Bayes in production ranking, and modern interval methods when a closed-form standard error is unavailable.

After working through it you should be able to recognise cross-entropy training as maximum likelihood, explain why moment matching is still used for Gaussian mixtures, implement a Beta–Binomial empirical-Bayes shrinker of the kind deployed in e-commerce search, and contrast a bootstrap interval with a conformal prediction set.

The prerequisites are the required lessons on point estimation (bias, variance, MLE, method of moments) and interval estimation (confidence intervals and the bootstrap), plus elementary Python with NumPy. No new probability axioms are introduced.

---

## Why estimation left the textbook

A ranking service, a language model, and a clustering pipeline all solve the same formal problem you have already studied: given a sample $$X_1,\ldots,X_n$$, produce $$\hat{\theta}$$ and, when a decision is costly, an interval around it. What changed between 2022 and 2026 is not the definition of an estimator. It is the scale at which the same definitions are executed, and the engineering constraints—latency, cold start, compute budgets—that decide *which* estimator is used.

Three production patterns dominate. Deep models minimise a negative log-likelihood, so they *are* MLE. Mixture models in speaker recognition, single-cell clustering, and anomaly detection still rely on moment matching when the likelihood is multimodal. Ranking and advertising systems shrink noisy rates toward a data-estimated prior—the empirical-Bayes move that Efron made precise and that large platforms still ship.

## Maximum likelihood is how modern networks learn

For i.i.d. observations the MLE maximises

$$\hat{\theta}_{\mathrm{MLE}} = \arg\max_\theta \sum_{i=1}^n \log p(x_i \mid \theta).$$

In a classifier the observation is a pair $$(x,y)$$ and $$p(y \mid x,\theta)$$ is a softmax. The average negative log-likelihood is exactly the cross-entropy loss used in every major deep-learning library. Training a network with stochastic gradient descent is therefore computational MLE: the score function $$\nabla_\theta \log p$$ is back-propagated rather than set to zero in closed form.

Autoregressive language models make the same identification token by token. Next-token training maximises

$$\sum_t \log p_\theta(w_t \mid w_{<t}),$$

which is MLE for a conditional categorical family. Hoffmann et al. (2022) treat that likelihood as an *empirical* object: they train more than 400 transformers and show that, for a fixed compute budget, the compute-optimal model scales parameters and training tokens in roughly equal proportion. The statistical content of the “Chinchilla” result is that the quality of an MLE is jointly determined by the dimension of $$\theta$$ and the sample size. That is the classical bias–variance story, read off a compute-optimal frontier rather than a textbook MSE table.

Two caveats keep the identification honest. First, regularisation, early stopping, and data augmentation mean the trained weights are a *penalised* MLE, not the raw maximiser. Second, over-parameterised networks can interpolate the training sample, so the classical “MLE is asymptotically normal of order $$n^{-1/2}$$” theorem does not literally apply. What survives is the *objective*: practitioners still report held-out negative log-likelihood, and scaling laws are still laws about that likelihood.

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

The likelihood is non-concave. Expectation–maximisation finds a local MLE and is the default in `sklearn.mixture.GaussianMixture`. The method of moments, introduced by Pearson in 1894 for a two-component univariate mixture, matches empirical moments to the moments implied by $$(\pi,\mu,\Sigma)$$.

In high dimension the relevant moments are tensors. Hsu and Kakade (2013) and Anandkumar, Ge, Hsu, Kakade, and Telgarsky (2014) showed that the first three moments of a spherical GMM determine the parameters through a symmetric tensor decomposition, with polynomial-time guarantees when the means are linearly independent. Those papers remain the citations that learning-theory courses use.

What is new is practicality. Forming a $$d$$-way moment tensor costs $$O(n^d)$$ storage. Pereira, Kileel, and Kolda (2022) give implicit expressions for GMM moment tensors so that moment matching can be run with $$O(n^2)$$ or $$O(n^3)$$ work for general covariances, and $$O(n)$$ work for diagonal ones, without materialising the tensor. The same idea continues in later implicit-moment work (2024–2025). The statistical message for this course is unchanged: when the likelihood surface is hostile, equate moments.

A one-dimensional illustration recovers Pearson’s original move.

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

In production, moment estimates are often used to *initialise* EM rather than to replace it—the combination inherits MoM’s global identification and EM’s local efficiency.

## Empirical Bayes in production ranking

Empirical Bayes estimates the prior from the same data that will be updated by it. For a click-through rate, a Beta–Binomial model is the workhorse. If item $$i$$ has $$c_i$$ clicks in $$n_i$$ impressions and the prior is $$\mathrm{Beta}(\alpha,\beta)$$,

$$\hat{p}_i = \frac{\alpha + c_i}{\alpha + \beta + n_i}.$$

The prior mean $$m = \alpha/(\alpha+\beta)$$ and the prior strength $$s = \alpha+\beta$$ are not guessed. They are estimated from the ensemble of items—by matching the mean and variance of the observed rates (a method-of-moments prior) or by maximising a marginal likelihood. New items with $$n_i = 0$$ are shown at the prior mean; items with huge $$n_i$$ are essentially the raw MLE $$c_i/n_i$$. That is James–Stein shrinkage in product clothing.

Han, Castells, Gupta, Xu, and Salaka (2022) deployed this pattern for cold start in Amazon product search: a non-behavioural model supplies the prior, and observed clicks and purchases update the posterior. An online experiment on 50 million queries raised new-product impressions by 13.53% and new-product purchases by 11.14%. Related 2023 work (EBRank) uses the same prior/posterior split to reduce exploitation bias in learning-to-rank.

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

The connection back to this chapter is literal. The prior parameters are a point estimate. The posterior mean is another point estimate. A credible interval from the Beta posterior, or a bootstrap of $$(\hat{\alpha},\hat{\beta})$$, is an interval estimate.

## Intervals when the analytic standard error is missing

The required interval-estimation lesson constructs

$$\bar{x} \pm z_{1-\alpha/2} \frac{s}{\sqrt{n}}$$

and then the bootstrap percentile interval. Those remain the right tools for a mean or a simple functional. Deep models break the derivation: the parameter is millions of weights, the loss is non-convex, and a Hessian-based standard error is not computable.

Two answers are used in 2022–2026 systems. The bootstrap (and its cousin, deep ensembles) resamples the training process and reads a percentile interval from the empirical distribution of the statistic—the same algorithm you already implemented, at higher cost. Conformal prediction, surveyed for machine-learning audiences by Angelopoulos and Bates (2023), wraps *any* fitted model in a prediction set $$C(x)$$ that satisfies

$$P\bigl(Y_{n+1} \in C(X_{n+1})\bigr) \ge 1-\alpha$$

under exchangeability, with no assumption that the model is correct. Split conformal uses a calibration fold to compute a quantile of nonconformity scores; the guarantee is finite-sample. It is not a confidence interval for $$\theta$$. It is a prediction interval for a new response, which is often what a deployed system actually needs.

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

## What a data scientist should take away

Choose the estimator that matches the *structure* of the problem, not the prestige of the model class. If the observation model is a well-specified exponential family and you can optimise it, use MLE (or a regularised version). If the likelihood is multimodal or you need a theoretically identified initialiser, match moments. If you have a large panel of similar units and many of them are sparse, shrink with a prior estimated from the panel. If you need a coverage guarantee around a black-box prediction, prefer conformal or the bootstrap to a naive normal interval.

## Challenges and extensions

MLE in deep nets is computationally successful and inferentially delicate: standard errors, model misspecification, and train–serve skew all break textbook asymptotics. Method-of-moments estimators can have large finite-sample variance even when they are consistent; implicit tensors reduce cost, not statistical noise. Empirical Bayes assumes exchangeability across items; a sudden assortment shift silently mis-estimates $$(\alpha,\beta)$$. Conformal coverage is marginal, not conditional on a rare slice of users.

Open directions that sit on top of this chapter include likelihood-free estimation (simulation-based inference, score matching), hierarchical empirical Bayes for many related ranking markets, and conformal methods under distribution shift.

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
