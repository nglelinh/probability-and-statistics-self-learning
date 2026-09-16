---
layout: post
title: 01-11-00 Applications and recent developments
chapter: "01"
order: 11
owner: nglelinh
lang: en
categories:
- chapter01
lesson_type: optional
---

Monte Carlo, cross-validation, and information criteria in this chapter are not classroom exercises you leave behind. They are the working language of how modern machine-learning systems *evaluate* themselves when a closed form is unavailable. This optional note connects those tools to three developments from roughly 2022–2024. The required lessons do not change; the point is to see *where* simulation, resampling, and model selection show up in production papers and libraries.

## Conformal prediction as a hold-out Monte Carlo

A classifier or regressor $$f$$ typically returns a point $$\hat{y}$$. High-stakes settings (diagnostics, ranking, content moderation) need a *set* $$C(x)$$ that covers the true label with a user-chosen probability. **Split conformal prediction** does this with the same computational idea as a validation split. Fit $$f$$ on a training fold; on a disjoint calibration fold of size $$n$$ compute nonconformity scores $$s_i = s(x_i, y_i)$$ (for example absolute residuals); take the empirical quantile

$$
\hat{q} = \text{Quantile}_{1-\alpha}\bigl(s_1,\ldots,s_n,+\infty\bigr).
$$

The prediction set $$C(x)=\{y : s(x,y)\le \hat{q}\}$$ then satisfies, under exchangeability,

$$
P\bigl(Y_{n+1}\in C(X_{n+1})\bigr) \ge 1-\alpha.
$$

No Gaussian residual, no correctly specified model—only the exchangeability that also justifies ordinary cross-validation. Angelopoulos and Bates’s monograph (*Foundations and Trends in Machine Learning*, 2023; [arXiv:2107.07511](https://arxiv.org/abs/2107.07511)) is the formulation most 2023–2026 software cites. The scikit-learn-contrib library [MAPIE](https://mapie.readthedocs.io/) (Cordier, Blot, Lacombe, Morzadec, Capitaine, Brunel, COPA 2023) wraps the same quantile around any estimator. When you later meet bootstrap intervals, reread this paragraph: conformal replaces “assume a sampling distribution” with “reuse a calibration sample,” which is Chapter 01 thinking.

## Simulation for LLM evaluation

There is no closed-form accuracy for a chat model. Liang, Bommasani, Lee, and colleagues (*Holistic Evaluation of Language Models*, [TMLR 2023](https://openreview.net/forum?id=iO4LZibEqW); [arXiv:2211.09110](https://arxiv.org/abs/2211.09110); toolkit [stanford-crfm/helm](https://github.com/stanford-crfm/helm)) treat evaluation as a designed experiment: 16 core scenarios, seven metrics (accuracy, **calibration**, robustness, fairness, bias, toxicity, efficiency), and 30 models run under a shared protocol. The law of large numbers from the opening lesson is what makes a multi-scenario average meaningful; a single headline number is a one-draw Monte Carlo.

Human preference is even less analytic. Chiang, Zheng, Sheng, Angelopoulos, Li, Li, Zhu, Zhang, Jordan, Gonzalez, and Stoica (*Chatbot Arena*, [ICML 2024](https://proceedings.mlr.press/v235/chiang24b.html); [arXiv:2403.04132](https://arxiv.org/abs/2403.04132)) collect anonymous pairwise votes and estimate a Bradley–Terry strength $$\theta_i$$ for each model. If $$i$$ beats $$j$$ with probability

$$
P(i \succ j) = \frac{e^{\theta_i}}{e^{\theta_i}+e^{\theta_j}},
$$

then ranking is a statistical estimation problem, not a leaderboard screenshot. Confidence intervals and adaptive pair sampling in that paper are the same Monte Carlo / resampling instincts as the birthday-problem simulation: you do not derive $$P(i\succ j)$$ from a likelihood of “intelligence”; you *count comparisons* and quantify uncertainty.

## What cross-validation actually estimates

The Chapter 01 CV lesson treats the CV score as an estimate of test error for “the” fitted model. Bates, Hastie, and Tibshirani (*Journal of the American Statistical Association*, 2023; [PMC11412612](https://pmc.ncbi.nlm.nih.gov/articles/PMC11412612/)) prove that, already for ordinary least squares, $$K$$-fold CV estimates the *average* prediction error of models trained on other draws of the training set—not the error of the specific $$\hat{f}$$ you will ship. Fold-wise errors are correlated (each point is used for both training and testing), so a naïve standard error is too small and the usual interval undercovers. Their remedy is a **nested** CV used only to estimate that variance, which is exactly the “use the data twice, carefully” warning from the required resampling lesson. In sklearn this is `cross_val_score` inside an outer loop, or `GridSearchCV` nested in another splitter—not a single optimistic `best_score_`.

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score

X, y = make_regression(n_samples=200, n_features=20, noise=8.0, random_state=0)
inner = KFold(n_splits=5, shuffle=True, random_state=0)
outer = KFold(n_splits=5, shuffle=True, random_state=1)
search = GridSearchCV(Ridge(), {"alpha": np.logspace(-2, 2, 9)}, cv=inner)
# Outer CV estimates prediction error of the *selection procedure*, not of one alpha.
scores = cross_val_score(search, X, y, cv=outer)
print("nested CV mean:", scores.mean(), "sd:", scores.std(ddof=1))
```

That snippet does not claim a new theorem. It is the Bates–Hastie–Tibshirani distinction in code: the quantity you report is an average over re-fitted, re-tuned models.

## What to carry forward

When a paper says “we conformalize a black-box,” “we report Arena scores with intervals,” or “we nest the tuner,” it is speaking this chapter. Later lectures add random variables and tests; they do not replace simulation, resampling, or the habit of asking *which* error a criterion estimates.

## Sources

1. A. N. Angelopoulos and S. Bates, “Conformal Prediction: A Gentle Introduction,” *Foundations and Trends in Machine Learning* 16(4), 2023. [DOI](https://doi.org/10.1561/2200000101) · [arXiv:2107.07511](https://arxiv.org/abs/2107.07511)
2. T. Cordier et al., “Flexible and Systematic Uncertainty Estimation with Conformal Prediction via the MAPIE Library,” COPA 2023. [docs](https://mapie.readthedocs.io/)
3. P. Liang et al., “Holistic Evaluation of Language Models,” *TMLR*, 2023. [OpenReview](https://openreview.net/forum?id=iO4LZibEqW) · [arXiv:2211.09110](https://arxiv.org/abs/2211.09110)
4. W.-L. Chiang et al., “Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference,” ICML 2024. [PMLR](https://proceedings.mlr.press/v235/chiang24b.html) · [arXiv:2403.04132](https://arxiv.org/abs/2403.04132)
5. S. Bates, T. Hastie, and R. Tibshirani, “Cross-Validation: What Does It Estimate and How Well Does It Do It?,” *JASA*, 2023. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11412612/)
