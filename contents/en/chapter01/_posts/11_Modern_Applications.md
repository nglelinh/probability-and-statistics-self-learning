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

It is tempting to treat Monte Carlo, cross-validation, and model selection as warm-up exercises you leave behind once “real” deep learning begins. In reality, they have become the exact working language of frontier AI evaluation and uncertainty quantification. When you deploy a black-box neural network in a hospital, rank large language models on Chatbot Arena, or report cross-validation error in a top journal, there is no neat closed-form Gaussian formula to save you—only **resampling**, **empirical quantiles**, and **paired-comparison estimation**.

This optional note connects Chapter 01’s core toolkit to three major 2022–2024 breakthroughs: **split conformal prediction**, **Monte Carlo / Bradley–Terry evaluation of LLMs**, and **what cross-validation actually estimates**.

---

## Background Knowledge Map: How Chapter 01 Tools Power Modern AI Evaluation

Three statistical themes tie these recent papers directly back to our opening lessons.

### 1. Distribution-Free Uncertainty (Conformal Prediction)
Classical confidence intervals assume Gaussian errors or large-sample asymptotics. **Conformal prediction** uses a held-out calibration split and order statistics to wrap *any* black-box predictor (tree, neural net, LLM) in a prediction set $$C(x)$$ with a finite-sample coverage guarantee—relying only on **exchangeability**.
- **Core questions:** How can we guarantee $$P(Y_{n+1} \in C(X_{n+1})) \ge 1-\alpha$$ without assuming the model is correctly specified?
- **Real-world reach:** Clinical decision support, autonomous-driving perception bounds, and financial forecasting intervals.

### 2. Paired-Comparison Modeling & Multi-Scenario Simulation (LLM Benchmarks)
Evaluating open-ended language models has no closed-form accuracy. **HELM** treats evaluation as a multi-scenario Monte Carlo design across seven metrics, while **Chatbot Arena** uses the **Bradley–Terry** logistic model to turn anonymous pairwise human votes into calibrated strength ratings with bootstrap confidence intervals.
- **Core questions:** How do we separate genuine model improvement from benchmark sampling noise?
- **Real-world reach:** Public LLM leaderboards (LMSYS Chatbot Arena), A/B ranking of search/recommendation systems, and sports rating engines (Elo).

### 3. Honest Resampling Inference (Nested Cross-Validation)
Standard $$K$$-fold CV reuses every point for both training and validation, creating subtle correlations across folds. Recent theoretical work (Bates, Hastie, Tibshirani, *JASA* 2023) clarified *which* error CV estimates and why **nested CV** is required for valid confidence intervals around generalization error.
- **Core questions:** Does CV estimate the error of *my specific fitted model* or the *average error of the fitting algorithm*?
- **Real-world reach:** High-stakes benchmark reporting, AutoML model selection, and reproducible ML papers.

---

## Conformal prediction as a hold-out Monte Carlo

A classifier or regressor $$f$$ typically returns a single point $$\hat{y}$$. High-stakes settings (diagnostics, ranking, content moderation) need a *set* $$C(x)$$ that covers the true label with a user-chosen probability $$1 - \alpha$$. **Split conformal prediction** does this with the exact same computational idea as a validation split: fit $$\hat{f}$$ on a training fold; on a disjoint calibration fold of size $$n$$ compute **nonconformity scores** $$s_i = s(x_i, y_i)$$ (for example, absolute residuals $$\lvert y_i - \hat{f}(x_i) \rvert$$); then take the finite-sample corrected empirical quantile

$$
\hat{q} = \text{Quantile}_{\frac{\lceil (n+1)(1-\alpha) \rceil}{n}}\bigl(s_1,\ldots,s_n\bigr).
$$

The prediction set $$C(x) = \{y : s(x,y) \le \hat{q}\}$$ then satisfies, under **exchangeability** alone,

$$
P\bigl(Y_{n+1} \in C(X_{n+1})\bigr) \ge 1 - \alpha.
$$

No Gaussian residual, no correctly specified model—only the exchangeability that also justifies ordinary cross-validation. Angelopoulos and Bates’s monograph (*Foundations and Trends in Machine Learning*, 2023; [arXiv:2107.07511](https://arxiv.org/abs/2107.07511)) is the formulation most 2023–2026 software cites, and the scikit-learn-contrib library [MAPIE](https://mapie.readthedocs.io/) (Cordier, Blot, Lacombe, Morzadec, Capitaine, Brunel, COPA 2023) wraps the same quantile around any estimator.

---

### Worked Example 1: Building a 90% Conformal Prediction Interval by Hand

Suppose your regression model predicts house prices on a tiny calibration fold of $$n = 9$$ unseen houses, producing sorted absolute residuals $$s_i = \lvert y_i - \hat{y}_i \rvert$$ (in $10k):

$$0.1,\; 0.2,\; 0.3,\; 0.4,\; 0.5,\; 0.6,\; 0.8,\; 1.1,\; 1.5.$$

You want a **90% coverage guarantee** ($$\alpha = 0.10$$) for a brand-new house $$X_{10}$$ whose point prediction is $$\hat{y}_{10} = 45.0$$:
1. **Compute the conformal rank index:**
   $$k = \lceil (n + 1)(1 - \alpha) \rceil = \lceil (9 + 1)(0.90) \rceil = \lceil 9.0 \rceil = \mathbf{9}.$$
2. **Read the 9th sorted residual:** $$\hat{q} = s_{(9)} = \mathbf{1.5}$$.
3. **Output the prediction interval:**
   $$C(X_{10}) = [\hat{y}_{10} - \hat{q},\; \hat{y}_{10} + \hat{q}] = [45.0 - 1.5,\; 45.0 + 1.5] = \mathbf{[43.5,\; 46.5]}.$$

**Why does this guarantee $$\ge 90\%$$ coverage?** Because all 10 residuals $$(s_1,\ldots,s_9, s_{10})$$ are exchangeable, the new residual $$s_{10}$$ is equally likely to land in any of the 10 rank positions! The chance it lands in positions 1 through 9 (i.e., $$s_{10} \le s_{(9)}$$) is **at least $$9/10 = 90\%$$**—even if your regression model is nonlinear, biased, or completely non-Gaussian!

---

## Simulation for LLM evaluation

There is no closed-form accuracy for a chat model. Liang, Bommasani, Lee, and colleagues (*Holistic Evaluation of Language Models*, [TMLR 2023](https://openreview.net/forum?id=iO4LZibEqW); [arXiv:2211.09110](https://arxiv.org/abs/2211.09110); toolkit [stanford-crfm/helm](https://github.com/stanford-crfm/helm)) treat evaluation as a designed experiment: 16 core scenarios, seven metrics (accuracy, **calibration**, robustness, fairness, bias, toxicity, efficiency), and 30 models run under a shared protocol. The Law of Large Numbers from our opening lesson is what makes a multi-scenario average meaningful; a single headline number is a noisy one-draw Monte Carlo.

Human preference is even less analytic. Chiang, Zheng, Sheng, Angelopoulos, Li, Li, Zhu, Zhang, Jordan, Gonzalez, and Stoica (*Chatbot Arena*, [ICML 2024](https://proceedings.mlr.press/v235/chiang24b.html); [arXiv:2403.04132](https://arxiv.org/abs/2403.04132)) collect anonymous pairwise votes and estimate a **Bradley–Terry** strength $$\theta_i$$ for each model. If model $$i$$ beats model $$j$$ with probability

$$
P(i \succ j) = \frac{e^{\theta_i}}{e^{\theta_i}+e^{\theta_j}} = \frac{1}{1 + e^{-(\theta_i - \theta_j)}},
$$

then leaderboard ranking is a statistical estimation problem, not a fixed trophy table. Bootstrap confidence intervals and adaptive pair sampling turn raw vote counts into honest uncertainty bars.

---

### Worked Example 2: Bradley–Terry Win Probabilities & Arena Elo Ratings by Hand

Suppose Chatbot Arena estimates latent strengths $$\theta_A = 1.0$$ and $$\theta_B = 0.0$$ for two LLMs:
1. **Predicted win probability of Model A over Model B:**
   $$P(A \succ B) = \frac{e^{1.0}}{e^{1.0} + e^{0.0}} = \frac{2.718}{3.718} \approx \mathbf{73.1\%}.$$
2. **Converting to the Chess/Arena Elo scale** (where 400 Elo points correspond to a $$10{:}1$$ win odds ratio, so $$R_i = 1000 + \frac{400}{\ln 10}\theta_i \approx 1000 + 173.7\,\theta_i$$):
   A gap of $$\theta_A - \theta_B = 1.0$$ equals **$$+174$$ Elo points**! Conversely, a narrow leaderboard lead of just **10 Elo points** corresponds to $$\Delta\theta = 10/173.7 \approx 0.0576$$, or a win probability of only $$\frac{1}{1+e^{-0.0576}} \approx \mathbf{51.4\%}$$—explaining why small Elo gaps with overlapping bootstrap intervals are statistically indistinguishable!

---

## What cross-validation actually estimates

The Chapter 01 CV lesson treats the CV score as an estimate of test error for “the” fitted model. Bates, Hastie, and Tibshirani (*Journal of the American Statistical Association*, 2023; [PMC11412612](https://pmc.ncbi.nlm.nih.gov/articles/PMC11412612/)) prove that, already for ordinary least squares, $$K$$-fold CV estimates the *average* prediction error of models trained on random draws of the training set—not the error of the specific $$\hat{f}$$ you will ship. Fold-wise errors are correlated (each point is used for both training and testing), so a naïve standard error is too small and the usual confidence interval undercovers. Their remedy is a **nested** CV used specifically to estimate that variance cleanly.

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

---

### Deep-Dive Explanation: Why Exchangeability Unifies Conformal, Arena, and Nested CV (Read Slowly)

These three 2022–2024 breakthroughs look completely different on the surface, yet they obey the exact same four-step statistical discipline.

#### Step 1: Never trust a single point estimate without a distribution or interval
A point prediction $$\hat{y}$$ hides how uncertain the model is; a single Arena Elo rank hides how few head-to-head votes separate #2 from #3; a single `GridSearchCV.best_score_` hides selection optimism.

#### Step 2: Replace fragile parametric assumptions with resampling & order statistics
Instead of assuming residuals are Gaussian (which fails on multimodal or heavy-tailed AI outputs), split conformal prediction relies solely on **exchangeability** of calibration and test points—and Chatbot Arena uses nonparametric **bootstrap** resampling over vote pairs.

#### Step 3: Guard the boundary between fitting and evaluating
Split conformal requires the calibration fold to be completely untouched during training; nested CV requires the outer test fold to be untouched during hyperparameter tuning. Cross that boundary even once, and your mathematical guarantee evaporates.

#### Step 4: Remember what breaks when the world drifts
Every guarantee above assumes tomorrow’s test data come from the *same* distribution as today’s calibration/CV data (exchangeability). Under **distribution shift** (new user demographics, new prompt styles, market regime change), even conformal coverage can drop—which is why adaptive conformal inference and continuous monitoring are active frontiers today.

---

## What to carry forward

When a paper says “we conformalize a black-box,” “we report Arena scores with intervals,” or “we nest the tuner,” it is speaking Chapter 01. Later lectures add random variables and hypothesis tests; they do not replace simulation, resampling, or the habit of asking *which* error a criterion estimates.

## Video tham khảo

- [A Tutorial on Conformal Prediction](https://www.youtube.com/watch?v=nql000Lu_iE) — Anastasios Nikolas Angelopoulos, English. **xem trước.** Angelopoulos teaches split conformal prediction: nonconformity scores on a calibration set, then a quantile — the procedure this note ties to arXiv:2107.07511.
- [Holistic Evaluation of Language Models (HELM) - Yifan Mai, Stanford University](https://www.youtube.com/watch?v=fLoYcXLG6ro) — PyTorch, English. Yifan Mai presents HELM, the multi-scenario evaluation this note reads as a Monte Carlo experiment for language models.
- [11.5 Nested CV for Algorithm Selection (L11 Model Eval. Part 4)](https://www.youtube.com/watch?v=XXFLFWHP9Nc) — Sebastian Raschka, English. Nested cross-validation for the whole model-selection procedure, the Bates–Hastie–Tibshirani warning in the note.

## Sources

1. A. N. Angelopoulos and S. Bates, “Conformal Prediction: A Gentle Introduction,” *Foundations and Trends in Machine Learning* 16(4), 2023. [DOI](https://doi.org/10.1561/2200000101) · [arXiv:2107.07511](https://arxiv.org/abs/2107.07511)
2. T. Cordier et al., “Flexible and Systematic Uncertainty Estimation with Conformal Prediction via the MAPIE Library,” COPA 2023. [docs](https://mapie.readthedocs.io/)
3. P. Liang et al., “Holistic Evaluation of Language Models,” *TMLR*, 2023. [OpenReview](https://openreview.net/forum?id=iO4LZibEqW) · [arXiv:2211.09110](https://arxiv.org/abs/2211.09110)
4. W.-L. Chiang et al., “Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference,” ICML 2024. [PMLR](https://proceedings.mlr.press/v235/chiang24b.html) · [arXiv:2403.04132](https://arxiv.org/abs/2403.04132)
5. S. Bates, T. Hastie, and R. Tibshirani, “Cross-Validation: What Does It Estimate and How Well Does It Do It?,” *JASA*, 2023. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11412612/)
