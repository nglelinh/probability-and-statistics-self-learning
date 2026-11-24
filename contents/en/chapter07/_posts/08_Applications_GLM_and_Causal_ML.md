---
layout: post
title: "07-08-00 Optional: GLMs, Mixed Models, and Causal Regression"
chapter: "07"
order: 8
owner: nglelinh
lang: en
categories:
- chapter07
lesson_type: optional
---

Run an ordinary A/B test on weekly user revenue, and you immediately hit a wall: a few “whale” shoppers spend $5,000 while most users spend $0, so your standard error is enormous—meaning you would need to wait eight weeks to detect a real 1% lift! Yet you already know how much each user spent *last* week, before the experiment even started. Subtracting out that predictable pre-experiment baseline cuts your variance in half—letting you finish the exact same experiment in **four weeks instead of eight**, for free!

This optional lesson takes Chapter 07’s linear regression tools straight into the four workhorses that modern industry and causal machine learning actually ship: **Generalized Linear Models (GLMs)** for non-Gaussian outcomes, **Mixed-Effects Models** for repeated user sessions, **CUPED / Lin regression adjustment** for randomized experiments, and **Double/Debiased Machine Learning (DML)** when treatment is observational rather than randomized.

---

## Background Knowledge Map: Four Extensions of Linear Regression

Every tool in this lesson relaxes one specific assumption of classical OLS $$Y = X\beta + \varepsilon$$.

### 1. Generalized Linear Models (GLMs) — Relaxing Gaussian Noise
Classical OLS lets predictions go negative and assumes constant variance. **GLMs** keep the clean linear predictor $$\eta = X\beta$$ while plugging in a **link function** $$g(\mathbb{E}[Y \mid X]) = \eta$$ and an **exponential-family likelihood** (Bernoulli/logit, Poisson/log, Tweedie).
- **Core questions:** How do we get calibrated, auditable models for counts, probabilities, and zero-inflated insurance claims?
- **Real-world reach:** Wide & Deep ad click models, actuarial insurance pricing, and epidemiology rate modeling.

### 2. Linear Mixed-Effects Models (LMM / GLMM) — Relaxing i.i.d. Rows
When the same user logs 30 sessions, treating all 30 rows as independent OLS observations artificially inflates your sample size and shrinks standard errors—manufacturing fake significance. **Mixed models** add a **random intercept** $$u_i \sim \mathcal{N}(0, \sigma_u^2)$$ per user/cluster to capture within-user correlation honestly.
- **Core questions:** How much variance lives *between* users versus *within* users?
- **Real-world reach:** Longitudinal clinical trials, neuroscience recordings, and session-level product logs.

### 3. Variance Reduction (CUPED) & Causal Machine Learning (Double ML)
In **randomized** experiments, regression isn’t needed for causal identification—it is used purely to crush noise variance (**CUPED**, Lin’s interacted OLS). In **observational** data (no coin flip), naive regularized ML suffers from **regularization bias**; **Double/Debiased ML** uses **cross-fitting** and **residual-on-residual regression** (Neyman orthogonality) to recover valid causal effects $$\tau$$.
- **Core questions:** When is a covariate safe to adjust for (pre-treatment vs. post-treatment)? How do we use tree/neural models for causal inference without regularization bias?
- **Real-world reach:** Experimentation platforms (Microsoft/Netflix/Airbnb), pricing elasticity, and marketplace policy evaluation (`EconML`).

---

## From a line to a GLM

OLS assumes $$Y = X\beta + \varepsilon$$ with $$\mathbb{E}[\varepsilon \mid X] = 0$$. A **Generalized Linear Model** (McCullagh & Nelder, 1989) keeps $$\eta = X\beta$$ and swaps in a link $$g(\mathbb{E}[Y \mid X]) = \eta$$ plus an exponential-family variance function. Logistic regression (Bernoulli + logit), Poisson log-linear models for counts, and Tweedie models for zero-heavy claims are all the same unified object.

Industry still fits GLMs at scale—often paired beside a neural net as **Wide & Deep** (Cheng et al., 2016)—because GLMs are calibrated, fast, and auditable when regulators or actuaries ask for coefficient tables.

```python
import numpy as np
from scipy.optimize import minimize

def poisson_glm_mle(X, y):
    """
    MLE for a Poisson log-link GLM without an offset.

    Args:
        X: (n, p) design, include a column of ones if you want an intercept
        y: non-negative counts

    Returns:
        beta_hat
    """
    def nll(beta):
        eta = X @ beta
        # y*eta - exp(eta) is the Poisson log-likelihood up to a constant
        return np.mean(np.exp(eta) - y * eta)

    start = np.zeros(X.shape[1])
    out = minimize(nll, start, method="L-BFGS-B")
    return out.x
```

## Mixed models when users are not i.i.d.

A/B logs and longitudinal data violate independence: the same user contributes many sessions. A **linear mixed model** writes

$$Y_{ij} = x_{ij}^\top \beta + u_i + \varepsilon_{ij}, \qquad u_i \sim \mathcal{N}(0, \sigma_u^2),$$

giving each user $$i$$ a random baseline $$u_i$$ (implemented in `lme4` by Bates et al., 2015; see also Yu et al., 2022). Practical rule: if treatment is assigned per user, either aggregate to **one row per user** or fit a **mixed model** with a user random intercept. Running raw OLS on event-level rows understates standard errors and creates false wins.

## Regression adjustment and CUPED

Randomization already identifies the average treatment effect (ATE). Regression here serves purely to **reduce variance**. Deng, Xu, Kohavi, and Walker (2013) introduced **CUPED** (Controlled-experiment Using Pre-Experiment Data): replace $$Y$$ by

$$Y^{\mathrm{cv}} = Y - \theta (X - \bar{X}), \qquad \theta = \frac{\mathrm{Cov}(Y,X)}{\mathrm{Var}(X)},$$

where $$X$$ is a **pre-experiment** covariate (such as last week’s spend). Because $$X$$ is independent of randomized treatment $$T$$, $$\mathbb{E}[Y^{\mathrm{cv}} \mid T=1] - \mathbb{E}[Y^{\mathrm{cv}} \mid T=0] = \text{ATE}$$ is completely unbiased, while variance shrinks by

$$\mathrm{Var}(Y^{\mathrm{cv}}) = \mathrm{Var}(Y)\,(1-\rho_{YX}^2).$$

Lin (2013) and Reluga, Ye, and Zhao (2022) further showed that including treatment-by-covariate interactions with sandwich standard errors asymptotically guarantees variance is *never* worse than raw difference-in-means, and Deng et al. (2023) extended CUPED to ratio metrics and ML-predicted baselines (CUPAC).

---

### Worked Example 1: CUPED Variance Reduction & Sample-Size Savings by Hand

Suppose your weekly user spend $$Y$$ has variance $$\operatorname{Var}(Y) = 100$$, and last week’s pre-experiment spend $$X$$ correlates with $$Y$$ at $$\rho_{YX} = 0.70$$ (with $$\operatorname{Var}(X) = 100$$, so $$\operatorname{Cov}(Y,X) = 70$$):

1. **Optimal CUPED coefficient:** $$\theta = \dfrac{\operatorname{Cov}(Y,X)}{\operatorname{Var}(X)} = \dfrac{70}{100} = \mathbf{0.70}$$.
2. **Adjusted variance:**
   $$\operatorname{Var}(Y^{\mathrm{cv}}) = 100\,(1 - 0.70^2) = 100\,(1 - 0.49) = \mathbf{51.0}!$$
3. **Practical impact on experiment speed:**
   Because required sample size scales linearly with variance ($$n \propto \sigma^2$$), cutting variance by **49%** cuts required users—and experiment wait time—nearly in **half**! If $$\rho = 0.80$$, variance drops by $$64\%$$ (nearly $$3\times$$ faster experiments) without changing a single line of randomization!

---

```python
import numpy as np

def cuped_ate(y, t, x):
    """
    Difference-in-means on a CUPED-adjusted outcome.

    Args:
        y: outcome
        t: 0/1 treatment
        x: pre-experiment covariate (same length)

    Returns:
        (ate_raw, ate_cuped, theta, var_reduction)
    """
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    y_cv = y - theta * (x - x.mean())
    ate_raw = y[t == 1].mean() - y[t == 0].mean()
    ate_cv = y_cv[t == 1].mean() - y_cv[t == 0].mean()
    reduction = 1 - np.var(y_cv, ddof=1) / np.var(y, ddof=1)
    return ate_raw, ate_cv, theta, reduction
```

Centering $$X$$ keeps the intercept readable without changing the ATE. In a randomized experiment you do **not** need Double ML—CUPED or Lin regression is already optimal and clean.

## Double machine learning when assignment is observational

When treatment $$T$$ is **not** randomized, we observe $$Y = \tau T + g(X) + \varepsilon$$ where confounder $$X$$ drives *both* $$T$$ and $$Y$$. Plugging a regularized ML model (Lasso, Gradient Boosting) directly into $$Y \sim \tau T + \hat{g}(X)$$ leaks shrinkage bias into $$\hat{\tau}$$.

Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey, and Robins (2018) solved this with **Double/Debiased ML (DML)**:
1. Predict $$Y$$ from $$X$$ and $$T$$ from $$X$$ out-of-fold (**cross-fitting**).
2. Form the two residualized variables:
   $$\tilde{Y} = Y - \hat{\mathbb{E}}[Y \mid X], \qquad \tilde{T} = T - \hat{\mathbb{E}}[T \mid X].$$
3. Regress $$\tilde{Y}$$ on $$\tilde{T}$$: $$\hat{\tau} = \dfrac{\sum \tilde{T}_i \tilde{Y}_i}{\sum \tilde{T}_i^2}$$.

---

### Worked Example 2: Partialing Out a Confounder (Frisch–Waugh / DML) by Hand

Suppose true causal effect is **$$\tau = 2.0$$**, but a confounder $$X \in \{0, 1\}$$ (e.g., “high-income user”) raises both treatment probability and baseline spending: $$Y = 2T + 10X$$. You observe 4 users:

| User | Confounder $$X$$ | Treatment $$T$$ | Outcome $$Y$$ | $$\hat{\mathbb{E}}[T \mid X]$$ | $$\tilde{T} = T - \hat{\mathbb{E}}[T \mid X]$$ | $$\hat{\mathbb{E}}[Y \mid X]$$ | $$\tilde{Y} = Y - \hat{\mathbb{E}}[Y \mid X]$$ |
|------|------------------|-----------------|---------------|--------------------------------|------------------------------------------------|--------------------------------|------------------------------------------------|
| 1 | $$0$$ | $$0$$ | $$0$$ | $$0.5$$ | $$-0.5$$ | $$1$$ | $$-1$$ |
| 2 | $$0$$ | $$1$$ | $$2$$ | $$0.5$$ | $$+0.5$$ | $$1$$ | $$+1$$ |
| 3 | $$1$$ | $$0$$ | $$10$$ | $$0.5$$ | $$-0.5$$ | $$11$$ | $$-1$$ |
| 4 | $$1$$ | $$1$$ | $$12$$ | $$0.5$$ | $$+0.5$$ | $$11$$ | $$+1$$ |

Now compute the residual-on-residual slope:
$$\hat{\tau}_{\text{DML}} = \frac{(-0.5)(-1) + (0.5)(1) + (-0.5)(-1) + (0.5)(1)}{(-0.5)^2 + 0.5^2 + (-0.5)^2 + 0.5^2} = \frac{2.0}{1.0} = \mathbf{2.0}!$$
Subtracting $$\hat{\mathbb{E}}[\,\cdot \mid X]$$ from *both* $$Y$$ and $$T$$ cleanly strips out the $$+10X$$ confounder! In high dimensions, **Neyman orthogonality** ensures that small estimation errors in $$\hat{m}_Y(X)$$ and $$\hat{m}_T(X)$$ multiply together ($$O(n^{-1/4}) \times O(n^{-1/4}) = O(n^{-1/2})$$), vanishing fast enough for valid confidence intervals!

---

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

def dml_ate(y, t, X, n_splits=5, random_state=0):
    """
    Cross-fit partial-linear DML for a scalar ATE.
    """
    y, t = np.asarray(y, float), np.asarray(t, float)
    n = len(y)
    y_res = np.zeros(n)
    t_res = np.zeros(n)
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train, test in folds.split(X):
        my = GradientBoostingRegressor(random_state=random_state)
        mt = GradientBoostingRegressor(random_state=random_state)
        my.fit(X[train], y[train])
        mt.fit(X[train], t[train])
        y_res[test] = y[test] - my.predict(X[test])
        t_res[test] = t[test] - mt.predict(X[test])
    tau = np.dot(t_res, y_res) / np.dot(t_res, t_res)
    return tau
```

Athey and Imbens (2019) and Wager and Athey (2018) (*Causal Forests*) extend the same residualization philosophy to heterogeneous treatment effects $$\tau(X)$$.

---

### Deep-Dive Explanation: Why Double ML Needs BOTH Residualization AND Cross-Fitting (Read Slowly)

Why couldn’t you just run Gradient Boosting for $$Y \sim T + X$$ and read off the treatment effect? Walk through these four steps slowly.

#### Step 1: Naive ML shrinkage steals credit from treatment $$T$$
Suppose $$X$$ is correlated with $$T$$. A regularized ML model shrinks $$X$$’s effect slightly toward zero to reduce variance—so the leftover confounding from $$X$$ spills directly into $$\hat{\tau}$$, creating a bias of order $$O(n^{-1/4})$$ that ruins confidence intervals!

#### Step 2: Residualizing BOTH $$Y$$ and $$T$$ creates Neyman orthogonality
By regressing $$\tilde{Y} = Y - \hat{m}_Y(X)$$ on $$\tilde{T} = T - \hat{m}_T(X)$$, first-order errors in $$\hat{m}_Y$$ and $$\hat{m}_T$$ cancel out! Only their **product** remains: if each ML model converges at rate $$n^{-1/4}$$, their product converges at $$n^{-1/2}$$—restoring exact asymptotic normality.

#### Step 3: Why cross-fitting is mandatory
If you compute $$\tilde{Y}$$ and $$\tilde{T}$$ on the *same* data used to train the Gradient Boosting trees, overfitting correlates the residuals and re-introduces bias! Training on Fold A and computing residuals on Fold B (**cross-fitting**) breaks that correlation completely.

#### Step 4: Remember the one assumption ML can never fix
Double ML eliminates *functional-form* and *regularization* bias—it **cannot** fix an **unmeasured confounder** missing from $$X$$! No algorithm replaces domain knowledge or true randomization.

---

## What to use when

| Setting | Tool | Why |
|---|---|---|
| Randomized experiment, Gaussian metric | Difference in means or CUPED / Lin | Identified by design; regression only cuts variance |
| Randomized experiment, counts or binaries | GLM or transformation, then CUPED on residual | Match likelihood to outcome scale |
| Repeated measures / nested users | Aggregate to assignment unit, or GLMM | Independence is false at the event level |
| Observational treatment, many covariates | Double ML or a causal forest | Removes regularization bias via orthogonality |

## Challenges and extensions

CUPED with a *post*-treatment covariate can subtract part of the effect; the 2013 paper is explicit that $$X$$ must be pre-experiment. Mixed models with a small number of clusters give anti-conservative standard errors. Double ML assumes no unmeasured confounding, which no library can test away. GLMs are miscalibrated under omitted interactions; a Tweedie model that ignores a seasonal offset will look precise and be wrong.

Extensions include targeted learning, policy learning from logged bandits, and the combination of CUPED with sequential tests from Chapter 06.

## Video tham khảo

- [Explaining generalized linear models (GLMs) | VNT #15](https://www.youtube.com/watch?v=1fZeBafD1nw) — Very Normal, English. **xem trước.** A GLM keeps a linear predictor and changes the link and the variance family — logistic and Poisson are the two cases this note names.
- [Simple Explanation of Mixed Models (Hierarchical Linear Models, Multilevel Models)](https://www.youtube.com/watch?v=5tOifM51ZOk) — Simplistics (QuantPsych), English. A random intercept when repeated observations on one user are not independent — the mixed-model section.
- [Double Machine Learning for Causal and Treatment Effects](https://www.youtube.com/watch?v=eHOjmyoPCFU) — Becker Friedman Institute University of Chicago, English. Chernozhukov on double/debiased machine learning: orthogonal scores and cross-fitting — the causal section’s procedure.

## Exercises

**Exercise 1 (conceptual).** A team regresses weekly revenue on a treatment dummy and twenty post-click covariates. They quote the treatment coefficient as “the causal lift.” Which assumption of CUPED/Lin have they broken, and what estimand might they have estimated instead?

**Exercise 2 (computational).** Simulate a randomised experiment with $$Y = 2 + 0.05 T + 0.8 X + \varepsilon$$, $$X \sim N(0,1)$$ observed before assignment. Compare the RMSE of the raw ATE and of `cuped_ate` over 500 replications. Then replace $$X$$ by a post-treatment variable $$X_{\mathrm{post}} = X + 0.5 T$$. What happens to the CUPED point estimate?

**Exercise 3 (computational).** Using `dml_ate`, generate observational data where $$T$$ depends on $$X$$ and $$Y$$ depends on both. Compare OLS of $$Y$$ on $$(T,X)$$, OLS of $$Y$$ on $$T$$ only, and DML. Repeat with a misspecified $$X$$ (drop a confounder). Which method fails silently?

**Exercise 4 (open).** Fit a Poisson or Tweedie GLM to a public count or insurance data set. Compare predicted means and a residual plot against a linear model on the raw counts. Then add a grouping factor (region, user, or policy year) and discuss whether a mixed model is warranted.

## References

Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015). Fitting linear mixed-effects models using lme4. *Journal of Statistical Software, 67*(1), 1–48. [https://doi.org/10.18637/jss.v067.i01](https://doi.org/10.18637/jss.v067.i01)

Cheng, H.-T., Koc, L., Harmsen, J., Shaked, T., Chandra, T., Aradhye, H., … Shah, H. (2016). Wide & Deep learning for recommender systems. In *Proceedings of the 1st Workshop on Deep Learning for Recommender Systems* (pp. 7–10). [https://doi.org/10.1145/2988450.2988454](https://doi.org/10.1145/2988450.2988454)

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal, 21*(1), C1–C68. [https://doi.org/10.1111/ectj.12097](https://doi.org/10.1111/ectj.12097)

Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). Improving the sensitivity of online controlled experiments by utilizing pre-experiment data. In *WSDM 2013* (pp. 123–132). [https://doi.org/10.1145/2433396.2433413](https://doi.org/10.1145/2433396.2433413)

Deng, A., et al. (2023). From augmentation to decomposition: A new look at CUPED in 2023. [https://arxiv.org/abs/2312.02935](https://arxiv.org/abs/2312.02935)

Lin, W. (2013). Agnostic notes on regression adjustments to experimental data: Reexamining Freedman’s critique. *The Annals of Applied Statistics, 7*(1), 295–318.

McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall. The still-cited GLM monograph.

Reluga, K., Ye, T., & Zhao, Q. (2022). A unified analysis of regression adjustment in randomized experiments. [https://arxiv.org/abs/2210.04360](https://arxiv.org/abs/2210.04360)

Yu, Z., Grieco, S. F., Holmes, T. C., Xu, X., et al. (2022). Beyond t-test and ANOVA: Applications of mixed-effects models for more rigorous statistical analysis in neuroscience research. *Neuron, 110*(1), 21–35. [https://doi.org/10.1016/j.neuron.2021.10.030](https://doi.org/10.1016/j.neuron.2021.10.030)
