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

This optional lesson does not re-teach OLS, ridge, lasso, or logistic regression. It takes those linear tools into the forms that industry and causal machine learning actually ship: generalized linear models and mixed models for non-Gaussian and clustered outcomes, regression adjustment (CUPED and Lin-style ANCOVA) for experiments, and double/debiased machine learning when the assignment mechanism is not a clean coin flip.

You should leave able to write a GLM log-likelihood, say when a random effect is required, implement CUPED as a control-variate adjustment, and sketch the residual-on-residual step that makes double ML regularisation-safe.

The required Chapter 07 lessons (simple regression, OLS, regularisation, logistic regression) and the Chapter 06 lesson on coefficient inference are the prerequisites. Matrix algebra at the level of $$(X^\top X)^{-1}$$ is enough.

---

## From a line to a GLM

OLS assumes

$$Y = X\beta + \varepsilon, \qquad E[\varepsilon \mid X] = 0.$$

A generalized linear model (McCullagh and Nelder, 1989) keeps the linear predictor $$\eta = X\beta$$ and changes two pieces: a link $$g(E[Y \mid X]) = \eta$$, and an exponential-family variance function. Poisson log-linear models for counts, Tweedie models for insurance claims with a point mass at zero, and the logistic model you already know are the same object. Logistic regression is the Bernoulli GLM with the logit link; its MLE is the cross-entropy minimiser from Chapter 05.

Industry still fits GLMs at scale because they are calibrated, cheap, and auditable. Recommendation and advertising stacks often keep a GLM (or a “wide” linear piece) next to a deep net: Cheng et al. (2016) named that pattern Wide & Deep. The linear piece memorises sparse IDs; the deep piece generalises. The statistical point is that a GLM is not a downgrade from a network. It is the likelihood that the network is approximating, plus an interpretable coefficient table when a regulator or a pricing actuary asks for one.

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

A/B logs and longitudinal product data violate independence. The same user contributes many sessions; the same query appears on many days. A linear mixed model writes

$$Y_{ij} = x_{ij}^\top \beta + u_i + \varepsilon_{ij}, \qquad u_i \sim \mathcal{N}(0, \sigma_u^2),$$

so the user effect $$u_i$$ is a random intercept. Generalized linear mixed models (GLMMs) put that random effect inside a GLM. Bates, Mächler, Bolker, and Walker (2015) made the computational stack (`lme4`) that most applied fields still call. Yu, Grieco, Holmes, Xu and colleagues (2022) argue, in a neuroscience primer that transfers verbatim to sessionised product data, that treating repeated measures as independent OLS rows is the default statistical error; a mixed model is the repair.

For an experimentation platform the practical rule is: if treatment is assigned at the user level, analyse at the user level (one aggregated outcome per user) *or* fit a mixed model with a user random effect. Analysing raw events with OLS understates the standard error and is a Chapter 06 validity failure dressed as a Chapter 07 regression.

## Regression adjustment and CUPED

Randomisation already identifies the average treatment effect (ATE). Regression is used to *reduce variance*, not to fix confounding. Deng, Xu, Kohavi, and Walker (2013) introduced CUPED (Controlled-experiment Using Pre-Experiment Data): replace the outcome $$Y$$ by

$$Y^{\mathrm{cv}} = Y - \theta X, \qquad \theta = \frac{\mathrm{Cov}(Y,X)}{\mathrm{Var}(X)},$$

where $$X$$ is a pre-experiment covariate, typically the same metric last week. The ATE on $$Y^{\mathrm{cv}}$$ equals the ATE on $$Y$$, and

$$\mathrm{Var}(Y^{\mathrm{cv}}) = \mathrm{Var}(Y)\,(1-\rho_{YX}^2).$$

A correlation of 0.7 halves the variance. That is a control variate from Monte Carlo, not a new causal trick.

Lin (2013) made the fully interacted OLS recommendation precise: include treatment, covariates, *and* treatment-by-covariate interactions, and use a heteroscedastic (sandwich) standard error. Reluga, Ye, and Zhao (2022) give a unified comparison of the ANOVA (difference-in-means), ANCOVA, and fully interacted (“ANHECOVA”) estimators and recover the fact that ANCOVA does *not* uniformly dominate the raw mean difference, while the interacted estimator does, asymptotically, under complete randomisation.

Deng and coauthors (2023) revisit CUPED ten years on and argue for an *augmentation* view that covers ratio metrics and, with care, in-experiment covariates. The 2022–2024 follow-ups (CUPAC, ML-adjusted CUPED, cross-fit nonlinear adjustment) are the same Chapter 07 idea with a more flexible regression for $$E[Y \mid X]$$.

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

Centering $$X$$ keeps the intercept readable; it does not change the ATE. In a randomised experiment you do **not** need double ML. Ordinary CUPED or Lin regression is enough, and it is what most platforms run.

## Double machine learning when assignment is observational

If treatment $$T$$ is not randomised, the regression

$$Y = \alpha + \tau T + X^\top \beta + \varepsilon$$

estimates a causal $$\tau$$ only under unconfoundedness and correct specification of $$E[Y \mid T,X]$$. Regularised ML used naively as a plug-in for the nuisance functions $$E[Y \mid X]$$ and $$E[T \mid X]$$ leaves a regularisation bias that does not vanish at $$n^{-1/2}$$.

Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey, and Robins (2018) fix that with *double/debiased ML*: estimate the two nuisances with cross-fitting, form residuals

$$\tilde{Y} = Y - \hat{E}[Y \mid X], \qquad \tilde{T} = T - \hat{E}[T \mid X],$$

and regress $$\tilde{Y}$$ on $$\tilde{T}$$. Neyman orthogonality makes the target moment insensitive to first-order nuisance errors; cross-fitting kills overfitting bias. Microsoft’s EconML library is the production implementation that applied teams actually call.

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

Athey and Imbens (2019) survey why economists—and, by the same math, data scientists in marketplaces—need this toolkit. Causal forests (Wager and Athey, 2018) extend the idea from a scalar ATE to heterogeneous effects. The regression chapter is the right home: the final step is still a regression, on residuals.

## What to use when

| Setting | Tool | Why |
|---|---|---|
| Randomised experiment, Gaussian metric | Difference in means or CUPED / Lin | Identified by design; regression only buys variance |
| Randomised experiment, counts or binaries | GLM or transformation, then CUPED on the GLM residual | Match the likelihood to the outcome |
| Repeated measures / nested users | Aggregate to the assignment unit, or GLMM | Independence is false at the event level |
| Observational treatment, many covariates | Double ML or a causal forest | Regularisation bias otherwise |

## Challenges and extensions

CUPED with a *post*-treatment covariate can subtract part of the effect; the 2013 paper is explicit that $$X$$ must be pre-experiment. Mixed models with a small number of clusters give anti-conservative standard errors. Double ML assumes no unmeasured confounding, which no library can test away. GLMs are miscalibrated under omitted interactions; a Tweedie model that ignores a seasonal offset will look precise and be wrong.

Extensions include targeted learning, policy learning from logged bandits, and the combination of CUPED with sequential tests from Chapter 06.

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
