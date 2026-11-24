---
layout: post
title: 01-08-00 Model Selection Criteria
chapter: "01"
order: 8
owner: nglelinh
lang: en
categories:
- chapter01
lesson_type: required
---

Add a completely useless random column—say, the last digit of each customer’s phone number—to a linear regression, and watch what happens to ordinary $$R^2$$: it **never goes down**, and it almost always ticks slightly *up*! Every extra parameter hands the model one more degree of freedom to absorb random training noise. Left unchecked, raw likelihood or $$R^2$$ will always crown the bloated, overfitted monster.

Cross-validation fights this empirically by holding out data. **Information criteria**—led by **AIC**, **BIC**, **Adjusted $$R^2$$**, and **Mallows’ $$C_p$$**—fight it *analytically*: they compute the exact mathematical “tax” you must charge for each extra parameter so that model selection rests on a principled formula rather than guesswork.

---

## Background Knowledge Map: Where Do Information Criteria Come From?

Model selection criteria look like simple one-line formulas, yet they descend from three deep branches of statistical theory.

### 1. Information Theory & Kullback–Leibler Divergence (AIC)
In 1974, Hirotugu Akaike asked: *how close is our fitted probability distribution $$q_{\hat{\theta}}$$ to the unknown true data-generating distribution $$p$$?* Measuring closeness by **Kullback–Leibler (KL) divergence**, he proved that the maximized log-likelihood $$\log L(\hat{\theta})$$ is optimistically biased by approximately $$k$$ (the number of fitted parameters)—producing **AIC** $$= -2\log L(\hat{\theta}) + 2k$$.
- **Core questions:** Which candidate model will predict *new* observations most accurately (minimizing out-of-sample KL loss)?
- **Real-world reach:** Time-series ARIMA order selection, ecological modeling, and generalized linear model comparison.

### 2. Bayesian Model Comparison & Laplace Approximation (BIC)
In 1978, Gideon Schwarz approached model selection from a completely different philosophy: *which model has the highest **posterior probability** $$P(M_j \mid \text{data})$$?* Integrating over parameter space via Laplace’s approximation yields **BIC** $$= -2\log L(\hat{\theta}) + k\log n$$—charging a penalty that grows with sample size $$n$$.
- **Core questions:** If the true sparse model sits inside our candidate list, which criterion guarantees we recover it as $$n \to \infty$$ (*model-selection consistency*)?
- **Real-world reach:** Choosing the number of clusters in Gaussian Mixture Models (GMMs), phylogenetic trees, and structural econometrics.

### 3. Degrees of Freedom & Unbiased Risk Estimation (Adjusted $$R^2$$ & Mallows’ $$C_p$$)
In classical Gaussian linear regression, training residual sum of squares $$\text{RSS}_p$$ underestimates expected squared prediction error by $$2p\sigma^2$$. Adding back $$2p\sigma^2$$ produces **Mallows’ $$C_p$$**, while dividing sums of squares by their respective **degrees of freedom** ($$n - p - 1$$ vs. $$n - 1$$) produces **Adjusted $$R^2$$**.
- **Core questions:** When does an extra predictor genuinely reduce noise variance rather than merely consuming a degree of freedom?
- **Real-world reach:** Variable screening in linear regression and subset-selection diagnostics.

---

## The selection problem

**The basic tradeoff:**
- A rich model fits training data tightly, risking high variance (overfitting).
- A parsimonious model risks bias (underfitting), yet often generalizes more reliably.

Every selection criterion below has the same two-part anatomy:

$$\text{Criterion} = \underbrace{\text{Lack of Fit}}_{\text{smaller is better}} + \underbrace{\text{Complexity Penalty}}_{\text{grows with } k}.$$

## Akaike information criterion (AIC)

AIC estimates relative out-of-sample prediction error (up to a constant):

$$\text{AIC} = -2\log L(\hat{\theta}) + 2k$$

where $$L(\hat{\theta})$$ is the maximized likelihood and $$k$$ is the number of estimated parameters.

**Rule:** prefer the **smaller** AIC. For Gaussian linear regression with unknown variance, this simplifies (up to an additive constant independent of the model) to:

$$\text{AIC} = n\log\!\left(\frac{\text{RSS}}{n}\right) + 2k + \text{const}.$$

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

def calculate_aic(y_true, y_pred, n_params):
    """AIC for a Gaussian linear model, using the training MSE as σ²."""
    n = len(y_true)
    mse = mean_squared_error(y_true, y_pred)
    log_likelihood = -n / 2 * np.log(2 * np.pi * mse) - n / 2
    aic = -2 * log_likelihood + 2 * n_params
    return aic

def model_selection_with_aic():
    """Compare polynomial degrees by AIC and BIC."""
    np.random.seed(42)
    X = np.linspace(0, 1, 50).reshape(-1, 1)
    y = np.sin(2 * np.pi * X).ravel() + np.random.normal(0, 0.2, 50)

    degrees = range(1, 11)
    aics = []
    bics = []

    for degree in degrees:
        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)

        n_params = X_poly.shape[1]
        aic = calculate_aic(y, y_pred, n_params)

        n = len(y)
        log_likelihood = -n / 2 * np.log(2 * np.pi * mean_squared_error(y, y_pred)) - n / 2
        bic = -2 * log_likelihood + n_params * np.log(n)

        aics.append(aic)
        bics.append(bic)

    plt.figure(figsize=(10, 6))
    plt.plot(degrees, aics, 'o-', label='AIC', linewidth=2)
    plt.plot(degrees, bics, 's-', label='BIC', linewidth=2)
    plt.xlabel('Polynomial degree')
    plt.ylabel('Information criterion')
    plt.title('Model selection: AIC vs BIC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()

    print(f"Optimal degree (AIC): {degrees[np.argmin(aics)]}")
    print(f"Optimal degree (BIC): {degrees[np.argmin(bics)]}")

# model_selection_with_aic()
```

## Bayesian information criterion (BIC)

BIC replaces AIC’s constant $$2$$ per parameter with $$\log(n)$$:

$$\text{BIC} = -2\log L(\hat{\theta}) + k\log(n)$$

**AIC versus BIC:**
- Because $$\log(8) \approx 2.08 > 2$$, BIC penalizes complexity more heavily than AIC for every sample size $$n \ge 8$$.
- **AIC** is *asymptotically efficient* for **prediction** (and asymptotically equivalent to Leave-One-Out CV!).
- **BIC** is *consistent* for **model identification** (if the true data-generating model has $$k_0$$ parameters, BIC recovers $$k_0$$ with probability $$\to 1$$ as $$n \to \infty$$, whereas AIC has a small positive chance of keeping 1–2 extra noise variables).

---

### Worked Example 1: Watching AIC and BIC Disagree by Hand ($$n = 50$$)

Suppose you fit two nested Gaussian regression models to $$n = 50$$ data points:
- **Model A (simpler):** $$k_A = 3$$ parameters, residual sum of squares $$\text{RSS}_A = 120$$.
- **Model B (richer):** $$k_B = 5$$ parameters (2 extra predictors), $$\text{RSS}_B = 110$$.

Compute the exact difference $$\Delta = \text{Score}_B - \text{Score}_A$$ (negative means **Model B wins**; positive means **Model A wins**):

1. **Improvement in fit ($$-2\Delta\log L$$):**
   $$n\log\!\left(\frac{\text{RSS}_B}{\text{RSS}_A}\right) = 50 \cdot \log\!\left(\frac{110}{120}\right) = 50 \cdot (-0.08701) = -4.35.$$
   Adding 2 parameters improved deviance by $$4.35$$ points. Is $$4.35$$ enough to pay the complexity tax?
2. **AIC check** (tax $$= 2\Delta k = 2(5 - 3) = +4.00$$):
   $$\Delta\text{AIC} = -4.35 + 4.00 = \mathbf{-0.35 < 0} \implies \textbf{AIC picks Model B!}$$
3. **BIC check** (tax $$= \Delta k \cdot \log(50) = 2 \times 3.912 = +7.82$$):
   $$\Delta\text{BIC} = -4.35 + 7.82 = \mathbf{+3.47 > 0} \implies \textbf{BIC picks Model A!}$$

**Lesson:** The exact same data produce opposite recommendations—not because one formula is “wrong,” but because AIC asks *“will these 2 features shave a tiny bit off prediction error?”* while BIC asks *“is the evidence strong enough to believe both features truly belong in the underlying mechanism?”*

---

## Adjusted $$R^2$$

$$R^2_{\text{adj}} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$$

**Plus:** intuitive scale (capped below 1) that immediately punishes useless predictors. **Minus:** restricted to linear regression with an intercept.

## Mallows’ $$C_p$$

$$C_p = \frac{\text{RSS}_p}{\hat{\sigma}^2} - n + 2p$$

where $$\hat{\sigma}^2$$ is the unbiased error variance estimated from the full model and $$p$$ is the number of parameters (including intercept).

**Rule of thumb:** an unbiased model has $$\mathbb{E}[\text{RSS}_p] \approx (n - p)\sigma^2$$, so $$\mathbb{E}[C_p] \approx (n - p) - n + 2p = p$$. Look for the smallest model where $$C_p \approx p$$ (and minimal $$C_p$$ overall).

---

### Worked Example 2: Adding a Junk Predictor — Why $$R^2$$ Lies and $$R^2_{\text{adj}}$$ Catches It

Suppose you have $$n = 20$$ observations and a 1-predictor linear model ($$p = 1$$) with $$R^2 = 0.800$$. You add a second predictor ($$p = 2$$) that is pure random noise, nudging $$R^2$$ up slightly to $$0.802$$. Check $$R^2_{\text{adj}}$$:

- **With $$p = 1$$:**
  $$R^2_{\text{adj}} = 1 - \frac{(1 - 0.800)(19)}{20 - 1 - 1} = 1 - \frac{3.80}{18} = \mathbf{0.7889}.$$
- **With $$p = 2$$:**
  $$R^2_{\text{adj}} = 1 - \frac{(1 - 0.802)(19)}{20 - 2 - 1} = 1 - \frac{3.762}{17} = \mathbf{0.7787}!$$

Ordinary $$R^2$$ climbed ($$0.800 \to 0.802$$), cheering on the junk variable. **Adjusted $$R^2$$ dropped** ($$0.7889 \to 0.7787$$), catching the bluff immediately!

---

### Deep-Dive Explanation: Where Does the Mysterious “$$+2k$$” in AIC Come From? (Read Slowly)

Why is the penalty in AIC specifically $$+2k$$—not $$+k$$ or $$+5k$$? Follow these four intuitive steps slowly.

#### Step 1: Training log-likelihood is doubly lucky
When you evaluate $$\log L(\hat{\theta})$$ on the *exact same* data used to pick $$\hat{\theta}$$, you evaluated the score at the peak of the random hill! If you evaluate that same $$\hat{\theta}$$ on a brand-new independent dataset of size $$n$$, the log-likelihood will be systematically lower.

#### Step 2: Expand around the true parameter $$\theta_0$$ with a 2nd-order Taylor series
Near $$\theta_0$$, the log-likelihood looks like a downward quadratic bowl controlled by the $$k \times k$$ Fisher Information Matrix $$\mathcal{I}$$. Because $$\hat{\theta}$$ sits at the top of the sample hill rather than at $$\theta_0$$, the training log-likelihood gains roughly $$\frac{1}{2}\chi^2_k$$ (expected value $$\frac{k}{2}$$) from fitting noise, while the *new-data* log-likelihood loses another $$\frac{1}{2}\chi^2_k$$ (expected value $$\frac{k}{2}$$) because $$\hat{\theta}$$ moved slightly away from $$\theta_0$$!

#### Step 3: Add both halves together
Total optimism bias in $$\log L(\hat{\theta})$$ is $$\frac{k}{2} + \frac{k}{2} = k$$! Multiplying by $$-2$$ (to turn log-likelihood into deviance, matching $$\chi^2$$ scales) turns that bias into **$$+2k$$**!

#### Step 4: Know when AIC breaks down
The Taylor derivation assumes $$n \gg k$$ (asymptotic regime). When $$n / k < 40$$ (small samples), AIC under-penalizes and can still overfit—so use the small-sample correction **AICc** $$= \text{AIC} + \dfrac{2k(k+1)}{n - k - 1}$$, or switch to cross-validation when models are nonparametric (like trees or neural nets) where counting “parameters $$k$$” is ambiguous.

---

## Video tham khảo

- [Model selection with AIC and AICc](https://www.youtube.com/watch?v=HOqHI53x9Go) — TileStats, English. **xem trước.** AIC penalizes the deviance by the parameter count and picks the smallest score — the formula and the rule in this lesson.
- [Machine Learning Fundamentals: Cross Validation](https://www.youtube.com/watch?v=fSytzGwwBVw) — StatQuest with Josh Starmer, English. The empirical alternative the lesson sets beside AIC and BIC: estimate prediction error by cross-validation instead of an analytic penalty.

## Exercises

**Exercise 1.** Implement BIC from scratch and compare it with AIC on the same polynomial path.

**Exercise 2.** Use AIC to drive forward or backward feature selection in linear regression.
