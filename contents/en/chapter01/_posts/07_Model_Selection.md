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

How do you choose between a simple model and a flexible one? Cross-validation gives an empirical answer. Information criteria such as AIC and BIC give a *theoretical* score that trades goodness-of-fit against complexity. This lesson introduces those tools so that model choice is a principle, not a vibe.

---

## The selection problem

**The basic tradeoff:**
- A rich model fits the training data and may overfit.
- A small model underfits but often generalizes more honestly.

We need a criterion that rewards fit and charges for extra parameters.

## Akaike information criterion (AIC)

AIC estimates expected prediction error:

$$\text{AIC} = -2\log L(\hat{\theta}) + 2k$$

where $$L(\hat{\theta})$$ is the maximized likelihood and $$k$$ is the number of parameters.

**Rule:** prefer the **smaller** AIC.

The first term measures fit (smaller is better). The second term is the complexity penalty.

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

BIC looks like AIC with a heavier penalty:

$$\text{BIC} = -2\log L(\hat{\theta}) + k\log(n)$$

**AIC versus BIC:**
- BIC penalizes more once $$n > 7$$ (because $$\log n > 2$$).
- BIC therefore tends to pick smaller models.
- AIC is usually the better *prediction* score.
- BIC is usually the better score if you want the “true” sparse model.

## Adjusted $$R^2$$

$$R^2_{\text{adj}} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$$

**Plus:** easy to explain. **Minus:** really a linear-model device.

## Mallows’ $$C_p$$

$$C_p = \frac{\text{RSS}_p}{\hat{\sigma}^2} - n + 2p$$

where $$\hat{\sigma}^2$$ comes from the full model.

**Rule of thumb:** look for a model with $$C_p \approx p$$.

## Exercises

**Exercise 1.** Implement BIC from scratch and compare it with AIC on the same polynomial path.

**Exercise 2.** Use AIC to drive forward or backward feature selection in linear regression.
