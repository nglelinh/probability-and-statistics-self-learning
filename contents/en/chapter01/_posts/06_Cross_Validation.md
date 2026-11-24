---
layout: post
title: 01-07-00 Cross-Validation and Resampling Methods
chapter: "01"
order: 7
owner: nglelinh
lang: en
categories:
- chapter01
lesson_type: required
---

Suppose you have only 100 patient records—expensive, hard-won clinical data—and you want to pick the best regularization strength $$\alpha$$ for a predictive model. If you lock away 30% as a validation set, you shrink your training set to 70 patients, and your choice of $$\alpha$$ might swing wildly depending on which 30 patients happened to land in the hold-out split. Yet if you score models on the training set itself, the most complex overfitted model wins every time!

**Cross-validation** and the **bootstrap** solve this dilemma through a remarkably clever idea: *reuse the data you already have, systematically holding out different slices in rotation* so every observation gets a turn as both teacher and exam question. This lesson builds $$K$$-fold CV, Leave-One-Out CV, Stratified CV, and the Bootstrap—complete with concrete hand calculations and honest guidance on when each method shines.

---

## Background Knowledge Map: What Statistical Ideas Power Resampling?

Resampling looks like simple data-shuffling, yet three deep statistical principles govern why it works—and when it fails.

### 1. Out-of-Sample Generalization & Empirical Risk
Training error systematically underestimates future test error because the model has already adapted to the noise in the training sample (*optimism bias*). **Resampling** simulates fresh test draws internally by partitioning the observed sample into disjoint fit and evaluation blocks.
- **Core questions:** How accurately does cross-validation estimate expected test error? How does fold size affect bias and variance of the error estimate?
- **Real-world reach:** Hyperparameter tuning (`GridSearchCV`), model comparison, and preventing overfitting across tabular, vision, and NLP workflows.

### 2. Exchangeability, Stratification, and Dependence Structure
Standard $$K$$-fold CV assumes observations are **exchangeable** (shuffling rows does not break the data-generating process). When classes are severely imbalanced (e.g., 2% fraud), random folds create noisy class ratios—fixed by **Stratified $$K$$-fold**. When data are ordered in time (stocks, weather) or grouped by patient/user, random shuffling leaks future or subject-level information—requiring **time-series** or **group splits**.
- **Core questions:** Which observations are truly independent? What structural boundaries must never be crossed between train and validation?
- **Real-world reach:** Fraud detection, clinical multi-hospital studies, and financial time-series forecasting.

### 3. Nonparametric Resampling (The Bootstrap)
Invented by Bradley Efron (1979), the **bootstrap** treats the empirical sample as a miniature stand-in for the unknown population, drawing $$n$$ observations *with replacement* $$B$$ times. Beyond estimating out-of-bag prediction error, it computes standard errors and confidence intervals for almost any statistic (medians, ratios, AUC) without assuming normality.
- **Core questions:** How much of the original sample is left “out-of-bag” in each bootstrap draw? When does the bootstrap accurately mimic the true sampling distribution?
- **Real-world reach:** Random Forests (bagging + out-of-bag error), A/B test confidence intervals for skewed metrics, and econometric standard errors.

---

## The problem: training error is the wrong score

Training error is a poor proxy for real performance:

- It falls monotonically as the model becomes more flexible.
- Test error on *unseen* data is what we actually care about.

Because we never have an infinite test set, we spend our finite training sample carefully to **estimate** test error.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression

def demonstrate_train_test_split():
    """A single 70/30 split already shows the U-shaped test-error curve."""
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=1, noise=10)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    degrees = range(1, 16)
    train_errors = []
    test_errors = []

    for degree in degrees:
        model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=0.1))
        model.fit(X_train, y_train)
        train_errors.append(np.mean((y_train - model.predict(X_train)) ** 2))
        test_errors.append(np.mean((y_test - model.predict(X_test)) ** 2))

    plt.figure(figsize=(10, 6))
    plt.plot(degrees, train_errors, 'o-', label='Training error', linewidth=2)
    plt.plot(degrees, test_errors, 's-', label='Test error', linewidth=2)
    plt.xlabel('Polynomial degree')
    plt.ylabel('Mean squared error')
    plt.title('Training vs test error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()

    optimal_degree = degrees[np.argmin(test_errors)]
    print(f"Optimal degree: {optimal_degree}")
    print(f"Test error at that degree: {min(test_errors):.4f}")

# demonstrate_train_test_split()
```

## $$K$$-fold cross-validation

Instead of betting everything on one random split, cut the data into $$K$$ disjoint folds (commonly $$K = 5$$ or $$K = 10$$):

1. Partition the sample into $$K$$ pieces of nearly equal size.
2. For each $$k = 1, \ldots, K$$:
   - hold fold $$k$$ out as validation,
   - train on the remaining $$K - 1$$ folds,
   - record the validation error $$\text{MSE}_k$$.
3. Average across all $$K$$ folds:

$$\text{CV}_{(K)} = \frac{1}{K}\sum_{k=1}^K \text{MSE}_k$$

**Why it helps:** every single observation is evaluated out-of-sample exactly once, reducing the random luck of a single train/validation cut.

---

### Worked Example 1: Computing LOOCV by Hand on a 3-Point Dataset

Let’s run Leave-One-Out Cross-Validation ($$K = n = 3$$) completely by hand on three observations $$y = \{2,\; 4,\; 6\}$$ using the simplest possible model: predict the **training sample mean** $$\hat{y} = \bar{y}_{\text{train}}$$.

| Fold | Held-out point $$y_i$$ | Training set | Training prediction $$\hat{y}_{(-i)}$$ | Squared error $$(y_i - \hat{y}_{(-i)})^2$$ |
|------|------------------------|--------------|----------------------------------------|--------------------------------------------|
| 1 | $$y_1 = 2$$ | $$\{4, 6\}$$ | $$(4+6)/2 = 5$$ | $$(2 - 5)^2 = 9$$ |
| 2 | $$y_2 = 4$$ | $$\{2, 6\}$$ | $$(2+6)/2 = 4$$ | $$(4 - 4)^2 = 0$$ |
| 3 | $$y_3 = 6$$ | $$\{2, 4\}$$ | $$(2+4)/2 = 3$$ | $$(6 - 3)^2 = 9$$ |

- **Training MSE** (fitting $$\bar{y} = 4$$ to all 3 points): $$\frac{(2-4)^2 + (4-4)^2 + (6-4)^2}{3} = \frac{8}{3} \approx 2.67$$.
- **LOOCV MSE:** $$\frac{9 + 0 + 9}{3} = \mathbf{6.00}$$!

Notice two illuminating facts: (1) Training MSE ($$2.67$$) is less than *half* of the honest out-of-sample LOOCV error ($$6.00$$); and (2) for any linear smoother, LOOCV satisfies an exact shortcut formula $$\text{CV}_{(n)} = \frac{1}{n}\sum_{i=1}^n \left(\frac{y_i - \hat{y}_i}{1 - h_{ii}}\right)^2$$ (here $$h_{ii} = 1/3$$, so $$\frac{1}{3}[(-2/(2/3))^2 + 0 + (2/(2/3))^2] = \frac{18}{3} = 6.00$$—letting you compute LOOCV for ordinary least squares in **one** fit!).

---

```python
def k_fold_cross_validation_demo():
    """Compare 5-fold CV with one holdout when choosing Ridge's α."""
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=5, noise=10)

    alphas = np.logspace(-2, 3, 20)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for alpha in alphas:
        model = Ridge(alpha=alpha)
        scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
        cv_scores.append(-scores.mean())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    single_split_scores = []

    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        single_split_scores.append(np.mean((y_test - pred) ** 2))

    plt.figure(figsize=(10, 6))
    plt.semilogx(alphas, cv_scores, 'o-', label='5-fold CV', linewidth=2, markersize=6)
    plt.semilogx(
        alphas, single_split_scores, 's--', label='Single train/test split',
        linewidth=2, markersize=6, alpha=0.7
    )
    plt.xlabel('Regularization parameter (α)')
    plt.ylabel('Mean squared error')
    plt.title('K-fold CV vs a single split')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()

    print(f"Optimal α (CV): {alphas[np.argmin(cv_scores)]:.4f}")
    print(f"Optimal α (single split): {alphas[np.argmin(single_split_scores)]:.4f}")

# k_fold_cross_validation_demo()
```

## Leave-one-out CV (LOOCV)

LOOCV is $$K$$-fold with $$K = n$$: each iteration holds out a single point.

$$\text{CV}_{(n)} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_{(-i)})^2$$

**Advantages:** each training set has $$n - 1$$ points, so the estimate of test error is nearly unbiased.

**Costs:** you train $$n$$ times (unless the hat-matrix shortcut applies), and variance can be surprisingly high because the $$n$$ training sets share $$n-2$$ identical points—so their errors are strongly positively correlated!

**When to use it:** small $$n$$ (say, under 100) and fast models (linear/ridge methods).

```python
from sklearn.model_selection import LeaveOneOut

def compare_cv_methods():
    """5-fold CV and LOOCV on a small regression problem."""
    np.random.seed(42)
    X, y = make_regression(n_samples=50, n_features=3, noise=5)
    alphas = np.logspace(-1, 2, 15)

    kfold_scores = []
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        kfold_scores.append(-scores.mean())

    loo = LeaveOneOut()
    loocv_scores = []
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        scores = cross_val_score(model, X, y, cv=loo, scoring='neg_mean_squared_error')
        loocv_scores.append(-scores.mean())

    plt.figure(figsize=(10, 6))
    plt.semilogx(alphas, kfold_scores, 'o-', label='5-fold CV', linewidth=2)
    plt.semilogx(alphas, loocv_scores, 's-', label='LOOCV', linewidth=2)
    plt.xlabel('α')
    plt.ylabel('CV error')
    plt.title('5-fold CV vs LOOCV')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()

    print(f"Optimal α (5-fold): {alphas[np.argmin(kfold_scores)]:.4f}")
    print(f"Optimal α (LOOCV):  {alphas[np.argmin(loocv_scores)]:.4f}")

# compare_cv_methods()
```

## Stratified $$K$$-fold (classification)

With imbalanced classes, a plain random fold can accidentally contain almost no minority labels. **Stratified $$K$$-fold** keeps the class mix in each fold closely matched to the full sample.

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

def stratified_cv_demo():
    """Show class balance and accuracy under ordinary vs stratified folds."""
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=3,
        n_redundant=0, weights=[0.8, 0.2], random_state=42
    )

    print(f"Overall share of class 1: {y.mean():.2%}")
    print()

    print("Ordinary K-fold:")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"  Fold {i + 1}: {y[val_idx].mean():.2%} class 1")

    print("\nStratified K-fold:")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"  Fold {i + 1}: {y[val_idx].mean():.2%} class 1")

    model = LogisticRegression()
    regular_scores = cross_val_score(
        model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42)
    )
    stratified_scores = cross_val_score(
        model, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    )

    print(f"\nOrdinary CV accuracy: {regular_scores.mean():.4f} ± {regular_scores.std():.4f}")
    print(f"Stratified CV accuracy: {stratified_scores.mean():.4f} ± {stratified_scores.std():.4f}")

# stratified_cv_demo()
```

## Bootstrap

The bootstrap is another resampling device, especially powerful for measuring estimator uncertainty and out-of-bag error.

**Algorithm:**
1. From $$n$$ observations, draw $$n$$ points **with replacement** (a bootstrap sample).
2. Train on that sample.
3. Score the points that were *not* drawn (**out-of-bag**, OOB).
4. Repeat $$B$$ times (often $$B = 100$$ or $$1000$$).

**Bootstrap estimate of test error:**

$$\text{Err}_{\text{boot}} = \frac{1}{B}\sum_{b=1}^B \frac{1}{\lvert C^{-b} \rvert}\sum_{i \in C^{-b}} L(y_i, \hat{f}^{*b}(x_i))$$

where $$C^{-b}$$ is the out-of-bag set for replicate $$b$$.

---

### Worked Example 2: The Famous $$36.8\%$$ Out-of-Bag Rule by Hand

When you draw $$n$$ observations *with replacement* from a dataset of size $$n$$, what fraction of observations are left untouched (out-of-bag) for validation?

For any fixed observation $$i$$:
- On a single draw, the chance of *missing* observation $$i$$ is $$1 - \frac{1}{n}$$.
- Across $$n$$ independent draws, the probability that observation $$i$$ is **never picked** is:
  $$P(i \in C^{-b}) = \left(1 - \frac{1}{n}\right)^n.$$

Plug in sample sizes $$n$$:
- $$n = 5$$: $$(4/5)^5 = 0.8^5 = \mathbf{32.77\%}$$ out-of-bag.
- $$n = 10$$: $$(9/10)^{10} = 0.9^{10} = \mathbf{34.87\%}$$ out-of-bag.
- $$n = 100$$: $$(99/100)^{100} = \mathbf{36.60\%}$$ out-of-bag.
- $$n \to \infty$$: $$\lim_{n\to\infty}\left(1 - \frac{1}{n}\right)^n = e^{-1} \approx \mathbf{36.79\%}$$!

Every bootstrap sample contains roughly **63.2% unique observations** (with duplicates filling the remaining 36.8% slots), leaving **36.8%** as a free built-in validation set! Because each bootstrap model trains on only ~63.2% unique data (smaller effective sample size than full $$n$$), raw OOB error tends to be slightly **pessimistic** (higher bias than 10-fold CV)—which is why methods like the `.632` bootstrap blend training and OOB error.

---

```python
def bootstrap_error_estimation(n_bootstrap=100):
    """Out-of-bag error versus 5-fold CV for Ridge."""
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=5, noise=10)
    n_samples = len(X)
    oob_errors = []

    for _ in range(n_bootstrap):
        bootstrap_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        oob_idx = np.array([i for i in range(n_samples) if i not in bootstrap_idx])
        if len(oob_idx) == 0:
            continue

        model = Ridge(alpha=1.0)
        model.fit(X[bootstrap_idx], y[bootstrap_idx])
        pred_oob = model.predict(X[oob_idx])
        oob_errors.append(np.mean((y[oob_idx] - pred_oob) ** 2))

    bootstrap_estimate = np.mean(oob_errors)
    cv_scores = cross_val_score(
        Ridge(alpha=1.0), X, y, cv=5, scoring='neg_mean_squared_error'
    )
    cv_estimate = -cv_scores.mean()

    print(f"Bootstrap estimate of test error: {bootstrap_estimate:.4f}")
    print(f"5-fold CV estimate: {cv_estimate:.4f}")

    plt.figure(figsize=(10, 6))
    plt.hist(oob_errors, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(
        bootstrap_estimate, color='red', linestyle='--', linewidth=2,
        label=f'Mean = {bootstrap_estimate:.4f}'
    )
    plt.xlabel('Out-of-bag error')
    plt.ylabel('Frequency')
    plt.title(f'Bootstrap error distribution ({n_bootstrap} iterations)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()

# bootstrap_error_estimation()
```

---

### Deep-Dive Explanation: Why 5-Fold and 10-Fold Beat Both Single Splits and LOOCV (Read Slowly)

Why do experienced practitioners almost always default to $$K = 5$$ or $$K = 10$$ instead of $$K = 2$$ or $$K = n$$ (LOOCV)? Follow these four steps slowly.

#### Step 1: Small $$K$$ (like $$K = 2$$) suffers from pessimistic bias
In 2-fold CV, every model trains on only $$50\%$$ of the data. If the learning curve is still steep at $$n/2$$, the 2-fold error reflects a weaker, half-data model—systematically **overestimating** the error of your final full-data model.

#### Step 2: Large $$K$$ (LOOCV, $$K = n$$) removes bias—but inflates correlation!
In LOOCV, every fold trains on $$n - 1$$ points, so bias virtually disappears. Yet any two LOOCV training sets share $$n - 2$$ identical points! Averaging $$n$$ highly correlated numbers does *not* shrink variance like averaging $$n$$ independent numbers does: $$\operatorname{Var}\bigl(\frac{1}{n}\sum Z_i\bigr) = \frac{\sigma^2}{n} + \frac{n-1}{n}\rho\sigma^2 \approx \rho\sigma^2$$!

#### Step 3: Why $$K = 5$$ or $$10$$ hits the sweet spot
With $$K = 5$$ (80% train) or $$K = 10$$ (90% train), each fold is large enough that bias is small, while the training folds differ enough to keep fold-to-fold correlation moderate—and you pay only 5–10 model fits instead of $$n$$.

#### Step 4: When you tune hyperparameters, nest the loop!
If you test 50 candidate hyperparameters with 5-fold CV and report the winning CV score as your final accuracy, that score is optimistically biased by selection. **Nested CV** wraps hyperparameter tuning inside an *inner* CV loop and evaluates the whole tuning procedure on an untouched *outer* CV loop.

---

## Which method when?

| Method | Typical use | Strength | Weakness |
|--------|-------------|----------|----------|
| **Train/test split** | Quick baseline | Fast, simple | High variance |
| **5-fold CV** | Default for most tables | Bias–variance compromise | Slight pessimistic bias on tiny $$n$$ |
| **10-fold CV** | Moderate $$n$$ | Lower bias than 5-fold | $$2\times$$ slower than 5-fold |
| **LOOCV** | Small $$n$$ ($$n < 100$$) | Lowest bias | High fold correlation & slow |
| **Stratified CV** | Imbalanced classification | Preserves class mix | Classification only |
| **Bootstrap** | Uncertainty estimates / CIs | Flexible, distribution-free | Slight OOB pessimism (63.2% unique) |

## Video tham khảo

- [Machine Learning Fundamentals: Cross Validation](https://www.youtube.com/watch?v=fSytzGwwBVw) — StatQuest with Josh Starmer, English. **xem trước.** Block picture of K-fold and leave-one-out, then using that same rotation to tune a parameter.
- [Bootstrapping Main Ideas!!!](https://www.youtube.com/watch?v=Xz0x-8-cgaQ) — StatQuest with Josh Starmer, English. Sampling with replacement and the out-of-bag idea — the bootstrap section at the end of the lesson.
- [Máy Học Ứng Dụng - Bài 11. Phương Pháp Đánh Giá Chéo (Cross-Validation) Trong Máy Học](https://www.youtube.com/watch?v=aKVgXrDBqgk) — OVNE-Giáo Dục Việt, Vietnamese. A Vietnamese walkthrough of the same rotation: estimate error on data the fit did not train on.

## Exercises

**Exercise 1: $$K$$-fold from scratch.** Write `my_kfold_cv(X, y, model, k=5)` without calling scikit-learn’s splitter.

**Exercise 2: Nested CV.** Implement nested cross-validation so that hyperparameter search and final scoring are not the same loop.

**Exercise 3: Time-series CV.** Look up `TimeSeriesSplit` and apply it to a series you must **not** shuffle.
