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

Model selection is one of the hardest practical problems in learning: how complicated should the model be, and how do you know it will work on new data? Cross-validation and other resampling methods give a data-driven answer. Instead of trusting a theoretical formula for test error, we reuse the sample we actually have.

---

## The problem: training error is the wrong score

Training error is a poor proxy for real performance:

- It falls as the model becomes more flexible.
- Test error on *new* data is what we care about.

We do not have an infinite test set. The workaround is to spend the training sample carefully so that we can **estimate** test error.

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

Instead of one split, cut the data into $$K$$ folds (commonly $$K = 5$$ or $$K = 10$$):

1. Partition the sample into $$K$$ pieces of similar size.
2. For each $$k = 1, \ldots, K$$:
   - hold fold $$k$$ out as validation,
   - train on the remaining $$K - 1$$ folds,
   - record the validation error.
3. Average:

$$\text{CV}_{(K)} = \frac{1}{K}\sum_{k=1}^K \text{MSE}_k$$

**Why it helps:** every point is used both for training and for validation; the estimate is stabler than a single split; the variance of the estimate drops.

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

$$\text{CV}_{(n)} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$

**Advantages:** the training set has $$n - 1$$ points; the estimate of test error is nearly unbiased.

**Costs:** you train $$n$$ times; variance is high because the training sets are almost identical.

**When to use it:** small $$n$$ (say, under 100) and a cheap model (linear methods).

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

With imbalanced classes, a plain random fold can accidentally contain almost no minority labels. **Stratified $$K$$-fold** keeps the class mix in each fold close to the mix in the full sample.

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

The bootstrap is another resampling device, especially useful for uncertainty.

**Algorithm:**
1. From $$n$$ observations, draw $$n$$ points **with replacement** (a bootstrap sample).
2. Train on that sample.
3. Score the points that were *not* drawn (out-of-bag).
4. Repeat $$B$$ times (often $$B = 100$$ or $$1000$$).

**Bootstrap estimate of test error:**

$$\text{Err}_{\text{boot}} = \frac{1}{B}\sum_{b=1}^B \frac{1}{|C^{-b}|}\sum_{i \in C^{-b}} L(y_i, \hat{f}^{*b}(x_i))$$

where $$C^{-b}$$ is the out-of-bag set for replicate $$b$$.

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

## Which method when?

| Method | Typical use | Strength | Weakness |
|--------|-------------|----------|----------|
| **Train/test split** | Quick baseline | Fast, simple | High variance |
| **5-fold CV** | Default for most tables | Bias–variance compromise | — |
| **10-fold CV** | Moderate $$n$$ | Lower bias than 5-fold | Slower |
| **LOOCV** | Small $$n$$ ($$n < 100$$) | Lowest bias | High variance, slow |
| **Stratified CV** | Imbalanced classification | Preserves class mix | — |
| **Bootstrap** | Uncertainty estimates | Flexible | More moving parts |

## Exercises

**Exercise 1: $$K$$-fold from scratch.** Write `my_kfold_cv(X, y, model, k=5)` without calling scikit-learn’s splitter.

**Exercise 2: Nested CV.** Implement nested cross-validation so that hyperparameter search and final scoring are not the same loop.

**Exercise 3: Time-series CV.** Look up `TimeSeriesSplit` and apply it to a series you must **not** shuffle.
