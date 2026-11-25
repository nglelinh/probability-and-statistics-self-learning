---
layout: post
title: 01-02-00 Overview of Supervised Learning
chapter: "01"
order: 2
owner: nglelinh
lang: en
categories:
- chapter01
lesson_type: required
---

Imagine two students preparing for a final exam using a past practice test. Student A refuses to study any formulas and simply guesses the average answer for every question—scoring poorly on both the practice test and the real exam. Student B memorizes every single answer key from the practice test word-for-word, scoring a flawless 100% on the practice sheet—and then bombs the real exam the moment the numbers change slightly.

Those two students are the living faces of **underfitting (high bias)** and **overfitting (high variance)**. Supervised learning is the art of learning from labeled past examples $$(X, Y)$$ so that your predictions hold up on *brand-new, unseen* data. This lesson builds the statistical-learning frame, separates regression from classification, derives the **bias–variance decomposition** with a concrete hand calculation, and shows why high-dimensional feature spaces quietly ambush naive models through the **curse of dimensionality**.

---

## Key ideas: What Fields Sit Behind Supervised Learning?

Supervised learning is not merely software engineering—it rests on three mathematical pillars.

### 1. Statistical Decision Theory (Loss & Risk)
Given random inputs $$X$$ and target $$Y$$ drawn from an unknown joint distribution $$P(X,Y)$$, **decision theory** defines a **loss function** $$L(Y, f(X))$$ measuring the cost of a wrong prediction and seeks the predictor $$f^*$$ that minimizes **expected risk** $$R(f) = \mathbb{E}[L(Y, f(X))]$$. Under squared error, the theoretical ideal predictor is the conditional expectation $$f^*(x) = \mathbb{E}[Y \mid X = x]$$.
- **Core questions:** Which loss function matches the real cost of errors (L2 vs. L1 vs. cross-entropy)? What is the theoretically lowest achievable error (*Bayes risk*)?
- **Real-world reach:** Credit scoring, demand forecasting, spam filtering, and medical triage models.

### 2. Estimation Theory and the Bias–Variance Tradeoff
Because we never know the true distribution $$P(X,Y)$$, we estimate $$\hat{f}$$ from a finite noisy training sample of size $$n$$. **Estimation theory** decomposes expected squared prediction error into **Bias²** (systematic misspecification), **Variance** (sensitivity to which training sample we happened to draw), and **Irreducible Noise** $$\sigma^2$$.
- **Core questions:** How does adding parameters trade lower bias for higher variance? How do regularization and ensembles rebalance that trade?
- **Real-world reach:** Choosing model complexity, Ridge/Lasso shrinkage, Random Forests (variance reduction), and Gradient Boosting (bias reduction).

### 3. High-Dimensional Geometry (The Curse of Dimensionality)
In 1D or 2D, nearby training points sit close together so local averaging works effortlessly. As the number of features $$p$$ climbs to 50 or 500, **high-dimensional geometry** behaves counterintuitively: almost all volume migrates to the outer boundary shell, and pairwise distances concentrate so that “nearest” neighbors are barely closer than average points.
- **Core questions:** How many samples $$n$$ are needed to cover a $$p$$-dimensional space? Why do we need structural assumptions (linearity, sparsity, manifolds) in high dimensions?
- **Real-world reach:** Genomics ($$p \gg n$$), recommendation systems, image/text embeddings, and dimensionality reduction (PCA).

---

## What is supervised learning?

A training set consists of labeled pairs $$(X, Y)$$:

- $$X = (X_1, X_2, \ldots, X_p)$$: the feature (predictor) vector
- $$Y$$: the target (response)

**Goal:** learn a function $$\hat{f}$$ from training data such that $$Y \approx \hat{f}(X)$$ on *unseen* future draws of $$X$$.

### Two main problem types

**1. Regression:** $$Y$$ is continuous  
Examples: house prices, temperature, sales.

**2. Classification:** $$Y$$ is categorical  
Examples: spam / not spam, handwritten digits, disease diagnosis.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split

def visualize_supervised_learning():
    """Contrast a regression cloud with a two-class classification cloud."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
    axes[0].scatter(X_reg, y_reg, alpha=0.6, s=50)
    axes[0].set_xlabel('Feature X')
    axes[0].set_ylabel('Target Y (continuous)')
    axes[0].set_title('Regression: predict a continuous value')
    axes[0].grid(True, alpha=0.3)

    X_clf, y_clf = make_classification(
        n_samples=200, n_features=2, n_redundant=0,
        n_informative=2, n_clusters_per_class=1, random_state=42
    )
    colors = ['red' if label == 0 else 'blue' for label in y_clf]
    axes[1].scatter(X_clf[:, 0], X_clf[:, 1], c=colors, alpha=0.6, s=50)
    axes[1].set_xlabel('Feature X₁')
    axes[1].set_ylabel('Feature X₂')
    axes[1].set_title('Classification: binary labels')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    # plt.show()

# visualize_supervised_learning()
```

## Loss functions and training error

A **loss** $$L(Y, f(X))$$ scores the discrepancy between the truth $$Y$$ and the prediction $$f(X)$$.

### Common losses

**Regression:**
- **Squared error (L2):** $$L(Y, f(X)) = (Y - f(X))^2$$ — heavily penalizes large errors; minimized by the conditional **mean**.
- **Absolute error (L1):** $$L(Y, f(X)) = \lvert Y - f(X) \rvert$$ — robust to outliers; minimized by the conditional **median**.

**Classification:**
- **0–1 loss:** $$L(Y, f(X)) = \mathbb{1}_{Y \neq f(X)}$$ — counts misclassifications directly, but non-differentiable.
- **Cross-entropy:** $$L(Y, f(X)) = -\sum_k y_k \log f_k(X)$$ — smooth, convex surrogate that rewards well-calibrated probabilities.

**Training error (empirical risk):**

$$\text{Err}_{\text{train}} = \frac{1}{n} \sum_{i=1}^n L(y_i, f(x_i))$$

This is error on data the model has already seen.

**Test error (expected prediction error):**

$$\text{Err}_{\text{test}} = \mathbb{E}[L(Y, f(X))]$$

This is the error we actually care about: performance on fresh draws from the population.

**The central tension:** training error keeps falling as the model gets more flexible; test error forms a U-shape and rebounds upward once the model starts memorizing noise. That rebound is **overfitting**.

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

def demonstrate_overfitting(degree_max=15):
    """Polynomial regression with a tiny sample: watch test error rebound."""
    np.random.seed(42)
    X = np.linspace(0, 1, 20).reshape(-1, 1)
    y_true = np.sin(2 * np.pi * X).ravel()
    y = y_true + np.random.normal(0, 0.2, X.shape[0])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    degrees = range(1, degree_max + 1)
    train_errors = []
    test_errors = []

    for degree in degrees:
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_poly, y_train)

        train_errors.append(mean_squared_error(y_train, model.predict(X_train_poly)))
        test_errors.append(mean_squared_error(y_test, model.predict(X_test_poly)))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(degrees, train_errors, 'o-', label='Training error', linewidth=2)
    plt.plot(degrees, test_errors, 's-', label='Test error', linewidth=2)
    plt.xlabel('Model complexity (polynomial degree)')
    plt.ylabel('Mean squared error')
    plt.title('Training error vs test error')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    X_plot = np.linspace(0, 1, 100).reshape(-1, 1)

    for degree in [1, 3, 9, 15]:
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_plot_poly = poly.transform(X_plot)

        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        plt.plot(X_plot, model.predict(X_plot_poly), label=f'Degree {degree}', linewidth=2)

    plt.scatter(X_train, y_train, color='black', s=50, alpha=0.5, label='Training data')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('The same data, four different degrees')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    # plt.show()

    optimal_degree = degrees[np.argmin(test_errors)]
    print(f"Degree with smallest test error: {optimal_degree}")
    print(f"Minimum test error: {min(test_errors):.4f}")

# demonstrate_overfitting()
```

## The bias–variance tradeoff

Suppose $$Y = f(x) + \varepsilon$$ where $$\mathbb{E}[\varepsilon] = 0$$ and $$\operatorname{Var}(\varepsilon) = \sigma^2$$. For any estimator $$\hat{f}(x)$$ trained on a random training set, its expected squared test error at a fixed point $$x$$ splits **exactly** into three pieces:

$$\mathbb{E}\bigl[(Y - \hat{f}(x))^2\bigr] = \underbrace{\bigl(\mathbb{E}[\hat{f}(x)] - f(x)\bigr)^2}_{\text{Bias}^2} + \underbrace{\operatorname{Var}(\hat{f}(x))}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible error}}$$

**1. Bias.** Error from a model that is too rigid to represent the true relationship. High bias is **underfitting** (fitting a straight line to a sine wave).

**2. Variance.** Error from a model that chases random fluctuations in the particular training sample. High variance is **overfitting** (a degree-15 polynomial wiggling wildly through 14 points).

**3. Irreducible error ($$\sigma^2$$).** Inherent randomness in $$Y$$ that *no* model can ever predict away.

---

### Worked Example 1: Exact Bias–Variance–Noise Accounting by Hand

Let’s make the decomposition completely concrete with zero simulation required. Suppose at a fixed point $$x_0$$, the true signal is $$f(x_0) = 10$$ and observations follow $$Y = 10 + \varepsilon$$ with noise variance $$\sigma^2 = 4$$. You collect a tiny training sample of two independent observations $$Y_1, Y_2$$, and compare **three** competing predictors for a new unseen draw $$Y_{\text{new}}$$:

| Predictor $$\hat{f}$$ | $$\mathbb{E}[\hat{f}]$$ | $$\text{Bias}^2 = (\mathbb{E}[\hat{f}]-10)^2$$ | $$\text{Variance} = \operatorname{Var}(\hat{f})$$ | Noise $$\sigma^2$$ | Total Test MSE |
|-----------------------|-------------------------|------------------------------------------------|---------------------------------------------------|--------------------|----------------|
| **A. Rigid constant:** $$\hat{f}_A \equiv 8$$ | $$8$$ | $$(8-10)^2 = 4$$ | $$0$$ | $$4$$ | **$$8$$** |
| **B. Memorize 1 point:** $$\hat{f}_B = Y_1$$ | $$10$$ | $$0$$ | $$\operatorname{Var}(Y_1) = 4$$ | $$4$$ | **$$8$$** |
| **C. Sample average:** $$\hat{f}_C = \frac{Y_1+Y_2}{2}$$ | $$10$$ | $$0$$ | $$\frac{4}{2} = 2$$ | $$4$$ | **$$6$$** |
| **D. Shrunk estimator:** $$\hat{f}_D = 0.8\bar{Y} + 0.2(9)$$ | $$9.8$$ | $$(9.8-10)^2 = 0.04$$ | $$0.8^2 \times 2 = 1.28$$ | $$4$$ | **$$5.32$$** |

**Look at row D!** By deliberately introducing a tiny bit of **bias** ($$\text{Bias}^2 = 0.04$$) via shrinkage toward a reasonable prior guess, we slashed variance from $$2.00$$ down to $$1.28$$—beating the unbiased sample mean ($$5.32 < 6.00$$)! That single 4-row table *is* the entire mathematical justification for **Ridge regression** and **regularization**.

---

```python
def bias_variance_decomposition(n_simulations=100):
    """Resample training sets and watch bias shrink while variance grows."""
    def true_function(x):
        return np.sin(2 * np.pi * x)

    x_test = np.array([0.5])
    y_true = true_function(x_test)

    degrees = [1, 3, 9, 15]
    results = {deg: {'predictions': [], 'bias': 0, 'variance': 0} for deg in degrees}

    for _ in range(n_simulations):
        X_train = np.random.uniform(0, 1, 20).reshape(-1, 1)
        y_train = true_function(X_train).ravel() + np.random.normal(0, 0.2, 20)

        for degree in degrees:
            poly = PolynomialFeatures(degree=degree)
            X_train_poly = poly.fit_transform(X_train)
            x_test_poly = poly.transform(x_test.reshape(-1, 1))

            model = LinearRegression()
            model.fit(X_train_poly, y_train)
            results[degree]['predictions'].append(model.predict(x_test_poly)[0])

    for degree in degrees:
        preds = np.array(results[degree]['predictions'])
        mean_pred = preds.mean()
        results[degree]['bias'] = (mean_pred - y_true[0]) ** 2
        results[degree]['variance'] = preds.var()
        results[degree]['mse'] = results[degree]['bias'] + results[degree]['variance']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].axhline(y_true[0], color='green', linestyle='--', linewidth=2, label='True value')
    for degree in degrees:
        axes[0].hist(results[degree]['predictions'], bins=20, alpha=0.5, label=f'Degree {degree}')
    axes[0].set_xlabel('Predicted value at x = 0.5')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Prediction distribution across training sets')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    x_pos = np.arange(len(degrees))
    bias_vals = [results[d]['bias'] for d in degrees]
    var_vals = [results[d]['variance'] for d in degrees]

    axes[1].bar(x_pos - 0.2, bias_vals, 0.4, label='Bias²', alpha=0.8)
    axes[1].bar(x_pos + 0.2, var_vals, 0.4, label='Variance', alpha=0.8)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f'Degree {d}' for d in degrees])
    axes[1].set_ylabel('Error')
    axes[1].set_title('Bias–variance decomposition')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    # plt.show()

    print("Bias–variance analysis")
    print("-" * 50)
    for degree in degrees:
        print(f"Degree {degree}:")
        print(f"  Bias²:    {results[degree]['bias']:.4f}")
        print(f"  Variance: {results[degree]['variance']:.4f}")
        print(f"  MSE:      {results[degree]['mse']:.4f}")
        print()

# bias_variance_decomposition()
```

## The curse of dimensionality

As the number of features $$p$$ grows, the feature space explodes exponentially:

1. **Sample size.** Matching a given density in $$p$$ dimensions needs on the order of $$n \propto c^p$$ points.
2. **Distances fade.** In high dimensions, almost every point sits near the outer shell, and nearest neighbors are nearly as far as distant points.
3. **Overfitting becomes effortless.** With more feature axes, a flexible model can easily align with pure noise.

---

### Worked Example 2: The Curse of Dimensionality in Two Back-of-the-Envelope Calculations

Imagine points drawn uniformly inside the $$p$$-dimensional unit hypercube $$[0,1]^p$$.

1. **Where does the volume hide?** Consider the inner core $$[0.1, 0.9]^p$$ (trimming just a 10% crust off every side). Its volume is $$0.8^p$$, so the fraction of points trapped in the outer 10% crust is $$1 - 0.8^p$$:
   - $$p = 1$$ (line segment): $$1 - 0.8^1 = 20\%$$ in the outer crust.
   - $$p = 2$$ (square): $$1 - 0.8^2 = 36\%$$ in the outer crust.
   - $$p = 10$$: $$1 - 0.8^{10} \approx 89.3\%$$ in the outer crust!
   - $$p = 50$$: $$1 - 0.8^{50} \approx 99.9986\%$$ in the outer crust! Virtually **every** training point is an edge/boundary point!
2. **How wide must a “local” neighborhood be to capture 10% of the data?**
   A sub-cube of side length $$\ell$$ has volume $$\ell^p = 0.10 \implies \ell = 0.10^{1/p}$$:
   - At $$p = 1$$: $$\ell = 0.10$$ (truly local!).
   - At $$p = 10$$: $$\ell = 0.10^{0.1} \approx 0.794$$ — you must stretch across **79.4% of every single feature’s range** just to collect 10% of the data! “Local” methods like $$k$$-nearest neighbors lose locality completely unless the data actually lie on a much lower-dimensional structure.

---

### Explanation: Deriving the Bias–Variance Decomposition Step by Step

Why do the cross terms vanish cleanly in $$\mathbb{E}[(Y - \hat{f})^2] = \text{Bias}^2 + \text{Variance} + \sigma^2$$? Walk through these four algebraic steps slowly—every statistician should verify this once by hand.

#### Step 1: Separate irreducible noise $$\varepsilon$$ from estimation error
Write $$Y = f + \varepsilon$$ where $$f = f(x)$$ is a fixed deterministic constant and $$\varepsilon$$ is independent of the training set (and therefore independent of $$\hat{f}$$), with $$\mathbb{E}[\varepsilon] = 0$$ and $$\mathbb{E}[\varepsilon^2] = \sigma^2$$:
$$\mathbb{E}\bigl[(f + \varepsilon - \hat{f})^2\bigr] = \mathbb{E}\bigl[(f - \hat{f})^2\bigr] + 2\underbrace{\mathbb{E}[\varepsilon]}_{0}\mathbb{E}[f - \hat{f}] + \underbrace{\mathbb{E}[\varepsilon^2]}_{\sigma^2} = \mathbb{E}\bigl[(f - \hat{f})^2\bigr] + \sigma^2.$$

#### Step 2: Add and subtract the average prediction $$\bar{f} = \mathbb{E}[\hat{f}]$$
Inside $$\mathbb{E}[(f - \hat{f})^2]$$, insert $$-\bar{f} + \bar{f}$$:
$$(f - \hat{f})^2 = \bigl((f - \bar{f}) + (\bar{f} - \hat{f})\bigr)^2 = (f - \bar{f})^2 + (\bar{f} - \hat{f})^2 + 2(f - \bar{f})(\bar{f} - \hat{f}).$$

#### Step 3: Take expectations and watch the cross term vanish
Since $$f - \bar{f}$$ is a deterministic constant and $$\mathbb{E}[\bar{f} - \hat{f}] = \bar{f} - \bar{f} = 0$$, the cross term disappears identically, leaving:
$$\mathbb{E}\bigl[(Y - \hat{f})^2\bigr] = (f - \bar{f})^2 + \mathbb{E}\bigl[(\hat{f} - \bar{f})^2\bigr] + \sigma^2 = \text{Bias}^2 + \text{Variance} + \sigma^2.$$

#### Step 4: Read the strategic takeaway for real modeling
- **Training error is high AND validation error is high:** You have **high bias** (underfitting) → add expressive features, increase model flexibility, or reduce regularization.
- **Training error is tiny BUT validation error is high:** You have **high variance** (overfitting) → gather more data, add regularization (Ridge/Lasso), reduce features, or average across bagged models (Random Forest).

---

```python
def demonstrate_curse_of_dimensionality():
    """Average distance from a reference point grows with dimension."""
    dimensions = [1, 2, 5, 10, 20, 50]
    n_samples = 1000

    avg_distances = []
    std_distances = []

    for dim in dimensions:
        data = np.random.uniform(0, 1, (n_samples, dim))
        distances = np.sqrt(((data[0] - data[1:]) ** 2).sum(axis=1))
        avg_distances.append(distances.mean())
        std_distances.append(distances.std())

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        dimensions, avg_distances, yerr=std_distances,
        marker='o', capsize=5, linewidth=2, markersize=8
    )
    plt.xlabel('Dimension (p)')
    plt.ylabel('Average distance')
    plt.title('Curse of dimensionality: distances inflate with p')
    plt.grid(True, alpha=0.3)
    # plt.show()

    print("Average distance by dimension:")
    for dim, avg, std in zip(dimensions, avg_distances, std_distances):
        print(f"  p={dim:2d}: {avg:.4f} ± {std:.4f}")

# demonstrate_curse_of_dimensionality()
```

## Video tham khảo

- [A Gentle Introduction to Machine Learning](https://www.youtube.com/watch?v=Gv9_4yMHFhI) — StatQuest with Josh Starmer, English. **xem trước.** The supervised frame: regression when Y is continuous and classification when Y is discrete, before the bias–variance section.
- [Machine Learning Fundamentals: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA) — StatQuest with Josh Starmer, English. Bias and variance on a fitted curve — why a more flexible model is not automatically a better predictor.
- [Statistical Learning: 2.3 Model Selection and Bias Variance Tradeoff](https://www.youtube.com/watch?v=pvcEQfcO3pk) — Stanford Online, English. The same tradeoff in the statistical-learning course this lesson is preparing you for.

## Exercises

**Exercise 1: Overfitting in a real table.** Using a housing-style regression data set (or another tabular set in scikit-learn), fit polynomial models of several degrees and plot training versus test error.

**Exercise 2: Bias–variance on $$f(x) = x^2$$.** Simulate the bias–variance decomposition for plain linear regression and for a degree-2 polynomial.

**Exercise 3: High-dimensional noise.** Draw data with $$p = 100$$ features of which only five matter. Fit a model and watch overfitting appear.
