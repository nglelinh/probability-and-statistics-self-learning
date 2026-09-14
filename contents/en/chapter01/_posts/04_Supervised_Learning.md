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

Supervised learning is the core of modern machine learning: we learn from labeled pairs in order to predict the label of a new point. This lesson builds a statistical-learning frame, separates regression from classification, and spends most of its energy on the **bias–variance tradeoff**—the reason a more complicated model is not automatically a better one.

---

## What is supervised learning?

A training set consists of pairs $$(X, Y)$$:

- $$X = (X_1, X_2, \ldots, X_p)$$: the feature (predictor) vector
- $$Y$$: the target (response)

**Goal:** learn a function $$f$$ such that $$Y \approx f(X)$$, well enough that the same $$f$$ is useful on *unseen* $$X$$.

### Two main problem types

**1. Regression:** $$Y$$ is continuous  
Examples: house prices, temperature, sales.

**2. Classification:** $$Y$$ is categorical  
Examples: spam / not spam, handwritten digits.

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
- **Squared error (L2):** $$L(Y, f(X)) = (Y - f(X))^2$$
- **Absolute error (L1):** $$L(Y, f(X)) = |Y - f(X)|$$

**Classification:**
- **0–1 loss:** $$L(Y, f(X)) = \mathbb{1}_{Y \neq f(X)}$$
- **Cross-entropy:** $$L(Y, f(X)) = -\sum_k y_k \log f_k(X)$$

**Training error (empirical risk):**

$$\text{Err}_{\text{train}} = \frac{1}{n} \sum_{i=1}^n L(y_i, f(x_i))$$

This is error on data the model has already seen.

**Test error (expected prediction error):**

$$\text{Err}_{\text{test}} = E[L(Y, f(X))]$$

This is the error we actually care about: performance on new draws.

**The central tension:** training error keeps falling as the model gets more flexible; test error can rise. That rise is overfitting.

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

Expected test error splits into three pieces:

$$\text{Expected test error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible error}$$

**1. Bias.** Error from a model that is too simple to represent the true relationship. High bias is **underfitting** (a straight line fit to a curve).

**2. Variance.** Error from a model that chases noise in the particular training sample. High variance is **overfitting** (a high-degree polynomial that interpolates the training points).

**3. Irreducible error.** Noise that no model can remove.

Making the model more flexible usually lowers bias and raises variance. The useful model is the one that balances the two, not the one that wins on the training set.

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

As the number of features $$p$$ grows, the space becomes sparse:

1. **Sample size.** Matching a given density in $$p$$ dimensions needs on the order of $$n \propto 2^p$$ points.
2. **Distances fade.** In high dimension, most pairs of points are comparably far apart.
3. **Overfitting is cheap.** There is more room to fit noise instead of signal.

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

## Exercises

**Exercise 1: Overfitting in a real table.** Using a housing-style regression data set (or another tabular set in scikit-learn), fit polynomial models of several degrees and plot training versus test error.

**Exercise 2: Bias–variance on $$f(x) = x^2$$.** Simulate the bias–variance decomposition for plain linear regression and for a degree-2 polynomial.

**Exercise 3: High-dimensional noise.** Draw data with $$p = 100$$ features of which only five matter. Fit a model and watch overfitting appear.
