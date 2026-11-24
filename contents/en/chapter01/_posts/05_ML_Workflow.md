---
layout: post
title: 01-10-00 A Practical Machine Learning Workflow
chapter: "01"
order: 10
owner: nglelinh
lang: en
categories:
- chapter01
lesson_type: required
---

Ask a veteran data scientist why real-world machine learning projects fail in production, and they will almost never blame the math of Ridge regression or Random Forests. They will tell you stories about **data leakage**—scaling a dataset before splitting off the test set, tuning hyperparameters on the final exam, or dropping outliers using global statistics that quietly peeked into the future. A model can post a dazzling $$R^2 = 0.94$$ inside a Jupyter notebook and collapse the day it meets real customers simply because the pipeline violated basic experimental hygiene.

Earlier lessons introduced individual statistical tools one at a time. This lesson assembles them into a disciplined, end-to-end **eight-step workflow** on the California Housing dataset—from exploratory data analysis and leak-free preprocessing through baselines, cross-validated tuning, one-shot test evaluation, and residual diagnostics.

---

## Background Knowledge Map: What Statistical Principles Govern an ML Pipeline?

A clean pipeline is not bureaucratic boilerplate—every step defends against a specific statistical failure mode.

### 1. Exploratory Data Analysis (EDA) & Robust Preprocessing
Before fitting any equation, **EDA** checks distributions, missingness, skewness, and extreme outliers. Skewed features (like population or income) stretch linear losses disproportionately; log transforms (`np.log1p`) and IQR outlier checks stabilize variance and prevent a handful of anomalous rows from hijacking parameter estimates.
- **Core questions:** Are features on wildly different scales? Are target values censored or heavy-tailed?
- **Real-world reach:** Financial fraud detection, real-estate pricing, clinical lab measurements, and sensor telemetry.

### 2. Data Splitting, Information Barriers & Leakage Prevention
Statistical generalization requires the test set to act as an **independent draw** from the unseen future population. Any step that learns parameters from data—computing means/standard deviations (`StandardScaler`), imputing medians, selecting features, or picking hyperparameters—*must* be fitted strictly inside the training split (wrapped in a `Pipeline`), never across the full dataset.
- **Core questions:** Which steps are “stateless” row transformations versus “stateful” estimators that can leak test information?
- **Real-world reach:** Reproducible clinical trials, quantitative trading backtests (preventing look-ahead bias), and production ML platforms.

### 3. Multi-Metric Evaluation & Residual Diagnostics
A single summary number like MSE can hide systematic bias, heteroscedasticity (error growing with predicted value), or sensitivity to a single bad prediction. Pairing **RMSE**, **MAE**, and **$$R^2$$** with **residual plots** reveals whether your model genuinely captured the signal or left structured patterns behind.
- **Core questions:** Are residuals centered near zero with roughly constant spread? Where does the model systematically under- or over-predict?
- **Real-world reach:** Model auditing, regulatory validation in banking/healthcare, and iterative error analysis.

---

## An end-to-end outline

A disciplined project walks through eight steps in strict order:

1. **Problem understanding and EDA**
2. **Data preprocessing**
3. **Feature engineering**
4. **Train / validation / test split**
5. **A baseline model**
6. **Model selection and hyperparameter tuning**
7. **Final evaluation (touch the test set once)**
8. **Interpretation and residual diagnostics**

---

### Worked Example 1: Catching Data Leakage by Hand on a 5-Point Dataset

Why is calling `StandardScaler().fit_transform(X)` *before* `train_test_split` such a serious mistake? Watch what happens on five tiny house-size measurements:

$$\text{Train} = \{10,\; 12,\; 14,\; 16\}, \qquad \text{Test} = \{100\} \text{ (one mansion outlier!)}.$$

1. **WRONG (Leaky — scale all 5 points together first):**
   - Combined mean: $$\mu_{\text{all}} = \frac{10+12+14+16+100}{5} = 30.4$$.
   - Combined standard deviation: $$\sigma_{\text{all}} \approx 34.86$$.
   - Standardized training values become $$\{-0.59,\; -0.53,\; -0.47,\; -0.41\}$$—every single normal house looks *negative* because the unseen test mansion silently dragged the mean up from 13 to 30.4!
2. **RIGHT (Leak-free — fit scaler on Train only, transform Test using Train’s $$\mu, \sigma$$):**
   - Training mean: $$\mu_{\text{train}} = 13.0$$, training std: $$\sigma_{\text{train}} = \sqrt{5} \approx 2.236$$.
   - Standardized training values: $$\{-1.34,\; -0.45,\; +0.45,\; +1.34\}$$ (cleanly centered!).
   - Standardized test value: $$\frac{100 - 13.0}{2.236} \approx +38.91$$—honestly flagging the test mansion as an extreme outlier, exactly as it would appear in live production!

**Rule of thumb:** Always split *first*, or wrap `StandardScaler` inside a scikit-learn `Pipeline` so `GridSearchCV` re-fits the scaler fresh inside every cross-validation fold.

---

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
```

## Step 1: Understand the problem and explore the data

```python
from sklearn.datasets import fetch_california_housing

def load_and_explore_data():
    """Load California housing and run a lightweight EDA."""
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    print("=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Number of samples: {len(df)}")
    print(f"Number of features: {len(df.columns) - 1}")
    print(f"\nTarget variable: {housing.target_names[0]}")
    print(f"\nFeatures:")
    for i, name in enumerate(housing.feature_names):
        print(f"  {i + 1}. {name}")

    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    print(df.describe())

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    print(df.isnull().sum())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(df['MedHouseVal'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Median house value (100k$)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of the target')

    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[0, 1],
                cbar_kws={'label': 'Correlation'})
    axes[0, 1].set_title('Correlation matrix')

    axes[1, 0].scatter(df['MedInc'], df['MedHouseVal'], alpha=0.3, s=10)
    axes[1, 0].set_xlabel('Median income')
    axes[1, 0].set_ylabel('Median house value')
    axes[1, 0].set_title('Income vs house value')

    axes[1, 1].boxplot(df['AveRooms'])
    axes[1, 1].set_ylabel('Average rooms')
    axes[1, 1].set_title('Distribution of average rooms')

    plt.tight_layout()
    # plt.show()

    return df

# df = load_and_explore_data()
```

## Step 2: Preprocess

```python
def preprocess_data(df):
    """Separate features from the target and trim a few extreme outliers."""
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']

    def remove_outliers(data, column, threshold=3):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        return data[(data[column] >= lower) & (data[column] <= upper)]

    df_clean = df.copy()
    for col in ['AveRooms', 'AveBedrms']:
        df_clean = remove_outliers(df_clean, col)

    print(f"Samples after outlier removal: {len(df_clean)} (dropped {len(df) - len(df_clean)})")

    X_clean = df_clean.drop('MedHouseVal', axis=1)
    y_clean = df_clean['MedHouseVal']
    return X_clean, y_clean

# X, y = preprocess_data(df)
```

## Step 3: Engineer features

```python
def create_features(X):
    """Build a few ratio and log features from the raw columns."""
    X_new = X.copy()

    X_new['RoomsPerHousehold'] = X['AveRooms'] / X['AveOccup']
    X_new['BedroomsPerRoom'] = X['AveBedrms'] / X['AveRooms']
    X_new['PopulationPerHousehold'] = X['Population'] / X['AveOccup']
    X_new['LogPopulation'] = np.log1p(X['Population'])
    X_new['LogMedInc'] = np.log1p(X['MedInc'])

    print("New features:")
    print("  - RoomsPerHousehold")
    print("  - BedroomsPerRoom")
    print("  - PopulationPerHousehold")
    print("  - LogPopulation")
    print("  - LogMedInc")

    return X_new

# X_engineered = create_features(X)
```

## Step 4: Split train / validation / test

```python
def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """Hold out a test set, then split the remainder into train and validation."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )

    print("Split sizes:")
    print(f"  Train:      {len(X_train):5d} ({len(X_train) / len(X) * 100:.1f}%)")
    print(f"  Validation: {len(X_val):5d} ({len(X_val) / len(X) * 100:.1f}%)")
    print(f"  Test:       {len(X_test):5d} ({len(X_test) / len(X) * 100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test
```

## Step 5: A baseline you can beat

```python
def build_baseline(X_train, y_train, X_val, y_val):
    """Mean predictor and ordinary least squares as reference models."""
    y_pred_mean = np.full(len(y_val), y_train.mean())
    mse_mean = mean_squared_error(y_val, y_pred_mean)
    r2_mean = r2_score(y_val, y_pred_mean)

    from sklearn.linear_model import LinearRegression

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_val_scaled)
    mse_lr = mean_squared_error(y_val, y_pred_lr)
    r2_lr = r2_score(y_val, y_pred_lr)

    print("BASELINE MODELS")
    print("=" * 50)
    print("Mean prediction:")
    print(f"  MSE: {mse_mean:.4f}")
    print(f"  R²:  {r2_mean:.4f}")
    print("\nLinear regression:")
    print(f"  MSE: {mse_lr:.4f}")
    print(f"  R²:  {r2_lr:.4f}")

    return mse_lr

# baseline_mse = build_baseline(X_train, y_train, X_val, y_val)
```

## Step 6: Compare models and tune

```python
def model_selection_pipeline(X_train, y_train, X_val, y_val):
    """Grid-search a few candidates and keep the validation winner."""
    preprocessor = StandardScaler()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)

    models = {
        'Ridge': {
            'model': Ridge(),
            'params': {'alpha': [0.1, 1, 10, 100, 1000]}
        },
        'Lasso': {
            'model': Lasso(max_iter=10000),
            'params': {'alpha': [0.001, 0.01, 0.1, 1, 10]}
        },
        'Random Forest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
        }
    }

    results = {}

    for name, config in models.items():
        print(f"\nTuning {name}...")

        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X_train_scaled, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_val_scaled)
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        results[name] = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'mse': mse,
            'r2': r2,
            'cv_score': -grid_search.best_score_
        }

        print(f"  Best params: {grid_search.best_params_}")
        print(f"  CV MSE: {-grid_search.best_score_:.4f}")
        print(f"  Val MSE: {mse:.4f}")
        print(f"  Val R²: {r2:.4f}")

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    for name, res in results.items():
        print(f"{name:15s}: Val MSE = {res['mse']:.4f}, R² = {res['r2']:.4f}")

    best_model_name = min(results, key=lambda x: results[x]['mse'])
    print(f"\nBest model: {best_model_name}")

    return results, best_model_name, preprocessor

# results, best_name, preprocessor = model_selection_pipeline(X_train, y_train, X_val, y_val)
```

## Step 7: Touch the test set once

---

### Worked Example 2: Computing MSE, RMSE, MAE, and $$R^2$$ by Hand on 4 Houses

Suppose your final model evaluates 4 unseen test houses (prices in $100k):
- Actual values: $$y = (2.0,\; 3.0,\; 4.0,\; 5.0)$$ (mean $$\bar{y} = 3.5$$)
- Predictions: $$\hat{y} = (2.2,\; 2.8,\; 4.1,\; 4.5)$$
- Residuals $$e_i = y_i - \hat{y}_i$$: $$(-0.2,\; +0.2,\; -0.1,\; +0.5)$$

Compute all four metrics directly:
1. **MSE:** $$\frac{(-0.2)^2 + 0.2^2 + (-0.1)^2 + 0.5^2}{4} = \frac{0.04 + 0.04 + 0.01 + 0.25}{4} = \frac{0.34}{4} = 0.085$$.
2. **RMSE:** $$\sqrt{0.085} \approx 0.2915$$ (roughly **$29,150** typical error in original units).
3. **MAE:** $$\frac{0.2 + 0.2 + 0.1 + 0.5}{4} = \frac{1.0}{4} = 0.25$$ (**$25,000** average absolute error). Notice $$\text{RMSE} > \text{MAE}$$ because squaring amplifies the single larger miss ($$+0.5$$) more heavily!
4. **$$R^2$$:** Total sum of squares $$\text{SST} = (-1.5)^2 + (-0.5)^2 + 0.5^2 + 1.5^2 = 5.0$$; residual sum of squares $$\text{SSR} = 0.34$$; so
   $$R^2 = 1 - \frac{0.34}{5.0} = 0.932 \quad (93.2\% \text{ of variance explained relative to the mean baseline!}).$$

---

```python
def final_evaluation(best_model, preprocessor, X_test, y_test):
    """Report test metrics and residual diagnostics."""
    X_test_scaled = preprocessor.transform(X_test)
    y_pred = best_model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))

    print("=" * 60)
    print("FINAL TEST SET PERFORMANCE")
    print("=" * 60)
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_test, y_pred, alpha=0.3, s=10)
    axes[0].plot(
        [y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        'r--', linewidth=2, label='Perfect prediction'
    )
    axes[0].set_xlabel('Actual values')
    axes[0].set_ylabel('Predicted values')
    axes[0].set_title('Predicted vs actual')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual plot')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    # plt.show()

# final_evaluation(results[best_name]['model'], preprocessor, X_test, y_test)
```

## Step 8: Interpret

```python
def interpret_model(model, feature_names):
    """Plot coefficients or impurity-based importances when they exist."""
    if hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
        title = 'Absolute feature coefficients'
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        title = 'Feature importances'
    else:
        print("This model does not expose a simple importance vector.")
        return

    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importances)), importances[indices], align='center')
    plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    # plt.show()

    print("\nTop 5 features:")
    for i in range(min(5, len(importances))):
        idx = indices[i]
        print(f"  {i + 1}. {feature_names[idx]:20s}: {importances[idx]:.4f}")

# interpret_model(results[best_name]['model'], X_train.columns.tolist())
```

## One function that runs the whole path

```python
def complete_ml_workflow():
    """Execute steps 1–8 in sequence."""
    print("STEP 1: LOAD & EXPLORE DATA")
    print("=" * 60)
    df = load_and_explore_data()

    print("\n\nSTEP 2: PREPROCESSING")
    print("=" * 60)
    X, y = preprocess_data(df)

    print("\n\nSTEP 3: FEATURE ENGINEERING")
    print("=" * 60)
    X_eng = create_features(X)

    print("\n\nSTEP 4: TRAIN / VAL / TEST SPLIT")
    print("=" * 60)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_eng, y)

    print("\n\nSTEP 5: BASELINE MODEL")
    print("=" * 60)
    baseline_mse = build_baseline(X_train, y_train, X_val, y_val)

    print("\n\nSTEP 6: MODEL SELECTION & TUNING")
    print("=" * 60)
    results, best_name, preprocessor = model_selection_pipeline(
        X_train, y_train, X_val, y_val
    )

    print("\n\nSTEP 7: FINAL EVALUATION")
    print("=" * 60)
    final_evaluation(results[best_name]['model'], preprocessor, X_test, y_test)

    print("\n\nSTEP 8: MODEL INTERPRETATION")
    print("=" * 60)
    interpret_model(results[best_name]['model'], X_train.columns.tolist())

    return results, best_name

# results, best_model_name = complete_ml_workflow()
```

---

### Deep-Dive Explanation: Why Three Splits and One Final Test Are Non-Negotiable (Read Slowly)

Why not save data by splitting only into Train and Test, tuning your hyperparameters on Test, and reporting the best score? Follow these four steps slowly.

#### Step 1: Every time you pick a winner on a dataset, you fit to its noise
Suppose you evaluate 100 random models on the same validation set. Even if every single model has identical true error, the *luckiest* model on that finite validation set will look noticeably better than average—a statistical bias called **selection optimism** (the winner’s curse).

#### Step 2: That is why Validation gets “spent” during model selection
Once you use the validation set to pick between Ridge, Lasso, and Random Forest (and their hyperparameters), the validation score is no longer an unbiased estimate of future performance—it has been used to make decisions!

#### Step 3: The Test set is your sealed envelope
By locking the test set away until every preprocessing rule, feature, model family, and hyperparameter is frozen, evaluating on Test *once* gives an honest, unbiased measurement of generalization error. If you peek at Test and go back to tweak the model, you have just turned Test into a second Validation set.

#### Step 4: Spot the subtle leakage still hiding in our prototype above!
Look closely at Step 2 (`preprocess_data`) in our script above: it trimmed outliers across `df` *before* splitting, and Step 6 fit `StandardScaler` once on `X_train` *outside* `GridSearchCV`! In Exercise 3 below, wrapping preprocessing inside a scikit-learn `Pipeline` fixes that subtle fold-to-fold leakage completely—an essential upgrade for production-grade ML.

---

## A short checklist

**Data understanding.** Do a real EDA. Look for missing values, outliers, and skewed features.

**Splitting.** Cut train / validation / test *before* you transform anything. Do not look at the test set until the end.

**Preprocessing.** Fit on train, transform on validation and test. Prefer a `Pipeline` so leakage is harder.

**Selection.** Start from a dumb baseline, compare several models, and use cross-validation.

**Tuning.** `GridSearchCV` or `RandomizedSearchCV` on the validation side—never on the test set.

**Final evaluation.** Score the test set once. Report more than one metric (MSE, $$R^2$$, MAE, …).

**Interpretation.** Feature importance, residual plots, and a sentence about *why* the model works.

## Video tham khảo

- [Train, Test, & Validation Sets explained](https://www.youtube.com/watch?v=Zi-0rlM4RDs) — deeplizard, English. **xem trước.** Why train, validation, and test must stay apart — the split step in this lesson’s pipeline.
- [One-Hot, Label, Target and K-Fold Target Encoding, Clearly Explained!!!](https://www.youtube.com/watch?v=589nCGeWG1w) — StatQuest with Josh Starmer, English. The lesson encodes categories with OneHotEncoder; this video explains one-hot before the other encodings.
- [Machine Learning Tutorial Python - 16: Hyper parameter Tuning (GridSearchCV)](https://www.youtube.com/watch?v=HdlDYng8g9s) — codebasics, English. GridSearchCV in scikit-learn for the hyperparameter step of the case study.

## Exercises

**Exercise 1: End-to-end project.** Run the full workflow on the diabetes data (or another regression table). Compare at least three models.

**Exercise 2: Feature engineering.** Invent at least five extra features for California housing and check whether they help.

**Exercise 3: Pipeline automation.** Wrap preprocessing, feature construction, and training in a single scikit-learn `Pipeline`.
