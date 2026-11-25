---
layout: post
title: "08-03-00 Optional: Modern ANOVA in Product Experiments"
chapter: "08"
order: 3
owner: nglelinh
lang: en
categories:
- chapter08
lesson_type: optional
---

A growth team designs four new checkout buttons—Control, Green, Large, and Express—and runs all $$\binom{4}{2} = 6$$ pairwise $$t$$-tests at $$\alpha = 0.05$. One pair flashes $$p = 0.03$$, so they celebrate and ship it—unaware that under a completely true null hypothesis, six unadjusted comparisons carry roughly a **26% false-alarm rate**! Meanwhile, down the hall, another team launches a new ranking model *and* a new snippet UI in two separate A/B tests—completely missing the fact that the new ranking model only lifts clicks *when paired* with the new snippet layout!

This optional lesson takes classical Chapter 08 **one-way ANOVA**, **two-way factorial layouts**, and **Tukey’s HSD** directly into modern multi-variant product experiments, showing how the classical variance decomposition $$SS_{\mathrm{Total}} = SS_{\mathrm{Between}} + SS_{\mathrm{Within}}$$ evolves naturally into **variance-component mixed models** when users or queries are clustered (Gelman, 2005; Yu et al., 2022).

---

## Key ideas: How ANOVA Powers Multi-Variant Experimentation

Three classical ideas from experimental design anchor modern A/B/n and factorial platforms.

### 1. Omnibus Variance Decomposition (One-Way ANOVA & $$F$$-Test)
Instead of spraying unadjusted pairwise $$t$$-tests across $$k$$ variants, **one-way ANOVA** compares **between-group signal** ($$MS_{\mathrm{Between}}$$) against **within-group noise** ($$MS_{\mathrm{Within}}$$) via a single omnibus $$F$$-ratio—acting as a clean statistical gate before post-hoc pairwise inspection (**Tukey HSD** or **Benjamini–Hochberg FDR**).
- **Core questions:** Do the $$k$$ variants differ more than random sampling noise predicts? Which post-hoc rule fits a small pre-registered family versus 50 exploratory slices?
- **Real-world reach:** A/B/n landing-page tests, multi-arm pricing tests, and clinical dose-response comparisons.

### 2. Factorial Designs & Interaction Effects (Two-Way ANOVA)
When two product changes launch together (Factor A × Factor B), testing them in a single **2×2 factorial layout** reuses 100% of your traffic for *both* main effects while explicitly measuring the **interaction term** $$(\alpha\beta)_{ij}$$—revealing synergy or conflict that separate A/B tests are mathematically blind to.
- **Core questions:** Does Feature A help only when Feature B is turned on? How do Type II vs. Type III sums of squares handle unbalanced cells?
- **Real-world reach:** Search ranking + UI snippet co-launches, industrial design of experiments (DoE), and combination drug therapies.

### 3. Hierarchical ANOVA & Variance Components (Mixed Models)
Following Andrew Gelman (2005), every row of an ANOVA table can be viewed as a **variance component** ($$\sigma^2_{\text{treatment}}, \sigma^2_{\text{user}}, \sigma^2_{\varepsilon}$$). When each user contributes dozens of repeated sessions, moving from plain `f_oneway` to a **random-intercept mixed model** fixes the denominator so clustered noise is never mistaken for treatment signal.
- **Core questions:** How much variation comes from user-to-user differences versus within-user randomness?
- **Real-world reach:** Repeated-measures UX studies, longitudinal app telemetry, and multi-market geo experiments.

---

## A/B/n is one-way ANOVA in a product coat

A platform testing $$k \ge 3$$ checkout buttons is fitting the one-way layout

$$Y_{ij} = \mu + \tau_i + \varepsilon_{ij}, \qquad \varepsilon_{ij} \sim \mathcal{N}(0,\sigma^2),$$

testing $$H_0: \tau_1 = \cdots = \tau_k = 0$$ via the omnibus $$F$$-statistic

$$F = \frac{MS_{\mathrm{Between}}}{MS_{\mathrm{Within}}}.$$

Pairwise $$t$$-tests without family correction inflate Type I error across all $$\binom{k}{2}$$ pairs. In industrial practice, assignment happens at the **user level**—analyzing raw session events as if they were i.i.d. rows understates $$MS_{\mathrm{Within}}$$ and manufactures fake significance.

---

### Worked Example 1: Computing a Complete One-Way ANOVA $$F$$-Test by Hand

Suppose three checkout variants each have $$n = 3$$ users with order values ($):
- **Variant A:** $$\{4,\; 5,\; 6\}$$ (mean $$\bar{y}_A = 5$$)
- **Variant B:** $$\{6,\; 7,\; 8\}$$ (mean $$\bar{y}_B = 7$$)
- **Variant C:** $$\{8,\; 9,\; 10\}$$ (mean $$\bar{y}_C = 9$$)

Grand mean across all $$N = 9$$ users: $$\bar{y} = 7$$. Compute every row of the ANOVA table directly:
1. **Between-group Sum of Squares ($$SS_B$$):**
   $$SS_B = 3(5-7)^2 + 3(7-7)^2 + 3(9-7)^2 = 12 + 0 + 12 = \mathbf{24}, \qquad df_B = k - 1 = 2 \implies MS_B = \frac{24}{2} = \mathbf{12.0}.$$
2. **Within-group Sum of Squares ($$SS_W$$):**
   Each group has sum of squared deviations $$(-1)^2 + 0^2 + 1^2 = 2$$, so
   $$SS_W = 2 + 2 + 2 = \mathbf{6}, \qquad df_W = N - k = 6 \implies MS_W = \frac{6}{6} = \mathbf{1.0}.$$
3. **Omnibus $$F$$-statistic:**
   $$F = \frac{MS_B}{MS_W} = \frac{12.0}{1.0} = \mathbf{12.0} \qquad (F_{2,6} \text{ critical value at } \alpha=0.05 \text{ is } 5.14 \implies p = \mathbf{0.008}!).$$

Verify the decomposition: $$SS_{\mathrm{Total}} = 24 + 6 = 30$$. Because between-group variance ($$12.0$$) is **12× larger** than within-group noise ($$1.0$$), the omnibus gate passes decisively—now you may safely run **Tukey’s HSD** to see which specific pairs differ!

---

```python
import numpy as np
from scipy import stats

def oneway_anova(groups):
    """
    Classical one-way ANOVA for a list of 1-d arrays.

    Returns:
        F, p_value, ss_between, ss_within
    """
    all_y = np.concatenate(groups)
    grand = all_y.mean()
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_w = sum(((g - g.mean()) ** 2).sum() for g in groups)
    df_b = len(groups) - 1
    df_w = len(all_y) - len(groups)
    F = (ss_b / df_b) / (ss_w / df_w)
    p = stats.f.sf(F, df_b, df_w)
    return F, p, ss_b, ss_w
```

Tukey’s HSD remains the default post-hoc when every pair is interesting in a balanced design; when exploring twenty ad-hoc segments after seeing the data, switch to Chapter 06 **Benjamini–Hochberg FDR**.

## Factorial launches are two-way ANOVA

Product teams rarely change one factor at a time. Crossing a new ranking model (Old / New) with a new snippet layout (Control / Variant) yields the two-way layout

$$Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \varepsilon_{ijk}.$$

The **interaction term** $$(\alpha\beta)_{ij}$$ captures whether two features amplify or cancel each other. Running two isolated A/B tests hides that interaction completely and wastes half your traffic.

---

### Worked Example 2: Spotting a Pure Interaction in a 2×2 Factorial Launch

Suppose average user clicks in a 2×2 launch (Model × Layout) have the following cell means:

| Ranking Model | Old Layout | New Layout | Row Mean |
|---------------|------------|------------|----------|
| **Old Model** | $$10$$ | $$10$$ | $$10$$ |
| **New Model** | $$10$$ | **$$18$$** | $$14$$ |
| **Column Mean** | $$10$$ | $$14$$ | Grand Mean $$= 12$$ |

Look what happens if you run two separate one-factor A/B tests:
- Test New Model *only* on Old Layout: $$10$$ vs. $$10$$ → **0 lift! Team kills the New Model!**
- Test New Layout *only* on Old Model: $$10$$ vs. $$10$$ → **0 lift! Team kills the New Layout!**

Both isolated A/B tests fail completely—yet inside a **2×2 factorial ANOVA**, the interaction contrast $$(18 - 10) - (10 - 10) = \mathbf{+8}$$ clicks lights up immediately, revealing that the New Model and New Layout unlock **+80% lift** together!

---

```python
import statsmodels.api as sm
from statsmodels.formula.api import ols

def two_way_anova_table(df):
    """
    Two-way ANOVA with interaction via OLS.

    df columns: y (float), model (A/B), layout (A/B)
    """
    fit = ols("y ~ C(model) * C(layout)", data=df).fit()
    return sm.stats.anova_lm(fit, typ=2)
```

Type II versus Type III sums of squares matter once cells are unbalanced—pre-register your choice rather than shopping after seeing $$p$$-values.

## From the ANOVA table to mixed models

Gelman (2005) reframed ANOVA as a hierarchical **variance-component model**:

$$Y_{ij} = \mu + \tau_{a[i]} + u_{b[i]} + \varepsilon_i$$

where batch $$b$$ (users, queries, markets, days) is a random effect with variance $$\sigma_u^2$$. As Yu et al. (2022) emphasize, running plain ANOVA on repeated observations from the same user ignores clustering. Follow three crisp rules:
1. One aggregated metric per assigned user → classical one-way/two-way ANOVA (or OLS) is valid.
2. Multiple repeated sessions per user → fit a **mixed model** with a user random intercept.
3. Read the variance components $$\sigma_u^2$$ vs. $$\sigma_\varepsilon^2$$ to understand how much noise is user-level vs. session-level.

```python
import numpy as np
import statsmodels.formula.api as smf

def mixed_treatment_test(df):
    """
    Random intercept for user; fixed treatment.

    df columns: y, treatment, user
    """
    model = smf.mixedlm("y ~ treatment", df, groups=df["user"])
    return model.fit()
```

---

### Explanation: Why ANOVA, Regression, and Mixed Models Are One Unified Machine

Students often learn $$t$$-tests, ANOVA, linear regression, and mixed models in separate chapters as if they were unrelated recipes. Walk through these four steps to see that they are all the *exact same* linear model.

#### Step 1: One-way ANOVA IS linear regression with dummy indicators
Coding $$k$$ variants with $$k-1$$ binary indicators $$D_2,\ldots,D_k$$ turns $$Y_{ij} = \mu + \tau_i + \varepsilon_{ij}$$ into ordinary OLS! The overall regression $$F$$-test is numerically identical to `f_oneway`.

#### Step 2: Two-way ANOVA simply adds product interaction columns
Adding $$C(\text{model}) \times C(\text{layout})$$ creates one extra product column whose regression coefficient tests whether the two features combine additively or synergistically.

#### Step 3: Why clustering breaks the classical $$F$$ denominator
Classical ANOVA divides by $$MS_{\mathrm{Within}}$$ assuming all $$N$$ rows are independent. If 100 users each contribute 50 sessions ($$N = 5{,}000$$), treating those 5,000 rows as independent fools the $$F$$-test into thinking you have $$50\times$$ more independent units than you actually do!

#### Step 4: Mixed models repair the denominator by splitting variance into two layers
Adding $$u_i \sim \mathcal{N}(0, \sigma_u^2)$$ separates within-user noise ($$\sigma_\varepsilon^2$$) from between-user noise ($$\sigma_u^2$$), restoring honest degrees of freedom and valid $$p$$-values on the treatment effect.

---

## Post-hoc choices that match the decision

| Decision after a significant omnibus | Tool | Why |
|---|---|---|
| Which of a few pre-registered variants wins? | Tukey HSD or a single planned contrast | Family is small and closed |
| Which of many slices moved? | Benjamini–Hochberg (Chapter 06) | Family is large and exploratory |
| Did anything move at all, for a ship/no-ship gate? | Omnibus $$F$$ or a pre-registered OEC | Avoids pairwise fishing |
| Are users or queries a source of noise? | Mixed model / variance components | Classical $$MS_{\mathrm{Within}}$$ is wrong |

A common 2022–2026 pattern on experimentation platforms is: omnibus test on the overall evaluation criterion, Tukey or a pre-registered contrast on the variants, and FDR on the fifty diagnostic metrics that nobody pre-registered. Mixing those three jobs into one unadjusted table is how A/B/n acquires a reputation for false wins.

## Challenges and extensions

ANOVA assumes constant variance and, in the classical derivation, Gaussian errors. Conversion rates near 0 or 1, and revenue with a point mass at zero, violate both; a GLM or a transformation is then the Chapter 07 companion, not a reason to abandon the decomposition. Unbalanced cells, missing variants, and sequential peeking (Chapter 06) all change the finite-sample Type I error of the $$F$$-test. Mixed models with few groups estimate $$\sigma_u^2$$ badly; aggregating may be safer.

Extensions include permutation ANOVA when the Gaussian assumption is implausible, aligned-rank transforms in human-computer interaction studies, and Bayesian hierarchical ANOVA when many weak factors share a prior—the setting Gelman originally had in mind.

## Video tham khảo

- [Full Factorial Design (DoE - Design of Experiments) Simply explained](https://www.youtube.com/watch?v=cHoYAyYHzB0) — numiqo, English. **xem trước.** A two-factor factorial — the design this note uses for a multi-factor launch instead of a pile of t-tests.
- [Follow Up ANOVA – Tukey’s HSD Post Hoc Test (12-5)](https://www.youtube.com/watch?v=P1j2CkOjWtM) — Research By Design, English. Tukey HSD after a significant F — one of the two post-hoc choices the note sets beside FDR.
- [The power of mixed-effects models | Longitudinal 3](https://www.youtube.com/watch?v=MOyjGnjAols) — Very Normal, English. When measurements nest in users or queries, a random variance component replaces a plain F table — the mixed-model section.

## Exercises

**Exercise 1 (conceptual).** A team runs four button colours and six pairwise $$t$$-tests, then ships the pair with the smallest $$p$$. Which two Chapter 08/06 mistakes did they combine? What single pre-registered contrast would have been cleaner?

**Exercise 2 (computational).** Simulate a balanced one-way design with $$k = 4$$ groups, $$n = 80$$, three means equal and one lifted by $$0.3\sigma$$. Compare the omnibus rejection rate and the number of false pairwise wins under unadjusted $$t$$-tests versus Tukey. Repeat with unequal $$n$$.

**Exercise 3 (computational).** Build a two-way factorial in which the new ranking model helps only with the new layout (pure interaction). Show that two separate A/B tests can both be non-significant while `two_way_anova_table` flags the interaction.

**Exercise 4 (open).** Take a public multi-group experiment or a UX study with repeated measures. Fit (i) one-way ANOVA on user-level aggregates, (ii) ANOVA on raw events, and (iii) a mixed model with a user intercept. Reconcile the three $$p$$-values with the variance-component view.

## References

Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015). Fitting linear mixed-effects models using lme4. *Journal of Statistical Software, 67*(1), 1–48. [https://doi.org/10.18637/jss.v067.i01](https://doi.org/10.18637/jss.v067.i01)

Gelman, A. (2005). Analysis of variance—why it is more important than ever. *Statistical Science, 20*(1), 1–31. [https://doi.org/10.1214/088342305000000016](https://doi.org/10.1214/088342305000000016)

Reluga, K., Ye, T., & Zhao, Q. (2022). A unified analysis of regression adjustment in randomized experiments. [https://arxiv.org/abs/2210.04360](https://arxiv.org/abs/2210.04360)

Tukey, J. W. (1949). Comparing individual means in the analysis of variance. *Biometrics, 5*(2), 99–114. The still-cited post-hoc procedure taught in the required lesson.

Yu, Z., Grieco, S. F., Holmes, T. C., Xu, X., et al. (2022). Beyond t-test and ANOVA: Applications of mixed-effects models for more rigorous statistical analysis in neuroscience research. *Neuron, 110*(1), 21–35. [https://doi.org/10.1016/j.neuron.2021.10.030](https://doi.org/10.1016/j.neuron.2021.10.030)
