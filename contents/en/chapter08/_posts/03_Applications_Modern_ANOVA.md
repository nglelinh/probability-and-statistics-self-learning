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

This optional lesson does not rewrite one-way ANOVA, the $$F$$-test, or Tukey’s HSD. It shows how those objects are the analysis of multi-variant product experiments, why a two-way layout is the right model for a factorial launch, and how the ANOVA table becomes a mixed model once users or queries are nested—the form Gelman argued for in 2005 and that 2022–2026 applied papers still recommend.

You should finish able to write an A/B/n experiment as a one-way ANOVA, add a second factor without running a handful of unadjusted $$t$$-tests, choose between Tukey and FDR for the post-hoc step, and say when the $$F$$-table should be replaced by a random-effect variance component.

The two required Chapter 08 lessons, plus the multiple-testing ideas from Chapter 06, are the prerequisites. The decomposition

$$SS_{\mathrm{Total}} = SS_{\mathrm{Between}} + SS_{\mathrm{Within}}$$

is assumed known.

---

## A/B/n is one-way ANOVA in a product coat

A platform that ships three checkout buttons is not running two A/B tests. It is comparing $$k \ge 3$$ means. The model is exactly the required one-way layout

$$Y_{ij} = \mu + \tau_i + \varepsilon_{ij}, \qquad \varepsilon_{ij} \sim \mathcal{N}(0,\sigma^2),$$

and the omnibus question is $$H_0: \tau_1 = \cdots = \tau_k = 0$$. The $$F$$-statistic

$$F = \frac{MS_{\mathrm{Between}}}{MS_{\mathrm{Within}}}$$

is the right first gate. Pairwise $$t$$-tests without a family correction are the mistake the required lesson already names: the Type I error grows as $$\binom{k}{2}$$.

What changed in industrial use is the *metric* and the *assignment unit*, not the algebra. Conversion is Bernoulli, so the Gaussian ANOVA is an approximation justified by the same CLT the interval chapter used, or one switches to a GLM (Chapter 07) and reads a likelihood-ratio test that is an ANOVA in the exponential family. Assignment is at the user level; analysing events as if they were i.i.d. rows understates $$MS_{\mathrm{Within}}$$ and manufactures significance.

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

Tukey’s HSD, already computed in the second required lesson, remains the default post-hoc when every pair is interesting and the design is balanced. When a team inspects twenty segments *after* seeing the data, they are no longer in Tukey’s world; they are in the Chapter 06 FDR world.

## Factorial launches are two-way ANOVA

Product teams rarely change one factor. A launch may cross a new ranking model (old / new) with a new snippet layout (control / variant). That is a two-way layout

$$Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \varepsilon_{ijk}.$$

The interaction $$(\alpha\beta)_{ij}$$ is the scientifically interesting term: the new model may help only when the snippet changes. Running two separate A/B tests hides that term and wastes sample. The ANOVA table splits $$SS$$ into main effects and interaction; each row is an $$F$$-test with its own degrees of freedom.

Reluga, Ye, and Zhao (2022), writing about regression adjustment, remind us that the one-way difference-in-means *is* the ANOVA estimator, and that adding covariates (ANCOVA) or interactions (ANHECOVA) changes the variance, not the causal identification, when treatment is randomised. A factorial experiment is the same reminder with two treatment factors instead of one treatment and a covariate.

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

Type II versus Type III sums of squares matter once the design is unbalanced—the usual state of an online experiment that did not hit its sample-size target on every cell. Report the type you pre-registered; do not shop.

## From the ANOVA table to mixed models

Gelman (2005) argued that ANOVA is *more* important once we treat every row of the table as a variance component, including the ones classically called fixed. A hierarchical model

$$Y_{ij} = \mu + \tau_{a[i]} + u_{b[i]} + \varepsilon_i$$

is an ANOVA in which batch $$b$$ (users, queries, markets, days) is a random effect. The $$F$$-table is then a summary of estimated $$\sigma^2$$s, not a ritual of stars.

That view is what modern applied papers operationalise. Yu et al. (2022) show how default ANOVA in laboratory science ignores the animal or cell as a cluster; the same paragraph applies to a user who contributes 40 sessions. Bates et al. (2015) is the computational reference. For a product experiment the practical translation is:

1. If you have one aggregated outcome per assigned unit, classical one-way or two-way ANOVA (or the equivalent OLS) is enough.
2. If you must keep repeated observations, put a random intercept on the assignment unit and read the treatment test from the mixed model, not from `scipy.stats.f_oneway` on the raw events.
3. Use the ANOVA *decomposition* to explain the mixed-model output: how much variance sits between users versus within users.

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

The fixed-effect $$p$$-value on `treatment` is then comparable to the ANOVA $$F$$-test, but the denominator uses the correct clustering.

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
