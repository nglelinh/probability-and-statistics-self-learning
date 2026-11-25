---
layout: post
title: 01-01-00 Introduction to Computational Statistics and Data-Driven Thinking
chapter: "01"
order: 1
owner: nglelinh
lang: en
categories:
- chapter01
lesson_type: required
---

In 1946, mathematician Stanislaw Ulam was recovering from a serious illness and passing the hours playing solitaire. Halfway through a game, he wondered: *what is the exact probability that a well-shuffled deck of Canfield solitaire comes out completely solvable?* He tried attacking it with classical combinatorics—counting permutations, tracking blocked cards—and quickly realized the algebra was a hopeless tangle. Then a thought struck him: why not simply deal 100 hands, count how many win, and let a fast electronic computer deal 100,000 hands in seconds? Together with John von Neumann, Ulam turned that bedside insight into the **Monte Carlo method**—the birth of modern computational statistics.

The goal of this opening lesson is not to hand you a stack of formulas to memorize. It is to shift how you think about randomness: away from *“can I solve this integral on paper?”* and toward *“can I simulate this system honestly on a computer and measure how accurate my simulation is?”* By the end, you will contrast classical (analytic) statistics with computational statistics, calculate Monte Carlo uncertainty by hand, and see the **Law of Large Numbers** turn raw repetition into rigorous statistical knowledge.

---

## Key ideas: What Fields Sit Behind Computational Statistics?

Before diving into code, it helps to see the three pillars of probability and statistics that make computational thinking work.

### 1. Probability Theory and Limit Theorems
**Probability theory** models random experiments using random variables, distributions, expectations $$\mu = \mathbb{E}[X]$$, and variances $$\sigma^2 = \operatorname{Var}(X)$$. Two cornerstone theorems bridge single random draws and reliable long-run behavior: the **Law of Large Numbers (LLN)** (sample averages converge to the true mean) and the **Central Limit Theorem (CLT)** (sample averages fluctuate around the true mean like a bell curve of width $$\sigma/\sqrt{n}$$).
- **Core questions:** How fast does an empirical average approach the true expectation? How many trials $$n$$ do we need to guarantee a given margin of error?
- **Real-world reach:** Insurance pricing, reliability engineering, clinical-trial sample-size planning, and polling accuracy.

### 2. Computational Statistics and Resampling (Monte Carlo & Bootstrap)
When real-world models involve correlated portfolios, non-normal tails, or complex nonlinear rules, closed-form formulas vanish. **Computational statistics** replaces paper-and-pencil derivations with algorithmic simulation: **Monte Carlo** simulates from a known probabilistic model, while the **Bootstrap** resamples directly from observed data to estimate uncertainty without assuming a textbook bell curve.
- **Core questions:** How do we estimate intractable probabilities, integrals, and confidence intervals using random sampling? How can we reduce simulation variance?
- **Real-world reach:** Financial risk modeling (Value-at-Risk), Bayesian MCMC inference, A/B testing with skewed revenue metrics, and physics particle transport.

### 3. Statistical Inference and Data-Driven Modeling
**Statistical inference** flips probability backwards: instead of knowing the mechanism and predicting the data, we observe finite noisy data and infer the unknown mechanism behind it—quantifying our uncertainty honestly rather than pretending a point estimate is exact.
- **Core questions:** Given a finite sample, what range of parameter values is plausible? How do we separate genuine signal from sampling noise?
- **Real-world reach:** Medical diagnostics, econometric forecasting, quality control, and machine-learning evaluation.

---

## Why computational statistics?

Suppose you want the probability that the sum of ten fair dice exceeds 40. A classical solution uses combinatorics or a normal approximation (the Central Limit Theorem). That can be elegant when the math cooperates. Change the problem slightly—say, the probability of a large joint loss in a portfolio of 50 correlated stocks with heavy-tailed jumps—and a closed-form expression is usually out of reach.

**Computational statistics** is what you do then. Instead of hunting for a perfect equation that describes the world, you *imitate* the world: generate synthetic copies of the experiment, repeat them thousands or millions of times, and count. Abstract probability becomes a concrete programming problem, and many questions that defeat analytic methods become transparent.

## Analytic versus computational statistics

| Aspect | Classical (Analytic) Statistics | Computational Statistics |
|--------|----------------------------------|--------------------------|
| **Core engine** | Closed-form theorems & parametric assumptions | Repeated random simulation & resampling |
| **Classic CI for a mean** | $$\bar{x} \pm 1.96 \frac{\sigma}{\sqrt{n}}$$ (relies on normality/CLT) | Bootstrap empirical percentiles from resampled data |
| **Strength** | Exact, instantaneous, reveals algebraic scaling laws | Handles messy, correlated, non-normal models easily |
| **Main cost** | Breaks down when assumptions fail | CPU time & Monte Carlo error of order $$O(1/\sqrt{n})$$ |

Neither replaces the other: great statisticians use analytic theory to understand *how* an estimator scales and computational methods to evaluate it when exact formulas do not exist.

## Simulation and the law of large numbers

The engine is **Monte Carlo simulation**. If you want to know an unknown probability $$p = P(A)$$, define an indicator random variable $$I_j = 1$$ if event $$A$$ happens on trial $$j$$ and $$I_j = 0$$ otherwise. After $$n$$ independent trials, your estimate is simply the empirical frequency $$\hat{p}_n = \frac{1}{n}\sum_{j=1}^n I_j$$.

What makes this mathematically legitimate is the **Law of Large Numbers**. As the number of trials $$n$$ grows, the sample mean converges in probability to the true expectation:

$$\lim_{n \to \infty} P\left( \left| \bar{X}_n - \mu \right| < \varepsilon \right) = 1$$

On a computer, $$n$$ is simply the number of loop iterations you can afford—and crucially, the **standard error** of your Monte Carlo estimate shrinks as

$$\operatorname{SE}(\hat{p}_n) = \sqrt{\frac{p(1-p)}{n}} \le \frac{1}{2\sqrt{n}}.$$

---

### Worked Example 1: The Birthday Problem by Hand vs. Monte Carlo Uncertainty

In a room of $$k$$ people (ignoring leap years), what is the probability that at least two share a birthday? Let’s compute the exact probability for $$k = 3$$ by hand, inspect $$k = 23$$, and calculate how accurate a 10,000-trial simulation actually is.

1. **Exact complement for $$k = 3$$ people:**
   - Person 1 can have any of 365 birthdays: $$\frac{365}{365} = 1$$.
   - Person 2 must avoid Person 1’s day: $$\frac{364}{365}$$.
   - Person 3 must avoid both: $$\frac{363}{365}$$.
   - Probability of *at least one match*:
     $$P(\text{match}, k=3) = 1 - \frac{365 \cdot 364 \cdot 363}{365^3} = 1 - 0.991796 = 0.008204 \approx 0.82\%.$$
2. **Why $$k = 23$$ crosses 50%:** With $$23$$ people, there are $$\binom{23}{2} = \frac{23 \times 22}{2} = 253$$ distinct pairs of people who can collide! Multiplying the complement gives
   $$P(\text{match}, k=23) = 1 - \prod_{i=0}^{22}\left(1 - \frac{i}{365}\right) \approx 0.5073.$$
3. **How noisy is a $$n = 10{,}000$$ Monte Carlo run?**
   Using $$p \approx 0.5073$$, the Monte Carlo standard error is
   $$\operatorname{SE}(\hat{p}_{10{,}000}) = \sqrt{\frac{0.5073 \times 0.4927}{10{,}000}} \approx \frac{0.50}{100} = 0.0050.$$
   So a 95% Monte Carlo confidence band is $$0.5073 \pm 1.96(0.0050) \approx [0.4975,\; 0.5171]$$! That explains why re-running the Python script below gives slightly different numbers like `0.5042` or `0.5118`—and why you should always report Monte Carlo error alongside a simulated probability.

---

## Example: the birthday problem

The computational solution requires no factorials at all—it just *replays the room* 10,000 times:

```python
import numpy as np

def simulate_birthday_problem(n_people, n_simulations=10000):
    """
    Estimate the probability that at least two people in a group
    of n_people share a birthday.

    Args:
        n_people (int): Number of people in the room.
        n_simulations (int): Number of independent rooms to simulate.

    Returns:
        float: Estimated probability (share of rooms with a collision).
    """
    count_duplicate = 0

    for _ in range(n_simulations):
        # Draw birthdays uniformly from day 1 through 365
        birthdays = np.random.randint(1, 366, size=n_people)

        if len(np.unique(birthdays)) < n_people:
            count_duplicate += 1

    probability = count_duplicate / n_simulations
    return probability

# Try the famous n = 23 case
n_people = 23
prob = simulate_birthday_problem(n_people)
print(f"Estimated collision probability with {n_people} people: {prob:.4f}")
```

There are no factorials in that function. It just replays the physical story. The estimate lands near `0.507`, which surprises almost everyone because human intuition thinks of 22 pairs involving *yourself*, forgetting the 253 total pairs in the room.

---

### Worked Example 2: The $$1/\sqrt{n}$$ Cost of Precision (Estimating $$\pi$$ by Dart-Throwing)

Suppose you throw $$n$$ random darts uniformly onto the square $$[-1,1] \times [-1,1]$$ (area $$4$$) and check whether each dart lands inside the unit circle $$x^2 + y^2 \le 1$$ (area $$\pi$$). Each dart hits with probability $$p = \pi/4 \approx 0.7854$$, so your estimator is $$\hat{\pi}_n = 4\hat{p}_n$$.

Let’s calculate the exact standard error of $$\hat{\pi}_n$$:
$$\operatorname{SE}(\hat{\pi}_n) = 4\sqrt{\frac{p(1-p)}{n}} = 4\sqrt{\frac{0.7854 \times 0.2146}{n}} \approx \frac{1.642}{\sqrt{n}}.$$

Plug in three simulation budgets:
- $$n = 100$$ darts: $$\operatorname{SE}(\hat{\pi}) \approx 0.164$$ (barely 1 reliable digit: $$3.1 \pm 0.3$$).
- $$n = 10{,}000$$ darts: $$\operatorname{SE}(\hat{\pi}) \approx 0.0164$$ (about 2 digits: $$3.14 \pm 0.03$$).
- $$n = 1{,}000{,}000$$ darts: $$\operatorname{SE}(\hat{\pi}) \approx 0.00164$$ (about 3 digits: $$3.142 \pm 0.003$$).

**Crucial scaling law:** To gain **one extra decimal digit** of accuracy (cutting error by $$10\times$$), naive Monte Carlo requires **$$100\times$$ as many simulations**! That universal $$1/\sqrt{n}$$ tax is the reason entire subfields of statistics (importance sampling, control variates, quasi-Monte Carlo) exist to shrink the numerator $$\sigma$$.

---

### Explanation: How Random Repetition Becomes Mathematical Certainty

Why does tossing virtual coins in a Python `for`-loop qualify as serious mathematics? Walk through these four steps slowly.

#### Step 1: Rewrite any probability or integral as an expectation
Every probability $$P(A)$$ is simply the expected value $$\mathbb{E}[\mathbb{1}_A]$$ of a $$0/1$$ indicator variable, and every integral $$\int_0^1 g(x)\,dx$$ is simply $$\mathbb{E}[g(U)]$$ where $$U \sim \text{Uniform}(0,1)$$. Once a problem is written as an expectation $$\mathbb{E}[X]$$, you can estimate it by generating i.i.d. copies $$X_1,\ldots,X_n$$ and averaging.

#### Step 2: Check Chebyshev’s inequality — why variance kills error
For any random variable with mean $$\mu$$ and finite variance $$\sigma^2$$, the sample mean $$\bar{X}_n$$ has the exact same mean $$\mu$$ and a shrunk variance $$\operatorname{Var}(\bar{X}_n) = \sigma^2 / n$$. Chebyshev’s inequality guarantees
$$P(|\bar{X}_n - \mu| \ge \varepsilon) \le \frac{\sigma^2}{n\varepsilon^2} \xrightarrow{n\to\infty} 0.$$
That two-line proof *is* the Weak Law of Large Numbers: averaging cancels independent random noise at rate $$\sigma/\sqrt{n}$$.

#### Step 3: Why dimension $$d$$ disappears from the convergence rate
If you approximate a $$d$$-dimensional integral using a deterministic grid of 10 points per axis, you need $$10^d$$ grid points—impossible when $$d = 50$$ stocks! Monte Carlo’s convergence rate $$\sigma/\sqrt{n}$$ has **no $$d$$ in the exponent**. That dimension-independence is the superpower that lets computational statistics conquer high-dimensional finance, physics, and Bayesian inference.

#### Step 4: Respect the two hidden failure modes
Monte Carlo is not magic. First, it can only be as honest as your **data-generating model** (garbage assumptions in → precise garbage out). Second, for **rare events** (say $$p = 10^{-8}$$), naive simulation might see zero hits in a million runs—demanding smarter techniques like importance sampling rather than brute-force loops.

---

## Video tham khảo

- [Law of Large Numbers - Explained and Visualized](https://www.youtube.com/watch?v=MntX3zWNWec) — Jeremy Blitz-Jones, English. **xem trước.** The sample mean settles on the expectation as the number of trials grows — the law of large numbers this lesson uses to justify Monte Carlo.
- [6. Monte Carlo Simulation](https://www.youtube.com/watch?v=OgO1gpXSUzU) — MIT OpenCourseWare, English. John Guttag explains why Ulam simulated solitaire instead of counting, then estimates an unknown quantity from random samples — the same move as the dice example and the pi exercise.
- [Check your intuition: The birthday problem - David Knuffke](https://www.youtube.com/watch?v=KtT_cgMzHx8) — TED-Ed, English. The birthday problem from the lesson: intuition underestimates the chance of a shared birthday, which a repeated experiment puts near one half for 23 people.

## Exercises

**Exercise 1: Biased coin.** Write a function that tosses a biased coin ($$p = 0.6$$ of heads) 100 times. Repeat the experiment 1000 times and plot a histogram of the number of heads.

**Exercise 2: Estimating $$\pi$$.** Use Monte Carlo to estimate $$\pi$$. Hint: throw points uniformly into the unit square and count how many land inside the inscribed disk.

**Exercise 3: Monty Hall.** Simulate the Monty Hall problem and show that switching roughly doubles the chance of winning relative to staying.

A later optional lesson, [01-11-00 Applications and recent developments]({% multilang_post_url contents/chapter01/11_Modern_Applications %}), maps Monte Carlo, cross-validation, and model selection onto conformal prediction and LLM evaluation (2023–2024).
