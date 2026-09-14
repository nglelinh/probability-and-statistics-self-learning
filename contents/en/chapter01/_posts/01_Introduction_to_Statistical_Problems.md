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

The goal of this opening lesson is not to hand you a stack of formulas. It is to change how you approach a statistical question: away from “solve it on paper” and toward “simulate it on a computer.” By the end you should be able to contrast classical (analytic) statistics with computational statistics, and to *see* the law of large numbers in an experiment rather than only in a proof.

---

## Why computational statistics?

Suppose you want the probability that the sum of ten fair dice exceeds 40. A classical solution uses combinatorics or a normal approximation (the central limit theorem). That can be elegant if the math cooperates. Change the problem slightly—say, the probability of a large loss in a portfolio of 50 correlated stocks—and a closed-form expression is usually out of reach.

**Computational statistics** is what you do then. Instead of hunting for a perfect equation that describes the world, you *imitate* the world: generate synthetic copies of the experiment, repeat them thousands or millions of times, and count. Abstract probability becomes a programming problem, and many questions that defeat analytic methods become straightforward.

## Analytic versus computational statistics

The tools differ. Analytic statistics leans on theorems and simplifying assumptions (for example, that data are exactly normal) to produce theoretically exact answers. A textbook confidence interval for a mean is

$$\bar{x} \pm 1.96 \frac{\sigma}{\sqrt{n}}$$

Computational statistics is willing to spend CPU time on repeated random trials. Instead of assuming normality, you can bootstrap the sample and read a confidence interval from the empirical distribution of the statistic. The computational route is typically more flexible, less tied to a named parametric family, and more intuitive for people who already think in code.

## Simulation and the law of large numbers

The engine is **Monte Carlo simulation**. The idea is almost too simple: if you want a probability, run the experiment many times and record the frequency.

What makes this legitimate is the **law of large numbers**. As the number of trials $$n$$ grows, the sample mean converges in probability to the true expectation:

$$\lim_{n \to \infty} P\left( \left| \bar{X}_n - \mu \right| < \varepsilon \right) = 1$$

On a computer, $$n$$ is just the number of loop iterations you can afford.

## Example: the birthday problem

A classic illustration: *in a room of $$k$$ people, what is the probability that at least two share a birthday?*

The analytic solution uses the complement and a bit of combinatorics. The computational solution is to *play the room* many times:

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

There are no factorials in that function. It just replays the physical story. The estimate is about 0.507, which usually surprises people: intuition systematically underweights collisions.

## Exercises

**Exercise 1: Biased coin.** Write a function that tosses a biased coin ($$p = 0.6$$ of heads) 100 times. Repeat the experiment 1000 times and plot a histogram of the number of heads.

**Exercise 2: Estimating $$\pi$$.** Use Monte Carlo to estimate $$\pi$$. Hint: throw points uniformly into the unit square and count how many land inside the inscribed disk.

**Exercise 3: Monty Hall.** Simulate the Monty Hall problem and show that switching roughly doubles the chance of winning relative to staying.
