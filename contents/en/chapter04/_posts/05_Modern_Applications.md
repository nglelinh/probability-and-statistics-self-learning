---
layout: post
title: 04-05-00 Applications and recent developments
chapter: "04"
order: 5
owner: nglelinh
lang: en
categories:
- chapter04
lesson_type: optional
---

Open a warehouse inventory dashboard at Amazon, a cloud-server latency monitor, or a frontier discrete diffusion model for text, and you might expect to find exotic mathematics completely divorced from undergraduate probability. Look under the hood, and you will find the exact named random variables from Chapter 04—**Bernoulli**, **Poisson**, **Negative Binomial**, **Categorical**, and **Gaussian**—acting as the probabilistic “output heads” of modern deep learning systems! Neural networks do not replace probability distributions; they simply learn rich mappings from raw context into the parameters $$(\lambda, \alpha, \pi)$$ of those exact families.

This optional note does not re-derive PMFs. Instead, it shows how three Chapter 04 families power modern 2020–2024 systems: **overdispersed count forecasting (DeepAR)**, **continuous-time event arrivals (Poisson & Neural Hawkes processes)**, and **discrete diffusion generative models (SEDD)**.

---

## Background Knowledge Map: Where Do Named Distributions Power Modern Systems?

Three pillars connect Chapter 04 random variables directly to production forecasting and generative AI.

### 1. Discrete Count Families & Overdispersion (Poisson vs. Negative Binomial)
A Poisson variable enforces the rigid identity $$\operatorname{Var}(N) = \mathbb{E}[N] = \lambda$$. Real-world retail orders, ad clicks, and support tickets are almost always **overdispersed** ($$\operatorname{Var} > \mathbb{E}$$): quiet days alternate with viral spikes. Adding a Gamma-distributed random rate turns Poisson into the **Negative Binomial** family, allowing mean and variance to decouple cleanly.
- **Core questions:** Why does a Gaussian loss fail on sparse nonnegative counts? How do we predict a full probability distribution of demand rather than a single average?
- **Real-world reach:** Retail inventory optimization (DeepAR), insurance claim frequency, and hospital ER admission forecasting.

### 2. Temporal Point Processes (Poisson & Hawkes Processes)
A **Poisson process** extends the Poisson random variable into continuous time: arrivals in disjoint windows are independent and memoryless. When one event triggers a cascade of follow-up events (earthquakes and aftershocks, viral tweets, high-frequency trades), the arrival rate $$\lambda(t)$$ becomes history-dependent—yielding **Hawkes** and **Neural Temporal Point Processes**.
- **Core questions:** How do we model the exact timestamps $$t_1 < t_2 < \cdots$$ of bursty asynchronous events?
- **Real-world reach:** Cloud traffic auto-scaling, algorithmic trading order books, seismology, and social-media cascade modeling.

### 3. High-Dimensional Discrete PMFs & Score-Ratio Diffusion
Every autoregressive LLM outputs a massive **Categorical PMF** over 50,000+ vocabulary tokens at each step. Newer **discrete diffusion models** (D3PM, SEDD) avoid left-to-right generation by learning probability *ratios* $$p(y)/p(x)$$ across discrete token states—enabling arbitrary infilling and parallel decoding.
- **Core questions:** How do we sample from complex discrete distributions when continuous gradients $$\nabla_x \log p(x)$$ do not exist?
- **Real-world reach:** Discrete text/DNA/protein generation, code infilling, and combinatorial molecular design.

---

## Count models: when the mean is not enough

Clicks, tickets, and items sold are nonnegative integers. A Poisson random variable $$N \sim \mathrm{Poisson}(\lambda)$$ satisfies

$$
P(N=k)=\frac{\lambda^k e^{-\lambda}}{k!},\qquad \mathbb{E}[N]=\mathrm{Var}(N)=\lambda.
$$

Real counts are routinely **overdispersed** ($$\mathrm{Var} > \mathbb{E}$$). The **Negative Binomial** distribution keeps a mean $$\mu$$ while adding a dispersion parameter $$\alpha > 0$$ so that $$\operatorname{Var}(N) = \mu + \alpha\mu^2$$. In production, scikit-learn’s [`PoissonRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html) and `TweedieRegressor` provide linear count models, while Salinas, Flunkert, Gasthaus, and Januschowski (*DeepAR*, [*International Journal of Forecasting* 36, 2020](https://doi.org/10.1016/j.ijforecast.2019.07.001); [arXiv:1704.04110](https://arxiv.org/abs/1704.04110)) attach a Negative Binomial (or Gaussian) likelihood head to a shared recurrent neural network so thousands of retail products borrow statistical strength from each other. The output is not a single point guess—it is a full predictive PMF you can sample to set safety-stock quantiles.

---

### Worked Example 1: Poisson vs. Negative Binomial on Bursty Retail Demand

Suppose an online store averages $$\mu = 4$$ orders per day for a jacket. Compare what a **Poisson** model versus a **Negative Binomial** model (with dispersion $$\alpha = 0.5$$) predicts about daily variability:

1. **Poisson($$\lambda = 4$$):**
   - Mean $$= 4$$, Variance $$= 4$$, Standard deviation $$= \sqrt{4} = \mathbf{2.00}$$.
   - Rough “mean + 2 SD” safety-stock threshold: $$4 + 2(2.00) = \mathbf{8}$$ jackets.
2. **Negative Binomial ($$\mu = 4,\; \alpha = 0.5$$):**
   - Mean $$= 4$$, Variance $$= \mu + \alpha\mu^2 = 4 + 0.5(16) = \mathbf{12}$$, Standard deviation $$= \sqrt{12} \approx \mathbf{3.46}$$.
   - Rough “mean + 2 SD” safety-stock threshold: $$4 + 2(3.46) \approx \mathbf{11}$$ jackets!

**Why this matters:** Both distributions share the *exact same* average demand (4 jackets/day), yet the Poisson model underestimates daily volatility by over 40%—causing frequent stockouts during sudden spikes! Choosing the right Chapter 04 likelihood head directly translates into real warehouse revenue.

---

```python
import numpy as np
from sklearn.linear_model import PoissonRegressor

rng = np.random.default_rng(0)
x = rng.normal(size=(400, 2))
lam = np.exp(0.4 * x[:, 0] - 0.2 * x[:, 1])
y = rng.poisson(lam)
model = PoissonRegressor(alpha=1e-4).fit(x, y)
print("mean predicted λ:", model.predict(x).mean(), "mean y:", y.mean())
```

If the printed means disagree badly, the log-link Poisson is the wrong family—not “the neural net failed.”

## Poisson processes in systems

A Poisson *random variable* counts events in a fixed window. A **Poisson process** lets that window slide: increments on disjoint intervals are independent, and the count on an interval of length $$t$$ is $$\mathrm{Poisson}(\lambda t)$$. Request arrivals, crash dumps, and trades are routinely modeled this way before anyone mentions a neural net. When past events *excite* the future (a popular post triggers more posts), the intensity $$\lambda(t)$$ becomes history-dependent: a **Hawkes process**, and then a **neural temporal point process**. Mei and Eisner (*The Neural Hawkes Process*, [NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/6463c88460bd63bbe256e495c63aa40b-Abstract.html)) replace the additive kernel with a continuous-time LSTM, and Shchur, Türkmen, Januschowski, and Günnemann (*IJCAI 2021*; [arXiv:2104.03528](https://arxiv.org/abs/2104.03528)) survey modern transformer and intensity-free point processes.

---

### Worked Example 2: Memoryless Poisson Arrivals vs. a Self-Exciting Hawkes Spike

Suppose an API server receives requests at baseline rate $$\mu = 3$$ requests/second.
1. **Homogeneous Poisson Process** over a $$t = 2$$-second window:
   - Total expected arrivals: $$\lambda t = 3 \times 2 = 6$$.
   - Probability of zero requests in 2 seconds: $$P(N(2) = 0) = e^{-6} \approx \mathbf{0.00248}$$ (0.25%).
   - Memorylessness: even if 5 requests arrived in the previous second, the rate for the next second stays frozen at $$3$$ req/s.
2. **Hawkes Process** $$\lambda(t) = \mu + \sum_{t_i < t} \alpha e^{-\beta(t - t_i)}$$ with jump $$\alpha = 2$$ and decay $$\beta = 1\text{ s}^{-1}$$:
   - The instant a request arrives at $$t_1 = 0$$, the intensity jumps from $$3$$ to $$\lambda(0^+) = 3 + 2 = \mathbf{5}$$ req/s!
   - After $$1$$ quiet second, the excitement decays exponentially to $$\lambda(1) = 3 + 2e^{-1} \approx \mathbf{3.74}$$ req/s.
   - This exact self-exciting term explains why server crashes and viral retweets arrive in bursts rather than smooth Poisson intervals!

---

## Discrete generative models as PMFs

An autoregressive language model is a huge categorical random variable at each site: $$X_t \mid X_{<t} \sim \mathrm{Categorical}(\pi_\theta(X_{<t}))$$. Diffusion models that conquered *images* relied on a continuous Gaussian score $$\nabla_x \log p_t(x)$$, which does not exist on discrete tokens. Lou, Meng, and Ermon (*Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution*, [ICML 2024](https://proceedings.mlr.press/v235/lou24a.html); [arXiv:2310.16834](https://arxiv.org/abs/2310.16834); code [Score-Entropy-Discrete-Diffusion](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion)) define a **score entropy** that matches discrete probability *ratios* $$p(y)/p(x)$$ to build **SEDD**—competitive with GPT-2-scale autoregressive models while supporting flexible infilling. Austin et al.’s D3PM ([NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/958c530554f2820e0d14e14d964d48d9-Abstract.html)) is the earlier discrete-diffusion formulation still cited beside SEDD.

The sampling-distribution lesson then explains why a Monte Carlo estimate of perplexity or coverage needs a stated $$n$$: you are looking at $$\bar{X}_n$$, not at one lucky generation.

---

### Deep-Dive Explanation: How a Neural Network Outputs a Probability Distribution (Read Slowly)

Students often ask: *“How does a deep learning model like DeepAR or SEDD connect to Chapter 04 PMFs?”* Follow these four steps slowly.

#### Step 1: Separate the feature extractor from the probability family
A neural network (RNN, Transformer, MLP) takes messy inputs (history, text, context) and produces a hidden vector $$h_t$$. By itself, $$h_t$$ has no probabilistic meaning until you map it to the parameters of a chosen distribution.

#### Step 2: Enforce parameter constraints with activation functions
Count rates $$\lambda$$ must be strictly positive ($$\lambda = \exp(w^\top h_t)$$ or $$\mathrm{softplus}$$); categorical probabilities $$\pi$$ must be nonnegative and sum to 1 ($$\pi = \mathrm{softmax}(W h_t)$$); Gaussian standard deviations $$\sigma$$ must be positive.

#### Step 3: Train by minimizing the negative log-PMF (Maximum Likelihood)
Instead of squared error $$(y - \hat{y})^2$$ (which secretly assumes a constant-variance Gaussian!), you plug the observed count or token directly into $$-\log P(Y = y \mid \text{parameters}(h_t))$$ and backpropagate.

#### Step 4: Sample from the output distribution at decision time
Once trained, you don’t just read the mean! You draw 1,000 Monte Carlo trajectories from the Negative Binomial or Categorical head to compute 95% inventory quantiles, tail crash probabilities, or diverse text generations.

---

## What to carry forward

When a paper says “negative-binomial likelihood,” “neural Hawkes intensity,” or “score entropy on tokens,” it is speaking Chapter 04. Estimation and tests come later; they act on these same named families.

## Video tham khảo

- [Regression with Count Data: Poisson and Negative Binomial](https://www.youtube.com/watch?v=uGKnoAw-PFQ) — Matthew E. Clapham, English. **xem trước.** Poisson and negative-binomial regression when the variance exceeds the mean — the count-model section of this note.
- [L22.2 Definition of the Poisson Process](https://www.youtube.com/watch?v=D_EGYzqmapc) — MIT OpenCourseWare, English. MIT’s definition of a Poisson process: independent increments and a Poisson count on an interval — the logs and request section.
- [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution – Paper Explained](https://www.youtube.com/watch?v=K_9wQ6LZNpI) — AI Coffee Break with Letitia, English. The SEDD paper this note cites: a discrete generative model that matches probability ratios, a PMF over tokens rather than a point forecast.

## Sources

1. D. Salinas, V. Flunkert, J. Gasthaus, and T. Januschowski, “DeepAR: Probabilistic forecasting with autoregressive recurrent networks,” *Int. J. Forecasting* 36(3), 2020. [DOI](https://doi.org/10.1016/j.ijforecast.2019.07.001) · [arXiv:1704.04110](https://arxiv.org/abs/1704.04110)
2. scikit-learn, `PoissonRegressor`. [docs](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html)
3. H. Mei and J. Eisner, “The Neural Hawkes Process,” NeurIPS 2017. [abstract](https://proceedings.neurips.cc/paper/2017/hash/6463c88460bd63bbe256e495c63aa40b-Abstract.html)
4. O. Shchur, A. C. Türkmen, T. Januschowski, and S. Günnemann, “Neural Temporal Point Processes: A Review,” IJCAI 2021. [arXiv:2104.03528](https://arxiv.org/abs/2104.03528)
5. A. Lou, C. Meng, and S. Ermon, “Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution,” ICML 2024. [PMLR](https://proceedings.mlr.press/v235/lou24a.html) · [arXiv:2310.16834](https://arxiv.org/abs/2310.16834)
6. J. Austin et al., “Structured Denoising Diffusion Models in Discrete State-Spaces,” NeurIPS 2021. [abstract](https://proceedings.neurips.cc/paper/2021/hash/958c530554f2820e0d14e14d964d48d9-Abstract.html)
