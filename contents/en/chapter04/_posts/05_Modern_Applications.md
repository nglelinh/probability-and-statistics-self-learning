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

Bernoulli, Poisson, negative binomial, Gaussians, and sampling distributions in this chapter are the named families that still sit under demand forecasts, server logs, and discrete generative models. This optional note does not re-derive PMFs. It shows three places those families are used, as-is or lightly extended, in 2020–2024 systems work.

## Count models: when the mean is not enough

Clicks, tickets, and items sold are nonnegative integers. A Poisson random variable $$N\sim\mathrm{Poisson}(\lambda)$$ has

$$
P(N=k)=\frac{\lambda^k e^{-\lambda}}{k!},\qquad \mathbb{E}[N]=\mathrm{Var}(N)=\lambda.
$$

Real counts are often **overdispersed** ($$\mathrm{Var}>\mathbb{E}$$). The negative binomial keeps a Poisson mean and adds a dispersion parameter—the same PMF as in the discrete-distributions lesson, now as a GLM likelihood. scikit-learn’s [`PoissonRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html) and `TweedieRegressor` are the linear production form. Salinas, Flunkert, Gasthaus, and Januschowski (*DeepAR*, [*International Journal of Forecasting* 36, 2020](https://doi.org/10.1016/j.ijforecast.2019.07.001); [arXiv:1704.04110](https://arxiv.org/abs/1704.04110)) put a negative-binomial (or Gaussian) head on a shared recurrent net so that thousands of related series borrow strength. The output is not a point forecast; it is a full predictive PMF you can sample for inventory. Chapter 04 is the head; the network is just a flexible map into $$(\lambda,\alpha)$$.

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

A Poisson *random variable* counts events in a fixed window. A **Poisson process** lets that window slide: increments on disjoint intervals are independent, and the count on an interval of length $$t$$ is $$\mathrm{Poisson}(\lambda t)$$. Request arrivals, crash dumps, and trades are routinely modeled this way before anyone mentions a neural net. When past events *excite* the future (a popular post triggers more posts), the intensity $$\lambda(t)$$ becomes history-dependent: a Hawkes process, then a neural temporal point process. Mei and Eisner (*The Neural Hawkes Process*, [NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/6463c88460bd63bbe256e495c63aa40b-Abstract.html)) replace the additive kernel with a continuous-time LSTM. Shchur, Türkmen, Januschowski, and Günnemann (*Neural Temporal Point Processes: A Review*, [IJCAI 2021](https://www.ijcai.org/proceedings/2021/600); [arXiv:2104.03528](https://arxiv.org/abs/2104.03528)) is the survey still cited for 2022–2025 follow-ups (transformers, intensity-free models). The Chapter 04 object you need is the Poisson count and the idea that a rate $$\lambda$$ can depend on time; the rest is a parameterized intensity.

## Discrete generative models as PMFs

An autoregressive language model is a huge categorical random variable at each site: $$X_t\mid X_{<t}\sim\mathrm{Categorical}(\pi_\theta(X_{<t}))$$. Diffusion models that won on *images* used a Gaussian score $$\nabla_x\log p_t(x)$$. Discrete tokens do not have that gradient. Lou, Meng, and Ermon (*Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution*, [ICML 2024](https://proceedings.mlr.press/v235/lou24a.html); [arXiv:2310.16834](https://arxiv.org/abs/2310.16834); code [Score-Entropy-Discrete-Diffusion](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion)) define a **score entropy** that matches probability *ratios* $$p(y)/p(x)$$ on a discrete state space and build SEDD. On language benchmarks, SEDD is competitive with GPT-2-scale autoregressive models and permits infilling, not only left-to-right sampling. You do not need stochastic-process theory to see the Chapter 04 claim: the data are a discrete RV; the model is a PMF (or a ratio of PMFs); sampling is a computational stand-in for “draw from $$p$$.” Austin et al.’s D3PM ([NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/958c530554f2820e0d14e14d964d48d9-Abstract.html)) is the earlier discrete-diffusion formulation still cited beside SEDD.

The sampling-distribution lesson then explains why a Monte Carlo estimate of perplexity or of a coverage rate needs a stated $$n$$: you are looking at $$\bar{X}_n$$, not at one lucky generation.

## What to carry forward

When a paper says “negative-binomial likelihood,” “neural Hawkes intensity,” or “score entropy on tokens,” it is speaking this chapter. Estimation and tests come later; they act on these same named families.

## Sources

1. D. Salinas, V. Flunkert, J. Gasthaus, and T. Januschowski, “DeepAR: Probabilistic forecasting with autoregressive recurrent networks,” *Int. J. Forecasting* 36(3), 2020. [DOI](https://doi.org/10.1016/j.ijforecast.2019.07.001) · [arXiv:1704.04110](https://arxiv.org/abs/1704.04110)
2. scikit-learn, `PoissonRegressor`. [docs](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html)
3. H. Mei and J. Eisner, “The Neural Hawkes Process,” NeurIPS 2017. [abstract](https://proceedings.neurips.cc/paper/2017/hash/6463c88460bd63bbe256e495c63aa40b-Abstract.html)
4. O. Shchur, A. C. Türkmen, T. Januschowski, and S. Günnemann, “Neural Temporal Point Processes: A Review,” IJCAI 2021. [arXiv:2104.03528](https://arxiv.org/abs/2104.03528)
5. A. Lou, C. Meng, and S. Ermon, “Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution,” ICML 2024. [PMLR](https://proceedings.mlr.press/v235/lou24a.html) · [arXiv:2310.16834](https://arxiv.org/abs/2310.16834)
6. J. Austin et al., “Structured Denoising Diffusion Models in Discrete State-Spaces,” NeurIPS 2021. [abstract](https://proceedings.neurips.cc/paper/2021/hash/958c530554f2820e0d14e14d964d48d9-Abstract.html)
