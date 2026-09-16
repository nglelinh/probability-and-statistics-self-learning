---
layout: post
title: 03-02-00 Applications and recent developments
chapter: "03"
order: 2
owner: nglelinh
lang: en
categories:
- chapter03
lesson_type: optional
---

Sample spaces, conditional probability, Bayes, and independence in this chapter are the same objects that production systems misuse when they treat a softmax score as a chance, a token likelihood as a belief, or a correlation as a cause. This optional note keeps the required derivations intact and only shows four places those objects appear in 2021–2023 computer-science practice.

## Calibration: when $$P(Y=1\mid X=x)$$ is a frequency

A model is **calibrated** if, among all $$x$$ with predicted probability $$p$$, about a fraction $$p$$ are positive:

$$
P\bigl(Y=1 \mid \hat{p}(X)=p\bigr) \approx p.
$$

Expected calibration error (ECE) bins $$\hat{p}$$ and compares the bin-wise frequency to the bin-wise mean score—the rare-disease example from the required lesson in histogram form. Guo, Pleiss, Sun, and Weinberger (*On Calibration of Modern Neural Networks*, [ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)) showed that deep nets can be accurate and badly calibrated; a one-parameter **temperature** $$T$$ on the logits, $$\mathrm{softmax}(z/T)$$, often fixes the reliability diagram. Minderer, Djolonga, Romijnders, Hubis, Zhai, Houlsby, Tran, and Lucic (*Revisiting the Calibration of Modern Neural Networks*, [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/8420d359404024567b5aefda1231af24-Abstract.html); [arXiv:2106.07998](https://arxiv.org/abs/2106.07998)) found the story is architecture-dependent: recent Vision Transformers can be *better* calibrated than older CNNs. sklearn’s [`CalibratedClassifierCV`](https://scikit-learn.org/stable/modules/calibration.html) (Platt scaling or isotonic regression on a hold-out) is the tabular version. Conditional probability is not a vibes score; it is a claim about long-run frequency.

## What an LLM means by “probability”

A language model defines a conditional distribution on the next token, $$P(x_t\mid x_{<t})$$, which is Bayes’ rule plus the chain rule on a huge discrete space. After RLHF, that conditional is a poor confidence: Tian, Mitchell, Zhou, Sharma, Rafailov, Yao, Finn, and Manning (*Just Ask for Calibration*, [EMNLP 2023](https://aclanthology.org/2023.emnlp-main.330/)) show that *verbalized* confidences (“I am 70% sure”) from ChatGPT / GPT-4 / Claude are often better calibrated than the model’s own token probabilities, cutting ECE by about half on TriviaQA, SciQ, and TruthfulQA. The Chapter 03 distinction is exact: $$P(\text{token}\mid\text{prompt})$$ is a likelihood in a generative story; $$P(\text{correct}\mid\text{answer})$$ is the posterior you wanted. They coincide only if you specified the right experiment.

## Naive Bayes that still ships

The spam-filter calculation in the required lesson is not a toy. sklearn’s [`MultinomialNB`](https://scikit-learn.org/stable/modules/naive_bayes.html) still filters mail and tags documents because, under the **conditional independence** assumption

$$
P(x_1,\ldots,x_d\mid y) = \prod_{j=1}^d P(x_j\mid y),
$$

the posterior $$P(y\mid x)$$ is a few counts and a multiply. Production use is honest about the lie: tokens are *not* independent given the class, yet the decision boundary can still work. When a Transformer beats Naive Bayes, it is because it models $$P(x_j\mid x_{-j},y)$$ instead of $$P(x_j\mid y)$$. That is the independence section of this chapter, not a new branch of mathematics.

## Light causal graphs: $$P(Y\mid X)$$ is not $$P(Y\mid \mathrm{do}(X))$$

Simpson’s paradox in the exercises is the warning that conditioning is not intervening. Sharma and Kiciman (*DoWhy*, [arXiv:2011.04216](https://arxiv.org/abs/2011.04216); [pywhy.org/dowhy](https://www.pywhy.org/dowhy/)) turn that warning into software: you draw a graph of assumed mechanisms, *identify* whether $$P(Y\mid\mathrm{do}(T))$$ is a function of observational conditionals, *estimate*, then *refute* (placebo, unobserved confounding checks). The library does not replace Bayes’ theorem. It refuses to pretend that $$P(\text{recovery}\mid\text{drug})$$ computed on hospital data is the effect of assigning the drug. A one-edge picture—treatment $$T$$, outcome $$Y$$, confounder $$Z$$—is enough to see why the required identity $$P(A\mid B)=P(A\cap B)/P(B)$$ is the wrong button for policy.

## What to carry forward

When a paper says “we report ECE,” “we asked the model for a verbal probability,” “we shipped MultinomialNB,” or “we identified the effect in DoWhy,” it is speaking this chapter. Random-variable lessons come next; they refine *how* we number events, not the meaning of $$P(\,\cdot\mid\,\cdot\,)$$.

## Sources

1. C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On Calibration of Modern Neural Networks,” ICML 2017. [PMLR](https://proceedings.mlr.press/v70/guo17a.html)
2. M. Minderer et al., “Revisiting the Calibration of Modern Neural Networks,” NeurIPS 2021. [abstract](https://proceedings.neurips.cc/paper/2021/hash/8420d359404024567b5aefda1231af24-Abstract.html) · [arXiv:2106.07998](https://arxiv.org/abs/2106.07998)
3. K. Tian et al., “Just Ask for Calibration,” EMNLP 2023. [ACL Anthology](https://aclanthology.org/2023.emnlp-main.330/)
4. scikit-learn, “Probability calibration” and “Naive Bayes.” [calibration](https://scikit-learn.org/stable/modules/calibration.html) · [NB](https://scikit-learn.org/stable/modules/naive_bayes.html)
5. A. Sharma and E. Kiciman, “DoWhy: An End-to-End Library for Causal Inference,” 2020. [arXiv:2011.04216](https://arxiv.org/abs/2011.04216) · [docs](https://www.pywhy.org/dowhy/)
