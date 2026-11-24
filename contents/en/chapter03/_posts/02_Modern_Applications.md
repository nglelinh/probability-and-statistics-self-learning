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

A weather app says “80% chance of rain,” and you expect it to rain on roughly four out of five days with that forecast. Now ask a modern deep neural network whether a tumor scan is malignant, and it prints `0.99`—yet when you check all scans where it printed `0.99`, only 82% were actually malignant! Something broke between the number coming out of `softmax` and the real conditional probability $$P(Y=1 \mid \hat{p}(X) = 0.99)$$.

Sample spaces, conditional probability, Bayes’ theorem, and independence are not dusty textbook exercises. They are the exact mathematical tools production AI systems misuse whenever they mistake a raw softmax score for a frequency, a next-token likelihood for a factual belief, or an observational conditional $$P(Y \mid X)$$ for a causal effect $$P(Y \mid \mathrm{do}(X))$$. This optional note traces four live battlegrounds of Chapter 03 probability in modern computer science (2021–2023).

---

## Background Knowledge Map: Where Does Chapter 03 Probability Live in Modern AI?

Three probability concepts sit directly behind calibration, LLMs, spam filters, and causal inference.

### 1. Conditional Probability & Calibration (Frequency Matching)
A probabilistic classifier outputs a score $$\hat{p}(X) \in [0,1]$$. **Calibration** demands that conditional on the model’s own forecast $$\hat{p}(X) = p$$, the true empirical frequency of positive outcomes actually equals $$p$$. **Expected Calibration Error (ECE)** measures the average gap between confidence and reality across probability bins.
- **Core questions:** Is an AI model overconfident? Can post-hoc **temperature scaling** or Platt scaling repair its reliability without hurting accuracy?
- **Real-world reach:** Medical diagnosis triage, autonomous driving risk thresholds, credit-default pricing, and weather forecasting.

### 2. Bayes’ Rule, Likelihoods & Conditional Independence
**Bayes’ theorem** links prior beliefs $$P(Y)$$ and generative likelihoods $$P(X \mid Y)$$ to the posterior $$P(Y \mid X)$$. Assuming **conditional independence** yields **Naive Bayes** (still widely deployed for text classification), while autoregressive LLMs factorize joint sequence probabilities via the **chain rule** $$P(x_1,\ldots,x_T) = \prod_t P(x_t \mid x_{<t})$$.
- **Core questions:** Why does a high next-token likelihood $$P(\text{token} \mid \text{prompt})$$ not equal a high probability that the statement is factually true?
- **Real-world reach:** Email spam filtering, document routing, speech recognition, and LLM uncertainty estimation.

### 3. Observational Conditioning vs. Causal Intervention ($$P(Y \mid X)$$ vs. $$P(Y \mid \mathrm{do}(X))$$)
Standard conditional probability $$P(Y \mid X = x)$$ describes what we *see* among individuals who happened to have $$X = x$$—a mixture of true causal effect and hidden **confounders** $$Z$$ (producing **Simpson’s paradox**). **Causal inference** (Pearl’s do-calculus, `DoWhy`) introduces the interventional distribution $$P(Y \mid \mathrm{do}(X = x))$$: what happens when we actively *set* $$X = x$$.
- **Core questions:** When can observational conditionals identify a causal effect via backdoor adjustment?
- **Real-world reach:** Healthcare policy evaluation, marketing attribution, pricing experiments, and algorithmic fairness.

---

## Calibration: when $$P(Y=1\mid X=x)$$ is a frequency

A model is **calibrated** if, among all $$x$$ with predicted probability $$p$$, about a fraction $$p$$ are positive:

$$
P\bigl(Y=1 \mid \hat{p}(X)=p\bigr) \approx p.
$$

**Expected Calibration Error (ECE)** partitions $$\hat{p}$$ into $$M$$ bins $$B_m$$ and computes the weighted gap between bin accuracy and bin confidence:

$$
\text{ECE} = \sum_{m=1}^{M} \frac{\lvert B_m \rvert}{N}\,\bigl\lvert \text{acc}(B_m) - \text{conf}(B_m) \bigr\rvert.
$$

Guo, Pleiss, Sun, and Weinberger (*On Calibration of Modern Neural Networks*, [ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)) showed that deep nets can be highly accurate yet severely overconfident—and that dividing logits by a single scalar **temperature** $$T > 1$$, $$\mathrm{softmax}(z/T)$$, often restores calibration without changing a single top-1 prediction! Minderer et al. (*NeurIPS 2021*; [arXiv:2106.07998](https://arxiv.org/abs/2106.07998)) added an architectural twist: modern Vision Transformers are often better calibrated out of the box than older CNNs. In tabular ML, scikit-learn’s [`CalibratedClassifierCV`](https://scikit-learn.org/stable/modules/calibration.html) (Platt scaling or isotonic regression) performs the same repair.

---

### Worked Example 1: Computing ECE and Temperature Scaling by Hand

Suppose a binary classifier evaluates $$N = 100$$ patients across two confidence bins:
- **Bin 1** (60 patients, avg predicted probability $$\text{conf} = 0.90$$): only 48 are positive ($$\text{acc} = 48/60 = 0.80$$).
- **Bin 2** (40 patients, avg predicted probability $$\text{conf} = 0.60$$): 26 are positive ($$\text{acc} = 26/40 = 0.65$$).

1. **Compute Expected Calibration Error (ECE):**
   $$\text{ECE} = \frac{60}{100}\lvert 0.80 - 0.90 \rvert + \frac{40}{100}\lvert 0.65 - 0.60 \rvert = 0.60(0.10) + 0.40(0.05) = 0.06 + 0.02 = \mathbf{0.08} \text{ (8\% calibration gap!)}.$$
2. **Apply Temperature Scaling ($$T = 2$$) to an overconfident logit vector $$z = (2.0,\; 0.0)$$:**
   - Before scaling ($$T = 1$$): $$\hat{p} = \dfrac{e^{2.0}}{e^{2.0} + e^{0.0}} = \dfrac{7.389}{8.389} \approx \mathbf{88.1\%}$$.
   - After scaling ($$T = 2$$): $$\hat{p}_T = \dfrac{e^{2.0/2}}{e^{2.0/2} + e^{0.0/2}} = \dfrac{e^1}{e^1 + 1} \approx \mathbf{73.1\%}$$.
   Notice the predicted class (Class 1) **did not change at all**, yet the overconfident 88.1% softened cleanly down to 73.1%!

---

## What an LLM means by “probability”

A language model defines a conditional distribution on the next token, $$P(x_t \mid x_{<t})$$, built from the chain rule over a vast discrete vocabulary. After RLHF fine-tuning, however, raw token probabilities become sharply overconfident. Tian, Mitchell, Zhou, Sharma, Rafailov, Yao, Finn, and Manning (*Just Ask for Calibration*, [EMNLP 2023](https://aclanthology.org/2023.emnlp-main.330/)) showed that *verbalized* confidences (“I am 70% sure”) from ChatGPT / GPT-4 / Claude often cut ECE roughly in half compared to token log-probabilities on TriviaQA, SciQ, and TruthfulQA. The Chapter 03 distinction is razor-sharp: $$P(\text{token} \mid \text{prompt})$$ is a **linguistic likelihood**, whereas $$P(\text{correct} \mid \text{answer})$$ is the **epistemic posterior** you actually wanted.

## Naive Bayes that still ships

The spam-filter calculation in Chapter 03 is not a classroom toy. scikit-learn’s [`MultinomialNB`](https://scikit-learn.org/stable/modules/naive_bayes.html) still filters mail and tags documents in production because, under **conditional independence**,

$$
P(x_1,\ldots,x_d\mid y) = \prod_{j=1}^d P(x_j\mid y),
$$

training reduces to counting word frequencies and prediction reduces to summing log-ratios. Production engineers know words are *not* independent given the class—so Naive Bayes probabilities are often pushed near 0 or 1 (poorly calibrated!), yet the **decision boundary** remains remarkably fast and accurate.

## Light causal graphs: $$P(Y\mid X)$$ is not $$P(Y\mid \mathrm{do}(X))$$

Simpson’s paradox is the classic proof that conditioning on an observational group is *not* the same as intervening. Sharma and Kiciman (*DoWhy*, [arXiv:2011.04216](https://arxiv.org/abs/2011.04216); [pywhy.org/dowhy](https://www.pywhy.org/dowhy/)) turn this into a four-step software workflow: **Model** a causal graph, **Identify** whether $$P(Y \mid \mathrm{do}(T))$$ can be computed from observational conditionals, **Estimate** it, and **Refute** it with placebo/confounder stress tests.

---

### Worked Example 2: Simpson’s Paradox by Hand — Why $$P(Y \mid T) \neq P(Y \mid \mathrm{do}(T))$$

A hospital records recovery rates ($$Y = 1$$) for a new drug ($$T = 1$$) versus standard care ($$T = 0$$), split by initial patient severity ($$Z$$):

| Patient Severity $$Z$$ | New Drug ($$T = 1$$) Recovery | Standard Care ($$T = 0$$) Recovery | Winner inside subgroup |
|------------------------|-------------------------------|------------------------------------|------------------------|
| **Mild** ($$Z = 0$$) | $$90 / 100 = \mathbf{90\%}$$ | $$320 / 400 = \mathbf{80\%}$$ | **New Drug (+10%)** |
| **Severe** ($$Z = 1$$) | $$200 / 400 = \mathbf{50\%}$$ | $$40 / 100 = \mathbf{40\%}$$ | **New Drug (+10%)** |
| **Pooled Total** | $$290 / 500 = \mathbf{58\%}$$ | $$360 / 500 = \mathbf{72\%}$$ | **Standard Care (+14%!)** |

Look at that jaw-dropping reversal!
- **Observational conditional:** $$P(Y=1 \mid T=1) = 58\% < 72\% = P(Y=1 \mid T=0)$$. Naive Bayes-table reading says the new drug is harmful!
- **Why it flipped:** Doctors gave the new drug mostly to **Severe** patients (400 vs. 100), so severity $$Z$$ confounded treatment $$T$$!
- **Backdoor interventional formula** (with $$P(Z=\text{Mild}) = 500/1000 = 0.5$$ and $$P(Z=\text{Severe}) = 0.5$$):
  $$P(Y=1 \mid \mathrm{do}(T=1)) = 0.5(90\%) + 0.5(50\%) = \mathbf{70\%},$$
  $$P(Y=1 \mid \mathrm{do}(T=0)) = 0.5(80\%) + 0.5(40\%) = \mathbf{60\%}!$$
  Giving the new drug actually **raises** recovery by **+10 percentage points**! Conditioning ($$58\%$$ vs. $$72\%$$) answered the wrong question; causal adjustment ($$70\%$$ vs. $$60\%$$) answered the policy question.

---

### Deep-Dive Explanation: Four Probability Questions to Ask Any AI System (Read Slowly)

Before trusting a number printed between `0.00` and `1.00` by a modern ML or LLM system, walk through these four Chapter 03 questions slowly.

#### Step 1: Is this number a calibrated frequency or just a ranking score?
Check a reliability diagram or ECE on a held-out set. A model with 0.95 AUC can still be wildly overconfident (ECE = 0.15) until you apply temperature scaling or isotonic calibration.

#### Step 2: What exact event is inside $$P(\,\cdot \mid \cdot\,)$$?
For an LLM, $$P(\text{"Paris"} \mid \text{prompt})$$ measures how likely human text continues with the word “Paris,” *not* necessarily $$P(\text{answer is factually true})$$. Never confuse a generative likelihood with a truth posterior.

#### Step 3: Which conditional independence assumptions were silently baked in?
Naive Bayes assumes features are independent given the label; many RAG/retrieval pipelines assume chunks are independent. Violating independence may leave top-1 classification intact while severely distorting predicted probabilities.

#### Step 4: Are we predicting (observational) or deciding (interventional)?
If you only want to *predict* who will default or recover under existing policies, $$P(Y \mid X)$$ is right. The moment you plan to *change* a policy, assign a drug, or offer a discount, you need $$P(Y \mid \mathrm{do}(X))$$—either via a randomized A/B test or an explicit causal graph.

---

## What to carry forward

When a paper says “we report ECE,” “we asked the model for a verbal probability,” “we shipped MultinomialNB,” or “we identified the effect in DoWhy,” it is speaking Chapter 03. Random-variable lessons come next; they refine *how* we number events, not the meaning of $$P(\,\cdot\mid\,\cdot\,)$$.

## Video tham khảo

- [Probability Calibration : Data Science Concepts](https://www.youtube.com/watch?v=AunotauS5yI) — ritvikmath, English. **xem trước.** Probability calibration: the frequency inside each bin should match the predicted score — the ECE and reliability diagram in this note.
- [Naive Bayes, Clearly Explained!!!](https://www.youtube.com/watch?v=O2L2Uv9pdDA) — StatQuest with Josh Starmer, English. Naive Bayes multiplies conditional probabilities — the independence assumption this note says is still shipping in mail filters.
- [Simpson's Paradox](https://www.youtube.com/watch?v=ebEkn-BiW5k) — minutephysics, English. Simpson’s paradox: conditioning on a group can reverse the conclusion — why the note separates P(Y|X) from P(Y|do(X)).

## Sources

1. C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On Calibration of Modern Neural Networks,” ICML 2017. [PMLR](https://proceedings.mlr.press/v70/guo17a.html)
2. M. Minderer et al., “Revisiting the Calibration of Modern Neural Networks,” NeurIPS 2021. [abstract](https://proceedings.neurips.cc/paper/2021/hash/8420d359404024567b5aefda1231af24-Abstract.html) · [arXiv:2106.07998](https://arxiv.org/abs/2106.07998)
3. K. Tian et al., “Just Ask for Calibration,” EMNLP 2023. [ACL Anthology](https://aclanthology.org/2023.emnlp-main.330/)
4. scikit-learn, “Probability calibration” and “Naive Bayes.” [calibration](https://scikit-learn.org/stable/modules/calibration.html) · [NB](https://scikit-learn.org/stable/modules/naive_bayes.html)
5. A. Sharma and E. Kiciman, “DoWhy: An End-to-End Library for Causal Inference,” 2020. [arXiv:2011.04216](https://arxiv.org/abs/2011.04216) · [docs](https://www.pywhy.org/dowhy/)
