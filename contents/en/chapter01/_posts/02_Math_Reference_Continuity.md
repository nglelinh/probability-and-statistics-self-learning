---
layout: post
title: 00-01-01 Continuity and Uniform Continuity
chapter: "01"
order: 3
owner: nglelinh
lang: en
categories:
- chapter01
lesson_type: required
---

Imagine training a statistical model by nudging its parameters slightly—say moving a weight by $$0.001$$—and watching the loss suddenly skyrocket by $$10{,}000$$. On paper, the loss function might be completely “continuous,” with no jumps or breaks anywhere. Yet numerically, your optimizer explodes! Why? Because ordinary continuity only promises that small input nudges keep output changes small *near one fixed point*—it makes no promise that the same step size works safely across the entire domain.

This short mathematical reference builds the three-level ladder of smoothness—**continuity**, **uniform continuity**, and **Lipschitz continuity**—that quietly controls whether gradient descent converges, whether probability densities behave cleanly, and why bounded domains save numerical algorithms from blowing up.

---

## Key ideas: Why Do Continuity Notions Matter in Statistics & ML?

Real analysis can feel abstract until you see where each definition shows up in statistical algorithms.

### 1. Real Analysis and $$\varepsilon$$–$$\delta$$ Precision
**Real analysis** replaces hand-waving (“draw it without lifting your pencil”) with exact quantifier logic: *for every target tolerance $$\varepsilon > 0$$, how small must the input perturbation $$\delta > 0$$ be?* Swapping the order of just two quantifiers turns pointwise continuity into uniform continuity.
- **Core questions:** Does a single step size $$\delta$$ work everywhere on the domain? Does a continuous function attain a global minimum on a closed bounded set (Weierstrass Extreme Value Theorem)?
- **Real-world reach:** Guaranteeing that maximum likelihood estimators exist and that numerical integration converges.

### 2. Optimization Theory and Lipschitz Smoothness
When you train logistic regression, ridge regression, or a neural network with gradient descent $$w_{t+1} = w_t - \eta \nabla L(w_t)$$, how large can you safely set the learning rate $$\eta$$? **Lipschitz continuity** gives the exact answer: if the loss (or its gradient) has Lipschitz constant $$L$$, you get a concrete global speed limit on how fast values or slopes can change.
- **Core questions:** How big can a step size $$\eta$$ be without overshooting the valley? How fast does gradient descent converge?
- **Real-world reach:** Tuning learning rates, gradient clipping in deep learning, and Wasserstein GANs (which explicitly enforce 1-Lipschitz critics).

### 3. Stability and Robustness in Statistical Learning
In statistics, data always carry measurement noise. If an estimator or prediction map is **Lipschitz** with a small constant $$L$$, small perturbations in the input data produce small, predictable changes in the output—the mathematical backbone of **adversarial robustness** and **differential privacy**.
- **Core questions:** How sensitive is a model’s prediction to tiny input noise or one altered training point?
- **Real-world reach:** Robust regression (Huber loss), adversarial defense in computer vision, and privacy-preserving data release.

---

## Continuity and uniform continuity

**Continuity** and **uniform continuity** describe how “predictable” a function is. They are closely related, but uniform continuity is the stronger global condition.

### Continuity at a point

A function $$f: A \to \mathbb{R}$$ is **continuous at** $$c \in A$$ if, for every $$\varepsilon > 0$$, there exists $$\delta > 0$$ such that for all $$x \in A$$, if

$$\lvert x - c \rvert < \delta$$

then

$$\lvert f(x) - f(c) \rvert < \varepsilon$$

Intuitively: whatever accuracy $$\varepsilon$$ you want in the output, you can find a small enough window around $$c$$ so that every input in that window maps into the $$\varepsilon$$-window around $$f(c)$$. The important catch is that $$\delta$$ may depend on both $$\varepsilon$$ *and* the point $$c$$.

A function is **continuous on** $$A$$ if it is continuous at every $$c \in A$$.

Polynomials, $$\sin(x)$$, $$\cos(x)$$, and $$e^x$$ are continuous on their usual domains. For example, $$f(x) = x^2 + 3x - 1$$ is continuous on $$\mathbb{R}$$.

### Uniform continuity

**Uniform continuity** asks for one $$\delta$$ that works everywhere at once. A function $$f: A \to \mathbb{R}$$ is **uniformly continuous on** $$A$$ if, for every $$\varepsilon > 0$$, there exists $$\delta > 0$$ such that for all $$x, y \in A$$, if

$$\lvert x - y \rvert < \delta$$

then

$$\lvert f(x) - f(y) \rvert < \varepsilon$$

The quantifiers are the whole story: here $$\delta$$ depends only on $$\varepsilon$$, not on a distinguished point. (By the **Heine–Cantor theorem**, every continuous function on a closed, bounded interval $$[a,b]$$ is automatically uniformly continuous!)

### Lipschitz continuity

**Lipschitz continuity** is a quantitative refinement. A function $$f: A \to \mathbb{R}$$ is **Lipschitz** (or **$$L$$-Lipschitz**) on $$A$$ if there exists $$L \geq 0$$ such that for all $$x, y \in A$$,

$$\lvert f(x) - f(y) \rvert \leq L \lvert x - y \rvert$$

The smallest such $$L$$ is the **Lipschitz constant** (or Lipschitz modulus) of $$f$$. Geometrically, if you draw a double cone of slope $$\pm L$$ centered at any point on the graph, the entire curve must stay outside the vertical interior of that cone.

**Useful facts:**

1. **Bounded rate of change.** The function cannot steepen faster than the linear rate $$L$$.
2. **Uniform continuity.** Every Lipschitz function is uniformly continuous (take $$\delta = \varepsilon / L$$ when $$L > 0$$).
3. **Almost-everywhere differentiability.** A Lipschitz function is differentiable almost everywhere, and $$\lvert f'(x) \rvert \leq L$$ wherever the derivative exists (Rademacher’s theorem).

**Examples:**
- $$f(x) = \lvert x \rvert$$ is 1-Lipschitz on $$\mathbb{R}$$
- $$f(x) = \sin(x)$$ is 1-Lipschitz on $$\mathbb{R}$$ because $$\lvert \cos(x) \rvert \leq 1$$
- $$f(x) = x^2$$ is not Lipschitz on $$\mathbb{R}$$, but it is Lipschitz on any bounded interval $$[-M, M]$$ with $$L = 2M$$

---

### Worked Example 1: Watching $$f(x) = x^2$$ Break Uniform Continuity by Hand

Why is $$f(x) = x^2$$ continuous everywhere on $$\mathbb{R}$$, yet **not** uniformly continuous on $$\mathbb{R}$$? Let’s plug in concrete numbers!

Fix output tolerance $$\varepsilon = 1$$. Suppose someone hands you *any* tiny global step size $$\delta > 0$$—say $$\delta = 0.01$$. Can we find two points $$x, y$$ closer than $$0.01$$ whose squared outputs jump by more than $$1$$?

Pick $$x = 100$$ and $$y = 100 + \frac{\delta}{2} = 100.005$$:
- Input gap: $$\lvert y - x \rvert = 0.005 < 0.01 = \delta$$. ✓
- Output jump:
  $$\lvert y^2 - x^2 \rvert = (y - x)(y + x) = 0.005 \times 200.005 = 1.000025 > 1 = \varepsilon!$$

In general, for *any* $$\delta > 0$$, choosing $$x_n = n$$ and $$y_n = n + \frac{\delta}{2}$$ gives $$\lvert y_n^2 - x_n^2 \rvert = n\delta + \frac{\delta^2}{4} \to \infty$$ as $$n \to \infty$$! Because the parabola gets infinitely steep as $$x \to \infty$$, no single $$\delta$$ can tame it globally on $$\mathbb{R}$$. Restrict it to $$[0, 1]$$, however, and the maximum slope is $$2$$, so $$\delta = \varepsilon/2$$ works everywhere!

---

### Worked Example 2: Computing Exact Lipschitz Constants for ML Loss Functions

In machine learning, loss functions directly inherit these properties. Compare two famous losses on the entire real line $$z \in \mathbb{R}$$:

1. **Logistic loss** $$\ell_{\text{log}}(z) = \log(1 + e^{-z})$$ (used in logistic regression):
   Differentiate:
   $$\ell'_{\text{log}}(z) = \frac{-e^{-z}}{1 + e^{-z}} = -\frac{1}{1 + e^z}.$$
   Since $$e^z > 0$$ for all $$z \in \mathbb{R}$$, we have $$-1 < \ell'_{\text{log}}(z) < 0$$ everywhere! By the Mean Value Theorem,
   $$\lvert \ell_{\text{log}}(z_1) - \ell_{\text{log}}(z_2) \rvert \le 1 \cdot \lvert z_1 - z_2 \rvert.$$
   So logistic loss is **1-Lipschitz on all of $$\mathbb{R}$$**—one single outlier can never cause an unbounded gradient explosion!
2. **Squared-error loss** $$\ell_{\text{sq}}(r) = r^2$$ (used in ordinary least squares):
   Its derivative is $$2r$$, which is **unbounded** on $$\mathbb{R}$$—so squared loss is *not* globally Lipschitz. A single huge residual $$r = 100$$ produces gradient $$200$$, pulling the regression line violently toward the outlier! That exact Lipschitz failure is why statisticians invented **Huber loss** (quadratic near $$0$$, linear with bounded slope $$\delta$$ in the tails).

---

### The hierarchy

The three notions nest cleanly (from strongest to weakest):

**Lipschitz continuity $$\subseteq$$ uniform continuity $$\subseteq$$ ordinary continuity**

| Property | Choice of $$\delta$$ | Control of slope | Example that satisfies this level but not the stronger one |
|----------|----------------------|------------------|------------------------------------------------------------|
| **Ordinary continuity** | $$\delta = \delta(\varepsilon, c)$$ | None | $$f(x) = x^2$$ on $$\mathbb{R}$$ |
| **Uniform continuity** | $$\delta = \delta(\varepsilon)$$ | Controls small-scale oscillations | $$f(x) = \sqrt{x}$$ on $$[0,1]$$ (slope $$\to \infty$$ at $$0$$) |
| **Lipschitz continuity** | $$\delta = \varepsilon / L$$ | Bounded linear slope $$\le L$$ | $$f(x) = \lvert x \rvert$$ on $$\mathbb{R}$$ (not differentiable at $$0$$, yet 1-Lipschitz!) |

### Worked comparisons

**Example 1: $$f(x) = x^2$$**
- On $$\mathbb{R}$$: continuous, not uniformly continuous ($$\lvert f'(x) \rvert = 2\lvert x \rvert$$ is unbounded)
- On $$[0, 1]$$: continuous, uniformly continuous, and Lipschitz with $$L = 2$$

**Example 2: $$f(x) = \sin(x)$$**
- On $$\mathbb{R}$$: continuous, uniformly continuous, and 1-Lipschitz

**Example 3: $$f(x) = \lvert x \rvert$$**
- On $$\mathbb{R}$$: continuous, uniformly continuous, and 1-Lipschitz
- Not differentiable at $$x = 0$$, but still Lipschitz

**Example 4: $$f(x) = \sqrt{x}$$**
- On $$[0, 1]$$: continuous and uniformly continuous, but not Lipschitz (the derivative $$1/(2\sqrt{x})$$ blows up at 0)
- On $$[a, 1]$$ with $$a > 0$$: Lipschitz with $$L = 1 / (2\sqrt{a})$$

---

### Explanation: Why Swapping Two Quantifiers Changes Everything

Students often wonder why real analysis makes such a fuss over moving “$$\forall c$$” in front of or behind “$$\exists \delta$$.” Walk through these four steps slowly to see the geometry behind the symbols.

#### Step 1: Think of continuity as a game between an Adversary and a Defender
The Adversary picks a target accuracy $$\varepsilon > 0$$. In **ordinary continuity**, the Adversary *also* reveals the location $$c$$ first—then the Defender gets to pick a custom window $$\delta(\varepsilon, c)$$ tailored to how steep the function happens to be right at $$c$$.

#### Step 2: In uniform continuity, the Defender must commit BEFORE seeing the location
Now reverse the order: the Adversary picks $$\varepsilon > 0$$, and the Defender must announce **one universal $$\delta(\varepsilon)$$** *before* the Adversary picks the pair of points $$x, y$$. If the function gets arbitrarily steep anywhere (like $$x^2$$ as $$x \to \infty$$ or $$1/x$$ as $$x \to 0^+$$), the Adversary simply moves into the steep zone and defeats any fixed $$\delta$$!

#### Step 3: Why closed bounded intervals $$[a,b]$$ rescue the Defender (Heine–Cantor)
On a compact interval $$[a,b]$$, there is nowhere for the Adversary to run off to infinity and no missing endpoint like $$0$$ in $$(0,1)$$ where a vertical asymptote can hide. Every continuous function on $$[a,b]$$ is automatically uniformly continuous—which is why regularization (restricting weights to a bounded ball $$\|w\| \le R$$) makes loss landscapes dramatically better behaved.

#### Step 4: Why ML prefers Lipschitz over bare uniform continuity
Uniform continuity guarantees a $$\delta(\varepsilon)$$ exists, but it doesn’t tell an algorithm *how big* $$\delta$$ is (look at $$\sqrt{x}$$ on $$[0,1]$$: its slope still explodes near $$0$$!). **Lipschitz continuity** hands you the explicit formula $$\delta = \varepsilon/L$$—and in smooth optimization, a Lipschitz gradient ($$\|\nabla f(x) - \nabla f(y)\| \le L\|x - y\|$$) directly hands you the guaranteed safe learning rate $$\eta \le 1/L$$!

---

## Video tham khảo

- [Limits, L'Hôpital's rule, and epsilon delta definitions | Chapter 7, Essence of calculus](https://www.youtube.com/watch?v=kfF40MiS7zA) — 3Blue1Brown, English. **xem trước.** A visual epsilon–delta account of limits and continuity, before the lesson tightens the definition to uniform continuity.
- [Lecture 17: Uniform Continuity and the Definition of the Derivative](https://www.youtube.com/watch?v=f_sNWn7zujU) — MIT OpenCourseWare, English. MIT’s real-analysis lecture puts uniform continuity next to the definition of the derivative — the lesson’s contrast between a point-dependent δ and one δ for the whole domain.
- [Epsilon-Delta Continuity | This Is What Continuity Actually Means | Visually Explained](https://www.youtube.com/watch?v=2SYsl3SomQc) — Demathify, English. A short picture of why δ may depend on the point c, which is the gap between continuity and uniform continuity in this note.
