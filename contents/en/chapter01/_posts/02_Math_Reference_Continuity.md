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

This short reference collects the continuity notions you will need when we talk about optimization, loss surfaces, and why some numerical methods behave smoothly while others do not.

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

The quantifiers are the whole story: here $$\delta$$ depends only on $$\varepsilon$$, not on a distinguished point.

### Lipschitz continuity

**Lipschitz continuity** is a quantitative refinement. A function $$f: A \to \mathbb{R}$$ is **Lipschitz** (or **$$L$$-Lipschitz**) on $$A$$ if there exists $$L \geq 0$$ such that for all $$x, y \in A$$,

$$\lvert f(x) - f(y) \rvert \leq L \lvert x - y \rvert$$

The smallest such $$L$$ is the **Lipschitz constant** (or Lipschitz modulus) of $$f$$.

**Useful facts:**

1. **Bounded rate of change.** The function cannot steepen faster than the linear rate $$L$$.
2. **Uniform continuity.** Every Lipschitz function is uniformly continuous (take $$\delta = \varepsilon / L$$ when $$L > 0$$).
3. **Almost-everywhere differentiability.** A Lipschitz function is differentiable almost everywhere, and $$\lvert f'(x) \rvert \leq L$$ wherever the derivative exists.

**Examples:**
- $$f(x) = \lvert x \rvert$$ is 1-Lipschitz on $$\mathbb{R}$$
- $$f(x) = \sin(x)$$ is 1-Lipschitz on $$\mathbb{R}$$ because $$\lvert \cos(x) \rvert \leq 1$$
- $$f(x) = x^2$$ is not Lipschitz on $$\mathbb{R}$$, but it is Lipschitz on any bounded interval

### The hierarchy

The three notions nest:

**continuity $$\subseteq$$ uniform continuity $$\subseteq$$ Lipschitz continuity**

1. **Local versus global.** Continuity is checked pointwise. Uniform continuity is a property of the whole function. Lipschitz continuity is global *and* quantitative.
2. **Choice of $$\delta$$.** Ordinary continuity allows $$\delta = \delta(\varepsilon, c)$$. Uniform continuity requires $$\delta = \delta(\varepsilon)$$ only. Lipschitz continuity gives the explicit rule $$\delta = \varepsilon / L$$.
3. **Control of slope.** Continuity says nothing about how fast $$f$$ can change. Uniform continuity keeps oscillations under control on small scales. Lipschitz continuity gives a linear bound.
4. **Converses fail.** Lipschitz $$\Rightarrow$$ uniformly continuous $$\Rightarrow$$ continuous, but not the other way around in general.

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
- On $$[0, 1]$$: continuous and uniformly continuous, but not Lipschitz (the derivative blows up at 0)
- On $$[a, 1]$$ with $$a > 0$$: Lipschitz with $$L = 1 / (2\sqrt{a})$$
