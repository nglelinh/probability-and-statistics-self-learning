---
layout: post
title: 00-01-02 Derivatives and Multivariable Calculus
chapter: "01"
order: 4
owner: nglelinh
lang: en
categories:
- chapter01
lesson_type: required
---

Stand blindfolded on a foggy mountain range and try to hike down to the lowest valley. You cannot see the whole landscape at once—all you can feel under your boots is the **local tilt** of the ground (which way is steepest downhill) and its **local curvature** (whether you are in a round bowl, on a narrow ridge, or at a mountain pass). That blindfolded hike is *literally* how modern statistics and machine learning fit models: whether you are solving ordinary least squares, maximizing a log-likelihood, or training a billion-parameter neural network, you are navigating a high-dimensional loss surface using **gradients**, **Hessians**, and the **chain rule**.

This reference turns multivariable calculus from a list of formulas into a working compass for statistical optimization—showing you how to compute gradients and Hessians by hand on real regression losses and why curvature separates slow zig-zagging from rapid convergence.

---

## Key ideas: Where Does Multivariable Calculus Power Statistics?

Three pillars connect undergraduate calculus directly to statistical estimation and machine learning.

### 1. Differential Calculus & First-Order Optimization (Gradients)
The **gradient** $$\nabla f(\mathbf{x})$$ generalizes the single-variable derivative $$f'(x)$$ to functions of $$p$$ parameters. It points in the exact direction of steepest ascent; moving in the opposite direction ($$-\nabla f$$) is **gradient descent**, the workhorse algorithm of modern machine learning.
- **Core questions:** Where does $$\nabla f(\mathbf{x}) = \mathbf{0}$$ (*critical points*)? Which direction shrinks the loss fastest at the current parameter vector?
- **Real-world reach:** Maximum likelihood estimation (MLE), logistic regression, stochastic gradient descent (SGD), and Adam optimizers.

### 2. Second-Order Geometry & Curvature (The Hessian Matrix)
Setting $$\nabla f(\mathbf{x}) = \mathbf{0}$$ only tells you the ground is flat—you might be at a minimum, a maximum, or a treacherous **saddle point**. The **Hessian matrix** $$\nabla^2 f(\mathbf{x})$$ captures local curvature in every direction. Its eigenvalues tell you whether the surface curves upward like a bowl (convex minimum) and how stretched or narrow the valley is (*condition number*).
- **Core questions:** Is a critical point a true local minimum? How does Newton–Raphson use $$\mathbf{H}^{-1}$$ to jump straight to the bottom of a quadratic bowl?
- **Real-world reach:** Fisher Information Matrix in statistical inference (computing standard errors of MLEs!), Newton–Raphson / L-BFGS solvers, and diagnosing ill-conditioned regression features.

### 3. Composition and Automatic Differentiation (The Chain Rule)
Statistical models are rarely single functions—they are nested pipelines: raw features $$\to$$ linear score $$\to$$ nonlinear link/activation $$\to$$ loss. The **multivariable chain rule** propagates sensitivity cleanly through any number of layers, forming the exact mathematical engine behind **backpropagation**.
- **Core questions:** How does nudging an early parameter affect the final loss several layers downstream?
- **Real-world reach:** Deep learning frameworks (PyTorch, JAX), sensitivity analysis, and delta-method variance propagation in statistics.

---

## Derivatives and rate of change

For a function of one variable, the derivative is the instantaneous slope—the local linear story of how the function moves.

### Basic formulas

**Slope between two points:**

$$\text{Slope} = \frac{y_2 - y_1}{x_2 - x_1}$$

**Derivative (instantaneous rate of change):**

$$f'(x_0) = \lim_{x_1 \to x_0} \frac{f(x_1) - f(x_0)}{x_1 - x_0} = \lim_{\Delta x \to 0} \frac{f(x_0 + \Delta x) - f(x_0)}{\Delta x}$$

Critical points of optimization problems are places where this rate of change is zero (or undefined). That is why derivatives show up the moment we fit models by minimizing a loss or maximizing a likelihood.

### Level curves

A **level curve** of $$f(x, y)$$ is the set of points where the function is constant:

$$f(x, y) = c$$

Level curves turn a 3-D loss surface into a 2-D topographic contour map:
- $$f(x, y) = x^2 + y^2$$ has circular contours $$x^2 + y^2 = c$$ (perfectly round bowl—easy to optimize).
- $$f(x, y) = x^2 + 10y^2$$ has narrow elliptical contours (a steep canyon—naive gradient descent zig-zags!).
- $$f(x, y) = x + y$$ has parallel straight-line contours $$x + y = c$$ (flat ramp).

---

## Multivariable building blocks

### Partial derivatives

For $$f(x_1, x_2, \ldots, x_n)$$, the **partial derivative** in $$x_i$$ is

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}$$

Only $$x_i$$ moves; every other coordinate stays frozen.

### Gradient

The **gradient** stacks all partial derivatives into a column vector:

$$\nabla f(\mathbf{x}) = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix}$$

It points in the direction of steepest increase and is always **perpendicular (orthogonal)** to the level curve passing through $$\mathbf{x}$$.

### Hessian

The **Hessian** collects all second partial derivatives into a symmetric $$n \times n$$ matrix (when second partials are continuous):

$$\nabla^2 f(\mathbf{x}) = \mathbf{H} = \begin{pmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{pmatrix}$$

If $$\nabla f(\mathbf{x}^*) = \mathbf{0}$$ and $$\mathbf{H}(\mathbf{x}^*)$$ is **positive definite** (all eigenvalues $$> 0$$; in 2D, $$H_{11} > 0$$ and $$\det \mathbf{H} > 0$$), then $$\mathbf{x}^*$$ is a strict local minimum. If eigenvalues have mixed signs, $$\mathbf{x}^*$$ is a **saddle point**.

---

### Worked Example 1: Finding the Least-Squares Minimum via Gradient & Hessian by Hand

Suppose we have just two data points $$(x_1, y_1) = (1, 2)$$ and $$(x_2, y_2) = (2, 3)$$, and we fit a line $$\hat{y} = w_0 + w_1 x$$ by minimizing the squared-error loss:

$$J(w_0, w_1) = \frac{1}{2}\Bigl[(w_0 + w_1 - 2)^2 + (w_0 + 2w_1 - 3)^2\Bigr].$$

Let’s solve this completely by hand:

1. **Compute the gradient $$\nabla J(w_0, w_1)$$:**
   $$\frac{\partial J}{\partial w_0} = (w_0 + w_1 - 2) + (w_0 + 2w_1 - 3) = 2w_0 + 3w_1 - 5,$$
   $$\frac{\partial J}{\partial w_1} = (w_0 + w_1 - 2)\cdot 1 + (w_0 + 2w_1 - 3)\cdot 2 = 3w_0 + 5w_1 - 8.$$
2. **Set $$\nabla J = \mathbf{0}$$:**
   Solving $$2w_0 + 3w_1 = 5$$ and $$3w_0 + 5w_1 = 8$$ gives the unique critical point **$$(w_0^*, w_1^*) = (1, 1)$$** (the line $$\hat{y} = 1 + x$$, which passes cleanly through both points with loss $$J(1,1) = 0$$!).
3. **Verify it is a global minimum using the Hessian $$\mathbf{H}$$:**
   $$\mathbf{H} = \begin{pmatrix} \frac{\partial^2 J}{\partial w_0^2} & \frac{\partial^2 J}{\partial w_0 \partial w_1} \\ \frac{\partial^2 J}{\partial w_1 \partial w_0} & \frac{\partial^2 J}{\partial w_1^2} \end{pmatrix} = \begin{pmatrix} 2 & 3 \\ 3 & 5 \end{pmatrix}.$$
   Check the 2D positive-definiteness test:
   - $$H_{11} = 2 > 0$$ ✓
   - $$\det(\mathbf{H}) = (2)(5) - (3)(3) = 10 - 9 = 1 > 0$$ ✓
   Since $$\mathbf{H}$$ is constant and positive definite everywhere, $$J$$ is **strictly convex**—guaranteeing that $$(1,1)$$ is the unique global minimum! (Notice also that $$\mathbf{H} = \mathbf{X}^\top \mathbf{X}$$—the exact Gram matrix of linear regression!)

---

### Worked Example 2: Gradient Descent vs. Newton’s Method in a Narrow Valley

Consider a narrow elliptical bowl $$f(x, y) = x^2 + 10y^2$$ whose true minimum is $$(0, 0)$$. Start at $$\mathbf{x}_0 = (1, 1)$$, where $$\nabla f(1,1) = (2, 20)^\top$$ and $$\mathbf{H} = \begin{pmatrix} 2 & 0 \\ 0 & 20 \end{pmatrix}$$.

1. **One step of Gradient Descent** ($$\mathbf{x}_1 = \mathbf{x}_0 - \eta \nabla f(\mathbf{x}_0)$$) with learning rate $$\eta = 0.08$$:
   $$\mathbf{x}_1 = \begin{pmatrix} 1 \\ 1 \end{pmatrix} - 0.08 \begin{pmatrix} 2 \\ 20 \end{pmatrix} = \begin{pmatrix} 0.84 \\ -0.60 \end{pmatrix}.$$
   Look what happened: $$y$$ overshot past $$0$$ to $$-0.60$$ (bouncing across the narrow canyon!), while $$x$$ barely crept from $$1$$ to $$0.84$$!
2. **One step of Newton’s Method** ($$\mathbf{x}_1 = \mathbf{x}_0 - \mathbf{H}^{-1}\nabla f(\mathbf{x}_0)$$):
   $$\mathbf{x}_1 = \begin{pmatrix} 1 \\ 1 \end{pmatrix} - \begin{pmatrix} 1/2 & 0 \\ 0 & 1/20 \end{pmatrix}\begin{pmatrix} 2 \\ 20 \end{pmatrix} = \begin{pmatrix} 1 \\ 1 \end{pmatrix} - \begin{pmatrix} 1 \\ 1 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}!$$

**Why this matters:** Multiplying by $$\mathbf{H}^{-1}$$ automatically scales down the steep $$y$$-direction (dividing by 20) and scales up the flat $$x$$-direction (dividing by 2), jumping straight to the exact minimum in **one step**! That same geometry explains why feature standardization (`StandardScaler`) makes gradient descent converge dramatically faster: standardizing features turns narrow ellipses back into round circles!

---

## Chain rule

Composite functions are the default in statistics and machine learning (a loss of a prediction of a feature map). The chain rule is how derivatives travel through that composition.

### One parameter

If $$z = f(x, y)$$ with $$x = g(t)$$ and $$y = h(t)$$,

$$\frac{dz}{dt} = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt}$$

### Several parameters

If $$z = f(x_1, \ldots, x_n)$$ and each $$x_i = x_i(t_1, \ldots, t_m)$$,

$$\frac{\partial z}{\partial t_j} = \sum_{i=1}^{n} \frac{\partial f}{\partial x_i} \frac{\partial x_i}{\partial t_j}$$

### Why this matters later

1. **Gradients of composed losses.** Almost every fitted model is a composition.
2. **Constraints.** A constraint $$g(x, y) = 0$$ often lets you substitute one variable for another (or introduce Lagrange multipliers).
3. **Algorithms.** Backpropagation and automatic differentiation are organized chain-rule applications.
4. **Sensitivity.** How a small change in a hyperparameter or input moves the fitted solution.

### Example: a simple constrained minimum

Minimize $$f(x, y) = x^2 + y^2$$ subject to $$g(x, y) = x + y - 1 = 0$$.

Eliminate $$y = 1 - x$$ and minimize

$$h(x) = f(x, 1 - x) = x^2 + (1 - x)^2$$

The chain rule gives

$$h'(x) = \frac{\partial f}{\partial x} \cdot 1 + \frac{\partial f}{\partial y} \cdot \frac{d(1 - x)}{dx} = 2x + 2(1 - x)(-1) = 4x - 2$$

Set $$h'(x) = 0$$ to get $$x = 1/2$$, hence the point $$(1/2, 1/2)$$. The same pattern—substitute, differentiate, set the derivative to zero—reappears whenever a statistical estimator is defined by an optimization problem.

---

### Explanation: From Taylor’s Expansion to Statistical Optimization

Every gradient method, Newton update, and asymptotic standard error in statistics grows out of one master formula: the **second-order Taylor expansion**. Follow these four steps slowly.

#### Step 1: Zoom into any smooth surface around your current guess $$\mathbf{x}$$
For a small step $$\mathbf{v}$$, any twice-differentiable loss $$f$$ behaves locally like a quadratic bowl:
$$f(\mathbf{x} + \mathbf{v}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^\top \mathbf{v} + \frac{1}{2}\mathbf{v}^\top \mathbf{H}(\mathbf{x}) \mathbf{v}.$$
The linear term $$\nabla f(\mathbf{x})^\top \mathbf{v}$$ controls the **tilt**; the quadratic term $$\frac{1}{2}\mathbf{v}^\top \mathbf{H}(\mathbf{x})\mathbf{v}$$ controls the **bowl curvature**.

#### Step 2: Why $$-\nabla f(\mathbf{x})$$ is guaranteed steepest descent
For a tiny step of fixed length $$\|\mathbf{v}\| = \epsilon$$, the linear term dominates. By Cauchy–Schwarz, $$\nabla f(\mathbf{x})^\top \mathbf{v} = \|\nabla f\|\,\|\mathbf{v}\|\cos\theta$$ is minimized when $$\cos\theta = -1$$—meaning $$\mathbf{v}$$ points straight opposite to $$\nabla f(\mathbf{x})$$.

#### Step 3: Why setting the gradient of the quadratic model to zero gives Newton’s method
Differentiate the quadratic Taylor model with respect to $$\mathbf{v}$$ and set it to $$\mathbf{0}$$:
$$\nabla f(\mathbf{x}) + \mathbf{H}(\mathbf{x})\mathbf{v} = \mathbf{0} \implies \mathbf{v} = -\mathbf{H}(\mathbf{x})^{-1}\nabla f(\mathbf{x}).$$
That is Newton’s step! When $$\mathbf{H}$$ is cheap to invert (a few dozen parameters, like logistic regression), Newton converges in 5–10 iterations. When $$p = 10^8$$ parameters (deep learning), forming an $$n\times n$$ Hessian is impossible—so we fall back on gradient descent + adaptive rescaling (Adam) or quasi-Newton approximations (L-BFGS).

#### Step 4: The statistical bonus — why curvature equals precision
In maximum likelihood estimation, the negative Hessian of the log-likelihood at the peak (the **observed Fisher information**) measures how *sharply peaked* the likelihood is. Sharp peak (large curvature) = narrow confidence interval (high precision); flat plateau (tiny curvature) = huge uncertainty! Calculus curvature and statistical certainty are literally the same object.

---

## Video tham khảo

- [The paradox of the derivative | Chapter 2, Essence of calculus](https://www.youtube.com/watch?v=9vKqVkMQHKk) — 3Blue1Brown, English. **xem trước.** The derivative as an instantaneous rate of change, matching the limit definition that opens the note.
- [Visualizing the chain rule and product rule | Chapter 4, Essence of calculus](https://www.youtube.com/watch?v=YG15m2VwSjA) — 3Blue1Brown, English. A visual chain rule for the multivariable composite-function section.
- [Partial Derivatives and the Gradient of a Function](https://www.youtube.com/watch?v=AXH9Xm6Rbfc) — Professor Dave Explains, English. Partial derivatives and the gradient — the tools the note uses for level curves and critical points.
