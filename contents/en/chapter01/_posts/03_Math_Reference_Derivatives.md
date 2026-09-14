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

This reference reviews derivatives and the multivariable calculus that sits underneath gradient-based learning, Newton-type updates, and a lot of later statistical optimization.

---

## Derivatives and rate of change

For a function of one variable, the derivative is the instantaneous slope—the local linear story of how the function moves.

### Basic formulas

**Slope between two points:**

$$\text{Slope} = \frac{y_2 - y_1}{x_2 - x_1}$$

**Derivative (instantaneous rate of change):**

$$f'(x_0) = \lim_{x_1 \to x_0} \frac{f(x_1) - f(x_0)}{x_1 - x_0} = \lim_{\Delta x \to 0} \frac{f(x_0 + \Delta x) - f(x_0)}{\Delta x}$$

Critical points of optimization problems are places where this rate of change is zero (or undefined). That is why derivatives show up as soon as we fit models by minimizing a loss.

### Level curves

A **level curve** of $$f(x, y)$$ is the set of points where the function is constant:

$$f(x, y) = c$$

Level curves let you draw a 3-D surface as a 2-D contour map.

**Examples:**
- $$f(x, y) = x^2 + y^2$$ has circular levels $$x^2 + y^2 = c$$
- $$f(x, y) = x + y$$ has parallel-line levels $$x + y = c$$

They tell you the topography of $$f$$, the directions of steepest ascent and descent, and where optima might sit.

---

## Multivariable building blocks

### Partial derivatives

For $$f(x_1, x_2, \ldots, x_n)$$, the **partial derivative** in $$x_i$$ is

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}$$

Only $$x_i$$ moves; every other coordinate stays put.

### Gradient

The **gradient** stacks the partials:

$$\nabla f(\mathbf{x}) = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix}$$

It points in the direction of steepest increase and is orthogonal to the level sets.

### Hessian

The **Hessian** collects second partials:

$$\nabla^2 f(\mathbf{x}) = \mathbf{H} = \begin{pmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{pmatrix}$$

The Hessian encodes curvature. We use it to classify critical points (minimum, maximum, saddle) and in second-order methods such as Newton’s method.

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
2. **Constraints.** A constraint $$g(x, y) = 0$$ often lets you substitute one variable for another.
3. **Algorithms.** Backpropagation and automatic differentiation are organized chain-rule applications.
4. **Sensitivity.** How a change in a hyperparameter moves the fitted solution.

### Example: a simple constrained minimum

Minimize $$f(x, y) = x^2 + y^2$$ subject to $$g(x, y) = x + y - 1 = 0$$.

Eliminate $$y = 1 - x$$ and minimize

$$h(x) = f(x, 1 - x) = x^2 + (1 - x)^2$$

The chain rule gives

$$h'(x) = \frac{\partial f}{\partial x} \cdot 1 + \frac{\partial f}{\partial y} \cdot \frac{d(1 - x)}{dx} = 2x + 2(1 - x)(-1) = 4x - 2$$

Set $$h'(x) = 0$$ to get $$x = 1/2$$, hence the point $$(1/2, 1/2)$$. The same pattern—substitute, differentiate, set the derivative to zero—reappears whenever a statistical estimator is defined by an optimization problem.
