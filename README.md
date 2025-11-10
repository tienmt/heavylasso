
# **heavylasso: Sparse Penalized Regression under Heavy-Tailed Noise**

The **heavylasso** package provides robust and efficient algorithms for sparse regression in the presence of heavy-tailed or non-Gaussian noise.  
It implements both the **Heavy Lasso** (based on a Student-t–type loss) and the **Exponential Lasso**, enabling users to perform regularized estimation under various robustness settings.

The Heavy Lasso method is inspired by:  
> **"Heavy Lasso: Sparse Penalized Regression under Heavy-Tailed Noise via Data-Augmented Soft-Thresholding."**

It combines coordinate descent with adaptive weighting derived from a Student-type loss, providing robustness against outliers and heavy-tailed residuals.

---

## **Installation**

Install the development version directly from GitHub:

```r
# install.packages("devtools")
devtools::install_github("tienmt/heavylasso")
library(heavylasso)
````

---

## **Overview**

The package implements two key estimators:

1. **Heavy Lasso (`heavylasso`)**

   * Uses a Student-t–type loss to handle heavy-tailed noise robustly.
   * The loss downweights extreme residuals via a data augmentation approach.

2. **Exponential Lasso (`expLasso`)**

   * Employs an exponential-type loss to achieve robustness to outliers with smooth penalization.
   * Suitable for data with asymmetric or non-Gaussian contamination.

Both methods include efficient coordinate descent solvers and cross-validation routines for tuning parameter selection.

---

## **Theoretical Background**

The standard Lasso solves the optimization problem

[
\min_\beta \frac{1}{2n} \sum_{i=1}^n (y_i - x_i^\top \beta)^2 + \lambda |\beta|_1.
]

However, when the noise distribution is heavy-tailed, the quadratic loss becomes unstable.
The **Heavy Lasso** replaces the squared loss by a **Student-t–type loss**, derived from the log-likelihood of a Student-t model:

[
L_{\text{Heavy}}(\beta) = \frac{1}{2n} \sum_{i=1}^n \nu \log!\left(1 + \frac{(y_i - x_i^\top \beta)^2}{\nu}\right),
]
where (\nu > 0) controls the heaviness of the tails.

The **Exponential Lasso** instead uses an **exponential loss**, providing a smooth, convex alternative:

[
L_{\text{Exp}}(\beta) = \frac{1}{n} \sum_{i=1}^n \frac{1}{\tau} \left( \exp!\left(\frac{(y_i - x_i^\top \beta)^2}{2\tau}\right) - 1 \right),
]
where (\tau > 0) controls sensitivity to large residuals.

Both losses are combined with the L1-penalty to encourage sparsity:
[
\min_\beta L(\beta) + \lambda |\beta|_1.
]

---

## **Example 1: Basic Heavy Lasso**

```r
# Simulate data with heavy-tailed errors
n <- 100; p <- 10
X <- matrix(rnorm(n * p), n, p)
beta_true <- c(1, -1, rep(0, p - 2))
y <- X %*% beta_true + rt(n, df = 3)  # heavy-tailed Student-t noise

# Fit Heavy Lasso model
fit <- heavylasso(X, y, lambda = 0.1)

# Print estimated coefficients
print(fit$coefficients)
```

---

## **Example 2: Cross-Validation for Heavy and Exponential Lasso**

```r
set.seed(123)
n_test <- 10000
n <- n_test + 100
p <- 120
s0 <- 10

# True sparse signal
beta_true <- rep(0, p)
beta_true[1:s0] <- c(rep(1, s0/2), rep(-1, s0/2))

# Design matrix
X1 <- matrix(rnorm(n * p), n, p)

# Heavy-tailed response (Student-t noise)
y1 <- X1 %*% beta_true + rt(n, df = 30)

# Train/test split
y <- y1[-(1:n_test), ]
X <- X1[-(1:n_test), ]
ytest <- y1[1:n_test, ]
xtest <- X1[1:n_test, ]

# Cross-validation for Heavy Lasso
cv.heavylasso1 <- cv_heavylasso(X, y, nu = IQR(y) * 3, lambdas = 1:50)
b_heavyt <- cv.heavylasso1$beta_best

# Cross-validation for Exponential Lasso
cv.explasso1 <- cv_expLasso(X, y, tau = 0.1, lambda = 1:50)
b_explasso <- cv.explasso1$beta_best

# Compare estimation accuracy
cat("MSE (Exponential Lasso):", sum((b_explasso - beta_true)^2), "\n")
cat("MSE (Heavy Lasso):", sum((b_heavyt - beta_true)^2), "\n")
```

---

## **Key Features**

* Robust estimation under heavy-tailed or contaminated noise
* Coordinate descent implementation for computational efficiency
* Cross-validation for automatic selection of tuning parameter (\lambda)
* Easy-to-use interface compatible with standard R workflows

---

## **References**

* Donoho, D. L., & Johnstone, I. M. (1994). *Ideal spatial adaptation by wavelet shrinkage*. **Biometrika**, 81(3), 425–455.
* Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
* Mai, T. T. (2025). *Heavy Lasso: Sparse Penalized Regression under Heavy-Tailed Noise via Data-Augmented Soft-Thresholding.* (Working paper)

```

---

