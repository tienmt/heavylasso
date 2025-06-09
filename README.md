# heavylasso
 This is based on the paper:  
**"Heavy Lasso: sparse penalized regression under heavy-tailed noise via data-augmented soft-thresholding."**

## Installation

Install the package using:

```r
devtools::install_github('tienmt/heavylasso')
library(heavylasso)

# simulate data
n <- 100; p <- 10
X <- matrix(rnorm(n * p), n, p)
beta_true <- c(1, -1, rep(0, p - 2))
y <- X %*% beta_true + rt(n, df = 3)  # heavy-tailed noise
 fit <- heavylasso(X, y, lambda = 0.1)
 print(fit$coefficients)
.
