#' Heavy-tailed Lasso Regression using an EM Algorithm
#'
#' Fits a robust linear regression model under a heavy-tailed error assumption
#' using an EM-like algorithm. The model incorporates \eqn{\ell_1}-penalization
#' (lasso) and assumes a Student-t distribution for the errors.
#'
#' @param X A numeric matrix of predictors of dimension \eqn{n \times p}.
#' @param y A numeric response vector of length \eqn{n}.
#' @param lambda Non-negative regularization parameter controlling sparsity.
#' @param nu Degrees of freedom for the Student-\eqn{t} distribution (default: 3).
#' @param sigma2 Fixed error variance, assumed known (default: 1).
#' @param max_iter Maximum number of EM iterations (default: 2000).
#' @param tol Convergence threshold based on mean squared difference of coefficients (default: 1e-11).
#' @param beta_init Optional numeric vector of initial coefficient values. If \code{NULL}, initialized to zero.
#' @author The Tien Mai (<the.tien.mai@fhi.no>)
#'
#' @references
#' Mai, T. T. (2025). Heavy Lasso: sparse penalized regression under heavy-tailed noise via data-augmented soft-thresholding. arXiv preprint arXiv:2506.07790.
#'
#' Mai, T. T. (2025). Exponential Lasso: robust sparse penalization under heavy-tailed noise and outliers with exponential-type loss.
#'
#' @return A list with the following components:
#' \describe{
#'   \item{coefficients}{Estimated regression coefficients \eqn{\hat{\beta}}.}
#'   \item{weights}{Observation-specific weights derived from the E-step.}
#'   \item{iterations}{Number of iterations completed before convergence.}
#' }
#'
#' @details
#' This function solves the following penalized regression problem under heavy-tailed errors:
#'
#' \deqn{ \min_{\beta} \sum_{i=1}^{n} w_i (y_i - x_i^\top \beta)^2 + \lambda \|\beta\|_1, }
#'
#' where the weights \eqn{w_i} are updated in the E-step assuming a Student-\eqn{t} likelihood:
#'
#' \deqn{ w_i = \frac{\nu + 1}{\nu + r_i^2 / \sigma^2} / 2, }
#'
#' and \eqn{r_i = y_i - x_i^\top \beta} is the residual for observation \eqn{i}. This arises from
#' modeling the errors with a scale-mixture of normals, which is equivalent to assuming:
#'
#' \deqn{ y_i = x_i^\top \beta + \epsilon_i, \quad \epsilon_i \sim t_\nu(0, \sigma^2). }
#'
#' The algorithm alternates between:
#' \itemize{
#'   \item \strong{E-step}: Update observation-specific weights \eqn{w_i}.
#'   \item \strong{M-step}: Solve a weighted lasso problem via coordinate descent.
#' }
#'
#' @section Algorithm:
#' \enumerate{
#'   \item Initialize \eqn{\beta^{(0)}}, set all weights \eqn{w_i = 1}.
#'   \item Repeat until convergence or maximum iterations:
#'     \enumerate{
#'       \item E-step: Compute residuals and update weights as above.
#'       \item M-step: Solve the weighted lasso problem using coordinate descent.
#'       \item Check for convergence based on squared difference in coefficients.
#'     }
#' }
#'
#' @examples
#' set.seed(123)
#' n <- 100; p <- 10
#' X <- matrix(rnorm(n * p), n, p)
#' beta_true <- c(1, -1, rep(0, p - 2))
#' y <- X %*% beta_true + rt(n, df = 3)  # heavy-tailed noise
#' fit <- heavylasso(X, y, lambda = 0.1)
#' print(fit$coefficients)
#'
#' @seealso \code{\link{select_lambda_ic}} for automated lambda selection.
#'
#' @export
#' @useDynLib heavylasso
#' @importFrom Rcpp sourceCpp
heavylasso <- function(X, y, lambda, nu = 3, sigma2 = 1, max_iter = 2000, tol = 1e-11
                       , beta_init = NULL ) {
  n <- length(y)
  p <- ncol(X)
  beta <- if (is.null(beta_init)) rep(0, p) else beta_init
  obj_null <- sum((nu + 1) / (nu + y^2))
  r <- y - X %*% beta
  w <- (nu + 1) / (nu + r^2)
  # Compute gradient = X^T y weighted by w
  grad <- as.vector(crossprod(X, w * r))
  active_set <- which(abs(grad) >= mean(abs(grad)) )
  for (iter in 1:max_iter) {
    beta <- coordinate_descent_update(X, y, beta, w,r, lambda, active_set)
    r <- y - X %*% beta
    w <- (nu + 1) / (nu + r^2)
    obj_new <- sum(w * r^2) + lambda * sum(abs(beta))
    if(abs(obj_new - obj_null) < tol *obj_null ){ break}
  }
  return(
    structure(
      list(coefficients = beta, weights = w, iterations = iter),
      class = "heavylasso"
    )
  )
}








#' Cross-validation Heavy-tailed Lasso Regression: Student loss, also known as HeavyLasso
#'
#' Fits a robust linear regression model under a heavy-tailed error assumption
#' using an EM-like algorithm. The model incorporates \eqn{\ell_1}-penalization
#' (lasso) and assumes a Student-t distribution for the errors.
#'
#' @param X A numeric matrix of predictors of dimension \eqn{n \times p}.
#' @param y A numeric response vector of length \eqn{n}.
#' @param lambda Non-negative regularization parameter controlling sparsity.
#' @param nu 	Robustness or temperature parameter (default: 3). The author recommend IQR(y)*4.
#' @param sigma2 Fixed error variance, assumed known (default: 1).
#' @param max_iter Maximum number of EM iterations (default: 2000).
#' @param tol Convergence threshold based on mean squared difference of coefficients (default: 1e-11).
#' @param beta_init Optional numeric vector of initial coefficient values. If \code{NULL}, initialized to zero.
#'
#' @author The Tien Mai (<the.tien.mai@fhi.no>)
#'
#' @references
#' Mai, T. T. (2025). Heavy Lasso: sparse penalized regression under heavy-tailed noise via data-augmented soft-thresholding. arXiv preprint arXiv:2506.07790.
#'
#' Mai, T. T. (2025). Exponential Lasso: robust sparse penalization under heavy-tailed noise and outliers with exponential-type loss.
#'
#' @return A list with the following components:
#' \describe{
#'   \item{coefficients}{Estimated regression coefficients \eqn{\hat{\beta}}.}
#'   \item{weights}{Observation-specific weights derived from the E-step.}
#'   \item{iterations}{Number of iterations completed before convergence.}
#' }
#'
#' @details
#' This function solves the following penalized regression problem under heavy-tailed errors:
#'
#' \deqn{ \min_{\beta} \sum_{i=1}^{n} w_i (y_i - x_i^\top \beta)^2 + \lambda \|\beta\|_1, }
#'
#' where the weights \eqn{w_i} are updated in the E-step assuming a Student-\eqn{t} likelihood:
#'
#' \deqn{ w_i = \frac{\nu + 1}{\nu + r_i^2 / \sigma^2} / 2, }
#'
#' and \eqn{r_i = y_i - x_i^\top \beta} is the residual for observation \eqn{i}. This arises from
#' modeling the errors with a scale-mixture of normals, which is equivalent to assuming:
#'
#' \deqn{ y_i = x_i^\top \beta + \epsilon_i, \quad \epsilon_i \sim t_\nu(0, \sigma^2). }
#'
#' The algorithm alternates between:
#' \itemize{
#'   \item \strong{E-step}: Update observation-specific weights \eqn{w_i}.
#'   \item \strong{M-step}: Solve a weighted lasso problem via coordinate descent.
#' }
#'
#' @section Algorithm:
#' \enumerate{
#'   \item Initialize \eqn{\beta^{(0)}}, set all weights \eqn{w_i = 1}.
#'   \item Repeat until convergence or maximum iterations:
#'     \enumerate{
#'       \item E-step: Compute residuals and update weights as above.
#'       \item M-step: Solve the weighted lasso problem using coordinate descent.
#'       \item Check for convergence based on squared difference in coefficients.
#'     }
#' }
#'
#' @examples
#' n_test = 10000
#' n <- n_test + 100
#' p <- 120
#' s0 = 10
#' beta_true <- rep(0, p) ;
#' beta_true[1:s0] <- c( rep(1, s0/2) , rep(-1, s0/2) )
#' X1 <- matrix(rnorm(n * p), n, p)
#'
#' y1 <- X1 %*% beta_true + rt(n, df = 30)
#' # rnorm(n,sd=3)  # rnorm(n)  # rcauchy(n) # rt(n, df = 3) #
#'
#' y = y1[-(1:n_test),]
#' X = X1[-(1:n_test),]
#' ytest = y1[ 1:n_test,] ;
#' xtest = X1[1:n_test,]
#'
#' cv.heavylasso1 <- cv_heavylasso(X, y, nu = IQR(y)*3,lambdas = 1:50 )
#' b_heavyt2 <- cv.heavylasso1$beta_best
#' cv.explasso1 <- cv_expLasso(X, y, tau = 0.1, lambda = 1:50 )
#' b_explasso <- cv.explasso1$beta_best
#'
#' sum( (b_explasso - beta_true)^2)
#' sum( (b_heavyt2 - beta_true)^2)
#'
#' @seealso \code{\link{select_lambda_ic}} for automated lambda selection.
#'
#' @export
#' @useDynLib heavylasso
#' @importFrom Rcpp sourceCpp
#' @name cv_heavylasso
NULL





#' Cross-validated Exponential Lasso (C++ implementation)
#'
#' Performs K-fold cross-validation for the exponential Lasso regression model,
#' implemented efficiently in C++ using Rcpp and Armadillo.
#'
#' @param X Numeric matrix of predictors (\eqn{n \times p}).
#' @param y Numeric response vector of length \eqn{n}.
#' @param lambdas Numeric vector of regularization parameters to evaluate.
#' @param nfolds Integer, number of cross-validation folds (default: 5).
#' @param tau Robustness or temperature parameter (default: 1).
#' @param max_iter Integer, maximum number of iterations (default: 400).
#' @param tol Numeric tolerance for convergence (default: 1e-4).
#' @author The Tien Mai (<the.tien.mai@fhi.no>)
#'
#' @references
#' Mai, T. T. (2025). Heavy Lasso: sparse penalized regression under heavy-tailed noise via data-augmented soft-thresholding. arXiv preprint arXiv:2506.07790.
#'
#' Mai, T. T. (2025). Exponential Lasso: robust sparse penalization under heavy-tailed noise and outliers with exponential-type loss.
#' @return A list with components:
#' \describe{
#'   \item{lambda_best}{The value of \code{lambda} giving the best cross-validation performance.}
#'   \item{beta_best}{Estimated coefficient vector at \code{lambda_best}.}
#'   \item{cv_errors}{Cross-validation error values for each lambda.}
#' }
#'
#' @details
#' This function calls the C++ routine \code{_heavylasso_cv_expLasso} via \code{.Call()},
#' which implements a robust coordinate descent algorithm for heavy-tailed Lasso regression.
#'
#' @examples
#' n_test = 10000
#' n <- n_test + 100
#' p <- 120
#' s0 = 10
#' beta_true <- rep(0, p) ;
#' beta_true[1:s0] <- c( rep(1, s0/2) , rep(-1, s0/2) )
#' X1 <- matrix(rnorm(n * p), n, p)
#'
#' y1 <- X1 %*% beta_true + rt(n, df = 30)
#' # rnorm(n,sd=3)  # rnorm(n)  # rcauchy(n) # rt(n, df = 3)
#'
#' y = y1[-(1:n_test),]
#' X = X1[-(1:n_test),]
#' ytest = y1[ 1:n_test,] ;
#' xtest = X1[1:n_test,]
#'
#' cv.heavylasso1 <- cv_heavylasso(X, y, nu = IQR(y)*3,lambdas = 1:50 )
#' b_heavyt2 <- cv.heavylasso1$beta_best
#' cv.explasso1 <- cv_expLasso(X, y, tau = 0.1, lambda = 1:50 )
#' b_explasso <- cv.explasso1$beta_best
#'
#' sum( (b_explasso - beta_true)^2)
#' sum( (b_heavyt2 - beta_true)^2)
#'
#' @seealso \code{\link{heavylasso}} for the main fitting function.
#'
#' @useDynLib heavylasso, .registration = TRUE
#' @export
#' @name cv_expLasso
NULL







#' Select Regularization Parameter via Information Criterion
#'
#' Chooses the optimal lasso penalty parameter \code{lambda} based on AIC or BIC
#' for a heavy-tailed lasso model.
#'
#' @param X A numeric matrix of predictors (n by p).
#' @param y A numeric response vector.
#' @param lambda_seq A vector of candidate lambda values to evaluate (default: 1:15).
#' @param criterion Information criterion to use for selection: \code{"AIC"} or \code{"BIC"}.
#' @param ... Additional arguments passed to \code{\link{heavylasso}}.
#' @author The Tien Mai (<the.tien.mai@fhi.no>)
#' @references
#' Mai, T. T. (2025). Heavy Lasso: sparse penalized regression under heavy-tailed noise via data-augmented soft-thresholding. arXiv preprint arXiv:2506.07790.
#'
#' Mai, T. T. (2025). Exponential Lasso: robust sparse penalization under heavy-tailed noise and outliers with exponential-type loss.
#'
#' @return A list with the following elements:
#' \describe{
#'   \item{beta_cv}{Estimated coefficients at the selected lambda.}
#'   \item{lambda_seq}{The sequence of lambda values evaluated.}
#'   \item{rss}{Residual sum of squares for each lambda.}
#'   \item{df}{Degrees of freedom (nonzero coefficients) for each lambda.}
#'   \item{ic_values}{AIC or BIC values corresponding to each lambda.}
#'   \item{best_lambda}{Selected lambda value with minimal criterion.}
#'   \item{best_ic}{Minimum value of the information criterion.}
#'   \item{criterion}{The information criterion used ("AIC" or "BIC").}
#' }
#'
#' @details This function fits a sequence of heavy-tailed lasso models and chooses
#' the best one using AIC or BIC. It assumes constant variance and uses the
#' same sigma2 for all models.
#'
#' @export
#' @examples
#' # Simulate data
#' set.seed(123)
#' n <- 100
#' p <- 20
#' X <- matrix(rnorm(n * p), nrow = n)
#' beta_true <- c(rep(2, 5), rep(0, p - 5))
#' y <- X %*% beta_true + rt(n, df = 3)  # heavy-tailed errors
#'
#' # Fit model and select lambda using AIC
#' result <- select_lambda_ic(X, y, lambda_seq = seq(5, 0.1, length.out = 10))
#'
#' # View selected lambda and coefficients
#' result$best_lambda
#' result$beta_cv
select_lambda_ic <- function(X, y, lambda_seq = 1:50,
                             criterion = c("AIC",'BIC'), ...) {
  criterion <- match.arg(criterion)
  n <- length(y)
  p <- ncol(X)
  lambda_seq <- sort(lambda_seq, decreasing = TRUE)
  X = scale(X)
  ic_vals <- numeric(length(lambda_seq))
  df_vals <- numeric(length(lambda_seq))
  rss_vals <- numeric(length(lambda_seq))

  for (i in seq_along(lambda_seq)) {
    lambda <- lambda_seq[i]
    fit <- heavylasso(X, y, lambda = lambda, ...)
    beta_hat <- fit$coefficients
    y_hat <- X %*% beta_hat
    residuals <- y - y_hat

    rss <- sum(residuals^2)
    df <- sum(beta_hat != 0)

    rss_vals[i] <- rss
    df_vals[i] <- df

    ic_vals[i] <- if (criterion == "AIC") {
      n * log(rss / n) + 2 * df
    } else if (criterion == "BIC")  {
      n * log(rss / n) + log(n) * df
    }
  }

  best_idx <- which.min(ic_vals)
  best_lambda <- lambda_seq[best_idx] #+1
  best_fit <- heavylasso(X, y, lambda = best_lambda, ...)

  list(
    beta_cv      = best_fit$coefficients,
    lambda_seq   = lambda_seq,
    rss          = rss_vals,
    df           = df_vals,
    ic_values    = ic_vals,
    best_lambda  = best_lambda,
    best_ic      = ic_vals[best_idx],
    criterion    = criterion
  )
}




#' Extract Coefficients from a Heavy-Tailed Lasso Model
#'
#' This function extracts the estimated regression coefficients from a fitted
#' heavy-tailed lasso model.
#'
#' @param object An object of class \code{"heavylasso"}, typically returned by \code{\link{heavylasso}}.
#' @param ... Additional arguments (ignored).
#' @author The Tien Mai (<the.tien.mai@fhi.no>)
#'
#' @references
#' Mai, T. T. (2025). Heavy Lasso: sparse penalized regression under heavy-tailed noise via data-augmented soft-thresholding. arXiv preprint arXiv:2506.07790.
#'
#' Mai, T. T. (2025). Exponential Lasso: robust sparse penalization under heavy-tailed noise and outliers with exponential-type loss.
#'
#' @return A numeric vector of estimated regression coefficients.
#' @export
coef.heavylasso <- function(object, ...) {
  if (!inherits(object, "heavylasso")) {
    stop("Input must be a 'heavylasso' object.")
  }
  object$coefficients
}




#' Predict Method for Heavy-Tailed Lasso Models
#'
#' Predicts responses from a heavy-tailed lasso regression model.
#'
#' @param object An object of class \code{"heavylasso"}.
#' @param newdata A numeric matrix or data frame with the same number of columns as the training data.
#' @param ... Additional arguments (currently unused).
#' @author The Tien Mai (<the.tien.mai@fhi.no>)
#' @references
#' Mai, T. T. (2025). Heavy Lasso: sparse penalized regression under heavy-tailed noise via data-augmented soft-thresholding. arXiv preprint arXiv:2506.07790.
#'
#' Mai, T. T. (2025). Exponential Lasso: robust sparse penalization under heavy-tailed noise and outliers with exponential-type loss.
#'
#' @return A numeric vector of predicted values.
#' @export
#'
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(100 * 10), 100, 10)
#' beta <- c(2, -1.5, 0, 0, 3, rep(0, 5))
#' y <- X %*% beta + rt(100, df = 3)
#' fit <- heavylasso(X, y, lambda = 1)
#' predict(fit, newdata = X)
predict.heavylasso <- function(object, newdata, ...) {
  if (!inherits(object, "heavylasso")) {
    stop("Input must be a 'heavylasso' object.")
  }
  if (missing(newdata)) {
    stop("'newdata' must be provided.")
  }
  if (!is.matrix(newdata)) {
    newdata <- as.matrix(newdata)
  }
  as.vector(newdata %*% object$coefficients)
}
