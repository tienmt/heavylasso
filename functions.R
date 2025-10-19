
omega <- 1      # scale
alpha <- 5     # shape (positive = right skew)
delta <- alpha / sqrt(1 + alpha^2)
xi <- -omega * delta * sqrt(2 / pi)  # location chosen so mean = 0

add_outliers_y <- function(y, proportion = 0.1, magnitude = 10) {
  n <- length(y)
  n_out <- floor(n * proportion)
  outlier_idx <- sample(1:n, n_out)
  y[outlier_idx] <- y[outlier_idx] + rnorm(n_out, mean = 0, sd = magnitude)
  y
}

cv_em_lasso_t <- function(X, y, lambda_seq = NULL, K = 5, nu = 3, sigma2 = 1,
                          max_iter = 2000, tol = 1e-14, beta_init = NULL,
                          lambda_min_ratio = 0.001, n_lambda = 20, seed = 123) {
  set.seed(seed)
  n <- nrow(X)
  p <- ncol(X)
  folds <- sample(rep(1:K, length.out = n))
  
  # Standardize X once for all folds
  X_scaled <- scale(X)
  
  # Auto-generate lambda_seq if not provided
  if (is.null(lambda_seq)) {
    # For t-distribution inspired weighting, we approximate residuals as y initially
    r <- y
    psi <- pmin(pmax(r, -sqrt(nu * sigma2)), sqrt(nu * sigma2))  # bounded like Huber
    lambda_max <- max(abs(t(X_scaled) %*% psi)) / n
    lambda_seq <- exp(seq(log(lambda_max),
                          log(lambda_max * lambda_min_ratio),
                          length.out = n_lambda))
  }
  
  cv_errors <- matrix(NA, nrow = K, ncol = length(lambda_seq))
  
  for (k in 1:K) {
    test_idx <- which(folds == k)
    train_idx <- setdiff(1:n, test_idx)
    
    X_train <- X_scaled[train_idx, , drop = FALSE]
    y_train <- y[train_idx]
    X_test  <- X_scaled[test_idx, , drop = FALSE]
    y_test  <- y[test_idx]
    
    for (i in seq_along(lambda_seq)) {
      lambda <- lambda_seq[i]
      
      fit <- em_lasso_t(X_train, y_train, lambda = lambda,
                        nu = nu, sigma2 = sigma2,
                        max_iter = max_iter, tol = tol,
                        beta_init = beta_init)
      
      beta_hat <- fit$coefficients
      y_pred <- X_test %*% beta_hat
      cv_errors[k, i] <- mean((y_test - y_pred)^2)
    }
  }
  
  mean_cv_errors <- colMeans(cv_errors)
  best_lambda <- lambda_seq[which.min(mean_cv_errors)]
  best_fit <- em_lasso_t(X, y, lambda = best_lambda,nu = nu, sigma2 = sigma2,
                         max_iter = max_iter, tol = tol,
                         beta_init = beta_init)
  list(beta_cv     = best_fit$coefficients,
       lambda_seq  = lambda_seq,
       cv_errors   = cv_errors,
       mean_cv_errors = mean_cv_errors,
       best_lambda = best_lambda)
}

selection_metrics <- function(true_support, selected_support, p = NULL) {
  # true_support: indices of truly relevant variables
  # selected_support: indices selected by your method
  # p: total number of variables (optional, required for TN)
  
  true_support <- as.integer(true_support)
  selected_support <- as.integer(selected_support)
  
  TP <- length(intersect(true_support, selected_support))
  FP <- length(setdiff(selected_support, true_support))
  FN <- length(setdiff(true_support, selected_support))
  if (!is.null(p)) {
    all_indices <- seq_len(p)
    TN <- length(setdiff(all_indices, union(true_support, selected_support)))
  } else {
    TN <- NA  # Cannot compute without knowing p
  }
  
  precision <- if ((TP + FP) == 0) NA else TP / (TP + FP)
  recall    <- if ((TP + FN) == 0) NA else TP / (TP + FN)
  specificity <- if (!is.na(TN) && (TN + FP) > 0) TN / (TN + FP) else NA
  f1 <- if (!is.na(precision) && !is.na(recall) && (precision + recall) > 0) {
    2 * precision * recall / (precision + recall)
  } else {
    NA
  }
  fdr <- if ((TP + FP) == 0) 0 else FP / (TP + FP)
  
  return(c(
    #'Precision' = precision,
    'TPR' = round(recall,2),
    #'F1' = f1,
    #'Specificity' = specificity,
    'FDR' =  round(fdr , 2 )
  ))
}

# --- M-step: coordinate descent with soft-thresholding ---
#for (j in 1:p) {
# partial residual excluding j
#   r_j <- y - X %*% beta + X[, j] * beta[j]
# weighted inner products
#  z_j <- sum(w * X[, j] * r_j)
#  A_j <- sum(w * X[, j]^2)
# soft-thresholding update
#  beta[j] <- sign(z_j) * max(abs(z_j) - lambda, 0) / A_j
#}

em_lasso_t <- function(X, y, lambda, nu = 3, max_iter = 400, tol = 1e-4, beta_init = NULL) {
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
    beta <- coordinate_descent_update_opt1(X, y, beta, w,r, lambda, active_set)
    r <- y - X %*% beta
    w <- (nu + 1) / (nu + r^2)
    obj_new <- sum(w * r^2) + lambda * sum(abs(beta))
    if(abs(obj_new - obj_null) < tol *obj_null ){ break}
  }
  list(coefficients = beta, weights = w, iterations = iter)
}
em_lasso_t <- compiler::cmpfun(em_lasso_t)



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
    fit <- heavylasso_opt(X, y, lambda = lambda, ...)
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
  best_fit <- em_lasso_t(X, y, lambda = best_lambda, ...)
  
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
select_lambda_ic <- compiler::cmpfun(select_lambda_ic)
#selection_metrics(true_support = 1:s0,selected_support = which(bet_heavyt !=0),p = p)
#selection_metrics(true_support = 1:s0,selected_support = which(bet_glmnet !=0) ,p = p)
#selection_metrics(true_support = 1:s0,selected_support = which(bet_LAD !=0),p = p)
#selection_metrics(true_support = 1:s0,selected_support = which(bet_Huber !=0),p = p)
#selection_metrics(true_support = 1:s0,selected_support = which(bet_TFRE !=0) ,p = p)

