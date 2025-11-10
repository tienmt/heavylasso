#include <Rcpp.h>
#include <random>
#include <cmath>
using namespace Rcpp;

// [[Rcpp::export]]
NumericVector coordinate_descent_update(
    NumericMatrix X,
    NumericVector y,
    NumericVector beta,
    NumericVector w,
    NumericVector r,
    double lambda,
    IntegerVector active_set  // new argument
) {
  int n = X.nrow();
  NumericVector beta_new = clone(beta);

  // Coordinate descent over provided active_set
  for (int idx = 0; idx < active_set.size(); ++idx) {
    int j = active_set[idx] - 1; // R to C++ index (if 1-based)

    // Add back contribution of current beta[j]
    for (int i = 0; i < n; ++i) {
      r[i] += X(i, j) * beta_new[j];
    }

    double z_j = 0.0;
    double A_j = 0.0;
    for (int i = 0; i < n; ++i) {
      double x_ij = X(i, j);
      z_j += w[i] * x_ij * r[i];
      A_j += w[i] * x_ij * x_ij;
    }

    double beta_j_new = 0.0;
    if (std::abs(z_j) > lambda) {
      beta_j_new = std::copysign(1.0, z_j) * (std::abs(z_j) - lambda) / A_j;
    }

    for (int i = 0; i < n; ++i) {
      r[i] -= X(i, j) * beta_j_new;
    }

    beta_new[j] = beta_j_new;
  }

  return beta_new;
}


// -------------------------------------------------------------
// Coordinate Descent Update (Weighted Lasso Step)
// -------------------------------------------------------------
void coordinate_descent_update_opt(
    const NumericMatrix& X,
    NumericVector& beta,
    NumericVector& w,
    NumericVector& r,
    double lambda,
    const IntegerVector& active_set
) {
  int n = X.nrow();

  for (int idx = 0; idx < active_set.size(); ++idx) {
    int j = active_set[idx] - 1; // R index -> C++

    // Add back old contribution
    for (int i = 0; i < n; ++i)
      r[i] += X(i, j) * beta[j];

    // Compute z_j and A_j
    double z_j = 0.0, A_j = 0.0;
    for (int i = 0; i < n; ++i) {
      double xij = X(i, j);
      z_j += w[i] * xij * r[i];
      A_j += w[i] * xij * xij;
    }

    // Soft-threshold update
    double beta_j_new = 0.0;
    if (std::abs(z_j) > lambda) {
      beta_j_new = std::copysign(1.0, z_j) * (std::abs(z_j) - lambda) / A_j;
      if (std::abs(beta_j_new) < 1e-3) beta_j_new = 0.0;
    }

    // Subtract new contribution
    for (int i = 0; i < n; ++i)
      r[i] -= X(i, j) * beta_j_new;

    beta[j] = beta_j_new;
  }
}

// -------------------------------------------------------------
// Main: Exponential-Type Robust Lasso (MM Algorithm)
// -------------------------------------------------------------
// [[Rcpp::export]]
List expLasso_opt(
    const NumericMatrix& X,
    const NumericVector& y,
    double lambda,
    double tau = 0.1,
    int max_iter = 400,
    double tol = 1e-4,
    Nullable<NumericVector> beta_init = R_NilValue,
    bool dynamic_active = true
) {
  int n = y.size();
  int p = X.ncol();

  // Initialize beta
  NumericVector beta(p);
  if (beta_init.isNotNull()) {
    beta = as<NumericVector>(beta_init);
  } else {
    std::fill(beta.begin(), beta.end(), 0.0);
  }

  // Initialize residuals r = y
  NumericVector r = clone(y);

  // Initialize weights for exponential loss
  NumericVector w(n);
  for (int i = 0; i < n; ++i)
    w[i] = std::exp(-0.5 * tau * r[i] * r[i]);

  // Active set = all predictors
  IntegerVector active_set(p);
  for (int j = 0; j < p; ++j) active_set[j] = j + 1;

  double obj_old = R_PosInf, obj_new = R_PosInf;
  int iter = 0;

  for (iter = 1; iter <= max_iter; ++iter) {
    // Weighted coordinate descent
    coordinate_descent_update_opt(X, beta, w, r, lambda, active_set);

    // Update weights (MM/EM step)
    for (int i = 0; i < n; ++i)
      w[i] = std::exp(-0.5 * tau * r[i] * r[i]);

    // Compute objective: surrogate quadratic loss + l1 penalty
    obj_new = 0.0;
    for (int i = 0; i < n; ++i)
      obj_new += w[i] * r[i] * r[i];
    for (int j = 0; j < p; ++j)
      obj_new += lambda * std::abs(beta[j]);

    // Check convergence
    if (std::abs(obj_new - obj_old) < tol * (obj_old + 1e-28))
      break;

    obj_old = obj_new;

    // (Optional) update active set every 10 iterations
    if (dynamic_active && iter % 10 == 0) {
      NumericVector grad(p);
      for (int j = 0; j < p; ++j) {
        double sum_j = 0.0;
        for (int i = 0; i < n; ++i) {
          sum_j += X(i, j) * w[i] * r[i];
        }
        grad[j] = sum_j;
      }
      double mean_abs_grad = 2* mean(abs(grad));
      std::vector<int> active;
      for (int j = 0; j < p; ++j) {
        if (std::abs(grad[j]) >= mean_abs_grad) active.push_back(j + 1);
      }
      active_set = wrap(active);
    }

  }

  return List::create(
    Named("coefficients") = beta,
    Named("weights") = w,
    Named("iterations") = iter,
    Named("objective") = obj_new
  );
}

// -------------------------------------------------------------
// Compute Mean Squared Error
// -------------------------------------------------------------
double compute_mse(NumericMatrix X, NumericVector y, NumericVector beta) {
  int n = X.nrow();
  int p = X.ncol();
  double mse = 0.0;

  for (int i = 0; i < n; ++i) {
    double pred = 0.0;
    for (int j = 0; j < p; ++j)
      pred += X(i, j) * beta[j];
    double err = y[i] - pred;
    mse += err * err;
  }
  return mse / n;
}

// -------------------------------------------------------------
// Cross-Validation for Exponential-Type Lasso
// -------------------------------------------------------------
// [[Rcpp::export]]
List cv_expLasso(
    NumericMatrix X,
    NumericVector y,
    NumericVector lambdas,
    int nfolds = 5,
    double tau = 1.0,
    int max_iter = 400,
    double tol = 1e-4
) {
  int n = X.nrow();
  int p = X.ncol();
  int nlambda = lambdas.size();

  // Create folds
  IntegerVector fold_id(n);
  for (int i = 0; i < n; i++)
    fold_id[i] = i % nfolds;
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(fold_id.begin(), fold_id.end(), g);

  NumericVector cv_errors(nlambda, 0.0);

  for (int f = 0; f < nfolds; f++) {
    // Train/test split
    std::vector<int> train_idx, test_idx;
    for (int i = 0; i < n; i++) {
      if (fold_id[i] == f) test_idx.push_back(i);
      else train_idx.push_back(i);
    }

    int n_train = train_idx.size();
    int n_test  = test_idx.size();
    NumericMatrix X_train(n_train, p), X_test(n_test, p);
    NumericVector y_train(n_train), y_test(n_test);

    for (int ii = 0; ii < n_train; ii++) {
      int i = train_idx[ii];
      y_train[ii] = y[i];
      for (int j = 0; j < p; j++) X_train(ii, j) = X(i, j);
    }
    for (int ii = 0; ii < n_test; ii++) {
      int i = test_idx[ii];
      y_test[ii] = y[i];
      for (int j = 0; j < p; j++) X_test(ii, j) = X(i, j);
    }

    // Loop over lambdas
    for (int l = 0; l < nlambda; l++) {
      double lambda = lambdas[l];

      List fit = expLasso_opt(X_train, y_train, lambda, tau, max_iter, tol);
      NumericVector beta = fit["coefficients"];

      double mse = compute_mse(X_test, y_test, beta);
      cv_errors[l] += mse;
    }
  }

  // Average CV error and find best lambda
  for (int l = 0; l < nlambda; l++)
    cv_errors[l] /= nfolds;

  double best_lambda = lambdas[0];
  double best_error = cv_errors[0];
  for (int l = 1; l < nlambda; l++) {
    if (cv_errors[l] < best_error) {
      best_error = cv_errors[l];
      best_lambda = lambdas[l];
    }
  }


  List fit_final = expLasso_opt(X, y, best_lambda, tau, max_iter, tol);

  NumericVector beta_best = fit_final["coefficients"];

  return List::create(
    Named("lambda_min") = best_lambda,
    Named("cv_errors") = cv_errors,
    Named("lambdas") = lambdas,
    Named("beta_best") = beta_best
  );
}


// [[Rcpp::export]]
List heavylasso_opt(
    const NumericMatrix& X,
    const NumericVector& y,
    double lambda,
    double nu = 3.0,
    int max_iter = 400,
    double tol = 1e-4,
    Nullable<NumericVector> beta_init = R_NilValue,
    bool dynamic_active = true
) {
  int n = y.size();
  int p = X.ncol();

  // Initialize beta
  NumericVector beta(p);
  if (beta_init.isNotNull()) {
    beta = as<NumericVector>(beta_init);
  } else {
    std::fill(beta.begin(), beta.end(), 0.0);
  }

  // Residuals r = y (since beta=0 initially)
  NumericVector r = clone(y);

  // Weights
  NumericVector w(n);
  for (int i = 0; i < n; ++i) {
    w[i] = (nu + 1.0) / (nu + r[i] * r[i]);
  }

  // Initial active set = all predictors
  IntegerVector active_set(p);
  for (int j = 0; j < p; ++j) active_set[j] = j + 1;

  double obj_new = R_PosInf, obj_old = R_PosInf;
  int iter = 0;

  for (iter = 1; iter <= max_iter; ++iter) {
    // Update beta & residuals
    coordinate_descent_update_opt(X, beta, w, r, lambda, active_set);

    // Update weights
    for (int i = 0; i < n; ++i) {
      w[i] = (nu + 1.0) / (nu + r[i] * r[i]);
    }

    // Objective = sum(w * r^2) + lambda * ||beta||_1
    obj_new = 0.0;
    for (int i = 0; i < n; ++i) obj_new += w[i] * r[i] * r[i];
    for (int j = 0; j < p; ++j) obj_new += lambda * std::abs(beta[j]);

    // Check convergence
    if (std::abs(obj_new - obj_old) < tol * (obj_old + 1e-28)) {
      break;
    }
    obj_old = obj_new;

    // (Optional) update active set every 10 iterations
    if (dynamic_active && iter % 10 == 0) {
      NumericVector grad(p);
      for (int j = 0; j < p; ++j) {
        double sum_j = 0.0;
        for (int i = 0; i < n; ++i) {
          sum_j += X(i, j) * w[i] * r[i];
        }
        grad[j] = sum_j;
      }
      double mean_abs_grad = 2* mean(abs(grad));
      std::vector<int> active;
      for (int j = 0; j < p; ++j) {
        if (std::abs(grad[j]) >= mean_abs_grad) active.push_back(j + 1);
      }
      active_set = wrap(active);
    }
  }

  return List::create(
    Named("coefficients") = beta,
    Named("weights") = w,
    Named("iterations") = iter,
    Named("objective") = obj_new
  );
}


// [[Rcpp::export]]
List cv_heavylasso(
    NumericMatrix X,
    NumericVector y,
    NumericVector lambdas,
    int nfolds = 5,
    double nu = 3.0,
    int max_iter = 400,
    double tol = 1e-4
) {
  int n = X.nrow();
  int p = X.ncol();
  int nlambda = lambdas.size();

  // Create fold indices
  IntegerVector fold_id(n);
  for (int i = 0; i < n; i++) {
    fold_id[i] = i % nfolds;
  }
  //on Window
  //std::random_shuffle(fold_id.begin(), fold_id.end());
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(fold_id.begin(), fold_id.end(), g);

  NumericVector cv_errors(nlambda, 0.0);

  for (int f = 0; f < nfolds; f++) {
    // Training and test split
    std::vector<int> train_idx;
    std::vector<int> test_idx;
    for (int i = 0; i < n; i++) {
      if (fold_id[i] == f) test_idx.push_back(i);
      else train_idx.push_back(i);
    }

    int n_train = train_idx.size();
    int n_test  = test_idx.size();

    NumericMatrix X_train(n_train, p), X_test(n_test, p);
    NumericVector y_train(n_train), y_test(n_test);

    for (int ii = 0; ii < n_train; ii++) {
      int i = train_idx[ii];
      y_train[ii] = y[i];
      for (int j = 0; j < p; j++) X_train(ii, j) = X(i, j);
    }
    for (int ii = 0; ii < n_test; ii++) {
      int i = test_idx[ii];
      y_test[ii] = y[i];
      for (int j = 0; j < p; j++) X_test(ii, j) = X(i, j);
    }

    // Loop over lambdas
    for (int l = 0; l < nlambda; l++) {
      double lambda = lambdas[l];

      // Fit model
      List fit = heavylasso_opt(X_train, y_train, lambda, nu, max_iter, tol);

      NumericVector beta = fit["coefficients"];

      // Compute test error
      double mse = compute_mse(X_test, y_test, beta);
      cv_errors[l] += mse;
    }
  }

  // Average errors
  for (int l = 0; l < nlambda; l++) {
    cv_errors[l] /= nfolds;
  }

  // Best lambda
  double best_lambda = lambdas[0];
  double best_error = cv_errors[0];
  for (int l = 1; l < nlambda; l++) {
    if (cv_errors[l] < best_error) {
      best_error = cv_errors[l];
      best_lambda = lambdas[l];
    }
  }

  List fit_final = heavylasso_opt(X, y, best_lambda, nu, max_iter, tol);

  NumericVector beta_best = fit_final["coefficients"];

  return List::create(
    Named("lambda_min") = best_lambda,
    Named("cv_errors") = cv_errors,
    Named("lambdas") = lambdas,
    Named("beta_best") = beta_best
  );
}
