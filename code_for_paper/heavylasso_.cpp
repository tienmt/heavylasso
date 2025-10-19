#include <Rcpp.h>
#include <random> 
using namespace Rcpp;

// [[Rcpp::export]]
NumericVector coordinate_descent_update_opt1(
    const NumericMatrix& X,
    const NumericVector& y,           // not used here but kept for signature parity
    NumericVector beta,
    const NumericVector& w,
    NumericVector r,
    double lambda,
    IntegerVector active_set
) {
  int n = X.nrow();
  // clone beta already done by value; beta is local copy (we will return updated)
  NumericVector beta_new = clone(beta);
  
  for (int idx = 0; idx < active_set.size(); ++idx) {
    int j = active_set[idx] - 1; // R -> C++ index
    double beta_old = beta_new[j];
    
    // Compute z = sum w * x * r and A = sum w * x^2 in one pass
    double z = 0.0;
    double A = 0.0;
    for (int i = 0; i < n; ++i) {
      double xij = X(i, j);
      double wi = w[i];
      double ri = r[i];
      z += wi * xij * ri;
      A += wi * xij * xij;
    }
    
    // z_adj corresponds to using residual excluding j: z_adj = z + beta_old * A
    double z_adj = z + beta_old * A;
    
    // safe guard for zero A (or extremely small)
    double beta_j_new = 0.0;
    if (A > 1e-12 && std::abs(z_adj) > lambda) {
      beta_j_new = std::copysign(1.0, z_adj) * (std::abs(z_adj) - lambda) / A;
    } else {
      beta_j_new = 0.0;
    }
    
    double diff = beta_j_new - beta_old;
    if (diff != 0.0) {
      // Update residual r = r - X_j * diff
      for (int i = 0; i < n; ++i) {
        r[i] -= X(i, j) * diff;
      }
    }
    
    beta_new[j] = beta_j_new;
  }
  
  return beta_new;
}


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
    int j = active_set[idx] - 1; // R index to C++
    
    // add back old beta contribution
    for (int i = 0; i < n; ++i) {
      r[i] += X(i,j) * beta[j];
    }
    
    // compute z_j and A_j
    double z_j = 0.0, A_j = 0.0;
    for (int i = 0; i < n; ++i) {
      double xij = X(i,j);
      z_j += w[i] * xij * r[i];
      A_j += w[i] * xij * xij;
    }
    
    // soft-threshold update
    double beta_j_new = 0.0;
    if (std::abs(z_j) > lambda) {
      beta_j_new = std::copysign(1.0, z_j) * (std::abs(z_j) - lambda) / A_j;
      if (std::abs(beta_j_new) < 1e-3) beta_j_new = 0.0;
    }
    
    // subtract new contribution from residual
    for (int i = 0; i < n; ++i) {
      r[i] -= X(i,j) * beta_j_new;
    }
    
    beta[j] = beta_j_new;
  }
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

double compute_mse(NumericMatrix X, NumericVector y, NumericVector beta) {
  int n = X.nrow();
  int p = X.ncol();
  double mse = 0.0;
  double pred;
  double err;
  
  for (int i = 0; i < n; i++) {
    pred = 0.0;
    for (int j = 0; j < p; j++) {
      pred += X(i, j) * beta[j];
    }
    err = y[i] - pred;
    mse += err * err;
  }
  return mse / n;
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
  
  return List::create(
    Named("lambda_min") = best_lambda,
    Named("cv_errors") = cv_errors,
    Named("lambdas") = lambdas
  );
}