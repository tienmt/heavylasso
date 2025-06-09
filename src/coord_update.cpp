#include <Rcpp.h>
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
