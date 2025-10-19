library(Rcpp) ; sourceCpp("heavylasso_.cpp") ; source('functions.R')
library(glmnet);library(tictoc);library(hqreg); library(TFRE); library(robustHD);library(sn)

n_test = 10000
n <- n_test + 300
p <- 1000
s0 = 10
beta_true <- rep(0, p) ; beta_true[1:s0] <- c( rep(1, s0/2) , rep(-1, s0/2) )
rho <- 0.0
Sigma <- outer(1:p, 1:p, function(i, j)rho^abs(i - j)) ; LL = chol(Sigma) 

glmnet_out = heavlasso_out = hvlassoCV_out = LAD_out = huber_out = TFRE_out = list()

for (ss in 1:50) {
  #noise_sn <- rsn(n, xi = xi, omega = omega, alpha = alpha)
  X1 <- matrix(rnorm(n * p), n, p)%*% LL
  y1 <- X1 %*% beta_true + rt(n, df = 3)  # noise_sn # rnorm(n)  # rcauchy(n) # rt(n, df = 3) #   # Student-t noise
  
  y = y1[-(1:n_test),]
  #y = add_outliers_y(y,proportion = 0.1)
  X = X1[-(1:n_test),]
  ytest = y1[ 1:n_test,] ; 
  xtest = X1[1:n_test,]
  
  fit_glmnet <- cv.glmnet(X, y, intercept = FALSE,nfolds = 5) ; bet_glmnet <- as.numeric(coef(fit_glmnet)[-1])
  fit_LAD = cv.hqreg(X, y,method ='quantile', nfolds = 5 ) ; bet_LAD <- coef(fit_LAD)[-1]
  fit_Huber = cv.hqreg(X, y,method ='huber', nfolds = 5) ; bet_Huber <- coef(fit_Huber)[-1]
  Obj_TFRE_Lasso <- TFRE(X, matrix(y,ncol = 1),second_stage = "none") ; bet_TFRE <- Obj_TFRE_Lasso$beta_TFRE_Lasso[-1]
  fit_em <- select_lambda_ic(X, y, nu = IQR(y)*4 ); bet_heavyt <- fit_em$beta_cv
  cvfit.heavylasso <- cv_heavylasso(X, y, nu = IQR(y)*4,lambdas = 1:50 ) ; bet_heavyt2 <- heavylasso_opt(X, y, nu = IQR(y)*4,lambda = cvfit.heavylasso$lambda_min)$coefficients
  
  
  heavlasso_out[[ss]] = c(sum( (bet_heavyt - beta_true)^2 ), sum( abs(bet_heavyt - beta_true) ) , mean( (X%*%bet_heavyt - X%*%beta_true)^2 ) , mean( (xtest%*% bet_heavyt - ytest)^2 ),
                          selection_metrics(true_support = 1:s0,selected_support = which(bet_heavyt !=0),p = p) )
  hvlassoCV_out[[ss]] = c(sum( (bet_heavyt2 - beta_true)^2 ), sum( abs(bet_heavyt2 - beta_true) ) , mean( (X%*%bet_heavyt2 - X%*%beta_true)^2 ) , mean( (xtest%*% bet_heavyt2 - ytest)^2 ),
                          selection_metrics(true_support = 1:s0,selected_support = which(bet_heavyt2 !=0),p = p) )
  glmnet_out[[ss]] = c(sum( (bet_glmnet - beta_true)^2 ), sum( abs(bet_glmnet - beta_true) ) , mean( (X%*%bet_glmnet - X%*%beta_true)^2 ) , mean( (xtest%*% bet_glmnet - ytest)^2 ),
                       selection_metrics(true_support = 1:s0,selected_support = which(bet_glmnet !=0),p = p) )
  LAD_out[[ss]] = c(sum( (bet_LAD - beta_true)^2 ), sum( abs(bet_LAD - beta_true) ) , mean( (X%*%bet_LAD - X%*%beta_true)^2 ) , mean( (xtest%*% bet_LAD - ytest)^2 ), 
                    selection_metrics(true_support = 1:s0,selected_support = which(bet_LAD !=0),p = p)  )
  huber_out[[ss]] = c(sum( (bet_Huber - beta_true)^2 ), sum( abs(bet_Huber - beta_true) ) , mean( (X%*%bet_Huber - X%*%beta_true)^2 ) , mean( (xtest%*% bet_Huber - ytest)^2 ),
                      selection_metrics(true_support = 1:s0,selected_support = which(bet_Huber !=0),p = p)  )
  TFRE_out[[ss]] = c(sum( (bet_TFRE - beta_true)^2 ), sum( abs(bet_TFRE - beta_true) ) , mean( (X%*%bet_TFRE - X%*%beta_true)^2 ) , mean( (xtest%*% bet_TFRE - ytest)^2 ),
                     selection_metrics(true_support = 1:s0,selected_support = which(bet_TFRE !=0),p = p)  )
  print(ss)
}
save.image(file = './output/sim_n300p1000s10_stu_.rda')

