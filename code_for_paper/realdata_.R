library(robustHD);library(Rcpp)
sourceCpp("coord_update.cpp") ; source('functions.R')
library(glmnet);library(tictoc);library(hqreg); library(TFRE); library(robustHD)

data("nci60")
# define response variable
Y <- protein[, 92]
# screen most correlated predictor variables
correlations <- apply(gene, 2, corHuber, Y)
keep <- partialOrder(abs(correlations), 150, decreasing = TRUE)
X <- gene[, keep]
X <- scale(X)

out_glmnet = out_lad = out_huber = out_tfre = out_hvt = c()

for (ss in 1:100) {
  test <- sample(1:nrow(X),size = 9,replace = FALSE)
  x <- X[-test,]
  y <- Y[-test]
  xtest <- X[test,]
  ytest <- Y[test]
  
  
  fit_glmnet <- cv.glmnet(x, y, intercept = FALSE,nfolds = 5) ; bet_glmnet <- as.numeric(coef(fit_glmnet)[-1])
  fit_LAD = cv.hqreg(x, y,method ='quantile', nfolds = 5 ) ; bet_LAD <- coef(fit_LAD)[-1]
  fit_Huber = cv.hqreg(x, y,method ='huber', nfolds = 5) ; bet_Huber <- coef(fit_Huber)[-1]
  Obj_TFRE_Lasso <- TFRE(x, matrix(y,ncol = 1),second_stage = "none") ; bet_TFRE <- Obj_TFRE_Lasso$beta_TFRE_Lasso[-1]
  fit_em <- select_lambda_ic(x, y, nu = IQR(y)*4, beta_init= bet_Huber,lambda_seq = 1:40, criterion = 'BIC' ); fit_em$best_lambda ; bet_heavyt <- fit_em$beta_cv
  
  out_glmnet[ss] = mean( (xtest%*% bet_glmnet - ytest)^2 ) 
  out_lad[ss]  = mean( (xtest%*% bet_LAD - ytest)^2 ) 
  out_huber[ss]  =mean( (xtest%*% bet_Huber - ytest)^2 ) 
  out_tfre[ss]  = mean( (xtest%*% bet_TFRE - ytest)^2 ) 
  out_hvt[ss]  = mean( (xtest%*% bet_heavyt - ytest)^2 ) 
  print(ss)
}
save.image(file = 'out_realdata_.rda')
mean(out_glmnet);sd(out_glmnet)
mean(out_lad );sd( out_lad )
mean( out_huber );sd( out_huber )
mean( out_tfre );sd( out_tfre)
mean( out_hvt);sd(out_hvt)

