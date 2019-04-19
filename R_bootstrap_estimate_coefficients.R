library(ISLR)
library(boot)

set.seed(10)
attach(Auto)  # attach the dataset
summary(Auto)
dim(Auto)  # 392, 9

# Using bootstrap to estimate the accuracy of Linear Regression Model
## this function returns the estimated coefficients of linear regression,
## in this case the estimated coefficients are the intercept(β0) and the slope(β1)
boot_fu <- function(data, index){
  return(coef(lm(mpg~horsepower, data = data, subset = index)))
}
boot_fu(Auto, 1:dim(Auto)[1])


# With boot() function, we can do bootstrap by
## computing the standard errors of 1,000 bootstrap estimates for the intercept and slope terms
## t1 is β0, t2 is β1, also standard errors for both
boot(data = Auto, statistic = boot_fu, R = 1000)


# Comparing with bootstrap, we can get standard errors and 
## estimated coefficients using standard formulas summary()
summary(lm(mpg~horsepower, data = Auto))$coef

# ------------------------------------------------ #
# Standard formulas makes assumptions,
# while bootstrap is nonparametric.
# So bootstrap results can be more reliable.
# ------------------------------------------------ #


# Using bootstrap to estimate coefficients of the quadratic model
# t1 is β0, t2 is β1, t3 is β2
## bootstrap results
set.seed(10)
boot_fu_qua <- function(data, index){
  return(coef(lm(mpg~horsepower+I(horsepower^2), data = data, subset = index)))
}
boot(data = Auto, statistic = boot_fu_qua, R = 1000)

# standard formulas results
summary(lm(mpg~horsepower+I(horsepower^2),data=Auto))$coef
