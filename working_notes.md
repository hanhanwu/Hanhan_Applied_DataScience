# Working Notes

## Regression Problem
* `mape` is still popular when you need to keep a balance between over estimation and under estimation
* Quantile regression with LGBM can help address over/under estimation
  * See how does `alpha` helps under/over estimation: http://ethen8181.github.io/machine-learning/ab_tests/quantile_regression/quantile_regression.html
  * In LGBM, set `objective` as "quantile", tune `alpha` 
