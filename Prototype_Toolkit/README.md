# Prototype Toolkit

## Fast Prototyping
### PyCaret
* Easy code, interactive visualization for model plot, model comparison, model interpretation, etc.
* [PyCaret Youtube Demos][1]
* [About PyCaret][5]
  * [Example - Plot Model][6]
    * It has many metrics to plot
  * [Example - Classification][7]
    * It's using SHAP for model interpretation
  * Many other things to explore at the left bar of its tutorials
* However, some functions are too basics
  * Association rules only has Apiori... Even Spark has FP-Growth...
  * Things like param tuning looks like a blackbox

## Param Tuning
* [Bayesian Optimization][2]
  * "This is a constrained global optimization package built upon bayesian inference and gaussian process, that attempts to find the maximum value of an unknown function in as few iterations as possible. This technique is particularly suited for optimization of high cost functions, situations where the balance between exploration and exploitation is important."
  * [Example for tuning CatBoost, LightGBM, XGBoost][3]
    * [Description][4]


[1]:https://www.youtube.com/channel/UCxA1YTYJ9BEeo50lxyI_B3g
[2]:https://github.com/fmfn/BayesianOptimization
[3]:https://github.com/dc-aichara/DS-ML-Public/blob/master/Medium_Files/hyp_tune.ipynb
[4]:https://medium.com/analytics-vidhya/hyperparameters-optimization-for-lightgbm-catboost-and-xgboost-regressors-using-bayesian-6e7c495947a9
[5]:https://github.com/pycaret/pycaret
[6]: https://pycaret.org/plot-model/
[7]:https://www.analyticsvidhya.com/blog/2020/05/pycaret-machine-learning-model-seconds/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29

