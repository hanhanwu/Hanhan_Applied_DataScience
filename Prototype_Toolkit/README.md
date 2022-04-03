# Prototype Toolkit

## Fast Prototyping
### [Some advanced tools][10]
* Tools like TPOT, MLBOX can do both params tuning and model selection
  * MLBox also do some data preprocessing
* Tools like HungaBunga, LazyPredict runs on a list of models and show you a list of results  

### PyCaret
* Easy code, interactive visualization for model plot, model comparison, model interpretation, etc.
* [PyCaret Youtube Demos][1]
* [About PyCaret][5]
  * [Example - Plot Model][6]
    * It has many metrics to plot
  * [Example - Classification][7]
    * It's using SHAP for model interpretation
  * [Example - Deploy Model on AWS][8]
  * [Example - Feature Engineering][9]
    * I haven't used "Trigonometry" or "Ploynomial" features before
  * More to explore at the left bar of its tutorials!
* However, some functions are too basics
  * Association rules only has Apiori... Even Spark has FP-Growth...
  * Things like param tuning looks like a blackbox

### [Mercury - Convert Jupyter Notebook to Web App][11]
* Mercury Github: https://github.com/mljar/mercury
* [An example][12]



[1]:https://www.youtube.com/channel/UCxA1YTYJ9BEeo50lxyI_B3g
[2]:https://github.com/fmfn/BayesianOptimization
[3]:https://github.com/dc-aichara/DS-ML-Public/blob/master/Medium_Files/hyp_tune.ipynb
[4]:https://medium.com/analytics-vidhya/hyperparameters-optimization-for-lightgbm-catboost-and-xgboost-regressors-using-bayesian-6e7c495947a9
[5]:https://github.com/pycaret/pycaret
[6]: https://pycaret.org/plot-model/
[7]:https://www.analyticsvidhya.com/blog/2020/05/pycaret-machine-learning-model-seconds/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[8]:https://pycaret.org/clustering/
[9]:https://pycaret.org/trigonometry-features/
[10]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/README.md#advanced-tools
[11]:https://mljar.com/mercury/
[12]: https://www.analyticsvidhya.com/blog/2022/04/how-to-convert-jupyter-notebook-into-ml-web-app/?utm_source=feedburner&utm_medium=email
