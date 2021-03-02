# Keep Up with Industry Updates 🌺

## [2020 Feb - Latest Scikit-Learn Updates][1]
* Some updates here have already appeared in R before, but still good to know now they are in python too
* Added Stacking
  * It can stack specified estimators
* Added Feature Importance with Permutation
  * By permutating the feature, it explores the importance of feature
* Added ROC-AUC supports multi-class
* Added Imputation missing data with KNN
* Extends Pruning to other tree based algorithms such as random forest and gradient boosting
  * It was implemented in Xgboost and LightGBM

## [MLESAC - Getting better regression results for outlier corrupted data][3]
* Maximum Likelihood Estimator Sample Consensus (MLESAC)
* It adopts the same sampling strategy as RANSAC to generate putative solutions but chooses the solution that maximizes the likelihood rather than just the number of inliers. 


## Useful Python Libraries
* [check this one][2]
  * `Joblib`
    * It helps speed up any function using `Parallel()`, we can compare this with Ray, to use it on Ray function to see the spped improvement
    * More efficient for data dumping and loading than pickle
  * `Black`
    * Auto formating to PEP8
    * Black gives you speed, determinism, and freedom from `pycodestyle` nagging about formatting :D Yeah, `pycodestyle` nags
  * `Pip-review`
    * Auto upgrade packages
    * Iteractively upgrade packages (ask for the approval for each package)
    


[1]:https://www.analyticsvidhya.com/blog/2020/02/everything-you-should-know-scikit-learn/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[2]:https://www.analyticsvidhya.com/blog/2021/01/5-python-packages-every-data-scientist-must-know/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[3]:https://www.analyticsvidhya.com/blog/2021/02/new-approach-for-regression-analysis-ransac-and-mlesac/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29#_ftn5
