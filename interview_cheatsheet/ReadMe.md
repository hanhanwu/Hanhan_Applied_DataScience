# Interview Cheatsheet 🍀

## Data Exploration & Preprocessing
* https://github.com/hanhanwu/Hanhan_Applied_DataScience#data-exploration
* https://github.com/hanhanwu/Hanhan_Applied_DataScience/blob/master/data_exploration_functions.py

## Model Training & Param Tuning Examples
* xgboost with cv: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/try_lightfGBM_cv.ipynb
  * `xgb.cv` doesn't work for some datasets, the split function might return error
    * Such as this dataset, I'm using sklearn cross validation to do it: https://github.com/hanhanwu/Hanhan_Applied_DataScience/blob/master/interview_cheatsheet/xgboost_CV_HPO.ipynb
      * Optuna is a great param tuning tool to be used with sklearn cross validation
    * However, `xgb.cv` could return the number of optimal n_estimators for xgboost, but sklearn `cross_val_score` noly returns the metrics of each fold. 
    * To help optimize params, bothneed to work with param tuning
* xgboost param tuning:
  * What often to tune: https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/Experiences.md#tune-xgboost
  * All params: https://xgboost.readthedocs.io/en/latest/parameter.html
* lightGBM with cv: https://github.com/hanhanwu/Hanhan_COLAB_Experiemnts/blob/master/link_prediction.ipynb
* Optuna param tuning: https://colab.research.google.com/github/optuna/optuna/blob/master/examples/quickstart.ipynb
  * random forest was used as the example
  * Optuna is a great param tuning tool to be used with sklearn cross validation
  * It's also fast and convenient to tune multiple models together

## Model Evaluation
* sklearn metrics: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
