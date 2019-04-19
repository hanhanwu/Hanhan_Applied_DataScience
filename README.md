# Hanhan_Applied_DataScience
Applied data science recommendations and tutorials

# Applied Recommendations
## Data Exploration
* [My Code - IPython][11]
  * When you are using `chi2` or `f_classif`, the features should have no NULL, no negative value.
* [Data Exploration Code I often use][42]
### Univariate Analysis
* Check distribution for continuous, categorical variables
  * For continuous variables, in many cases, I just check percentile at min, 1%, 5%, 25%, 50%, 75%, 95%, 99% and max. This is easier to implement and even more straightforward to find the outliers.
  * For categorical data, you can also find whether there is data inconsistency issue, and based on the cause to preprocess the data
* Check null percentage
* Check distinct values count and percentage
  * Pay attention to features with 0 variance or low variance, think about the causes. Features with 0 variance should be removed before model training.
  * For categorical features, besides count of unique values, we can also use [Simpson's Diversity Index][50]

### Bivariate Analysis
* Check correlation between every 2 continuous variables
* 2 way table or Stacked Column Chart - for 2 variable variables, check count, percentage of group by the 2 variables
* Check chi-square test between 2 categorical features (similar to correlation for continuous features)
  * probability 0 means the 2 variables are dependent; 1 means independent; a value x in [0,1] range means the dependence between the 2 variables is at `(1-x)*100%`
  * [sklearn chi-square][4]
* Check ANOVA between categorical and continuous variables
  * [Example to check ANOVA f-value][2]
  * [sklearn ANOVA][3]
  * Lower F-score the higher dependent between the variables
  * Besides ANOVA, we could calculate t-score or z-score (less than 30 records)
* For features that are highly dependent on each other, in parametrics algorithms these features should be removed to reduce error; in nonparametrics algorithms, I still recommend to remove them, because for algorithms such as trees tend to put highly dependent features at the same importance level, which won't contribute feature importance much.
* Feature selection based on the dependency between features and the label
  * Select features that have stronger correlation with the label, because these features tend to contribute to the label values.
  * correlation: continuous features vs continuous label
  * chi-square: categorical features vs categorical label
    * Cramer’s V for Nominal Categorical Variable (categorical values in numerical format)
    * Mantel-Haenszed Chi-Square for ordinal categorical variable (categorical values in order)
  * ANOVA: categorical features vs cotinuous label; or vice versa
  
### Deal With Missing Values
* <b>Check whether missing values appear in different values with different probability.</b> This may help understand whether the missing value is missing completely at random (missing with same probability for different values) or whether there are certain values tend to have more missing values and why.
* Deletion
  * List wise deletion - remove the whole list
  * Pair wise deletion - remove missing values for each column, but each column may end up with difference number of records
  * It's safer to use deletion when the missing values are completely missing at random, since deletion will reduce the records and may reduce the prediction power
* Impute with mean/median/mode
  * General Imputation - Replace missing values with selected value in the whole column
  * Similar Case Imputation - For different group of values, impute with the selected value from that group
* Impute with special values, such as "MISSING", -1, etc.
* KNN Imputation
* Model prediction to compare whether imputing missing values will help

### Deal With Outliers
* To check whether there is ourliers, I normally check distribution, even just by using percentile. And only deal with a very small percentage of outliers.
  * It's recommended that to use the model run on raw features without any preprocessing, this is the baseline result. Then run the same model with preprocessed data, to compare. So that you will know how much does preprocessing affect your prediction in each project.
* Sometimes I also check boxplot, but the data I am dealing with tend to have large amount of outliers, and imputing data above & below 1.5*IQR will reduce the prediction power too much.
* <b>Better to know what caused the outliers</b>, this may help you decide how to deal with them
* To deal with ourliers, I normally use:
  * Simply replace the outliers with NULL
  * Replace outliers with median/mean/mode, or a special value
  * Binning the feature
  * Just leave it there
  * Anoter suggest method is to build seperate models for normal data and outliers when there are large amount of outliers which are not caused by errors. In industry, you may not be allowed to build seperate models but it's still a method to consider.
  
### Dimensional Reduction
* Check <b>missing values</b>, for those with high percentage of missing values, you may want to remove them.
* Check variance
* Deal with collinearity issues
  * Correlation matrix to check the correlation between pairs of features - Collinearity
  * VIF (Variance Inflation Factor) to check the correlation exists between 3+ features but could not be found in any pair of features - Multicollinearity
    * How to use this method, check the code [here][11], "Check Multicollinearity" section.
    * Normally when VIF is between 5 and 10, there could be multicollineary issue of the feature. When VIF > 10, it's too high and the feature should be removed.
    * 2 major methods to deal with features with high VIF
      * Remove it
      * Combine features with high VIF into 1 feature
    * [The implementation of using VIF to drop features][39]
    * [Description of VIF][40]
      * The Variance Inflation Factor (VIF) is a measure of colinearity among predictor variables within a multiple regression. It is calculated by taking the the ratio of the variance of all a given model's betas divide by the variane of a single beta if it were fit alone.
* Use tree models to find feature importance
  * Better to remove highly correlated features before doing this. Some tree model will put highly correlated features all as important if one of them is highly ranked
* Dimensional Reduction Algorithms
  * [sklearn decomposition][15]
* Feature Selection Methods
  * [sklearn feature selection][16]
  
### Feature Engineering
* Scaling the features
  * Sometimes, you want all the features to be normalized into the same scale, this is especially helpful in parametric algorithms.
  * [sklearn scaling methods on data with outliers][5]
    * PowerTransformer is better in dealing with outliers
    * Sometimes, you just want to scale the values between 0 and 1 so MaxMinScaler is still popular
* Transform nonlinear relationship to linear relationship, this is not only easier to comprehend, but also required for parametric algorithms
  * scatter plot to check the relationship; we can also use pandas cross tab method
  * Such as `log`, here are a list of methods for nonlinear transformation: https://people.revoledu.com/kardi/tutorial/Regression/nonlinear/NonLinearTransformation.htm
  * PCA - It can convert the whole feature set into normalized linear combination
    * [How to use PCA for dimensional reduction/feature selection][29]
    * The components are independent from each other, earlier components capture more variance. Each principle component is also a linear combination of original features
* Convert skewed distribution to symmetric distribution, this is also prefered for parametric algorithms. Skewed data could reduce the impact from lower values which may also important to the prediction.
  * `log` to deal with righ skewness
  * squre root, cube root
  * `exp` to deal with left skewness
  * `power`
  * binning
* derived features
* one-hot features
* Methods to deal with categorical features
  * [Categorical Encoding Methods][37]
    * It has one-hot and other label encoding methods
    * `TargetEncoder` can be used when the categorical column contains lots of uniquen values (High-Cardinality), such as zipcode
  * Convert to value frequency or response rate, aggregated value. Also for categorical value, we can use part of the value and convert to numerical values

## Deal With Imbalanced Data
### Sampling Methods
* There are oversampling, undersampling and synthetic sampling (which is also oversampling), combined sampling (oversampling + undersampling). In practice, I tried different methods in different projects, so far non of them worked well in both training and testing data.
### Cost Sensitive Learning
* This method is becoming more and more popular recently. Majorly you just set the class weights based on the importance of false positive and false negative.
* In practice, it is worthy to know more from the customers or the marketing team, trying to understand the cost of TP/TN or FP/FN.
* Example-dependent Cost Sensitive method
  * [Costcla][36] - it's a package that do model prediction including cost matrix
    * The cost matrix is example dependent. Each row has the cost of [FP, FN, TP, TN]. So when you are creating this cost matrix yourself, your training & testing data all records the costs, each row of your data is an example. [Sample Model Prediction][36]
    * The drawback is, testing data also needs the cost matrix, but in practice, you may not know. However this method can still be used in train-validation dataset, in order to find the optimal model.
### Thresholding
* When the prediction result is in probability format, we can change the threshold of class prediction. By default the reshold is 50-50. With the evaluation metric, better to <b>draw a curve with thresholds as x-axis and evaluation result as y-axis</b>, so that we will know which threshold to choose is better.
<img src="https://github.com/hanhanwu/Hanhan_Applied_DataScience/blob/master/images/error_rate_thresholds.png" width="500" height="200">


### Other Methods
* Clustering & Multiple Model Training
  * Cluster the majority class into multiple non-overlapped clusters. For each cluster, train them with the minority class and build a model. Average the final prediction
  * You don't need testing data here, but the drawback is you always need the label which won't be the case in practice. When there are more data, you also need to operate on all the data, which can be time & computational consuming.

### Reference
* [Data Exploration Guidance][1]
* [Impact of Data Size on Model Performance][6]
  * It's using deep learning, but can inspire on multiple other models, especially those using stochastic methods
  * Use bootstrap methods + multiple seeds to do multiple run, cross validatio are suggested. Because for stochastic methods, different seed could lead to different direction and optimal results. It could also help dealing with the high variance issue in a model.
  * "As such, we refer to neural network models as having a low bias and a high variance. They have a low bias because the approach makes few assumptions about the mathematical functional form of the mapping function. They have a high variance because they are sensitive to the specific examples used to train the model."
  * Larger training set tend to lead to better prediction results, smaller test case may not always be the case. So 7:3 or 8:2 train-test ratio is always a good start.
* [Guidance on dealing with imbalnced data 1][30], [Guidance on dealing with imbalnced data 2][31]
  
## Models
[My code][12]
### Which Model to Choose
* Linear or Nonlinear
  * In the code [here][11], we can use residual plot to check whether there is linear/non-linear relationship between the feature set and the label, to decide to use linear/nonlinear model.
    * If the residual plot is showing funnel shape, it indicates non-constant variance in error terms (heteroscedasticity), which also tends to have residuals increase with the response value (Y). So we can also try to use a concave function (such as log, sqrt) to transform `Y` to a much smaller value, in order to reduce heteroscedasticity. 
    * As we can see in the code, after transforming `Y` with `log`, the residual plot was showing a much more linear relationship (the residuals are having a more constant variance)
* TPOT Automatic Machine Learning
  * It's a very nice tool that helps you find the model with optimized param, and also export the python code for using the selected model [TPOT Examples][7]
  * [TPOT Params for Estimators][8]
    * 10-fold of cross validation, 5 CV iterations by default
    * TPOT will evaluate `population_size + generations × offspring_size` pipelines in total. So when the dataset is large, it can be slow.
  * [All the classifiers that TPOT supports][9]
  * [All the regressors that TPOT supports][10]
  * Different TPOT runs may result in different pipeline recommendations. TPOT's optimization algorithm is stochastic in nature, which means that it uses randomness (in part) to search the possible pipeline space.
  * <b>The suggestion here is:</b> Run TPOT multiple times with different random_state, narrow down to fewer models and do further evaluation (see below spot check pipeline). If the data size is large, try to reduce `population_size + generations × offspring_size`, `cv` and use `subsample` 
  * [Sample exported TPOT python file][13]
* Spot-check Model Evaluation
  * After you have a few list of models, you want to quickly check which one performs better. What I'm doing here is, for each model, use all the training data but with stratified kfold cross validation. Finally it evaluates the average score and the score variance.
    * Some people think it's important to use bootstrap, which split the data into multiple folds and run on each fold or it will use different seeds to run the same model multiple times. I'm not using bootstrap here, because the first solution is similar to stratified kfold cross valiation, the second solution, I will use it when I have finalized 1 model, and use bootstrap to deliver the final evaluation results.

## Algorithms Details
### SVM
* [How did margin generated][33] - margin has maximized orthogonal distance between the cloest points in each category and the hyperplane, these closest points are supporting vectors
* [How does kernal SVM work for nonlinear data][34]
  * Gaussian kernal is `rbf` kernal in sklearn

## Model Evaluation
### Before Evaluation
* There are things we can do to make the evaluation more reliable:
  * Hold-out Data
    * If the dataset is large enough, it's always better to have a piece of hold-out data, and always use this piece of data to evaluate the models
  * Run the same model multiple times
    * You can try different seeds with the whole dataset
    * Bootstrap - run the model on randomly selected samples, finally aggregate the results
      * <b>Resample with replacement</b>
      * UNIFORMALY random draw
      * The advantage of draw with replacement is, the distribution of the data won't be changed when you are drawing
      * [sklearn methods in data spliting][28]
      * Statistics proved that Bootstrap has close estimates as using the true population, when you have selected enough amount of samples (similar to central theorem)
      * [My code - Using Bootstrap to Estimate the Coefficients of Linear Regression][51]
        * Applied on linear regression, and quadratic linear regression
        * It compares bootstrap results and standard formulas results. Even though both got the same estimated coefficients, standard formulas tend to show lower standard errors. However sinece standard formulas makes assumptions, while bootstrap is nonparametric, bootstrap tends to be more reliable

  * Cross Validation
    * We can calculate the average evaluation score and the score variance to compare model performance
    * In sklearn, we can just use [cross_val_score][23], which allows us either to use integer as stratified kfold cv folds, or [cross validation instances][24]
### Evaluation Methods
* "Probability" means predict continuous values (such as probability, or regression results), "Response" means predict specific classes
* [sklearn classification evaluation metrics][19]
* [sklearn regression evaluation metrics][20]
  * Logloss works better for discrete numerical target; MSE/RMSE works better for continuous target
* [sklearn clustering metrics][21]
  * Still needs labels
  * Sometimes, in simply do not have the label at all. You can:
    * Compare with the datasets that have label, at the risk of non-transformable
    * Check predicted results distribution, comparing between different rounds
    * Your customer will expect a percentage for each label, compare with that...
* Suggestions to improve scores
    * Making the probabilities less sharp (less confident). This means adjusting the predicted probabilities away from the hard 0 and 1 bounds to limit the impact of penalties of being completely wrong.
    * Shift the distribution to the naive prediction (base rate). This means shifting the mean of the predicted probabilities to the probability of the base rate, such as 0.5 for a balanced prediction problem.
    * [Reference][41]
* How to Compare 2 Models After Each has been Running Multiple Rounds
  * [Recommended t-table][49]
<img src="https://github.com/hanhanwu/Hanhan_Applied_DataScience/blob/master/images/compare_2_model_multi_rounds_results.png" width="700" height="400">


### After Evaluation - The confidence/reliability of prediction
* [Calibration][22]
* [Concordant & Discordant][25]
* [KS Test][26] - Kolmogorov-Smirnov (K-S) chart is a measure of the degree of separation between the positive and negative distributions.

## Time Series Analysis Cheatsheet
* Data Exploration
  * Lagged Features
    * When you features are created by lagged data, you can check the correlation between these lags. If the correlation is close to 0, it means current values have no correlation to previous time values
* If you want a quick forecast, such as hourly, daily, weekly, yearly forecast, try Facebook Prophet
  * [My Prophet Python code][43]
  * [My Prophet R code][44]
* [My code - time series functions][45]
* [11 Forecasting methods][46]
  * When to use which model
  * It seems that methods do not have `forcast` but only have `predict` will only predict the next value. With `forecast` we can predict the next period of values
* [My code with 7 methods][47]
  * Some are not in above 11 methods
  * Simple, Exponential & Weighted Moving Average
    * Simple moving average (SMA) calculates an average of the last n prices
    * The exponential moving average (EMA) is a weighted average of the last n prices, where the weighting decreases exponentially with each previous price/period. In other words, the formula gives recent prices more weight than past prices.
    * The weighted moving average (WMA) gives you a weighted average of the last n prices, where the weighting decreases with each previous price. This works similarly to the EMA.
  * [Grid Search to tune ARIMA params][48]

#### Reference
[My model evaluation previous detailed summary][27]

## Tools
### R Tools
#### Data Manipulation Tools
* `dplyr` - majorly to do query related operation, such as "select", "group by", "filter", "arrage", "mutate", etc.
* `data.table` - fast, even more convenient than data.frame, can also do queries inside
* `ggplot2`
* `reshape2` - reshape the data, such as "melt", "dcast", "acast" (reversed melt)
* `readr` - read files faster, different functions supports different files
* `tidyr` - makes the data "tidy". "gather" is similar to above "melt" in `reshape2`; "seperate", "sperate_d" could help seperate 1 column into multiple columns and vice versa, etc.
* `lubridate` - deal witb datetime
* [My code of R 5 packages for dealing with missing values][17]
  * `MICE` - it assumes that the missing data are Missing at Random (MAR), which means that the probability that a value is missing depends only on observed value and can be predicted using them.
  * `Amelia` - It assumpes that All variables in a data set have Multivariate Normal Distribution (MVN). It uses means and covariances to summarize data. 
  * `missForest` - It builds a random forest model for each variable. Then it uses the model to predict missing values in the variable with the help of observed values.
  * `Hmisc` - It automatically recognizes the variables types and uses bootstrap sample and predictive mean matching to impute missing values. <b>You don’t need to separate or treat categorical variable</b>. It assumes linearity in the variables being predicted.
  * `mi` - It allows graphical diagnostics of imputation models and convergence of imputation process. It uses bayesian version of regression models to handle issue of separation. Imputation model specification is similar to regression output in R. It automatically detects irregularities in data such as high collinearity among variables. Also, it adds noise to imputation process to solve the problem of additive constraints.
  * <b>Recommend to start with missForest, Hmisc, MICE</b>, and then try others

#### reference
* [7 R data manipulation tools][14]
* [My code of R 5 packages for dealing with missing values][17]
  * [Original tutorial][18]
  
### Python Tools
* [Some quick methods in pandas][32]
  * pivot table - generate aggregate results for multiple columns
  * multi-indexing - using the values of multiple columns as the index to locate
  * cross-tab - this can be used to check whether a feature affects the label with percentage value
  * cut - binning
  
# [My Production Solutions Notes][38]
* It records my simple solutions in production deployment.


[1]:https://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/
[2]:https://chrisalbon.com/machine_learning/feature_selection/anova_f-value_for_feature_selection/
[3]:https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif
[4]:https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2
[5]:https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
[6]:https://machinelearningmastery.com/impact-of-dataset-size-on-deep-learning-model-skill-and-performance-estimates/
[7]:https://github.com/EpistasisLab/tpot
[8]:https://epistasislab.github.io/tpot/api/
[9]:https://github.com/EpistasisLab/tpot/blob/master/tpot/config/classifier.py
[10]:https://github.com/EpistasisLab/tpot/blob/master/tpot/config/regressor.py
[11]:https://github.com/hanhanwu/Hanhan_Applied_DataScience/blob/master/data_exploration.ipynb
[12]:https://github.com/hanhanwu/Hanhan_Applied_DataScience/blob/master/model_selection.ipynb
[13]:https://github.com/hanhanwu/Hanhan_Applied_DataScience/blob/master/tpot_bigmart_pipeline0.py
[14]: https://www.analyticsvidhya.com/blog/2015/12/faster-data-manipulation-7-packages/
[15]:https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition
[16]:https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection
[17]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/5R_packages_for_missing_values.R
[18]:https://www.analyticsvidhya.com/blog/2016/03/tutorial-powerful-packages-imputing-missing-values/?utm_content=buffer916b5&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
[19]:https://scikit-learn.org/stable/modules/classes.html#classification-metrics
[20]:https://scikit-learn.org/stable/modules/classes.html#regression-metrics
[21]:https://scikit-learn.org/stable/modules/classes.html#clustering-metrics
[22]:https://github.com/hanhanwu/Hanhan_Data_Science_Resources2/blob/master/about_calibration.ipynb
[23]:https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score
[24]:https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
[25]:https://www.listendata.com/2014/08/modeling-tips-calculating-concordant.html
[26]:https://stackoverflow.com/questions/10884668/two-sample-kolmogorov-smirnov-test-in-python-scipy?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
[27]:https://github.com/hanhanwu/Hanhan_Data_Science_Resources2#model-evaluation
[28]:https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
[29]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/make_sense_dimension_reduction.ipynb
[30]:https://www.analyticsvidhya.com/blog/2016/03/practical-guide-deal-imbalanced-classification-problems/?utm_content=buffer929f7&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
[31]:https://stackoverflow.com/questions/26221312/dealing-with-the-class-imbalance-in-binary-classification
[32]:https://www.analyticsvidhya.com/blog/2016/01/12-pandas-techniques-python-data-manipulation/?utm_content=bufferfa8d9&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
[33]:https://nlp.stanford.edu/IR-book/html/htmledition/support-vector-machines-the-linearly-separable-case-1.html
[34]:https://albahnsen.com/2018/09/13/machine-learning-algorithms-explained-support-vector-machines/
[35]:https://github.com/albahnsen/CostSensitiveClassification
[36]:http://albahnsen.github.io/CostSensitiveClassification/ThresholdingOptimization.html
[37]:https://github.com/scikit-learn-contrib/categorical-encoding
[38]:https://github.com/hanhanwu/Hanhan_Applied_DataScience/blob/master/Simple%20Production%20Solutions.ipynb
[39]:https://www.kaggle.com/ffisegydd/sklearn-multicollinearity-class
[40]:https://etav.github.io/python/vif_factor_python.html
[41]:https://machinelearningmastery.com/how-to-score-probability-predictions-in-python/
[42]:https://github.com/hanhanwu/Hanhan_Applied_DataScience/blob/master/data_exploration_functions.py
[43]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/try_prophet.ipynb
[44]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/try_prophet_R.R
[45]:https://github.com/hanhanwu/Hanhan_Applied_DataScience/blob/master/time_series_functions.py
[46]:https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/
[47]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/time_series_forecasting.ipynb
[48]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/ARIMA_grid_search.ipynb
[49]:http://www.sjsu.edu/faculty/gerstman/StatPrimer/t-table.pdf
[50]:https://geographyfieldwork.com/Simpson'sDiversityIndex.htm
[51]:https://github.com/hanhanwu/Hanhan_Applied_DataScience/blob/master/R_bootstrap_estimate_coefficients.R
