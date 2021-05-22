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

## General Model Pros & Cons
| Model | Pros | Cons | Description |
| --- | --- | --- | --- |
| Bayesian Classifier | Fast in training and querying large dataset; And can be constantly trained on new coming data without the old data | Cannot deal with the combinations of features caused outcome changes | The basic idea of Bayesian classifier is to get the label's probability by multiplying the probability of each independent features, for example, the category of a text where each token is the feature and the category is the label |
| Decision Tree | Easy to interpret; no feature preprocessing is needed; can handle the combination of features; can handle misxed data type in features (but decision tree regressor cannot) | Not good at dealing with numerical dependent data, a regressor tree can divide the data into mean values with lowest variance, but when the data is complex, the tree can grow very large; higher variance that a small change in the data could largely affect the model output | |
| SVM | Better for high dimensional data (especailly when the # of dimensions is higher than the # of observations); Fast prediction once the hyperplane is found; Affected less by the outliers since the support vectors only use points near the margins | It can be tricky to find the right kernel function when the classes are in nonlinear relationship; It doesn't do well on overlapped classes | SVM is trying to find a line (hyperplane) that can seperate classes clearly by maximize the distances between support vectors (data points near the margins). For nonlinear relationship between classes, we can try kernel trick to convert to a linear problem. Kernel trick is a method that can converts higher dimensional data into lower dimensions without using coordiates of points, but using the inner products between images of all pairs of data | 
| KNN | The reasonging process is easy to understand; | Needs all the data to present when adding new data and therefore can be time consuming; Needs to scale all the features into the same range | It using distance methods to find K most similar objects and the averaged value is the prediction value, `K>1` can reduce the impact from the outlier |

### Unsupervised
* Clustering
  * Hierarchical Clustering: Closest objects grouped together, and later it's the centroid of each group to decide which groups to form a larger group
    * Tree like structure, dendrogram
  * k-means: Divided into distinct groups using distances
  * Density based such as DBSCAN
* Multidimensional Scaling
  * Convert 2+ dimensions into 2 dimensions
  * Original distance using distance methods such as euclidean distance (2+ features), then plot on 2D space with some initial distances. Through multiple iteration that each iteration is trying to reduce the total distance error. Stop until the error cannot be reduced further
* NMF (Non-negative Matrix factorization)
  * It's a method can be used to categorize the data without using labels. Such as categorize text into topics
  * It starts with initialized features matrics and weights matrix, keep updating them according to update rules. Repeat until the product of features matrix and weights matrics is close enough to the data matrix
  
### Optimization
* Simulated Annealing: It's trying to improve the solution by determining the cost for a similar solution that's a small distance away and ina random direction from the solution in the question. If the cost is lower, it became the new solution, otherwise it became the new solution with a certain probability which depends on the tenperature
  * Therefore, with the termperature is higher initially, it tends to accept worse solution (in order to avoid stucking in a local minimum)
  * When the termperature is 0, the algorithm stops
* Genetic Algorithms: It starts with random soltions and in each iteration, it selects the strongest solutions (those with optimal costs) to do mutation or crossover to create a new population, repeat until the population has not improved over several generations

## Relevant Links
### Statistics
* A/B test: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/tree/master/Applied_Statistics/ABTest_Experiments 
* CI, p-value, t-test, z-test
  * https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/pvalue_ttest_ztest.md
  * https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/Learning_Notes/interval_estimation.md 

## Interview Questions
* When tuning tree model's params, would that affect the feature importance?
  * It could. For example, when change the max depth of the model, a single feature (such as numerical feature) could be splited more than once, and the change in max depth could affect this type spliting and therefore affect the feature importance
* When there are lots of features, how could you tell which features are useful?
  * Check the feature correlation with the label first
  * Feature importance, sometimes even go deeper to check how does each feature affect the positive/negative class
* How to interpret p-value, coefficients, r-square in linear regression output
  * Coefficient indicates the change in the response variable when a unit of an independent variable changes while all the other independent variables stay constant
  * The null hypothesis is a coefficient is 0, so when p-value is larger than the significant level, accept the null hypothesis, then the coefficient is 0 and the independent variable is not important to the linear regression model
  * R-Square = Expected Variance/Total Variance. It's the percentage of the response variable variation that is explained by the model.
    * Better to check residual plot with R square, if there is a pattern in the residual plot instead of the randomness, then even if R square is high, there is still unexplained pattern in the data
* How are you going to find the root cause
  * Check correlation, feaure importance to find potenail factors
  * To check cause, might use A/B test to experiment: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/ABTest_Experiments/detailed_ABTest_Experiment.ipynb
    * Factor X in 2 groups x1, x2
    * H0: x1-x2=0
    * Estimate sample size if necessary, which needs the probaility of Type I, II error, detectable effect, baseline metrics, std1 and std2 can be caluclated from detectable effect and baseline metrics
    * To check whether the differece between x1, x2 is expected (if so, accept H0), we need confidence interval, which needs p_hat and the margin of error
      * `p_hat = control group metric/2 group totoal metric`
      * `margin of error` needs z score (calculated from confidence level), std
* Explain the difference between L1, L2
  * https://www.quora.com/What-is-the-difference-between-L1-and-L2-regularization-How-does-it-solve-the-problem-of-overfitting-Which-regularizer-to-use-and-when
  * https://www.analyticsvidhya.com/blog/2021/05/complete-guide-to-regularization-techniques-in-machine-learning/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * The square, circle regions in L1, L2 are the feasible region for the regularization
  * The contours represent the different loss values for unregularized regression model
  * Because the square is more angular, it's more likely for the contours to get min loss when a coefficient is 0 (the location is the intersection point of the feasible region boundry and the x, y axis, x, y are the coefficients). This is also why L1 tend to be used for feature selection since some features will get 0 coefficient
* Explain the math behind PCA
  * `SNR (signal to noise ratio) = P_signal/P_noise`, signal represents all the valid values within a variable's value range, they are the wanted components; the noise part is caused by random factors and should be removed. The objective of PCA is trying to increase the signal content while reducing the noise.
  * PCA is basically one type of Singular Value Decomposition (SVD), meaning we are breaking or decomposing a larger value (i.e. a singular value) into smaller values.
    * PCA covariance matrix converts (larger values) to eigenvectors (smaller values, priciple components), A*A_transform, A is the feature matrix
    * Earlier principle components captures more covariance info
* How does random forest, xgboost handle missing data
  * Random forest tend to use average/mode values (either by imputation with average, either by rough average/mode, either by an averaging/mode based on proximities)
  * Xgboost put missing data record to the right node by default (if there is no mssing data in the training), otherwise it decides left or right node depends on the minimized loss; similar for lightGBM
