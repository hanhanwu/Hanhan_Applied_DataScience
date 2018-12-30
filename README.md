# Hanhan_Applied_DataScience
Applied data science recommendations and tutorials

## Applied Recommendations
### Data Exploration
#### Univariate Analysis
* Check distribution for continuous, categorical variables
  * For continuous variables, in many cases, I just check percentile at min, 1%, 5%, 25%, 50%, 75%, 95%, 99% and max. This is easier to implement and even more straightforward to find the outliers.
  * For categorical data, you can also find whether there is data inconsistency issue, and based on the cause to preprocess the data
* Check null percentage
* Check distinct values count and percentage
  * Pay attention to features with 0 variance or low variance, think about the causes. Features with 0 variance should be removed before model training.

#### Bivariate Analysis
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
  
#### Deal With Missing Values
* <b>Check whether missing values appear in different values with different probability.</b> This may help understand whether the missing value is missing completely at random (missing with same probability for different values) or whether there are certain values tend to have more missing values and why.
* Deletion
  * List wise deletion - remove the whole list
  * Pair wise deletion - remove missing values for each column, but each column may end up with difference number of records
  * It's safer to use deletion when the missing values are completely missing at random, since deletion will reduce the records and may reduce the prediction power
* Impute with mean/median/mode
  * General Imputation - Replace missing values with selected value in the whole column
  * Similar Case Imputation - For different group of values, impute with the selected value from that group
* KNN Imputation
* Model prediction to compare whether imputing missing values will help

#### Reference
* [Data Exploration Guidance][1]


[1]:https://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/
[2]:https://chrisalbon.com/machine_learning/feature_selection/anova_f-value_for_feature_selection/
[3]:https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif
[4]: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2
