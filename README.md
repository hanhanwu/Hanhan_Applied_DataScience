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

#### Bivariate Analysis
* Check correlation between every 2 continuous features
* 2 way table or Sracked Column Chart - for 2 variable features, check count, percentage of group by the 2 features
* Check chi-square test between 2 categorical features (similar to correlation for continuous features)
  

#### Reference
* [Data Exploration Guidance][1]


[1]:https://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/
