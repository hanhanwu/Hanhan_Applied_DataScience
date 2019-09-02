# Case Study Notes

## Airbnb Experiences - Machine Learning Powered Search Ranking

* When creating personalization features it is very important not to “leak the label”
  * Such as exposing some information that happened after the event used for creating the label.
* When using GBDT (gradient boost decision tree), one does not need to worry much about scaling the feature values, or missing values (they can be left as is). However, one important factor to take into account is that, unlike in a linear model, <b>using raw counts as features in a tree-based model to make tree traversal decisions may be problematic when those counts are prone to change rapidly in a fast growing marketplace. In that case, it is better to use ratios of fractions.</b>
* [Reference][1]


[1]:https://medium.com/airbnb-engineering/machine-learning-powered-search-ranking-of-airbnb-experiences-110b4b1a0789
