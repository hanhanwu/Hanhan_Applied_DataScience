# check metrics strings can be used in "scoring" param
## NOTE: `from sklearn.metrics import` does not related to the strings used in `scoring`
# https://scikit-learn.org/stable/modules/model_evaluation.html
import sklearn
sorted(sklearn.metrics.SCORERS.keys())
