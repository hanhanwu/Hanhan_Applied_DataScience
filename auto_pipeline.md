# Build Customized Data Science Pipeline

In the industry, you do need to build a customized data science pipeline to solve most of the problems in your company. 

## Industry Design Examples
* [Google AutoML Table][7]
  * From structured data to dashboard, the whole system design is very smooth 

## General Architecture

To be added...
* [Some points to be considered][5]

## Core Parts

### Param Tuning (HPO)
* [Hyperopt][1]
* [Optuna][2]
  * [Quick start][3] - You will see how to tune param with multiple estimators and params, nice visualization too
* [Optuna vs Hyperopt][4]
  * It compared the 2 tools from different important aspects, and in every aspects, Optuna appears to be no worse

### Model Selection
* [Google Model Search][8]
  * "The Model Search system consists of multiple trainers, a search algorithm, a transfer learning algorithm and a database to store the various evaluated models. The system runs both training and evaluation experiments for various ML models (different architectures and training techniques) in an adaptive, yet asynchronous fashion. While each trainer conducts experiments independently, all trainers share the knowledge gained from their experiments. At the beginning of every cycle, the search algorithm looks up all the completed trials and uses beam search to decide what to try next. It then invokes mutation over one of the best architectures found thus far and assigns the resulting model back to a trainer." 
  * [Model Search Intro][9]

### Security Threats to Machine Learnig Systems
* [Some threats before/during/after model training, interesting][6]



[1]:https://github.com/hyperopt/hyperopt
[2]:https://github.com/optuna/optuna
[3]:https://github.com/hanhanwu/Hanhan_COLAB_Experiemnts/blob/master/optuna_quickstart.ipynb
[4]:https://towardsdatascience.com/optuna-vs-hyperopt-which-hyperparameter-optimization-library-should-you-choose-ed8564618151
[5]:https://www.analyticsvidhya.com/blog/2021/01/a-look-at-machine-learning-system-design/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[6]:https://www.analyticsvidhya.com/blog/2021/01/security-threats-to-machine-learning-systems/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[7]:https://cloud.google.com/automl-tables
[8]:https://github.com/google/model_search
[9]:https://ai.googleblog.com/2021/02/introducing-model-search-open-source.html?m=1
