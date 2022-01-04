# Build Customized Data Science Pipeline

In the industry, you do need to build a customized data science pipeline to solve most of the problems in your company. 

## Industry Design Examples
* [Google AutoML Table][7]
  * From structured data to dashboard, the whole system design is very smooth 

## Pipeline Tools
* Luigi
* Airflow
* [Orchest][20]
  * The pipeline it allows you to build can use both `.py` and ipython notebooks, looks convenient

## Cloud Platforms
* [Google Cloud Platform (GCP) for ML][22]

## General Architecture
* [Some points to be considered][5]

## Core Parts

### Param Tuning (HPO)
* [Hyperopt][1]
* [Optuna][2]
  * [Quick start][3] - You will see how to tune param with multiple estimators and params, nice visualization too
  * [Optuna vs Hyperopt][4]
    * It compared the 2 tools from different important aspects, and in every aspects, Optuna appears to be better overall
    * Optuna's TPE appears better than hyperopt's Adaptive TPE
  * [Using Optuna in different models][14]
* [Keras Tuner][17]
  * It can be used to tune neural networks, the user interface is similar to Optuna
  * [Different types of tuners][18]: hyperband, bayesian optimization and random search
  * It also provides a tuner for sklearn models, [sklearn tuner][16]
* [FLAML][19] 
  * In some cases, FLAML can be more efficient than optuna in param tuning and even deliver better testing performance within a shorter time
  * It developed 2 searching algorithms (CFO, Blend Search), CFO works faster with higher testing performance in many cases
* Bayesian Optimizaton
  * It considers past model info to select params for the new model
  * [An example][15] 
    * Bayes_opt may not be faster than hyperopt but you can stop whenever you want and get current best results. It also shows the tuning progress that contains which value got selected in each trial
 

### Model Selection
* [Google Model Search][8]
  * "The Model Search system consists of multiple trainers, a search algorithm, a transfer learning algorithm and a database to store the various evaluated models. The system runs both training and evaluation experiments for various ML models (different architectures and training techniques) in an adaptive, yet asynchronous fashion. While each trainer conducts experiments independently, all trainers share the knowledge gained from their experiments. At the beginning of every cycle, the search algorithm looks up all the completed trials and uses beam search to decide what to try next. It then invokes mutation over one of the best architectures found thus far and assigns the resulting model back to a trainer." 
  * [Model Search Intro][9]
* [MLJAR is a nice automl tool][21]
  * Besides EDA, model selection and param tuning, it will stack models at the end to achieve better results 

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
[14]:https://www.kaggle.com/dixhom/bayesian-optimization-with-optuna-stacking
[15]:https://www.analyticsvidhya.com/blog/2021/05/bayesian-optimization-bayes_opt-or-hyperopt/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[16]:https://keras.io/api/keras_tuner/tuners/sklearn/
[17]:https://keras.io/keras_tuner/
[18]:https://keras.io/api/keras_tuner/tuners/
[19]:https://github.com/microsoft/FLAML
[20]:https://orchest.readthedocs.io/en/latest/getting_started/quickstart.html
[21]:https://github.com/mljar/mljar-supervised
[22]:https://www.analyticsvidhya.com/blog/2022/01/google-cloud-platform/?utm_source=feedburner&utm_medium=email
