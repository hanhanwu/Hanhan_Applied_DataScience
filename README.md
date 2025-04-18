# Hanhan_Applied_DataScience
Applied data science recommendations and tutorials

# [AI Notes][199]

# Older Notes
* [Better 4 Industry][67]

# Important Tips
* Sampling strategies should be documented.
* Split data into train and test first, then do data preprocessing on training data. After model trained, apply the final data preprocessing methods on the testing data.
* [Working Notes][173] 🌺
* [Deep Learning tips][171]

## Research Resources
* I especially like [Papers with Code][88] 🌺
  * You can also find papers here under Methods groups: https://paperswithcode.com/methods

# Applied Recommendations
## [Stakeholder Notes][93]

## [Prototyping][81]
* Prototyping Tookit
* Need big data environment, [cloudxlab saves efforts in environment setup][111]
### [My Story Telling Collections][113]

## [Build Auto Data Science Pipeline][114]
* Pipeline tools
* Platforms
* Core Components
  * Feature Store
  * HPO
  * etc. 
  
## MLOps
* 🌺 [Made with ML][198]
* Google ML Engineering Guides
  * [MLOps architecture][181]
  * [ML Pipeline best suggestions][182]
* Docker
  * [Docker for beginner][164] 
* [Deploy deep learning web application with TensorFlow.js][157]
  * It includes image labeling, modeling with Google Colab and deployment on TensorFlow.js 
* [Build ML Web Application with Flask][90]
  * [How to deploy flask app to Azure][101]
  * Flask vs FastAPI
    * Personally, I didn't find fastapi is better, because when I was using a HTML template to get user typed input and generate the model output, with flask it was simple, you just need to render the template. Using fastapi with template rendering was much more complex.
* [ML App with Streamlit][145]
  * The UI and the 3D visualization built 
* [Deploy ML service with BentoML][94]
  * [About BentoML][95]
  * It allows you to deploy the trained model as REST API (built directly for you), docker image or python loadable file
  * When your docker image is running, you can use it as REST API locally to, just go to http://localhost:5000
  * Correction
    * For docker, the bento command should be `bentoml containerize IrisClassifier:latest -t iris-classifier:latest`
      * "iris-classifier" is the image name, "latest" is the tag name
* [MLOps library][146]
  * [Here's a simple example][147] with [code][148], you can write a pipeline in .py, then specify the devops workflow in .yaml to run not only ML pipeline but also other devops work
  * Might be Azure ML specific, check [getting started here][149]
* ZenML: https://github.com/zenml-io/zenml
* Inferrd: https://docs.inferrd.com/
  * Easily deploy your model on GPUs 
* More Open Source MLOps Tools
  * [DVC Studio, MLFlow][150] 
  * DVC Studio: https://github.com/iterative/dvc
  * MLFlow: https://github.com/mlflow/mlflow
* [About Kubernetes][151]
  * Conscepts
  * Commands
  * How to deploy your application using Kubernetes together with Docker 
  * [More suggestions about Kubernetes in production][153]
  * [More deployment walkthrough][154]
  
## [Applied Data Engineering][92]
* Rational databases
* Big Data Systems
* Big Data Cloud Platforms

## Code Speed Up Methods
### Ray
* Ray vs Python multiprocessing vs Python Serial coding
  * [Code example][79]
  * [My code example][117]
  * [Blog details][80]
  * To sum up, Ray is faster than python multiprocessing. It seems that the basic steps are initialize actors, and for each actor, call the function you want to be processed distributedly.
  
## AI Ethics
* [Different tools for measuring AI Ethics][162]
  * Questions to think when using these tools:
    * If a model can't pass the AI Ethics measurement, can we get suggestions to be able to passs the measurement?
* [Google AI Principles][166]
  * It suggests to evaluate model performance for each segment/subgroup too, to see whether there could be bias
  * Data exploration can also disclose some bias in the data, so [Google introduced Facets for some data exploration][167] 

### Federated Learning
* [In this paper, a simple application of federated learning][140]
  * Shared & aggregated weights, instead of shared data 
* Federated Learning is a distributed machine learning approach which enables model training on a large corpus of decentralised data.
  * [The aggregation algorithm used in federated learning][110] - it's mainly to address data imbalance between clients and data non-representativeness. The algorithm is to sum up the weighted model parameter from each selected client's model as the general model param, the weight is the percentage of data records of the client among all the selected clients' records.
* [FedML: A Research Library and Benchmark for Federated Machine Learning][103]
  * It describes the system design, new programming interface, application examples, benchmark, dataset, and some experimental results.
* [Tensorflow built-in federated learning][104]
  * The way how this example works is, it radomly choose some clients' data (or use aggregation function to do that more efficiently), then sends the multi-clients' data to the remote server to train a model
  * [Detailed tensorflow and keras federated learning ipyhton notebook][105]
  * [All the tensorflow federated tutorials][107]
* How to Customize Federated Learning
  * [Inspiration][115]
    * It captures the core part of FL,  global model global params (weights), local model local params (weights), aggregated params (weights) are used to update the global model and then distributed to the local model for next iteration of params aggregation. Repeat.
    * The FL aggregation method was built upon deep learning and used for update the weights. If we want to apply FL in classical ML method, we can borrow the idea and instead of updating weights, we can update params.
    * It also compared SGD with FL, and FL does have certain optimization function.
    

## Data Quality Check
When you got the data from the client or from other teams, better to check the quality first.
### Label Quality Check
* Overall Check
  * Data Imbalance
* Within Each Group
  * "Group" here can be each account/user/application/etc.
  * How does different labels distribute within each group
  * Better to understand why
* Label Quality
  * [Confident Learning][191]: `Cleanlab` is a python package provides insights on potential mistakenly labeled data
### Data Drift Check
* [When to use which metrics to measure drift or compare distributions][197]
  * KS vs PSI vs WD
  * It also has KL, JS, which both can be applied to numerical distributions or categorical distributions.
    * Details on how did I apply these methods can be found in: https://github.com/hanhanwu/Hanhan_Applied_DataScience/blob/master/data_exploration_functions.py
* [Different methods to detect concept drift, covariate drift][139]
  * How to use K-S test to compare 2 numerical distributions: https://stackoverflow.com/questions/10884668/two-sample-kolmogorov-smirnov-test-in-python-scipy
    * K-S test can be used even when the 2 distributions are in different length, and it's non-parametric
    * But I found, no matter it's K-S test or wasserstein distance, even when 2 distributions look similar, K-S' null hypothesis can be rejected (indicating 2 distributions are not identical) and wasserstein distance can be large... 
  * Chi-square is used for comparing categorical features' distributions, it's non-parametric too but requires the 2 distributions share the same length...
  * [PSI is used when your data has normal fluctuations and you mainly care about significant drift in numerical data][142] 
* [Continual Learning with Ensembling models][192]
  * Add trained model's knowledge to the new data, new model

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
* Use q-q plot to check whether the data distribution aligns with an assumed distribution
  * [Example][134]
  * When the distribution aligns, the q-q plot will show a roughly straight line
  * q-q plot can also be used to check skewness

### Bivariate Analysis
* Check correlation between every 2 continuous variables
  * [More methods to check correlations][137]
    * Point Biserial Correlation, Kendall Rank Correlation, Spearman’s Rank Correlation, Pearson Correlation and its limitation (linear only, cant be used for ordinal data) 
  * Person vs Spearman
    * Pearson correlation evaluated the linear relationship between two continuous variables. 
      * A relationship is linear when a change in one variable is associated with a proportional change in the other variable. 
    * Spearman evaluates a monotonic relationship. 
      * A monotonic relationship is one where the variables change together but not necessarily at a constant rate.
  * Check chi-square test between 2 categorical features (similar to correlation for continuous features)
    * (p-value) probability 0 means the 2 variables are dependent; 1 means independent; a value x in [0,1] range means the dependence between the 2 variables is at `(1-x)*100%`
      * H0 (Null Hypothesis) = The 2 variables to be compared are independent. Normally when p-value < 0.05 we reject H0 and the 2 variables are NOT independent.
    * [sklearn chi-square][4]
  * NOTE: ["Same" distribution doesn't mean same correlation, see this example!][183]
  * Check ANOVA between categorical and continuous variables
    * [Example to check ANOVA f-value][2]
    * [sklearn ANOVA][3]
    * Lower F-score the higher dependent between the variables
    * Besides ANOVA, we could calculate t-score or z-score (less than 30 records)
* Comparing with correlation, Mutual Information (MI) measures the non-linear dependency between 2 random variables
  * Larger MI, larger dependency 
* [Using PPS (predictive powerscore) to check asymmetrical correlation][174]
  * Sometimes x can predict y but y can't predict x, this is asymmetrical 
  * Besides using it as correlation between features, PPS can also be used to check stronger predictors
* 2 way table or Stacked Column Chart - for 2 variable variables, check count, percentage of group by the 2 variables
* For features that are highly dependent on each other, in parametrics algorithms these features should be removed to reduce error; in nonparametrics algorithms, I still recommend to remove them, because for algorithms such as trees tend to put highly dependent features at the same importance level, which won't contribute feature importance much.
* Feature selection based on the dependency between features and the label
  * Select features that have stronger correlation with the label, because these features tend to contribute to the label values.
  * correlation: continuous features vs continuous label
  * chi-square: categorical features vs categorical label
    * Cramer’s V for Nominal Categorical Variable (categorical values in numerical format)
    * Mantel-Haenszed Chi-Square for ordinal categorical variable (categorical values in order)
  * ANOVA: categorical features vs cotinuous label; or vice versa
  
### Regression Coefficients
* [Unstandardized vs Standardized Regression Coefficients][126]
  * Unstandardized coefficient α are used for independent variables when they all kept their own units (such as kg, meters, years old, etc.). It means when there  is one unit change in the independent variable, there is α cahnge in the dependent variable y.
    * However, because those independent variables are in different units, based on unstandardized coefficients, we cannot tell which feature is more important
  * So we need standardized coefficient β, larger abs(β) indicates the feature is more important
    * Method 1 to calculate β is to convert each observation of an independent variable to <b>0 mean and 1 std</b>
      *  new_x_i = (x_i - Mean_x) / Std_x, do this for both independent and dependent variables
      *  Then get β
    * Method 2
      * Linear Regression: β_x = α_x * (std_x / std_y)
      * Logistic Regression: β_x = (sqrt(3)/pi) * α_x * std_x
    * It means when there is 1 std increase in the independent variable, there is β std increase in the dependent variable
  
### Deal With Missing Values
* [MissForest][200]
  * It can get more accurate results when imputing the missing at random
  * How it works:
    * - Step 1: To begin, impute the missing feature with a random guess — Mean, Median, etc.
    * - Step 2: Model the missing feature using Random Forest.
    * - Step 3: Impute ONLY originally missing values using Random Forest’s prediction.
    * - Step 4: Back to Step 2. Use the imputed dataset from Step 3 to train the next Random Forest model.
    * - Step 5: Repeat until convergence (or max iterations). 
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
* [LGBM Imputation][160]
  * It uses LGBM to impute missing values 
* [Miss Forest, Mice Forest][165]
  * Handle missing values with random forest
  * It explains how
  * Works for both numerical, categorical features 
* Model prediction to compare whether imputing missing values will help

### Deal With Outliers
* To check whether there is ourliers, I normally check distribution, even just by using percentile. And only deal with a very small percentage of outliers.
  * It's recommended that to use the model run on raw features without any preprocessing, this is the baseline result. Then run the same model with preprocessed data, to compare. So that you will know how much does preprocessing affect your prediction in each project.
* Sometimes I also check boxplot, but the data I am dealing with tend to have large amount of outliers, and imputing data above & below 1.5*IQR will reduce the prediction power too much.
* <b>Better to know what caused the outliers</b>, this may help you decide how to deal with them
* Decide which are outliers
  * Distributions
  * Boxplot, 1.5IQR
  * [Modified Z score][108]
    * Check `MAD` value
* [ML methods can be used for anomaly detection][172]
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
      * Remove it: Often start with removing features with highest VIF
      * Combine features with high VIF into 1 feature
    * [The implementation of using VIF to drop features][39]
    * [Description of VIF][40]
      * The Variance Inflation Factor (VIF) is a measure of colinearity among predictor variables within a multiple regression. It is calculated by taking the the ratio of the variance of all a given model's betas divide by the variane of a single beta if it were fit alone. <b>VIF score of an independent variable represents how well the variable is explained by other independent variables.</b>
      * `VIF = 1/(1-R^2)`, higher R^2 means higher correlation between the variable and other variables, so higher VIF indicates higher multicollinearity.
* Use tree models to find feature importance
  * Better to remove highly correlated features before doing this. Some tree model will put highly correlated features all as important if one of them is highly ranked
* Dimensional Reduction Algorithms
  * <b>Better to use standardize data before applying these methods</b>, by making data on the same scale, distance calculation makes more sense.
  * [sklearn decomposition][15]
  * [The intermediate result from autoencoder - encoder][60]
  * [sklearn manifold learning][118]
    * [Each algorithm][119]
  * [How does PCA work][184], great way to explain "eigenvalue" also plot the explained variance of each principle component
    * Unsupervised, linear
    * Needs data standardization
  * [How does LDA work][185] 
    * Linear, Supervised, as it's trying to separate classes as much as possible
    * Needs data standardization
  * [How does t-SNE work][177]
    * Linear, Unsupervised, Non-parametric (no explicit data mapping function), therefore mainly used in visualization 
    * "Similarity" & normal distribution in original dimension, "similarity" & t-distribution in lower dimension. 
      * Both probability distributions are used to calculate similarity socres
      * t-distribution has "flatter" shape with longer tails, so that data points can be spread out in lower dimensions, to avoid high density of gathering
    * "Perplexity" determines the density
    * Needs data standardization
  * [How does Isomap work][187]
    * Unsupervised, Nonlinear, with the help of [MDS (multidimensional scaling)][186], it will try to keep both global and local (between-point distance) structure
      * Linear method like PCA focuses on keeping global structure more
  * [How does LLE work][188]
    * Nonlinear, Unsupervised
    * Focusing on local structure more, more efficient than Isomap as it doesn't need to calculate pair-wise distance, but Isomap works better when you want to keep both global and local structure
  * [How does UMAP work][189]
    * Supports both supervised & unsupervised
    * Trying to maintain both local & global structures
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
  * Binning
  * Such as `log`, here are a list of methods for nonlinear transformation: https://people.revoledu.com/kardi/tutorial/Regression/nonlinear/NonLinearTransformation.htm
    * 🌺 Some data do need to use log, such as "income" in banking data. There is a common assunption that, income data is log-normally distributed. Applying `log` on the better can be better.
    * This can be applied to features that have inevitable "outliers" that you should not remove, but it could make the overall distribution hard to see. When there are non-positive values, sometimes you might need to use `np.log(df + 1) * np.sign(df)`
  * Kernel functions
  * PCA - It can convert the whole feature set into normalized linear combination
    * [How to use PCA for dimensional reduction/feature selection][29]
    * The components are independent from each other, earlier components capture more variance. Each principle component is also a linear combination of original features
* To calculate skewness
  * The difference between the mean and mode, or mean and median, will tell you how far the distribution departs from symmetry.
  * Pearson Mode skewness: `(mean - mode)/std`
  * Pearson Mode skewness alternative: `3*(mean - median)/std`
  * https://www.statisticshowto.datasciencecentral.com/pearson-mode-skewness/
* Convert skewed distribution to symmetric distribution, this is also prefered for parametric algorithms. Skewed data could reduce the impact from lower values which may also important to the prediction.
  * `log` to deal with righ skewness
  * squre root, cube root
  * `exp` to deal with left skewness
  * `power`
  * binning
* derived features
* one-hot features
* decision tree paths as the feature
* Methods to deal with categorical features
  * [Some simple categorical encoding methods][141]
    * I often use label encoding to convert categorical target value to numerical format, and use ordinal encoding to convert categorical target value to ordinal numerical format
  * [LightGBM offers good accuracy with built-in integer-encoded categorical features. LightGBM applies Fisher (1958) to find the optimal split over categories][122]. [You can also have categorical features as direct input][123], [but need to specify that as categorical type][124]
  * [10+ Built-in Categorical Encoding Methods][37]
    * Target Encoder is a popular method, and [this paper][155] tells the drawbacks of target encoding for reducible bias, and indicating that using smoothing regularization can reduce such bias. [We can do smoothing regularization through the param here][156]
    * [Params for each encoding method][84]
    * [More descriptions of some of the encoding methods][85] 
      * Base N creates less dimensions while represent the data effciently, if you choose the proper N
      * "In target encoding, for numerical target, we calculate the mean of the target variable for each category and replace the category variable with the mean value; for categorical target, the posterior probability of the target replaces each category"
        * Target encoding should only be applied to the training data to avoid target leakage
        * 2 methods mentioned in the article to reduce target leakage/overfitting
        * When the categories in training and testing data are distributed improperly, the categories may assume extreme value
    * Better to remove highly correlated features after applying these encoding methods. Even if the feature correlations affect tree models less, too many dimensions are not optimal to tree models either, might lead to large trees.
  * [Beta Target Encoding][128]
    * requires minimal updates and can be used for online learning
    * [Check its python implementation][129], supports numerical target  
  * Concat multiple categorical features as 1 feature
  * Convert to value frequency or response rate, aggregated value. Also for categorical value, we can use part of the value and convert to numerical values
* Normalize data into [0, 1] without min, max
  * We can use sigmoid function `exp(x)/(exp(x) + 1)` to normalize any real value into 0, 1 range
  * If we check the curve of sigmoid function, you can see that for any x that belongs to a real number, y is always between 0 and 1.
* <b>Data Transformation</b>
  * Transform non-normal distribution into normal distribution
    * Besides minmax mentioned above, [sklean has multiple transform methods][78], some are even robust to outlisers.
  * In [this example][82], using box-cox method can convert distributions with multiple peaks into normal distribution
    * The requirement of box-cox is, all the values have to be positive.

## Deal With Imbalanced Data
### Semi-supervised learning for imbalanced data
* [2020 paper - Rethinking the Value of Labels for Improving Class-Imbalanced Learning][102]
* Inspirations
  * Train the model without label first to generate the predicted labels as new feature for the next model learning (semi-supervised learning)
  * More relevant new data input might reduce test error
  * Smaller data imbalance ratio for unlabeled data also reduce unlabeled data test error
  * The use of T-SNE for visualization to check class boundary
    * This is data dependent, not all the dataset could show obvious class boundary
### Sampling Methods
* There are oversampling, undersampling and synthetic sampling (which is also oversampling), combined sampling (oversampling + undersampling). In practice, I tried different methods in different projects, so far non of them worked well in both training and testing data.
* [imbalanced-learn][99] has multiple advanced sampling methods
* [Compare sample feature and population feature distributions][116]
  * Numerical features: KS (Kolmogorov-Smirnov) test
  * Categorical features: (Pearson’s) chi-square test
* [Probability * Non-probability Sampling Methods][138]
### Cost Sensitive Learning (class weights)
* This method is becoming more and more popular recently. Majorly you just set the class weights based on the importance of false positive and false negative.
* In practice, it is worthy to know more from the customers or the marketing team, trying to understand the cost of TP/TN or FP/FN.
* Example-dependent Cost Sensitive method
  * [Costcla][36] - it's a package that do model prediction including cost matrix
    * The cost matrix is example dependent. Each row has the cost of [FP, FN, TP, TN]. So when you are creating this cost matrix yourself, your training & testing data all records the costs, each row of your data is an example. [Sample Model Prediction][36]
    * The drawback is, testing data also needs the cost matrix, but in practice, you may not know. However this method can still be used in train-validation dataset, in order to find the optimal model.
### Thresholding
* When the prediction result is in probability format, we can change the threshold of class prediction. By default the threshold is 50-50. With the evaluation metric, better to <b>draw a curve with thresholds as x-axis and evaluation result as y-axis</b>, so that we will know which threshold to choose is better.
<img src="https://github.com/hanhanwu/Hanhan_Applied_DataScience/blob/master/images/error_rate_thresholds.png" width="500" height="200">

* [Here's a method to find the best threold when your evaluation metric is ROC-AUC or precision-recall][158]

### Given Partial Labels
* When the labels are given by the client, without being able to work with any business expert, there can be many problems. Especially when they only gave you 1 class of the labels, and you might need to assume the rest of the data all belong to the other label, it can be more problematic.
* Here're the suggestions
  * Better not to build the machine learning pipeline at the early stage, since the pipeline can create much more limitation in the work when you want to try different modeling methods.
  * If not trust the label quality, try clustering first, and check the pattern in each cluster, and compare with the given labels.
  * Communicate with the business experts frequently if possible, to understand how did the labels get generated.
  * If your model is supervised method, and if using all the data there will be huge data imbalance issue, better to choose representative sample data as the assumed label, and try to reduce the size, in order to reduce the data imbalance issue.

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
### Baseline Models for R&D
* During R&D, there will be more and more fancy methods, especially those in deep learning. Are they going to perform better comparing with these easy-to-implement baselines?
* Categorize based on words/characters
  * Naive Bayesian
  * Fisher Methods
  * You can do self-implementation, [check chapter 6][125]
* Find themes of a piece of text
  * Non-Negative Matrix Factorization 
  * You can do self-implementation, [check chapter 10][125]

### Which Model to Choose
* Linear or Nonlinear
  * In the code [here][11], we can use residual plot to check whether there is linear/non-linear relationship between the feature set and the label, to decide to use linear/nonlinear model.
    * If the residual plot is showing funnel shape, it indicates non-constant variance in error terms (heteroscedasticity), which also tends to have residuals increase with the response value (Y). So we can also try to use a concave function (such as log, sqrt) to transform `Y` to a much smaller value, in order to reduce heteroscedasticity. 
    * As we can see in the code, after transforming `Y` with `log`, the residual plot was showing a much more linear relationship (the residuals are having a more constant variance)
* TPOT Automatic Machine Learning
  * [TPOT for model selection][12] 
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
* [Tools other than TPOT][135]
### Notes for Evaluation Methods in Model Selection
* R-Square, RSS (residual sum of squares), will decrease when there are more features, but the test error may not drop. <b>Therefore, R-Square, RSS should not be used for selecting models that have different number of features.</b>
  * `RSS = MSE*n`
  * `R-square = explained variation / total variation`, it means the percentage of response variable variance explained by the model.
    * It's between 0% to 100%
    * 0% indicates that the model explains none of the variability of the response data around its mean.
    * 100% indicates that the model explains all the variability of the response data around its mean.
### Cross Validation
* When there is time order in the data
  * Solution - Forward Chaining
    * Step 1: training(1), testing(2)
    * Step 2: training(1,2), testing(3)
    * Step 3: training(1,2,3), testing(4)
    * ...
  * Application - [sklearn time series split][77] 
  
### Multi-Tag Prediction
* [A simple example using OneVsRest][136]

### More about Ensembling
* We know ensembling models tend to work better in many cases, such as Xgboost and LightGBM
#### Super Learner
* [Super Learner example][100]
  * It trains on a list of base models, using the predictions as the input of a meta model to predict the targets, and finally might be better than a specific "best model"


## Algorithms Details
### RBF Kernel
* Radial Basis Function kernel (RBF kernel) is used to determine edge weights in affinity matrix, a matrix defines a pairwise relationship points.
  * Sklrean's implementation of RBF kernel looks slightly different as it replaces 1/2sigma^2 with a hyperparameter gamma. The effect is the same as it allows you to control the smoothness of the function. High gamma extends the influence of each individual point wide, hence creating a smooth transition in label probabilities. Meanwhile, low gamma leads to only the closest neighbors having influence over the label probabilities.  
* [Reference][175]

### Semi Supervised Learning
* When you have labeled data for all classes:
  * [How does label propagation work][176]
  * [How does label spreading work][175]
    * To be more accurate, I would say the difference between hard clamping and soft clamping is the weight (alpha) on original label is 0 or not, [check sklearn's description][178]
  * [How does self-training classifier work][190]
* Only has positive & ublabled data
  * [PU learning - How does E&N work][179]
  * How to evaluate PU learning without negative label
    * [Evaluation Method Proposal 1][180]

### SVM
* [How did margin generated][33] - margin has maximized orthogonal distance between the cloest points in each category and the hyperplane, these closest points are supporting vectors
* [How does kernal SVM work for nonlinear data][34]
  * Gaussian kernal is `rbf` kernal in sklearn
  
### Decision Tree
* [How to choose `ccp_alpha` for pruning to reduce overfitting][96]

### Gradient Descent
* It's almost the oldest optimization function, trying to find the local minimum through iteration.
* [An easy way to understand it][98]
* [How to implement gradient descent][97]
  * `X` is feature set, `y` is lables, `theta` is (initial) weight; `alpha` is learning rate; `m` is number of records
  
### [Adaboost vs Gradient Boost][109]
* The final comparision table is very helpful
  * weights vs gradient
  * tree depth
  * classifier weights
  * data variance capture
  
### L1 vs L2
* L1 regularization adds the penalty term in cost function by adding the absolute value of weight(Wj) parameters, while L2 regularization adds the squared value of weights(Wj) in the cost function.
* L1 pushes weight `w` towards 0 no matter it's positive or negative, therefore, L1 tend to be used for feature selection
  * If w is positive, the regularization L1 parameter λ>0 will push w to be less positive, by subtracting λ from w. If w is negative, λ will be added to w, pushing it to be less negative. Hence, this has the effect of pushing w towards 0.
* As for dealing with multicollinearity, L1, L2 and Elastic Net (uses both l1, L2) could all do some help

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
* There is a suggestion for model evaluation, which is, choose and train models on the same dataset and averaging the evaluation results. The idea here is, "combined models increase prediction accuracy"
* "Probability" means predict continuous values (such as probability, or regression results), "Response" means predict specific classes
#### Evaluation for Time Series Data
* The model doesn't have to be forecasting/time series models only, it can be classification model too with features generated using sliding window/tumbling window.
* No matter which model to use, when there is time series in the data, better to use walk-forward evaluation, which is to train on historical but predict on the future data.
  * Better to sort the input data in time order, even though the model doesn't check historical records and the features already added time series value, if the dta is not sorted in time order, when the later records appeared before its historical record during training, it can still create bias

#### [sklearn classification evaluation metrics][19]
* ROC Curve vs. Precision-Recall Curve
  * ROC curve requires the 2 classes are balanced. Precision-Recall curve is better at being used on imbalanced dataset.
  * Both of them have TPR (recall/sensitivity, the percentage of TP cases are described as positive), the difference is in precision and FPR.
    * `FPR = FP/(FP + TN)`, when the negative class is much larger than positive class, FPR can be small.
    * `Precision = TP/(TP + FP)`, it indicates the percentage of TP are correct. When the negative class is much larger than the positive class, this can be affected less, comparing with ROC.
* Logloss is used for <b>probability output</b>, therefore it's not used as a regression method in sklearn
  * In deep learning loss function doesn't have to be logloss, it can simple be "mean_absolute_error"

#### [sklearn regression evaluation metrics][20]
* MSE/RMSE works better for continuous target
* `R-square = explained variation / total variation`, it means the percentage of response variable variance explained by the model.
  * Interpretation
    * 1 is the best, meaning no error
    * 0 means your regression is no better than taking the mean value
    * Negative value means you are doing worse than the mean value
  * It's between 0% to 100%
    * 0% indicates that the model explains none of the variability of the response data around its mean.
    * 100% indicates that the model explains all the variability of the response data around its mean.
  * R-square cannot tell bias, so have to be used with residual plot.
    * When the residual plot dots are randomly dispersed around the horizontal axis, a linear model is better for the data, otherwise a nonlinear model is better.
  * Lower R-square doesn't mean the model is bad.
    * For unpredictable things, lower R-square is expected. Or R-squared value is low but you have statistically significant predictors, it's also fine.
  * High R-square doesn't mean good model.
    * The residual plot may not be random and you would choose a nonlinear model rather than a linear model.
* Residual Plot
  * In regression model, `Response = (Constant + Predictors) + Error  = Deterministic + Stochastic`
    * Deterministic - The explanatory/predictive information, explained by the predictor variables.
    * Stochastic - The unpredictable part, random, the error. Which also means the explanatory/predictive information should be in the error.
  * Therefore, residual plot should be random, which means the error only contains the Stochastic part.
  * So when the residual plot is not random, that means the predictor variables are not capturing some explanatory information leaked in the residual plot. Posibilities when predictive info could leak into the residual plot:
    * Didn't capture a missing variable.
    * Didn't capture a missing higher-order term of a variable in the model to explain the curvature.
    * Didn't capture the interaction between terms already in the model.
    * The residual is correlated to one or more variables, and these variables didn't get captured in the model.
      * Better to check the correlation between residuals and variables.
    * Autocorrelation - Adjacent residuals are correlated to each other.
      * Typically this situation appears in time series, when you can use one residual to predict the next residual.
      * Test autocorrelation with ACF, PACF
        * ACF (AutoCorrelation Function), PACF (Partial AutoCorrelation Function)
        * [More description about ACF, PACF][63]
        * When there are 1+ lags have values beyond the interval in ACF, or PACF, then there is autocorrelation.
      * [Test autocorrelation with python statsmodels][64]
      * [What is Durbin Watson test for autocorrelation][66]
        * [How to tell it's autocorrelation with Durbin Watson test result][65]
* References
  * [R-square][61]
  * [Why need residual plot in regression analysis][62]
#### [sklearn clustering metrics][21]
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
* KS Test - Kolmogorov-Smirnov (K-S) chart is a measure of the difference between the y_true and y_pred distributions for each class respectively. It's a method can be used to compare 2 samples.
  * KS test vs t-test: Imagine that the population means were similar but the variances were very different. The Kolmogorov-Smirnov test could pick this difference up but the t-test cannot

## Time Series Specific
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
* Combing multiple models forecasts
  * Combining multiple forecasts leads to increased forecast accuracy.
  * Choose and train models on the same time series and averaging the resulting forecasts
  * [Reference][57]
  
  
## Optimization Problem

### [Param Tuning in Industry Pipelines][132]

### Linear Programming
* This is a common solution used in industry problems. With multiple constraint functions and 1 function for optimized solution.
* [My code - Linear Programming to Get Optimized List of Youtube Videos][52]
  * Only supports minimized problem, so you need to adjust the optimization function if it's supposed to be maximized
* [A detailed introduction of applied linear programming][53]
* [How does Simplex Method work][54]
  * Use this solution when 2D drawing is no longer enough
* [Python Linear Programming - scipy.optimize.linprog][55]
  * [How to use this library...][56]
  * Only supports minimized problem, so you need to adjust the optimization function if it's supposed to be maximized
* [LP Modeler - PuLP][159]
  * [Pulp example with concept description][161], concept includes:
    * convex vs concave vs non-convex
    * infeasible or unbounded solution 

### [Pareto Front][127] & [MADM][130]
* Pareto Front
  * When you have multiple evluation metrics (or objective functions), this method helps to find those records with at least 1 metric wroks better.
  * With this method, it helps remove more unnecessary records --> more efficient.
  * The idea can be modified for other use cases, such as instead of having at least 1 metric works better, we can define all the metrics or a certain metrics work better, etc.
* MADM (Multi criteria decision analysys methods)
  * The methods listed in [skcriteria][130] are mainly using linear programming 
* Use Cases
   * Param Tuning  - Pareto Front improves the efficiency, MADM find the optimal option
  
## Model Explainability
### Dashboard Tools
* Tableau
* [Spotfire][143]
* [Redash][144]
* [ExplainerDashboard][152]

### [Interpret Clustering with Decision Tree][168]

### Plot Decision Tree Boundry
* [Example][169]
  * [Code][170] 

### [Interpretable Machine Learning][131]
* It has summarized different methods in ML model interpretations, such as NN interpretation, model-agnostic interpretation, etc.
### Lime - Visualize feature importance for all machine learning models
* Their GitHub, Examples and the Paper: https://github.com/marcotcr/lime
  * Interpretable: The explanation must be easy to understand depending on the target demographic
  * Local fidelity: The explanation should be able to explain how the model behaves for individual predictions
  * Model-agnostic: The method should be able to explain any model
  * Global perspective: The model, as a whole, should be considered while explaining it
* The tool can be used for both classification and regression. The reason I put it here is because it can show feature importance even for blackbox models. In industry, the interpretability can always finally influence whether you can apply the more complex methods that can bring higher accuracy. Too many situations that finally the intustry went with the most simple models or even just intuitive math models. This tool may help better intrepretation for those better models.
* My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Better4Industry/lime_interpretable_ML.ipynb
  * It seems that GitHub cannot show those visualization I have created in IPython. But you can check LIME GitHub Examples
  * LIME requires data input to be numpy array, it doesn't support pandas dataframe yet. So that's why you can see in my code, I was converting the dataframe, lists all to numpy arraies.
* NOTES
  * Currently they have to use predicted probability in `explain_instance()` function
  * You also need to specify all the class names in `LimeTabularExplainer`, especially in classification problem, otherwise the visualization cannot show classes well
  
### SHAP
* SHAP value is a type of feature importance value: https://christophm.github.io/interpretable-ml-book/shap.html#shap-feature-importance
* It uses Shapley values at its core and is aimed at explaining each individual record.
  * "Shapley value for each feature is basically trying to find the correct weight such that the sum of all Shapley values is the difference between the predictions and average value of the model. In other words, Shapley values correspond to the contribution of each feature towards pushing the prediction away from the expected value."
    * So higher shaply value indicates the feature pushes the prediction towards the positive class more.
    * So lower shaply value indicates the feature pushes the prediction towards the negative class more.
* [More details from my practice][68], including shap decision plot

### [ELI5][70]
* It supports feature importance interpretation for those famous emsembling models, as well as deep learning model output interpretation.
* [Tutorials][69]

### [Yellowbrick][71]
* The visualization generated from this library are more like sklearn stype, basic but useful. It has multiple visualizers: 
  * Feature Analysis, Target visualizer, Regression/Classfication/Clustering visualizers, Model Selection visualizer, Text Modeling visualizers.
* [Find all its visualizers][72]

### [Tensorflow Lucid][75]
* You can <b>visualize neural networks</b> without any prior setup. 
* [Google Colab Notebooks][76]

### [Alibi][73]
* It might be better for deep learning. So far I haven't seen how impressive this tool is~
* [Examples][74]
  * Much more text than visualization


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
* How does R output use p-value in regression
  * In R, after we have applied regression, we will see coefficient values as well as p-values.
  * The null hypothesis is, coefficient is 0 for a variable (no effect to the model). 
  * So when p-value is lower than the significance level, reject the null hypothesis, which also means the variable should be included in the model. When p-value is high, accept the null hypothesis, and the variable should be removed from the model.

#### Reference
* [7 R data manipulation tools][14]
* [My code of R 5 packages for dealing with missing values][17]
  * [Original tutorial][18]
  
### Python Tools
* [MLFlow - Machine Learning Life Cycle Platform][58]
  * Record and query experiments: code, data, config, and results.
  * Packaging format for reproducible runs on any platform.
  * General format for sending models to diverse deployment tools.
  * [My Code to try it][59]
* [Pymongo - Get access to MongoDB easier][86]
  * [A tutorial][87]
  
# [My Production Solutions Notes][38]
### Notes in Speed Up
* Better not use pandas `iterrows()` but to store each record as a dictionary and store in a list for iteration. Iterating pandas rows can be much slower.

## [Learning Notes][133]
* How to apply Monte Carlo Simulation
* 3 types of t-test
* Regression plots, comparing model output with random output
* [Promising New Methods Mentioned in My Garden][193]

# [Keep up with Industry Updates][91]



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
[52]:https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/DEF_CON_video_list_linear_optimization.ipynb
[53]:https://www.analyticsvidhya.com/blog/2017/02/lintroductory-guide-on-linear-programming-explained-in-simple-english/?fbclid=IwAR3f6DSNp_vVdvET5SNIdv6AMhBnbF5prwFNk2tSQ2pnoRVW9V8T0R_IZYU&utm_medium=social&utm_source=facebook.com
[54]:https://www.youtube.com/watch?v=XK26I9eoSl8
[55]:https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
[56]:https://stackoverflow.com/questions/30849883/linear-programming-with-scipy-optimize-linprog
[57]:https://medium.com/datadriveninvestor/systematic-solution-for-time-series-forecasting-in-real-business-problems-2816747799d6
[58]:https://github.com/mlflow/mlflow/blob/master/docs/source/quickstart.rst
[59]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/ML_Tools/Readme.md#mlflow
[60]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/ReadMe.md#more-about-autoencoder
[61]:https://blog.minitab.com/blog/adventures-in-statistics-2/regression-analysis-how-do-i-interpret-r-squared-and-assess-the-goodness-of-fit
[62]:https://blog.minitab.com/blog/adventures-in-statistics-2/why-you-need-to-check-your-residual-plots-for-regression-analysis
[63]:https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/
[64]:https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html
[65]:https://www.statisticshowto.datasciencecentral.com/durbin-watson-test-coefficient/
[66]:http://www.math.nsysu.edu.tw/~lomn/homepage/class/92/DurbinWatsonTest.pdf
[67]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/tree/master/Better4Industry
[68]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Better4Industry/Feature_Selection_Collection/ReadMe.md#shap
[69]:https://eli5.readthedocs.io/en/latest/tutorials/index.html
[70]:https://github.com/TeamHG-Memex/eli5
[71]:https://github.com/DistrictDataLabs/yellowbrick
[72]:https://www.scikit-yb.org/en/latest/api/index.html
[73]:https://github.com/SeldonIO/alibi
[74]:https://docs.seldon.io/projects/alibi/en/latest/overview/getting_started.html
[75]:https://github.com/tensorflow/lucid#community
[76]:https://github.com/tensorflow/lucid#notebooks
[77]:https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
[78]:https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py
[79]:https://gist.github.com/robertnishihara/2b81595abd4f50a049767a040ce435ab
[80]:https://towardsdatascience.com/10x-faster-parallel-python-without-python-multiprocessing-e5017c93cce1
[81]:https://github.com/hanhanwu/Hanhan_Applied_DataScience/tree/master/Prototype_Toolkit
[82]:https://www.analyticsvidhya.com/blog/2020/06/introduction-anova-statistics-data-science-covid-python/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[83]:https://www.analyticsvidhya.com/blog/2020/07/difference-between-sql-keys-primary-key-super-key-candidate-key-foreign-key/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[84]:http://contrib.scikit-learn.org/category_encoders/index.html
[85]:https://www.analyticsvidhya.com/blog/2020/08/types-of-categorical-data-encoding/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[86]:https://github.com/mongodb/mongo-python-driver
[87]:https://www.analyticsvidhya.com/blog/2020/08/query-a-mongodb-database-using-pymongo/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[88]:https://paperswithcode.com/
[89]:https://www.analyticsvidhya.com/blog/2020/08/a-beginners-guide-to-cap-theorem-for-data-engineering/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[90]:https://github.com/hanhanwu/Hanhan_Applied_DataScience/tree/master/flask_basics
[91]:https://github.com/hanhanwu/Hanhan_Applied_DataScience/blob/master/industry_updates.md
[92]:https://github.com/hanhanwu/Hanhan_Applied_DataScience/tree/master/data_engineering
[93]:https://github.com/hanhanwu/Hanhan_Applied_DataScience/blob/master/stakeholder_notes.md
[94]:https://colab.research.google.com/github/bentoml/BentoML/blob/master/guides/quick-start/bentoml-quick-start-guide.ipynb
[95]:https://github.com/bentoml/BentoML
[96]:https://www.analyticsvidhya.com/blog/2020/10/cost-complexity-pruning-decision-trees/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[97]:https://stackoverflow.com/questions/17784587/gradient-descent-using-python-and-numpy
[98]:https://www.analyticsvidhya.com/blog/2020/10/how-does-the-gradient-descent-algorithm-work-in-machine-learning/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[99]:https://github.com/scikit-learn-contrib/imbalanced-learn
[100]:https://machinelearningmastery.com/super-learner-ensemble-in-python/
[101]:https://www.analyticsvidhya.com/blog/2020/10/how-to-deploy-machine-learning-models-in-azure-cloud-with-the-help-of-python-and-flask/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[102]:https://github.com/YyzHarry/imbalanced-semi-self
[103]:https://arxiv.org/pdf/2007.13518.pdf
[104]:https://www.tensorflow.org/federated
[105]:https://github.com/tensorflow/federated/blob/master/docs/tutorials/federated_learning_for_image_classification.ipynb
[106]:https://towardsdatascience.com/federated-learning-3097547f8ca3
[107]:https://github.com/tensorflow/federated/tree/master/docs/tutorials
[108]:https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
[109]:https://www.analyticsvidhya.com/blog/2020/10/adaboost-and-gradient-boost-comparitive-study-between-2-popular-ensemble-model-techniques/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[110]:https://github.com/hanhanwu/readings/blob/master/federated-learning_aggregation.pdf
[111]:https://cloudxlab.com/#pricing
[112]:https://prezi.com/welcome#/about-you
[113]:https://github.com/hanhanwu/Hanhan_Applied_DataScience/blob/master/story_telling/README.md
[114]:https://github.com/hanhanwu/Hanhan_Applied_DataScience/blob/master/auto_pipeline.md
[115]:https://towardsdatascience.com/federated-learning-a-step-by-step-implementation-in-tensorflow-aac568283399
[116]:https://www.analyticsvidhya.com/blog/2020/11/big-data-to-small-data-welcome-to-the-world-of-reservoir-sampling/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[117]:https://github.com/hanhanwu/Hanhan_Break_the_Limits/blob/master/Bank_Fantasy/Golden_Bridge/recommendation_experiments/merchant_similarity.ipynb
[118]:https://scikit-learn.org/stable/modules/manifold.html
[119]:https://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold
[120]:https://github.com/anish-lakkapragada/SeaLion
[121]:https://github.com/anish-lakkapragada/SeaLion/tree/main/examples
[122]:https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html
[123]:https://lightgbm.readthedocs.io/en/latest/Features.html#optimal-split-for-categorical-features
[124]:https://medium.com/analytics-vidhya/lightgbm-for-regression-with-categorical-data-b08eaff501d1
[125]:https://github.com/ShawnLeee/the-book/blob/master/pybooks/Programming%20Collective%20Intelligence.pdf
[126]:https://www.analyticsvidhya.com/blog/2021/03/standardized-vs-unstandardized-regression-coefficient/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[127]:https://en.wikipedia.org/wiki/Pareto_efficiency#Pareto_frontier
[128]:https://mattmotoki.github.io/beta-target-encoding.html
[129]:https://www.kaggle.com/mmotoki/beta-target-encoding
[130]:https://scikit-criteria.readthedocs.io/en/latest/api/madm/madm.html
[131]:https://christophm.github.io/interpretable-ml-book/index.html
[132]:https://github.com/hanhanwu/Hanhan_Applied_DataScience/blob/master/auto_pipeline.md#param-tuning-hpo
[133]:https://github.com/hanhanwu/Hanhan_Applied_DataScience/blob/master/Learning_Notes.md
[134]:https://www.analyticsvidhya.com/blog/2021/09/q-q-plot-ensure-your-ml-model-is-based-on-the-right-distributions/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[135]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice#advanced-tools
[136]:https://www.analyticsvidhya.com/blog/2021/09/onevsrest-classifier-for-predicting-multiple-tags-of-research-articles/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[137]:https://www.analyticsvidhya.com/blog/2021/09/different-type-of-correlation-metrics-used-by-data-scientist/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[138]:https://www.analyticsvidhya.com/blog/2021/09/a-complete-guide-on-sampling-techniques/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[139]:https://www.analyticsvidhya.com/blog/2021/10/mlops-and-the-importance-of-data-drift-detection/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[140]:https://link.springer.com/content/pdf/10.1007/s10922-021-09609-5.pdf
[141]:https://www.kdnuggets.com/2021/05/deal-with-categorical-data-machine-learning.html
[142]:https://github.com/mwburke/population-stability-index
[143]:https://www.analyticsvidhya.com/blog/2021/12/introduction-to-tibco-spotfire-for-interactive-data-visualization-and-analysis/?utm_source=feedburner&utm_medium=email
[144]:https://github.com/getredash/redash
[145]:https://www.analyticsvidhya.com/blog/2021/12/ml-hyperparameter-optimization-app-using-streamlit/?utm_source=feedburner&utm_medium=email
[146]:https://github.com/microsoft/MLOps
[147]:https://www.analyticsvidhya.com/blog/2021/04/bring-devops-to-data-science-with-continuous-mlops/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[148]:https://github.com/amitvkulkarni/Bring-DevOps-to-Machine-Learning-with-CML
[149]:https://github.com/microsoft/MLOpsPython/blob/master/docs/getting_started.md
[150]:https://www.analyticsvidhya.com/blog/2022/01/overview-of-mlops-with-open-source-tools/?utm_source=feedburner&utm_medium=email
[151]:https://www.analyticsvidhya.com/blog/2022/01/a-comprehensive-guide-on-kubernetes/?utm_source=feedburner&utm_medium=email
[152]:https://github.com/oegedijk/explainerdashboard
[153]:https://www.analyticsvidhya.com/blog/2022/01/a-basic-guide-to-kubernetes-in-production/?utm_source=feedburner&utm_medium=email
[154]:https://www.analyticsvidhya.com/blog/2022/01/deploying-ml-models-using-kubernetes/?utm_source=feedburner&utm_medium=email
[155]:https://arxiv.org/pdf/2201.11358.pdf
[156]:https://contrib.scikit-learn.org/category_encoders/targetencoder.html
[157]:https://www.analyticsvidhya.com/blog/2021/12/custom-object-detection-on-the-browser-using-tensorflow-js/?utm_source=feedburner&utm_medium=email
[158]:https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
[159]:https://www.analyticsvidhya.com/blog/2022/02/optimal-resource-allocation-using-python/?utm_source=feedburner&utm_medium=email 
[160]:https://github.com/analokmaus/kuma_utils/blob/master/preprocessing/imputer.py
[161]:https://www.analyticsvidhya.com/blog/2022/03/linear-programming-discrete-optimization-with-pulp/?utm_source=feedburner&utm_medium=email
[162]:https://datakind-ai-ethics.netlify.app/#/fairness/scorecard
[163]:https://www.analyticsvidhya.com/blog/2021/04/numba-for-data-science-make-your-py-code-run-1000x-faster/
[164]:https://www.analyticsvidhya.com/blog/2022/04/docker-tutorial-for-beginners-part-i/?utm_source=feedburner&utm_medium=email
[165]:https://www.analyticsvidhya.com/blog/2022/05/handling-missing-values-with-random-forest/?utm_source=feedburner&utm_medium=email
[166]:https://ai.google/principles
[167]:https://github.com/PAIR-code/facets
[168]:https://www.analyticsvidhya.com/blog/2022/05/adding-explainability-to-clustering/?utm_source=feedburner&utm_medium=email
[169]:https://ubc-cs.github.io/cpsc330/lectures/22_deployment-conclusion.html
[170]:https://github.com/UBC-CS/cpsc330/blob/master/lectures/code/plotting_functions.py#L30
[171]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/learning_notes.md#tips-on-improving-deep-learning-performance
[172]:https://www.analyticsvidhya.com/blog/2021/04/dealing-with-anomalies-in-the-data/
[173]:https://github.com/hanhanwu/Hanhan_Applied_DataScience/blob/master/working_notes.md
[174]:https://www.analyticsvidhya.com/blog/2020/12/using-predictive-power-score-to-pinpoint-non-linear-correlations/
[175]:https://towardsdatascience.com/how-to-benefit-from-the-semi-supervised-learning-with-label-spreading-algorithm-2f373ae5de96
[176]:https://scikit-learn.org/stable/modules/semi_supervised.html#label-propagation
[177]:https://towardsdatascience.com/t-sne-machine-learning-algorithm-a-great-tool-for-dimensionality-reduction-in-python-ec01552f1a1e
[178]:https://scikit-learn.org/stable/modules/semi_supervised.html#label-propagation
[179]:https://towardsdatascience.com/semi-supervised-classification-of-unlabeled-data-pu-learning-81f96e96f7cb
[180]:https://link.springer.com/content/pdf/10.1007/978-3-662-44415-3_24.pdf
[181]:https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
[182]:https://developers.google.com/machine-learning/guides/rules-of-ml
[183]:https://stats.stackexchange.com/questions/594659/is-one-variable-with-theoretical-distribution-and-a-second-variable-with-same-ob/594660#594660
[184]:https://towardsdatascience.com/pca-principal-component-analysis-how-to-get-superior-results-with-fewer-dimensions-7a70e8ab798c
[185]:https://towardsdatascience.com/lda-linear-discriminant-analysis-how-to-improve-your-models-with-supervised-dimensionality-52464e73930f
[186]:https://towardsdatascience.com/mds-multidimensional-scaling-smart-way-to-reduce-dimensionality-in-python-7c126984e60b
[187]:https://towardsdatascience.com/isomap-embedding-an-awesome-approach-to-non-linear-dimensionality-reduction-fc7efbca47a0
[188]:https://towardsdatascience.com/lle-locally-linear-embedding-a-nifty-way-to-reduce-dimensionality-in-python-ab5c38336107
[189]:https://towardsdatascience.com/umap-dimensionality-reduction-an-incredibly-robust-machine-learning-algorithm-b5acb01de568
[190]:https://towardsdatascience.com/self-training-classifier-how-to-make-any-algorithm-behave-like-a-semi-supervised-one-2958e7b54ab7
[191]:https://l7.curtisnorthcutt.com/confident-learning
[192]:https://towardsdatascience.com/you-dont-need-neural-networks-to-do-continual-learning-2ed3bfe3dbfc
[193]:https://github.com/lady-h-world/My_Garden/blob/main/reading_pages/Graden_Museum/weaponry.md
[194]:https://developer.nvidia.com/blog/categorical-features-in-xgboost-without-manual-encoding/
[195]:https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html#optimal-partitioning
[196]:https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
[197]:https://www.evidentlyai.com/blog/data-drift-detection-large-datasets
[198]:https://madewithml.com/
[199]:https://github.com/hanhanwu/Hanhan_Applied_DataScience/blob/master/AI_Learning_Notes.md
[200]:https://github.com/yuenshingyan/MissForest
