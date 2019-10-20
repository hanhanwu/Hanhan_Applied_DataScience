# Learning Notes 🌟

## Target Encoder
* [Original Paper][1]
* This paper presents a simple data-preprocessing scheme that transforms high-cardinality categorical attributes into quasicontinuous scalar attributes suited for use in regression-type models. The key transformation used in the proposed scheme is one that maps each instance (value) of a high-cardinality categorical to the probability estimate of the target attribute. * In a classification scenario, the numerical representation corresponds to the posterior probability of the target, conditioned by the value of the categorical attribute. In a prediction scenario, the numerical representation corresponds to the expected value of the target given the value of the categorical attribute.
* <b>In a word, for high cardinality categorical feature, try this encoding method</b>


## Monte Carlo Simulation
* [The basic idea of Monte Carlo Simulation][2] is, you generate large amount of random "lines" that all satisfy the same distribution. So that you will have a bunch of records to evaluate your model. The example here is pretty good.
* [Hamiltonian Monte Carlo (HMC)][3] is a Markov chain Monte Carlo (MCMC) algorithm. Hamiltonian dynamics can be used to produce distant proposals for the Metropolis algorithm, thereby avoiding the slow exploration of the state space that results from the diffusive behaviour of simple random-walk proposals.
  * [To get the distribution of your data][4]
    * You can try the `fit_and_plot` to get the data distribution
    * To choose a better distribution, you can use ks test suggested. In this example, significance level 0.05 is used, and <b>the null hypothesis is "the distribution matches the real data distribution", when ks test p value is lower than the significance level, you reject the null hypothesis</b>. But if there is no choice, I would choose the higher p value, cuz that distribution might be closer
    * With the distribution, you can generate samples with HMC


## Pseudo Ground Truth
* Found the inspiration from the method used [here][5] - Psuedo Ground Truth Generation
* In the industry, there can be so many cases that you don't have complete labels of the data or you even have no label at all, and no matter whether you are using unsupervised method, you may drop into a situation where you still hope there could be some psuedo labels to give a certain level of insights or even help further model development.
* The inspiration got here is to use multiple estimators, use them to make some prediction, get max/avg/median results as the pseudo label. 
  * For example there is a dataset, we are not sure which is real fraud, nonfarud. We can try multiple unsupervised or even supervised estimators to make the prediction, choose the aggregated results to label the data or label part of the data that we have more confidence. 
  * With the pseudo labeled data, we can use it as training data for supervised learning, and may achieve better accuracy in the final prediction.


## Bayes Classification Methods
* Bayes' Theorem vs Naive Bayesian
  * This week, a task makes me think about the independence between attributes.
  * Bayes' Theorem
    * `P(H|X) = P(X|H)*P(H)/P(X)`
      * `P(H)` is the prior probability, which is independent from X.
      * `P(X|H)` is the posterior probability of X conditioned on H. `P(H|X)` is the posterior probability of H conditioned on X.
        * When `X` is categorical value, you just calculate P(X|H) with each value of X
        * When `X` is continuous value, continuous values are assumed to have normal distribution, so you just need to calcualte the mean μ and the stand deviation δ,and use values between [μ-δ, μ+δ] to caluclate P(X|H)
    * For example, "H" indicates a transaction is fraud; "X" is the feature set that includes transaction amount, transaction time, such as $5000, at midnight. So the problem is we want to calculate the probability when a transaction of $5000 at midnight is fraud (`P(H|X)`). So we need the probability of transaction of $5000 at midnight (`P(X)`); the probability of fraud regardless of transaction amount and time (`P(H)`); and the probability of transaction amount=$5000 at midnight when it's fraud.
  * Naive Bayesian requires features to be independent to each other, given the class label. So that the calculation can be much easier. Here's why:
    * `P(Ci|X) = P(X|Ci)*P(Ci)/P(X)` is the problem Naive Bayesian needs to solve in order to get P(Ci|X).
    * P(X|Ci) can be very complex to calculate unless we assume each feature is independent given class label Ci, so that `P(X|Ci) = P(x1|Ci)*P(x2|Ci)*...*P(xn|Ci)`. This is the core for understanding whether in practice, we should make sure attributes are independent from each other conditioned on H.
  * <b>Bayes' Could Get > 1 Probability?</b>
    * Real world problem is always more complex, have you ever met a situation that your Bayes' Theorem got ">1" probability? This is not supposed to happen, but here's the reason and the solution.
    * <b>Reason:</b> Let's look at Bayes' Theorem formula again `P(Ci|X) = P(X|Ci)*P(Ci)/P(X)`. `P(X)` is calculated in the whole population, `P(X|Ci)` is calculated within class Ci. So if the cases almost all exist in class Ci, then P(X) will be smaller than P(X|Ci), and you may get the final probability larger than 1.
    * <b>Solution 1 - change the population range:</b> Change the population range, and see is there any way to make X could happen across classes
    * <b>Solution 2 - Force the values into 0 and 1:</b>
      * We can use sigmoid function `exp(x)/(exp(x) + 1)` to normalize any real value into 0, 1 range
      * If we check the curve of sigmoid function, you can see that for any x that belongs to a real number, y is always between 0 and 1.
  * <b>Laplacia Correction</b>
    * When one of the P(xi|Ci) is 0, the whole P(X|Ci) will become 0, which means it cancles out all the other posterior probability conditioned on Ci.
    * To deal with this problem, with "Laplacia Correction", you can add 1 to each dividend when calculating each P(xi|Ci), so that you can avoid 0 probability and "corrected" probability is still close to the "uncorrected" ones
* Likelihood vs Probability
  * Likelihood `L(H|D) = k * P(D|H)`
    * The likelihood of a hypothesis (H) given some data (D) is proportional to the probability of obtaining D given that H is true, multiplied by an arbitrary positive constant k
    * For conditional probability, the hypothesis is treated as a given and the data are free to vary. For likelihood, the data are a given and the hypotheses vary.
    * When you are comparing the likelihood of 2 cases, you need to know/assume the distribution. With the distribution function, you calculate the probabilities P1, P2 for the 2 cases. Then likelihood ratio `LR = P1/P2` to tell you which case is more likely happen.
    * With more data records in the sample, your distribution can be narrower. Below is the sample distribution (likelihood function), left has 10 records, right has 100 records. The vertical dotted line marks the hypothesis best supported by the data. The 2 blue dots represent tge likelihood of 2 cases, higher the better. On the right side, even though the likelihood ratio between 2 cases is larger, both 2 have very low position in the likelihood function.
    <img src="https://github.com/hanhanwu/Hanhan_Applied_DataScience/blob/master/images/likelihood.png" width="700" height="400">
    
    * Considering the issue we are seeing above, here comes <b>Bayes Factor</b>.
      * A Bayes factor is a weighted average likelihood ratio based on the prior distribution specified for the hypotheses.
      * The likelihood ratio is evaluated at each point of the prior distribution and weighted by the probability we assign that value.
      * [How to calculate Bayesian Factor in Python][6]
        * They added prior in `beta_binom` function
  * Reference: https://alexanderetz.com/2015/04/15/understanding-bayes-a-look-at-the-likelihood/
  
## Linear Discriminant Analysis (LDA)
* When using R, it will return priori probability, which is `nk/n`, the estimated values
* LDA vs Bayesian
  * The LDA classifier assumes that the observations within each class come from a normal distribution with a class-specific mean vector and a common variance, and <b>plugging estimates for these parameters into the Bayes classifier</b>.
  * In a word, LDA is a method that plugs in estimated mean, variance and priori probability into logged function of bayes' theorem which was written with normal density function
  <img src="https://github.com/hanhanwu/Hanhan_Applied_DataScience/blob/master/images/bayesian_vs_LDA.png" width="600" height="400">
  
* LDA vs QDA (Quadratic Discriminant Analysis)
  * Same as LDA, QUA assumes that observations from each class are drawn from a Gaussian distribution, and plugs in estimates into Bayes' theorem
  * Different from LDA, QDA assumes each class has its own covariance matrix, while LDA assumes classes are sharing the same covariance matrix
  * When there's shared covariance matrix, Bayes decision boundry is linear; when each class has its own covariance matrix, Bayes decision boundry is non-linear
  * When there is less training observations so that the variance of each class is small, LDA is better than QDA; when there is larger amount of training observations that the variance of each class is larger, QDA is better than LDA
* LDA, Logistic Regression, QDA and KNN
  * When the true decision boundry is linear, LDA and Logictic regression can be better
    * Both LDA and Logictics Regression are linear function of `x`
    * The way they got coefficients are different. Logistic regresion uses maximum likelihood, LDA uses estimated mean and variance from a normal distribution
  * When the decision boundry is moderate non-linear, QDA maybe better
  * When the decison boundry is more complicated, non-parametric methods such as KNN can be better

## How to Understand Logictic Regression Output in R
* [Sample Code in R][7]
* Sample output
<img src="https://github.com/hanhanwu/Hanhan_Applied_DataScience/blob/master/images/logistic_regression_output.png" width="400" height="200">

* The target in this sample is "Direction" which has value "Up" or "Down"
* You can check p-value or z-value (opposite trend as p-value) to decide whether to reject NUll hypothesis, if reject the null hypothesis, it means the feature and the target has no clear correlation
* Lag1 has the lowest p-value and negative coefficient, indicating that when there is positive return yesterday, it's less likely to have the direction goes up today.
* 0.145 p-value is still larger than 0.05 (95% confidence level), therefore fail to reject the null hypothesis, which means there is correlation between Lag1 and Direction
  * <b>Larger the p-value is, the more likely that null hypothesis is correct, fail to reject it</b>
  
## 3 Types of t-test
* The assumptions of t-test
  * The data should follow a continuous or ordinal scale
  * The observations in the data should be randomly selected
  * The data should resemble a bell-shaped curve
  * Large sample size should be taken for the data to approach a normal distribution
  * Variances among the groups should be equal
* One Sample t-test - Compare group mean with population/theoretical mean
  * To compute t-value, `t = (m-μ) * sqrt(n)/s`
    * t - t value
    * m - group mean
    * μ - population/theoretical mean
    * n - the number of records in the group, sample size
    * s - standard deviation of the group
  * R t-test
    * In R, you just need to type `t.test(df$col,mu=10)`, mu is the population/theoretical mean here
    <img src="https://github.com/hanhanwu/Hanhan_Applied_DataScience/blob/master/images/one_sample_t_test.png" width="400" height="200">
    
    * To understand the output
      * Method 1 - Compare calculated t value with [t-table][11]
        * At the confidence level & degree of freedom (df = n-1), if calculated t is smaller than the value in t-table, then fail to reject the null hypothesis.
        * Null Hypothesis tends to be ".... are correlated", "..... are the same", etc.
      * Method 2 - Compare p-value with confidence level
        * If p-value is higher than 1 - confidence level, then fail to reject null hypothesis.
        * For example, cofidence level is 95%, p value is higher than 0.05, then accept the null hypothesis
* Two Sample t-test - Compare the means of 2 groups
  * [How to calculate t-value for 2 sample t-test][12]
  * R t-test
    * In R, you just need to type `t.test(df1$col, df2$col,var.equal = T)`
    <img src="https://github.com/hanhanwu/Hanhan_Applied_DataScience/blob/master/images/two_sample_t_test.png" width="400" height="200">
    
* Paired Sample t-test - Compare the means for 1 group at 2 times/conditions
* [Reference][12]

## Comparing Distribution vs Checking Identicalness vs Checking Distance
* Recently I just realized, when comparing 2 distributions, even if they look very close in visualization, t-test or chi2 may still show larger difference then those having more visual difference. So currently, `K-L score` is what I'm using now.
  * <b>K-L score works better in normal distribution, for other types of distribution, it can return Inf</b>
  * However, K-L score does have its limitation. For example, 1 distribution is almost flat while the other has a very high peak, and k-l score could be 0. When this happens, using t-test statistics used [here][13] may help.
* Also `pearson correlation` can be better than t-test, chi2 when the date values in a list are not in the same scale.

## Prove better than Random Guessing
* The baseline of ROC is the random guess line, below it means the results are worse than random guess.
  * [Understanding ROC random guessing][8]
  * [A breif but easy-to-understand description][9]
  * [There's a theory about using Precision-Recall curve to check randomness][10]
    * It says the baseline should a horizontal line
    * I personally think it a vertical line. Because think about an extreme case, you predict all the results as positive (majority class), then recall will be constant, while precision won't. Same in this case, ROC also has baseline as random guessing

[1]:https://dl.acm.org/citation.cfm?id=507538
[2]:https://www.pythonforfinance.net/2016/11/28/monte-carlo-simulation-in-python/
[3]:https://pythonhosted.org/pyhmc/
[4]:https://mikulskibartosz.name/monte-carlo-simulation-in-python-d63f0cfcdf6f
[5]:https://static1.squarespace.com/static/56368331e4b08347cb5555e1/t/5c47d75bb91c915700195753/1548212060246/SCP_draft.pdf
[6]:https://docs.pymc.io/notebooks/Bayes_factor.html
[7]:https://github.com/hanhanwu/Hanhan_Applied_DataScience/blob/master/R_logistic_regression.R
[8]:https://medium.com/datadriveninvestor/understanding-roc-auc-curve-7b706fb710cb
[9]:https://stats.stackexchange.com/questions/46502/why-is-the-roc-curve-of-a-random-classifier-the-line-x-y
[10]:https://stats.stackexchange.com/questions/251175/what-is-baseline-in-precision-recall-curve
[11]:http://www.sjsu.edu/faculty/gerstman/StatPrimer/t-table.pdf
[12]:https://www.analyticsvidhya.com/blog/2019/05/statistics-t-test-introduction-r-implementation/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[13]:https://github.com/hanhanwu/Hanhan_Applied_DataScience/blob/master/Simple%20Production%20Solutions.ipynb
