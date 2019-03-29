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
    * For example, "H" indicates a transaction is fraud; "X" is the feature set that includes transaction amount, transaction time, such as $5000, at midnight. So the problem is we want to calculate the probability when a transaction of $5000 at midnight is fraud (`P(H|X)`). So we need the probability of transaction of $5000 at midnight (`P(X)`); the probability of fraud regardless of transaction amount and time (`P(H)`); and the probability of transaction amount=$5000 at midnight when it's fraud.
  * Naive Bayesian requires features to be independent to each other, given the class label. So that the calculation can be much easier. Here's why:
    * `P(Ci|X) = P(X|Ci)*P(Ci)/P(X)` is the problem Naive Bayesian needs to solve in order to get P(Ci|X).
    * P(X|Ci) can be very complex to calculate unless we assume each feature is independent given class label Ci, so that `P(X|Ci) = P(x1|Ci)*P(x2|Ci)*...*P(xn|Ci)`. This is the core for understanding whether in practice, we should make sure attributes are independent from each other conditioned on H.
  * <b>Bayes' Could Get > 1 Probability?</b>
    * Real world problem is always more complex, have you ever met a situation that your Bayes' Theorem got ">1" probability? This is not supposed to happen, but here's the reason and the solution.
    * <b>Reason:</b> Let's look at Bayes' Theorem formula again `P(Ci|X) = P(X|Ci)*P(Ci)/P(X)`. `P(X)` is calculated in the whole population, `P(X|Ci)` is calculated within class Ci. So if the cases almost all exist in class Ci, then P(X) will be smaller than P(X|Ci), and you may get the final probability larger than 1.
    * <b>Solution:</b> Change the population range, and see is there any way to make X could happen across classes
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
* LDA vs Bayesian
  * The LDA classifier assumes that the observations within each class come from a normal distribution with a class-specific mean vector and a common variance, and <b>plugging estimates for these parameters into the Bayes classifier</b>.


[1]:https://dl.acm.org/citation.cfm?id=507538
[2]:https://www.pythonforfinance.net/2016/11/28/monte-carlo-simulation-in-python/
[3]:https://pythonhosted.org/pyhmc/
[4]:https://mikulskibartosz.name/monte-carlo-simulation-in-python-d63f0cfcdf6f
[5]:https://static1.squarespace.com/static/56368331e4b08347cb5555e1/t/5c47d75bb91c915700195753/1548212060246/SCP_draft.pdf
[6]:https://docs.pymc.io/notebooks/Bayes_factor.html
