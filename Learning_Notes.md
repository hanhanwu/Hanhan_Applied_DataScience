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



[1]:https://dl.acm.org/citation.cfm?id=507538
[2]:https://www.pythonforfinance.net/2016/11/28/monte-carlo-simulation-in-python/
[3]:https://pythonhosted.org/pyhmc/
[4]:https://mikulskibartosz.name/monte-carlo-simulation-in-python-d63f0cfcdf6f
