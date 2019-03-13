# autocorrelation plot
autocorrelation_plot(series)
plt.show()


# check whether the time series is strictly stationary with both ADF and KPSS
## My original code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/time_series_stationary_measures.ipynb
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12,center=False).mean()
    rolstd = timeseries.rolling(window=12,center=False).std()
    rolcov = timeseries.rolling(window=12,center=False).cov()

    # Plot rolling statistics:
    plt.figure(figsize=(9,7))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='green', label='Rolling Mean')
    std = plt.plot(rolstd, color='red', label = 'Rolling Std')
    cov = plt.plot(rolstd, color='purple', label = 'Rolling Cov')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    # Perform Augmented Dickey-Fuller test:
    print 'Results of Augmented Dickey-Fuller (ADF) Test:'
    dftest = adfuller(timeseries.iloc[:,0].values, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput
    print
    
    # Perform KPSS
    print 'Results of KPSS Test:'
    kpsstest = kpss(timeseries.iloc[:,0].values, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print kpss_output
