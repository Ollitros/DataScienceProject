import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import cv2 as cv
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import grangercausalitytests
from dateutil.parser import parse

sns.set()


# Import as Dataframe
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'])
print(df.head())

'''

    First overlook and plotting 

'''


def plot_val_by_years(df, x, y, title="", xlabel='Date', ylabel='Value', dpi=100):
    plt.figure(figsize=(16, 5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()


def plot_val_seasonaly():

    # Prepare data
    df['year'] = [d.year for d in df.date]
    df['month'] = [d.strftime('%b') for d in df.date]
    years = df['year'].unique()

    # Prep Colors
    np.random.seed(100)
    mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(years), replace=False)

    # Draw Plot
    plt.figure(figsize=(14, 10), dpi=80)
    for i, y in enumerate(years):
        if i > 0:
            plt.plot('month', 'value', data=df.loc[df.year == y, :], color=mycolors[i], label=y)
            plt.text(df.loc[df.year == y, :].shape[0] - .9, df.loc[df.year == y, 'value'][-1:].values[0], y,
                     fontsize=12, color=mycolors[i])

    # Decoration
    plt.gca().set(xlim=(-0.3, 11), ylim=(2, 30), ylabel='$Drug Sales$', xlabel='$Month$')
    plt.yticks(fontsize=12, alpha=.7)
    plt.title("Seasonal Plot of Drug Sales Time Series", fontsize=20)
    plt.show()


def draw_boxplot():

    # Draw Plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 7), dpi=80)
    sns.boxplot(x='year', y='value', data=df, ax=axes[0])
    sns.boxplot(x='month', y='value', data=df.loc[~df.year.isin([1991, 2008]), :])

    # Set Title
    axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=18)
    axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=18)
    plt.show()


# Plot data through years
plot_val_by_years(df, x=df.date, y=df.value, title='Monthly anti-diabetic drug sales in Australia from 1992 to 2008.')

# Plot data through months by years
plot_val_seasonaly()

# Plot boxplots
draw_boxplot()


'''

    Patterns in a time series
    Any time series may be split into the following components: Base Level + Trend + Seasonality + Error

'''


fig, axes = plt.subplots(1, 3, figsize=(20, 4), dpi=100)

pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/guinearice.csv',
            parse_dates=['date'], index_col='date').plot(title='Trend Only', legend=False, ax=axes[0])

pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/sunspotarea.csv',
            parse_dates=['date'], index_col='date').plot(title='Seasonality Only', legend=False, ax=axes[1])

pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/AirPassengers.csv',
            parse_dates=['date'], index_col='date').plot(title='Trend and Seasonality', legend=False, ax=axes[2])
plt.show()


'''
    
        Additive and multiplicative time series
    
    Depending on the nature of the trend and seasonality, a time series 
    can be modeled as an additive or multiplicative, 
    wherein, each observation in the series can be expressed 
    as either a sum or a product of the components:
    
    Additive time series:
    Value = Base Level + Trend + Seasonality + Error
    
    Multiplicative Time Series:
    Value = Base Level x Trend x Seasonality x Error
    
    
        How to decompose a time series into its components?
        
    You can do a classical decomposition of a time series by 
    considering the series as an additive or multiplicative 
    combination of the base level, trend, seasonal index and the residual.
    
    The seasonal_decompose in statsmodels implements this conveniently.
    
'''

# Import Data
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'], index_col='date')

# Multiplicative Decomposition
result_mul = seasonal_decompose(df['value'], model='multiplicative', extrapolate_trend='freq')

# Additive Decomposition
result_add = seasonal_decompose(df['value'], model='additive', extrapolate_trend='freq')

# Plot
plt.rcParams.update({'figure.figsize': (10, 10)})
result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
result_add.plot().suptitle('Additive Decompose', fontsize=22)
plt.show()

# Extract the Components ----
# Actual Values = Product of (Seasonal * Trend * Resid)
df_reconstructed = pd.concat([result_mul.seasonal, result_mul.trend, result_mul.resid, result_mul.observed], axis=1)
df_reconstructed.columns = ['seas', 'trend', 'resid', 'actual_values']
print(df_reconstructed.head())


'''

        Stationary and Non-Stationary Time Series
        
    Stationarity is a property of a time series. A stationary 
    series is one where the values of the series is not a function of time.
    
    That is, the statistical properties of the series like mean, variance 
    and autocorrelation are constant over time. Autocorrelation of the series 
    is nothing but the correlation of the series with its previous values, 
    more on this coming up.
    
    A stationary time series id devoid of seasonal effects as well.
    
    So how to identify if a series is stationary or not? Let’s plot some examples to make it clear:
 
'''

image = cv.imread('stationary.png', 0)
cv.imshow('image', image)
cv.waitKey(0)
cv.destroyAllWindows()


'''

        So why does a stationary series matter? why am I even talking about it?

    I will come to that in a bit, but understand that it is possible to make nearly 
    any time series stationary by applying a suitable transformation. Most statistical 
    forecasting methods are designed to work on a stationary time series. The first step 
    in the forecasting process is typically to do some transformation to convert a 
    non-stationary series to stationary.

'''


'''

        How to make a time series stationary?
        
    You can make series stationary by:
    
        Differencing the Series (once or more)
        Take the log of the series
        Take the nth root of the series
        Combination of the above
        
    The most common and convenient method to stationarize the series is by differencing the 
    series at least once until it becomes approximately stationary.
    
    
        So what is differencing?
    
    If Y_t is the value at time ‘t’, then the first difference of Y = Yt – Yt-1. In simpler terms, 
    differencing the series is nothing but subtracting the next value by the current value.
    
    If the first difference doesn’t make a series stationary, you can go for the second differencing. And so on.
    
    For example, consider the following series: [1, 5, 2, 12, 20]
    
    First differencing gives: [5-1, 2-5, 12-2, 20-12] = [4, -3, 10, 8]
    
    Second differencing gives: [-3-4, -10-3, 8-10] = [-7, -13, -2]

'''


'''

        Why make a non-stationary series stationary before forecasting?\
        
    Forecasting a stationary series is relatively easy and the forecasts are more reliable.
    An important reason is, autoregressive forecasting models are essentially linear regression 
    models that utilize the lag(s) of the series itself as predictors.
    
    We know that linear regression works best if the predictors (X variables) are not correlated 
    against each other. So, stationarizing the series solves this problem since it removes any persistent 
    autocorrelation, thereby making the predictors(lags of the series) in the forecasting models nearly independent.
    
    Now that we’ve established that stationarizing the series important, how do you check if a 
    given series is stationary or not?

'''


'''

        How to test for stationarity?
        
    The stationarity of a series can be established by looking at the plot of the series like we did earlier.
    
    Another method is to split the series into 2 or more contiguous parts and computing the 
    summary statistics like the mean, variance and the autocorrelation. If the stats are quite 
    different, then the series is not likely to be stationary.
    
    Nevertheless, you need a method to quantitatively determine if a given series is stationary or not. 
    This can be done using statistical tests called ‘Unit Root Tests’. There are multiple variations of 
    this, where the tests check if a time series is non-stationary and possess a unit root.
    
        There are multiple implementations of Unit Root tests like:
    
        Augmented Dickey Fuller test (ADH Test)
        Kwiatkowski-Phillips-Schmidt-Shin – KPSS test (trend stationary)
        Philips Perron test (PP Test)
        The most commonly used is the ADF test, where the null hypothesis is the time series 
        possesses a unit root and is non-stationary. So, id the P-Value in ADH test is less 
        than the significance level (0.05), you reject the null hypothesis.
    
    The KPSS test, on the other hand, is used to test for trend stationarity. The null hypothesis and the 
    P-Value interpretation is just the opposite of ADH test. The below code implements these two tests using 
    statsmodels package in python.

'''

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'])

# ADF Test
result = adfuller(df.value.values, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')

# KPSS Test
result = kpss(df.value.values, regression='c')
print('\nKPSS Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[3].items():
    print('Critial Values:')
    print(f'   {key}, {value}')


'''

            How to detrend a time series?
    
'''

# Using statmodels: Subtracting the Trend Component.
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'], index_col='date')
result_mul = seasonal_decompose(df['value'], model='multiplicative', extrapolate_trend='freq')
detrended = df.value.values - result_mul.trend
plt.plot(detrended)
plt.title('Drug Sales detrended by subtracting the trend component', fontsize=16)
plt.show()


'''

        How to deseasonalize a time series?
        
    There are multiple approaches to deseasonalize a time series as well. Below are a few:

    - 1. Take a moving average with length as the seasonal window. This will smoothen in series in the process.

    - 2. Seasonal difference the series (subtract the value of previous season from the current value)

    - 3. Divide the series by the seasonal index obtained from STL decomposition

'''

# Subtracting the Trend Component.
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'], index_col='date')

# Time Series Decomposition
result_mul = seasonal_decompose(df['value'], model='multiplicative', extrapolate_trend='freq')

# Deseasonalize
deseasonalized = df.value.values / result_mul.seasonal

# Plot
plt.plot(deseasonalized)
plt.title('Drug Sales Deseasonalized', fontsize=16)
plt.plot()
plt.show()


'''

        How to treat missing values in a time series?
        
    Sometimes, your time series will have missing dates/times. That means, the data was not captured 
    or was not available for those periods. It could so happen the measurement was zero on those days, 
    in which case, case you may fill up those periods with zero.

    Secondly, when it comes to time series, you should typically NOT replace missing values with the mean 
    of the series, especially if the series is not stationary. What you could do instead for a quick and 
    dirty workaround is to forward-fill the previous value.

    However, depending on the nature of the series, you want to try out multiple approaches before concluding. 
    Some effective alternatives to imputation are:

    Backward Fill
    Linear Interpolation
    Quadratic interpolation
    Mean of nearest neighbors
    Mean of seasonal couterparts
    
    To measure the imputation performance, I manually introduce missing values to the time series, impute it 
    with above approaches and then measure the mean squared error of the imputed against the actual values.

'''


'''
    
        What is autocorrelation and partial autocorrelation functions?
        
    Autocorrelation is simply the correlation of a series with its own lags. 
    If a series is significantly autocorrelated, that means, the previous values 
    of the series (lags) may be helpful in predicting the current value.
    
    Partial Autocorrelation also conveys similar information but it conveys the pure 
    correlation of a series and its lag, excluding the correlation contributions from the intermediate lags.

'''

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv')

# Calculate ACF and PACF upto 50 lags
# acf_50 = acf(df.value, nlags=50)
# pacf_50 = pacf(df.value, nlags=50)

# Draw Plot
fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)
plot_acf(df.value.tolist(), lags=50, ax=axes[0])
plot_pacf(df.value.tolist(), lags=50, ax=axes[1])
plt.show()


'''

        Lag Plots
        
    A Lag plot is a scatter plot of a time series against a lag of itself. It is 
    normally used to check for autocorrelation. If there is any pattern existing in 
    the series like the one you see below, the series is autocorrelated. If there is 
    no such pattern, the series is likely to be random white noise.

    In below example on Sunspots area time series, the plots get more and more scattered as the n_lag increases.

'''

from pandas.plotting import lag_plot
plt.rcParams.update({'ytick.left' : False, 'axes.titlepad':10})

# Import
ss = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/sunspotarea.csv')
a10 = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv')

# Plot
fig, axes = plt.subplots(1, 4, figsize=(10,3), sharex=True, sharey=True, dpi=100)
for i, ax in enumerate(axes.flatten()[:4]):
    lag_plot(ss.value, lag=i+1, ax=ax, c='firebrick')
    ax.set_title('Lag ' + str(i+1))

fig.suptitle('Lag Plots of Sun Spots Area \n(Points get wide and scattered with increasing lag -> lesser correlation)\n', y=1.15)

fig, axes = plt.subplots(1, 4, figsize=(10,3), sharex=True, sharey=True, dpi=100)
for i, ax in enumerate(axes.flatten()[:4]):
    lag_plot(a10.value, lag=i+1, ax=ax, c='firebrick')
    ax.set_title('Lag ' + str(i+1))

fig.suptitle('Lag Plots of Drug Sales', y=1.05)
plt.show()


'''

        How to estimate the forecastability of a time series?
        
    The more regular and repeatable patterns a time series has, the easier it is to forecast. The ‘Approximate Entropy’ can be used to quantify the regularity and unpredictability of fluctuations in a time series.

    The higher the approximate entropy, the more difficult it is to forecast it.

    Another better alternate is the ‘Sample Entropy’.

    Sample Entropy is similar to approximate entropy but is more consistent in estimating the complexity even for smaller time series. For example, a random time series with fewer data points can have a lower ‘approximate entropy’ than a more ‘regular’ time series, whereas, a longer random time series will have a higher ‘approximate entropy’.

    Sample Entropy handles this problem nicely. See the demonstration below.

    
'''

ss = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/sunspotarea.csv')
a10 = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv')
rand_small = np.random.randint(0, 100, size=36)
rand_big = np.random.randint(0, 100, size=136)


def ApEn(U, m, r):
    """Compute Aproximate entropy"""
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0)**(-1) * sum(np.log(C))

    N = len(U)
    return abs(_phi(m+1) - _phi(m))


print(ApEn(ss.value, m=2, r=0.2*np.std(ss.value)))     # 0.651
print(ApEn(a10.value, m=2, r=0.2*np.std(a10.value)))   # 0.537
print(ApEn(rand_small, m=2, r=0.2*np.std(rand_small))) # 0.143
print(ApEn(rand_big, m=2, r=0.2*np.std(rand_big)))     # 0.716


def SampEn(U, m, r):
    """Compute Sample entropy"""
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for j in range(len(x)) if i != j and _maxdist(x[i], x[j]) <= r]) for i in range(len(x))]
        return sum(C)

    N = len(U)
    return -np.log(_phi(m+1) / _phi(m))


print(SampEn(ss.value, m=2, r=0.2*np.std(ss.value)))      # 0.78
print(SampEn(a10.value, m=2, r=0.2*np.std(a10.value)))    # 0.41
print(SampEn(rand_small, m=2, r=0.2*np.std(rand_small)))  # 1.79
print(SampEn(rand_big, m=2, r=0.2*np.std(rand_big)))      # 2.42


'''
    
        Why and How to smoothen a time series?
        
    Smoothening of a time series may be useful in:
    
    Reducing the effect of noise in a signal get a fair approximation of the noise-filtered series.
    The smoothed version of series can be used as a feature to explain the original series itself.
    Visualize the underlying trend better
    So how to smoothen a series? Let’s discuss the following methods:
    
    Take a moving average
    Do a LOESS smoothing (Localized Regression)
    Do a LOWESS smoothing (Locally Weighted Regression)
    Moving average is nothing but the average of a rolling window of defined width. But you must choose 
    the window-width wisely, because, large window-size will over-smooth the series. For example, a window-size 
    equal to the seasonal duration (ex: 12 for a month-wise series), will effectively nullify the seasonal effect.
    
    LOESS, short for ‘LOcalized regrESSion’ fits multiple regressions in the local neighborhood of each point. 
    It is implemented in the statsmodels package, where you can control the degree of smoothing using frac argument 
    which specifies the percentage of data points nearby that should be considered to fit a regression model.

'''


# from statsmodels.nonparametric.smoothers_lowess import lowess
# plt.rcParams.update({'xtick.bottom' : False, 'axes.titlepad':5})

# # Import
# df_orig = pd.read_csv('datasets/elecequip.csv', parse_dates=['date'], index_col='date')

# # 1. Moving Average
# df_ma = df_orig.value.rolling(3, center=True, closed='both').mean()

# # 2. Loess Smoothing (5% and 15%)
# df_loess_5 = pd.DataFrame(lowess(df_orig.value, np.arange(len(df_orig.value)), frac=0.05)[:, 1], index=df_orig.index, columns=['value'])
# df_loess_15 = pd.DataFrame(lowess(df_orig.value, np.arange(len(df_orig.value)), frac=0.15)[:, 1], index=df_orig.index, columns=['value'])

# # Plot
# fig, axes = plt.subplots(4,1, figsize=(7, 7), sharex=True, dpi=120)
# df_orig['value'].plot(ax=axes[0], color='k', title='Original Series')
# df_loess_5['value'].plot(ax=axes[1], title='Loess Smoothed 5%')
# df_loess_15['value'].plot(ax=axes[2], title='Loess Smoothed 15%')
# df_ma.plot(ax=axes[3], title='Moving Average (3)')
# fig.suptitle('How to Smoothen a Time Series', y=0.95, fontsize=14)
# plt.show()


'''
        How to use Granger Causality test to know if one time series is helpful in forecasting another?
        
    Granger causality test is used to determine if one time series will be useful to forecast another.
    
    How does Granger causality test work?
    
    It is based on the idea that if X causes Y, then the forecast of Y based on previous values of Y AND the previous 
    values of X should outperform the forecast of Y based on previous values of Y alone.
    
    So, understand that Granger causality should not be used to test if a lag of Y causes Y. Instead, it is generally 
    used on exogenous (not Y lag) variables only.
    
    It is nicely implemented in the statsmodel package.
    
    It accepts a 2D array with 2 columns as the main argument. The values are in the first column and the predictor (X) 
    is in the second column.
    
    The Null hypothesis is: the series in the second column, does not Granger cause the series in the first. 
    If the P-Values are less than a significance level (0.05) then you reject the null hypothesis and conclude 
    that the said lag of X is indeed useful.
    
    The second argument maxlag says till how many lags of Y should be included in the test.
    
    In the above case, the P-Values are Zero for all tests. 
    So the ‘month’ indeed can be used to forecast the Air Passengers.
    
'''

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'])
df['month'] = df.date.dt.month
result = grangercausalitytests(df[['value', 'month']], maxlag=2)
print(result)