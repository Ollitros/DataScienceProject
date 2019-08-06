import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import cv2 as cv
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
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