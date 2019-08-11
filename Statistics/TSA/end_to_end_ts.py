import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import matplotlib


matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
warnings.filterwarnings("ignore")
sns.set()


def load_data():

    # Load data
    df = pd.read_excel("data/Superstore.xls")
    print(df.head(), df.shape)

    # Select data with "Furniture" category
    furniture = df.loc[df['Category'] == 'Furniture']
    print(furniture.head(), furniture.shape)

    # Print time limits
    print(furniture['Order Date'].min(), furniture['Order Date'].max())

    return furniture, df


def preprocess_data(data):

    # Remove useless columns
    cols = data.columns.values.tolist()

    columns = ['Sales', 'Order Date']
    for column in columns:
        if column in cols:
            cols.remove(column)

    data.drop(cols, axis=1, inplace=True)
    data = data.sort_values('Order Date')

    # Check for missing values
    print('\n Missing values:\n', data.isnull().sum())

    # Grouping and handling with indexes
    data = data.groupby('Order Date')['Sales'].sum().reset_index()
    print('\nGrouped data:\n', data.head())

    # Set column 'Order Date' ad index
    data = data.set_index('Order Date')
    print('\nIndexes:\n', data.index)

    # Take mean of every day in each month
    years = data['Sales'].resample('MS').mean()
    print('\nAll years:\n', years)
    print('\nParticular year:\n', years['2015'])

    # Print preprocessed data
    print('\nPreprocessed data:\n', data.head(), data.shape)

    # Visualize time-series
    years.plot()
    plt.title('Original time-series')
    plt.show()

    # Decompose time-series on  trend, seasonality, and noise
    decomposition = sm.tsa.seasonal_decompose(years, model='additive')
    decomposition.plot()
    plt.show()

    return years


def arima_forecast(data):

    # Make the GRID SEARCH to find out best parameters for model
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    print(data.values)

    # ADF Test
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(data.values, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print('Critial Values:')
        print(f'   {key}, {value}')


    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(data, order=param, seasonal_order=param_seasonal,
                                                enforce_stationarity=False, enforce_invertibility=False)
                results = mod.fit(disp=False)
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue

    # The above output suggests that SARIMAX(1, 1, 1) x(1, 1, 0, 12) yields
    # the lowest AIC value of 297.78.Therefore we should consider this to be optimal option.

    # Fit ARIMA model
    mod = sm.tsa.statespace.SARIMAX(data,
                                    order=(1, 1, 1),
                                    seasonal_order=(1, 1, 0, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit(disp=False)
    print(results.summary().tables[1])

    # We should always run model diagnostics to investigate any unusual behavior.
    # It is not perfect, however, our model diagnostics suggests that the model residuals are near normally distributed.

    # results.plot_diagnostics()
    # plt.show()

    # Validating forecasts
    #  To help us understand the accuracy of our forecasts, we compare predicted sales to real
    #  sales of the time series, and we set forecasts to start at 2017–01–01 to the end of the data.

    pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
    pred_ci = pred.conf_int()
    ax = data['2014':].plot(label='observed')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Furniture Sales')
    plt.legend()
    plt.show()

    data_forecasted = pred.predicted_mean
    data_truth = data['2017-01-01':]
    mse = ((data_forecasted - data_truth) ** 2).mean()
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
    print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

    # Producing and visualizing forecasts
    pred_uc = results.get_forecast(steps=100)
    pred_ci = pred_uc.conf_int()
    ax = data.plot(label='observed', figsize=(14, 7))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Furniture Sales')
    plt.legend()
    plt.show()


def combined(df):

    furniture = df.loc[df['Category'] == 'Furniture']
    office = df.loc[df['Category'] == 'Office Supplies']
    print(furniture.shape, office.shape)

    cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country',
            'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name',
            'Quantity', 'Discount', 'Profit']

    furniture.drop(cols, axis=1, inplace=True)
    office.drop(cols, axis=1, inplace=True)

    furniture = furniture.sort_values('Order Date')
    office = office.sort_values('Order Date')

    furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()
    office = office.groupby('Order Date')['Sales'].sum().reset_index()

    furniture = furniture.set_index('Order Date')
    office = office.set_index('Order Date')

    y_furniture = furniture['Sales'].resample('MS').mean()
    y_office = office['Sales'].resample('MS').mean()

    furniture = pd.DataFrame({'Order Date': y_furniture.index, 'Sales': y_furniture.values})
    office = pd.DataFrame({'Order Date': y_office.index, 'Sales': y_office.values})

    store = furniture.merge(office, how='inner', on='Order Date')
    store.rename(columns={'Sales_x': 'furniture_sales', 'Sales_y': 'office_sales'}, inplace=True)

    print(store.head())

    plt.figure(figsize=(20, 8))
    plt.plot(store['Order Date'], store['furniture_sales'], 'b-', label='furniture')
    plt.plot(store['Order Date'], store['office_sales'], 'r-', label='office supplies')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title('Sales of Furniture and Office Supplies')
    plt.legend()
    plt.show()

    first_date = store.ix[np.min(list(np.where(store['office_sales'] > store['furniture_sales'])[0])), 'Order Date']
    print("Office supplies first time produced higher sales than furniture is {}.".format(first_date.date()))

    return store, office


def main():

    # Load data
    furniture, df = load_data()

    # Preprocess data
    furniture = preprocess_data(furniture)

    # ARIMA forecasting
    arima_forecast(furniture)

    # Combine
    store, office = combined(df)


if __name__ == '__main__':

    main()