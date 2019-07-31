import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import norm, skew
from sklearn.preprocessing import LabelEncoder

sns.set()


def load_dataset():

    train = pd.read_csv('data/raw_data/train.csv')
    test = pd.read_csv('data/raw_data/test.csv')
    print(train.columns)

    description = train['SalePrice'].describe()
    print(description)

    # Histogram of target variable
    sns.distplot(train['SalePrice'])
    plt.show()

    # Skewness and Kurtosis of target variable
    print("Skewness: %f" % train['SalePrice'].skew())
    print("Kurtosis: %f" % train['SalePrice'].kurt())

    return train, test


def make_corr_analysis(train):

    # Correlation matrix of all variables
    corrmat = train.corr()
    f, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(corrmat, square=True)
    plt.show()

    # SalePrice`s correlation matrix
    k = 10  # number of variables for heatmap
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(train[cols].values.T)
    sns.set(font_scale=1.25)
    f, ax = plt.subplots(figsize=(10, 10))
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                     xticklabels=cols.values)
    plt.show()

    # Take only those variable which are filtered after MULTICOLINEARITY test
    # And make scatter plot
    cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    sns.pairplot(train[cols])
    plt.show()


def handle_missing_data(train, test):

    # Delete columns with high rate of missing vales
    train = train.drop(['Utilities'], axis=1)
    test = test.drop(['Utilities'], axis=1)

    sale_price = train.loc[:, train.columns == 'SalePrice']

    train = train.loc[:, train.columns != 'SalePrice']
    print(len(train.columns))

    # Unite train and test dataset
    data = pd.concat((train, test)).reset_index(drop=True)

    missing_values = data.isnull().sum().sort_values(ascending=False)
    percent_data = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([missing_values, percent_data], axis=1, keys=['Total', 'Percent'])
    print(missing_data)

    data = data.drop((missing_data[missing_data['Total'] > 4]).index, 1)

    for col in ('GarageArea', 'GarageCars'):
        data[col] = data[col].fillna(0)
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF'):
        data[col] = data[col].fillna(0)

    data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].mode()[0])
    data["Functional"] = data["Functional"].fillna("Typ")
    data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])
    data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])
    data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])
    data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])
    data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])
    data['MSSubClass'] = data['MSSubClass'].fillna("None")

    print(data.isnull().sum().max())

    # ### This module should be in other function but i want it be there. I case my rest
    # Ok, now we are dealing with the big boss. What do we have here?
    #   I) Something that, in general, presents skewness.
    #   II) A significant number of observations with value zero (houses without basement).
    #   III) A big problem because the value zero doesn't allow us to do log transformations.
    # To apply a log transformation here, we'll create a variable that can get the effect of having or not having basement
    # (binary variable). Then, we'll do a log transformation to all the non-zero observations, ignoring those
    # with value zero. This way we can transform data, without losing the effect of having or not basement.

    # create column for new variable (one is enough because it's a binary categorical feature)
    # if area>0 it gets 1, for area==0 it gets 0
    data['HasBsmt'] = pd.Series(len(data['TotalBsmtSF']), index=data.index)
    data['HasBsmt'] = 0
    data.loc[data['TotalBsmtSF'] > 0, 'HasBsmt'] = 1

    # Transform data
    data.loc[data['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(data['TotalBsmtSF'])

    # Histogram and normal probability plot
    sns.distplot(train[train['TotalBsmtSF'] > 0]['TotalBsmtSF'])
    plt.title('TotalBsmtSF with Norm')
    plt.show()
    #
    fig = plt.figure()
    res = stats.probplot(data[data['TotalBsmtSF'] > 0]['TotalBsmtSF'], plot=plt)
    plt.title('TotalBsmtSF ProbPlot after')
    plt.show()

    train = data[:1460]
    test = data[1460:]

    train = pd.concat((train, sale_price), axis=1)

    return train, test


def handle_outliers(train):

    # Standardizing data to find outliers
    saleprice_scaled = StandardScaler().fit_transform(train['SalePrice'][:, np.newaxis])
    low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
    high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]
    print('outer range (low) of the distribution:')
    print(low_range)
    print('\nouter range (high) of the distribution:')
    print(high_range)

    # From the pairplot we detected two variables with some outliers
    # Lets again visualize and handle with them

    # Handle with FIRST variable
    data = pd.concat([train['SalePrice'], train['GrLivArea']], axis=1)
    data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0, 800000))
    plt.title('Outliers Plot 1')
    plt.show()

    # Deleting points by visualizing analysis
    drop_indexes = train.loc[(train['GrLivArea'] >= 4000) & (train['SalePrice'] <= 300000)]
    print(drop_indexes)

    for point in drop_indexes['Id']:
        train = train.drop(train[train['Id'] == point].index)

    # Handle with SECOND variable
    # So, these outliers are not so critical and we wont drop them
    data = pd.concat([train['SalePrice'], train['TotalBsmtSF']], axis=1)
    data.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0, 800000))
    plt.title('Outliers Plot 2')
    plt.show()

    return train


def handle_normality(train, test):

    fig = plt.figure()
    res = stats.probplot(train['SalePrice'], plot=plt)
    plt.title('SalePrice ProbPlot before')
    plt.show()

    # Histogram and normal probability plot
    # Applying log transformation to make SalePrice from Unnormal to Normal distribution
    train['SalePrice'] = np.log(train['SalePrice'])

    fig = plt.figure()
    res = stats.probplot(train['SalePrice'], plot=plt)
    plt.title('SalePrice ProbPlot After')
    plt.show()

    sns.distplot(train['SalePrice'])
    plt.title('SalePrice In NORM')
    plt.show()

    # # #
    # # # Lets do it with GrLivArea
    # # #

    fig = plt.figure()
    res = stats.probplot(train['GrLivArea'], plot=plt)
    plt.title('GrLivArea ProbPlot before')
    plt.show()

    train['GrLivArea'] = np.log(train['GrLivArea'])
    test['GrLivArea'] = np.log(test['GrLivArea'])

    fig = plt.figure()
    res = stats.probplot(train['GrLivArea'], plot=plt)
    plt.title('GrLivArea ProbPlot After')
    plt.show()

    sns.distplot(train['GrLivArea'])
    plt.title('GrLivArea In NORM')
    plt.show()

    # # # #
    # # # # Lets do it again with TotalBsmtSF
    # # # #

    fig = plt.figure()
    res = stats.probplot(train['TotalBsmtSF'], plot=plt)
    plt.title('TotalBsmtSF ProbPlot before')
    plt.show()

    sns.distplot(train['TotalBsmtSF'])
    plt.title('TotalBsmtSF Without NORM')
    plt.show()

    return train


def handle_homo(train):
    plt.scatter(train['GrLivArea'], train['SalePrice'])
    plt.title('Homoscedasticity GrLivArea')
    plt.show()

    plt.scatter(train[train['TotalBsmtSF'] > 0]['TotalBsmtSF'], train[train['TotalBsmtSF'] > 0]['SalePrice'])
    plt.title('Homoscedasticity TotalBsmtSF')
    plt.show()


def make_categorical(data):
    # Make some label encoding
    # MSSubClass=The building class
    data['MSSubClass'] = data['MSSubClass'].apply(str)

    # Changing OverallCond into a categorical variable
    data['OverallCond'] = data['OverallCond'].astype(str)

    # Year and month sold are transformed into categorical features.
    data['YrSold'] = data['YrSold'].astype(str)
    data['MoSold'] = data['MoSold'].astype(str)

    cols = ('ExterQual', 'ExterCond', 'HeatingQC', 'KitchenQual', 'Functional',
            'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond',
            'YrSold', 'MoSold', 'LandSlope')
    # process columns, apply LabelEncoder to categorical features
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(data[c].values))
        data[c] = lbl.transform(list(data[c].values))

    data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
    print('Shape all_data: {}'.format(data.shape))

    return data


def handle_skew(data):
    numeric_feats = data.dtypes[data.dtypes != "object"].index

    # Check the skew of all numerical features
    skewed_feats = data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    print("\nSkew in numerical features: \n")
    skewness = pd.DataFrame({'Skew': skewed_feats})
    print(skewness)

    skewness = skewness[abs(skewness) > 0.75]
    print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

    from scipy.special import boxcox1p

    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:

        # all_data[feat] += 1
        if feat == 'Id':
            continue
        data[feat] = boxcox1p(data[feat], lam)

    return data


def handle_dummy(train, test):

    print(len(train.columns))
    print(len(test.columns))

    sale_price = train.loc[:, train.columns == 'SalePrice']

    train = train.loc[:, train.columns != 'SalePrice']
    print(len(train.columns))

    # Unite train and test dataset
    train = train.reset_index(drop=True)
    data = pd.concat([train, test])

    # Make categorical transformation
    data = make_categorical(data)

    # Handle with skew data
    data = handle_skew(data)

    # Get dummies
    data = pd.get_dummies(data)

    train = data[:1458]
    test = data[1458:]

    train['SalePrice'] = sale_price.values

    print(sale_price)
    print(len(train.columns))
    print(len(test.columns))

    return train, test


def main():

    # ###
    # 1) Load the data from csv
    # ###

    train, test = load_dataset()

    # ###
    # 2) Make correlation analysis
    # ###

    make_corr_analysis(train=train)

    # ###
    # 3) Handle missing data
    # ###
    # ###
    # Deleting such values is easy way, but we can input numeric missing data by median or smth similar,
    # not numeric by None and so on.
    # ###

    train, test = handle_missing_data(train, test)

    # ###
    # 4) Outliers
    # ###

    train = handle_outliers(train)

    # ###
    # 5) Normality, Homoscedasticity and Dummies
    # ###
    # ###
    # 5.1) Normality
    # ###

    train = handle_normality(train, test)

    # ###
    # 5.2) Homoscedasticity
    # ###
    # ###
    # The best choice to check Homoscedasticity -> visualize it. There are all well.
    # ###

    handle_homo(train)

    # ###
    # 5.3) Dummy values
    # ###

    train, test = handle_dummy(train, test)

    # Save dataset
    train.to_csv('data/processed_data/train_processed.csv', index=False)
    test.to_csv('data/processed_data/test_processed.csv', index=False)


if __name__ == "__main__":
    main()