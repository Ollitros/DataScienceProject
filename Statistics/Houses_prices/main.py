import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# ###
# 1) Load the data from csv
# ###

train = pd.read_csv('data/houses_prices/train.csv')
test = pd.read_csv('data/houses_prices/test.csv')
print(train.columns)

description = train['SalePrice'].describe()
print(description)

# Histogram of target variable
sns.distplot(train['SalePrice'])
plt.show()

# Skewness and Kurtosis of target variable
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())

# ###
# 2) Make correlation analysis
# ###

# Correlation matrix of all variables
corrmat = train.corr()
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corrmat,  square=True)
plt.show()

# SalePrice`s correlation matrix
k = 10 # number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(10, 10))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# Take only those variable which are filtered after MULTICOLINEARITY test
# And make scatter plot
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size=2.5)
plt.show()

# ###
# 3) Handle missing data
# ###

# Calculate percentage of missing data
missing_values = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([missing_values, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data)

# Delete variables with missing data except 'Electrical' -> drop only one row with missing value
train = train.drop((missing_data[missing_data['Total'] > 1]).index, 1)
train = train.drop(train.loc[train['Electrical'].isnull()].index)
print(train.isnull().sum().max())


