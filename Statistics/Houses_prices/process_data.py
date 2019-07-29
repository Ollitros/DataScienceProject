import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import norm
sns.set()

# ###
# 1) Load the data from csv
# ###

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
sns.pairplot(train[cols])
plt.show()

# ###
# 3) Handle missing data
# ###
# ###
# Deleting such values is easy way, but we can input numeric missing data by median or smth similar,
# not numeric by None and so on.
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


# ###
# 4) Outliers
# ###

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

# ###
# 5) Normality, Homoscedasticity and Dummies
# ###
# ###
# 5.1) Normality
# ###

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

fig = plt.figure()
res = stats.probplot(train['GrLivArea'], plot=plt)
plt.title('GrLivArea ProbPlot After')
plt.show()

sns.distplot(train['GrLivArea'])
plt.title('GrLivArea In NORM')
plt.show()

# # #
# # # Lets do it again with TotalBsmtSF
# # #

fig = plt.figure()
res = stats.probplot(train['TotalBsmtSF'], plot=plt)
plt.title('TotalBsmtSF ProbPlot before')
plt.show()

sns.distplot(train['TotalBsmtSF'])
plt.title('TotalBsmtSF Without NORM')
plt.show()

# Ok, now we are dealing with the big boss. What do we have here?
#   I) Something that, in general, presents skewness.
#   II) A significant number of observations with value zero (houses without basement).
#   III) A big problem because the value zero doesn't allow us to do log transformations.
# To apply a log transformation here, we'll create a variable that can get the effect of having or not having basement
# (binary variable). Then, we'll do a log transformation to all the non-zero observations, ignoring those
# with value zero. This way we can transform data, without losing the effect of having or not basement.

# create column for new variable (one is enough because it's a binary categorical feature)
# if area>0 it gets 1, for area==0 it gets 0
train['HasBsmt'] = pd.Series(len(train['TotalBsmtSF']), index=train.index)
train['HasBsmt'] = 0
train.loc[train['TotalBsmtSF'] > 0, 'HasBsmt'] = 1

# Transform data
train.loc[train['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(train['TotalBsmtSF'])

# Histogram and normal probability plot
sns.distplot(train[train['TotalBsmtSF'] > 0]['TotalBsmtSF'])
plt.title('TotalBsmtSF with Norm')
plt.show()

fig = plt.figure()
res = stats.probplot(train[train['TotalBsmtSF'] > 0]['TotalBsmtSF'], plot=plt)
plt.title('TotalBsmtSF ProbPlot after')
plt.show()

# ###
# 5.2) Homoscedasticity
# ###
# ###
# The best choice to check Homoscedasticity -> visualize it. There are all well.
# ###

plt.scatter(train['GrLivArea'], train['SalePrice'])
plt.title('Homoscedasticity GrLivArea')
plt.show()

plt.scatter(train[train['TotalBsmtSF'] > 0]['TotalBsmtSF'], train[train['TotalBsmtSF'] > 0]['SalePrice'])
plt.title('Homoscedasticity TotalBsmtSF')
plt.show()

# ###
# 5.3) Dummy values
# ###

# Convert categorical variable into dummy
train = pd.get_dummies(train)
print(train.columns)
print(train)

train.to_csv('data/processed_data/train_processed.csv')
