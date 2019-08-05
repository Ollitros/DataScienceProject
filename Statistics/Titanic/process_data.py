import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
sns.set()


def load_data():
    # Load data
    train = pd.read_csv('data/raw_data/train.csv')
    test = pd.read_csv('data/raw_data/test.csv')

    # Print some data info
    print('Shape of train:', train.shape)
    print('Shape of test:', test.shape)
    print('\nColumns in train:\n', train.columns)
    print('\nColumns in test:\n', test.columns)

    return train, test


def make_some_overview(train):

    train_corr = train.copy()

    cols = ('Sex', 'Embarked')

    for c in cols:
       lbl = LabelEncoder()
       lbl.fit(list(train_corr[c].values))
       train_corr[c] = lbl.transform(list(train_corr[c].values))

    # Make correlation analysis
    corrmat = train_corr.corr()
    f, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(corrmat, square=True)
    plt.show()

    print(train_corr.describe())

    cols = corrmat.nlargest(10, 'Survived')['Survived'].index
    cm = np.corrcoef(train_corr[cols].values.T)
    sns.set(font_scale=1.25)
    f, ax = plt.subplots(figsize=(10, 10))
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                     xticklabels=cols.values)
    plt.show()

    # Print most correlated features
    print(train_corr[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived',
                                                                                                    ascending=False))
    print(train_corr[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))

    # Print most no-correlated features
    print(train_corr[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))

    print(train_corr[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))

    # Visualize Age-Survived distribution
    g = sns.FacetGrid(train_corr, col='Survived')
    g.map(plt.hist, 'Age', bins=20)
    plt.show()

    # Visualize Age-Pclass-Survived distribution
    grid = sns.FacetGrid(train_corr, col='Survived', row='Pclass', size=2.2, aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    grid.add_legend()
    plt.show()

    # Visualize Sex-Pclass-Survived  from Embarked distribution
    grid = sns.FacetGrid(train, row='Embarked', size=2.2, aspect=1.6)
    grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
    grid.add_legend()
    plt.show()

    # Visualize Sex-Fare-Survived  from Embarked distribution
    grid = sns.FacetGrid(train, row='Embarked', col='Survived', size=2.2, aspect=1.6)
    grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
    grid.add_legend()
    plt.show()


def handle_missing_data(data):

    # Drop "Name" column
    data = data.drop(['Name'], axis=1)
    print('\nShape of combined train and test after drop <Name> column: ', data.shape)

    # Drop "Ticket" column
    data = data.drop(['Ticket'], axis=1)
    print('\nShape of combined train and test after drop <Ticket> column: ', data.shape)

    # Handle with missing values
    missing_values = data.isnull().sum().sort_values(ascending=False)
    percent_data = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([missing_values, percent_data], axis=1, keys=['Total', 'Percent'])
    print('Missing data:\n', missing_data)

    # Drop "Cabin" column because missing val > 70%
    data = data.drop(['Cabin'], axis=1)
    print('\nShape of combined train and test after drop <Cabin> column: ', data.shape)

    # Check again
    missing_values = data.isnull().sum().sort_values(ascending=False)
    percent_data = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([missing_values, percent_data], axis=1, keys=['Total', 'Percent'])
    print('Missing data:\n', missing_data)

    # Fill some missing data
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())

    # Combine several features into one
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = 0
    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1

    # Delete unimportant features
    data = data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

    print(data.head())
    # Check again
    print('Missing data:', data.isnull().sum().max())

    return data


def make_data_categorical(data, survived):
    # # Make data categorical
    # cols = ('Sex', 'Embarked')

    # for c in cols:
    #     lbl = LabelEncoder()
    #     lbl.fit(list(data[c].values))
    #     data[c] = lbl.transform(list(data[c].values))

    # Split data again
    data = pd.get_dummies(data)
    train = data[0:891].reset_index(drop=True)
    train['Survived'] = survived.values
    test = data[891:].reset_index(drop=True)

    print('Shape of train:', train.shape)
    print('Shape of test:', test.shape)

    return train, test


def main():

    # Load dataset
    train, test = load_data()

    # Do some data overview
    make_some_overview(train)

    # Combine train and test in one dataset
    survived = train.loc[:, train.columns == 'Survived']
    train = train.loc[:, train.columns != 'Survived']

    data = pd.concat([train, test], sort=False).reset_index(drop=True)
    print('\nShape of combined train and test: ', data.shape)

    # Handle missing data
    data = handle_missing_data(data)

    # Make data categorical
    train, test = make_data_categorical(data, survived)

    # Save data
    train.to_csv('data/processed_data/train_processed.csv', index=False)
    test.to_csv('data/processed_data/test_processed.csv', index=False)


if __name__ == '__main__':
    main()