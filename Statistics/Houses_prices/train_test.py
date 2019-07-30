import pandas as pd
import numpy as np
from sklearn.linear_model import *
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, RobustScaler, MinMaxScaler, Normalizer
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor


#Validation function
n_folds = 5


def rmsle_cv_test(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(testX)
    rmse= np.sqrt(-cross_val_score(model, testX, testY, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


def rmsle_cv_train(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(trainX)
    rmse= np.sqrt(-cross_val_score(model, trainX, trainY, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


train_data = pd.read_csv('data/processed_data/train_processed.csv')
test_data = pd.read_csv('data/processed_data/test_processed.csv')

trainX = np.asarray(train_data.loc[:, train_data.columns != 'SalePrice'].values)
trainY = np.asarray(train_data['SalePrice'].values)

trainX = np.nan_to_num(trainX)
trainY = np.nan_to_num(trainY)

trainX, testX, trainY, testY = train_test_split(trainX, trainY, test_size=0.2)
print(trainX.shape, trainY.shape)
print(testX.shape, testY.shape)

# Ensemble learning
model = make_pipeline(Lasso(alpha=0.5))
model = BaggingRegressor(base_estimator=model, n_estimators=100)
model.fit(trainX, trainY)
prediction = model.predict(testX)

# Evaluation
mean_squared_error_score = mean_squared_error(testY, prediction)
val_score = cross_val_score(model, testX, testY)
r2_score = r2_score(testY, prediction)
print("mean_squared_error_score: ", mean_squared_error_score)
print("val_score: ", val_score, val_score.mean())
print("r2_score: ", r2_score)

print(prediction[0:10])
print(trainY[0:10])

test = model.predict(test_data.values)
test_transformed = np.exp(test)
submission = pd.DataFrame(data={'Id': np.int32(test_data['Id'].values), 'SalePrice': test_transformed})
submission.to_csv('data/my_submission.csv', index=False)


# score = rmsle_cv_train(model)
# print("\nTrain Score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# score = rmsle_cv_test(model)
# print("\nTest Score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


