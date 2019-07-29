import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, RobustScaler
from sklearn.ensemble import BaggingRegressor


train_data = pd.read_csv('data/processed_data/train_processed.csv')
trainX = np.asarray(train_data.loc[:, train_data.columns != 'SalePrice'].values)
trainY = np.asarray(train_data['SalePrice'].values)

trainX, testX, trainY, testY = train_test_split(trainX, trainY, test_size=0.2)
print(trainX.shape, trainY.shape)
print(testX.shape, testY.shape)

# Grid Search
model = make_pipeline(RobustScaler(), Lasso(random_state=1))
features_params = np.linspace(0, 0.005, num=100)
param_grid = [{'lasso__alpha': features_params}]
gs = GridSearchCV(estimator=model, param_grid=param_grid)

gs = gs.fit(trainX, trainY)
best_params = gs.best_params_

# Ensemble learning
model = make_pipeline(RobustScaler(), Lasso(alpha=best_params['lasso__alpha'], random_state=1))
model = BaggingRegressor(base_estimator=model, n_estimators=10)
model.fit(trainX, trainY)
prediction = model.predict(testX)

# Evaluation
print(best_params)
mean_squared_error_score = mean_squared_error(testY, prediction)
val_score = cross_val_score(model, trainX, trainY)
r2_score = r2_score(testY, prediction)
print("mean_squared_error_score: ", mean_squared_error_score)
print("val_score: ", val_score, val_score.mean())
print("r2_score: ", r2_score)

print(prediction[0:10])
print(trainY[0:10])


with open("data/results.txt", "a") as file:
    file.write("|||\nModel: {},\n mean_squared_error_score:{}, val_score: {}, r2_score: {}\n\n".format(
        model, mean_squared_error_score, val_score.mean(), r2_score))
