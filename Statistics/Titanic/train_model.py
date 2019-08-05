import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import  StandardScaler, RobustScaler, MinMaxScaler, Normalizer
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


train_data = pd.read_csv('data/processed_data/train_processed.csv')
test_data = pd.read_csv('data/processed_data/test_processed.csv')

trainX = np.asarray(train_data.loc[:, train_data.columns != 'Survived'].values)
trainY = np.asarray(train_data['Survived'].values)

trainX, testX, trainY, testY = train_test_split(trainX, trainY, test_size=0.2)
print(trainX.shape, trainY.shape)
print(testX.shape, testY.shape)

# Ensemble learning
model = make_pipeline(RandomForestClassifier(n_estimators=100))
model = BaggingClassifier(base_estimator=model, n_estimators=100)
model.fit(trainX, trainY)
prediction = model.predict(testX)


# Evaluation
mean_squared_error_score = mean_squared_error(testY, prediction)
val_score = cross_val_score(model, testX, testY)
print("mean_squared_error_score: ", mean_squared_error_score)
print("val_score: ", val_score, val_score.mean())
print('Train score:', model.score(trainX, trainY))
print('Test score', model.score(testX, testY))

print(prediction[0:10])
print(trainY[0:10])

test = model.predict(test_data)

submission = pd.DataFrame(data={'PassengerId': np.int32(test_data['PassengerId'].values), 'Survived': test})
submission.to_csv('data/my_submission.csv', index=False)




