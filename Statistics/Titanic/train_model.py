import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import  StandardScaler, RobustScaler, MinMaxScaler, Normalizer
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from keras.layers import Dense, Dropout, Input, BatchNormalization, Activation
from keras.models import Model
from keras.utils import to_categorical


def make_model(input_shape, targets):

    inputs = Input(shape=input_shape)
    x = Dense(256, input_dim=input_shape)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)

    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)

    x = Dense(targets, activation='softmax')(x)

    model = Model(inputs, x)
    model.summary()

    return model


train_data = pd.read_csv('data/processed_data/train_processed.csv')
test_data = pd.read_csv('data/processed_data/test_processed.csv')

trainX = np.asarray(train_data.loc[:, train_data.columns != 'Survived'].values)
trainY = np.asarray(train_data['Survived'].values)

trainX, testX, trainY, testY = train_test_split(trainX, trainY, test_size=0.2)
print(trainX.shape, trainY.shape)
print(testX.shape, testY.shape)

# Ensemble learning
# model = make_pipeline(RandomForestClassifier(n_estimators=100))
# model = BaggingClassifier(base_estimator=model, n_estimators=1)
# model.fit(trainX, trainY)
# prediction = np.int32(model.predict(testX))

trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

trainX = trainX / 255
testX = testX / 255

model = make_model(input_shape=(9, ), targets=2)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(trainX, trainY, batch_size=32, epochs=1000, validation_data=(testX, testY))
prediction = np.int32(model.predict(testX))

# Evaluation
mean_squared_error_score = mean_squared_error(testY, prediction)
# val_score = cross_val_score(model, testX, testY)
print("mean_squared_error_score: ", mean_squared_error_score)
# print("val_score: ", val_score, val_score.mean())
# print('Train score:', model.score(trainX, trainY))
# print('Test score', model.score(testX, testY))

print(prediction[0:10])
print(trainY[0:10])

test = np.int32(model.predict(test_data))
test = np.argmax(test, axis=1)

submission = pd.DataFrame(data={'PassengerId': np.int32(test_data['PassengerId'].values), 'Survived': test})
submission.to_csv('data/my_submission.csv', index=False)




