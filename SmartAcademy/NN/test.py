import pandas as pd
import glob, os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras.layers import *
from keras import datasets, utils, backend
from keras import regularizers
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint

train = pd.read_csv('train.csv')
test = pd.read_csv('val.csv')

print(train.head())
print(test.head())

del test['id']
del test['era']
del test['data_type']

train_y = train['target'].values
del train['target']
train_x = train.copy().values

test_y = test['target'].values
del test['target']
test_x = test.copy().values

x_train, val_x, y_train, val_y = train_test_split(train_x, train_y, test_size=0.3)

input_shape = train_x.shape

y_train = utils.to_categorical(y_train, 2)
test_y = utils.to_categorical(test_y, 2)
val_y = utils.to_categorical(val_y, 2)

min_max = preprocessing.StandardScaler()
x_train = min_max.fit_transform(x_train)
test_x = min_max.transform(test_x)
val_x = min_max.transform(val_x)

model_seq = Sequential([
    InputLayer(input_shape=(input_shape[1], )),
    Dense(8, kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    Activation('relu'),

    Dense(4, kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    Activation('relu'),

    Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.001))
])

model_seq.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


