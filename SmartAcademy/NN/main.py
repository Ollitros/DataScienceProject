import pandas as pd
import glob
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import *
from keras import datasets, utils, backend
from keras import regularizers
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.feature_selection import RFECV, RFE
from sklearn.linear_model import LogisticRegression



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

y_train = utils.to_categorical(y_train, 2)
test_y = utils.to_categorical(test_y, 2)
val_y = utils.to_categorical(val_y, 2)

model_checkpoint = ModelCheckpoint('model-{epoch:02d}-{val_acc:.2f}_.h5', verbose=1, save_best_only=True)

min_max = preprocessing.MinMaxScaler()
x_train = min_max.fit_transform(x_train)
test_x = min_max.transform(test_x)
val_x = min_max.transform(val_x)


batch_size = 2048

epochs = 500


input_shape = (x_train.shape[1], )
best_model = Sequential([
    InputLayer(input_shape=input_shape),
    Dense(1024,
          kernel_initializer='random_normal'),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.25),


    Dense(2,
          kernel_initializer='random_normal', activation='softmax')

])
best_model.compile(optimizer='adam',
                   loss='categorical_crossentropy', metrics=['accuracy'])


print(best_model.summary())
best_model_history = best_model.fit(x_train, y_train, epochs=epochs,
                                   batch_size=batch_size, validation_data=(val_x, val_y),
                                   callbacks=[model_checkpoint], verbose=0)
evals = best_model.evaluate(test_x, test_y)
print(evals)
best_model.save_weights('last_epoch_.h5')


import glob

evals = []
files = []
for file in glob.glob("*_.h5"):
    best_model.load_weights(file)
    eval = best_model.evaluate(test_x, test_y)
    evals.append(eval)
    files.append(file)

for eval, file in zip(evals, files):
    print(file, eval)
