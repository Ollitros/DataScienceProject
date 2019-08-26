import glob
import pandas as pd
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
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
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,  roc_curve, balanced_accuracy_score


def auc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


x_train = pd.read_csv('data/data_for_model/x_train_not_balanced.csv')
y_train = pd.read_csv('data/data_for_model/y_train_not_balanced.csv')

x_test = pd.read_csv('data/data_for_model/x_test.csv')
y_test = pd.read_csv('data/data_for_model/y_test.csv')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

y_train = utils.to_categorical(y_train, 2)
y_test = utils.to_categorical(y_test, 2)

batch_size = 1024
epochs = 1000
input_shape = (x_train.shape[1], )

best_model = Sequential([
    InputLayer(input_shape=input_shape),
    Dense(2048, kernel_regularizer='l1', kernel_initializer='random_normal'),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),

    Dense(1024, kernel_regularizer='l1', kernel_initializer='random_normal'),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),

    Dense(512, kernel_regularizer='l1', kernel_initializer='random_normal'),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),

    Dense(256, kernel_regularizer='l1', kernel_initializer='random_normal'),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),


    Dense(2, kernel_regularizer='l1', kernel_initializer='random_normal', activation='softmax')

])


best_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[auc])
print(best_model.summary())


model_checkpoint = ModelCheckpoint('ch/model-{epoch:02d}-{val_loss:.2f}_.h5', verbose=1, save_best_only=True)
best_model_history = best_model.fit(x_train, y_train, epochs=epochs,
                                    batch_size=batch_size, validation_split=0.3, verbose=1, callbacks=[model_checkpoint])
best_model.save_weights('ch/last_epoch_.h5')


y_pred = best_model.predict(x_test)
print('Balanced:', balanced_accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
print('AUC: ', auc(y_test, y_pred))

probs = y_pred[:, 1]
auc = roc_auc_score(np.argmax(y_test, axis=1), probs)
fpr, tpr, thresholds = roc_curve(np.argmax(y_test, axis=1), probs)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr)
plt.show()
print('SKLEARN AUC: ', auc)


evals = []
files = []
aucs = []
for file in glob.glob("ch/*.h5"):
    print(file)
    best_model.load_weights(file)
    y_pred = best_model.predict(x_test)

    eval = balanced_accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    auc = roc_auc_score(np.argmax(y_test, axis=1), y_pred[:, 1])
    evals.append(eval)
    files.append(file)
    aucs.append(auc)

for eval, file, auc in zip(evals, files, aucs):
    print(file, eval, auc)
