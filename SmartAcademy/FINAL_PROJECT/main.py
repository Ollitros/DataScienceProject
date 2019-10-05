import glob
import pandas as pd
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import *
from keras import datasets, utils, backend as K
from keras import regularizers
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.feature_selection import RFECV, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,  roc_curve, balanced_accuracy_score
sns.set()


def conv_block(inputs, channel_axis, growth, residuals):
    x = BatchNormalization()(inputs)
    x = Activation('relu')(x)
    x = Conv2D(64, 1, use_bias=False)(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', use_bias=False)(x)
    x = Dropout(0.25)(x)
    residuals.append(x)
    x = concatenate(residuals, axis=channel_axis)

    return x


def transition_block(inputs, reduction, channel_axis):

    x = BatchNormalization()(inputs)
    x = Activation('relu')(x)
    x = Conv2D(64, 1)(x)
    x = AveragePooling2D(2, strides=2)(x)
    return x


def dense_block(x, blocks, channel_axis):
    residuals = list()
    residuals.append(x)
    for i in range(blocks):
        x = conv_block(x, channel_axis, 16, residuals)

    return x


def DenseNet(input_shape, blocks):

    channel_axis =1
    # Preliminary convolution
    inputs = Input(shape=input_shape)
    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(inputs)
    x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='conv1/relu')(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPool2D(strides=1, padding='same')(x)

    x = dense_block(x, blocks[0], channel_axis=channel_axis)
    x = transition_block(x, 0.5, channel_axis=channel_axis)
    x = dense_block(x, blocks[1], channel_axis=channel_axis)
    x = transition_block(x, 0.5, channel_axis=channel_axis)
    x = dense_block(x, blocks[2], channel_axis=channel_axis)
    x = transition_block(x, 0.5, channel_axis=channel_axis)
    x = dense_block(x, blocks[3], channel_axis=channel_axis)

    # Final part
    x = GlobalAveragePooling2D()(x)
    print(x)
    # x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.25)(x)
    dense = Dense(2, activation='softmax')(x)

    model = Model(inputs=[inputs], outputs=[dense])

    return model


def auc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


x_train = pd.read_csv('data/data_for_model/x_train.csv')
y_train = pd.read_csv('data/data_for_model/y_train.csv')

x_test = pd.read_csv('data/data_for_model/x_test.csv')
test_y = pd.read_csv('data/data_for_model/y_test.csv')

print(x_train.shape, y_train.shape)
print(x_test.shape, test_y.shape)

y_train = utils.to_categorical(y_train, 2)
y_test = utils.to_categorical(test_y, 2)


del x_train['WEEKDAY_APPR_PROCESS_START_FRIDAY_y']
del x_train['WEEKDAY_APPR_PROCESS_START_MONDAY_y']
del x_train['WEEKDAY_APPR_PROCESS_START_SATURDAY_y']
del x_train['WEEKDAY_APPR_PROCESS_START_SUNDAY_y']
del x_train['WEEKDAY_APPR_PROCESS_START_THURSDAY_y']
del x_train['WEEKDAY_APPR_PROCESS_START_TUESDAY_y']
del x_train['WEEKDAY_APPR_PROCESS_START_WEDNESDAY_y']
del x_train['FLAG_LAST_APPL_PER_CONTRACT_N']

del x_test['WEEKDAY_APPR_PROCESS_START_FRIDAY_y']
del x_test['WEEKDAY_APPR_PROCESS_START_MONDAY_y']
del x_test['WEEKDAY_APPR_PROCESS_START_SATURDAY_y']
del x_test['WEEKDAY_APPR_PROCESS_START_SUNDAY_y']
del x_test['WEEKDAY_APPR_PROCESS_START_THURSDAY_y']
del x_test['WEEKDAY_APPR_PROCESS_START_TUESDAY_y']
del x_test['WEEKDAY_APPR_PROCESS_START_WEDNESDAY_y']
del x_test['FLAG_LAST_APPL_PER_CONTRACT_N']


print(x_train.shape, y_train.shape)
print(x_test.shape, test_y.shape)

x_train = np.reshape(x_train.values, (-1, 21, 22, 1))
x_test = np.reshape(x_test.values, (-1, 21, 22, 1))

print(x_train.shape, y_train.shape)
print(x_test.shape, test_y.shape)

batch_size = 256
epochs = 100

# input_shape = (x_train.shape[1], )
input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
blocks = [1, 1, 1, 1]

# best_model = Sequential([
#     InputLayer(input_shape=input_shape),
#     Dense(1024, kernel_regularizer='l2', kernel_initializer='random_normal'),
#     BatchNormalization(),
#     Activation('relu'),
#     Dropout(0.5),
#
#     Dense(256,  kernel_regularizer='l2', kernel_initializer='random_normal'),
#     BatchNormalization(),
#     Activation('relu'),
#     Dropout(0.5),
#
#
#     Dense(2, kernel_initializer='random_normal', activation='softmax')
#
# ])

best_model = DenseNet(input_shape=input_shape, blocks=blocks)
best_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[auc])
print(best_model.summary())


model_checkpoint = ModelCheckpoint('ch/model-{epoch:02d}-{val_loss:.2f}_.h5', verbose=1, save_best_only=True)
best_model_history = best_model.fit(x_train, y_train, epochs=epochs,
                                    batch_size=batch_size, validation_data=(x_test, y_test), verbose=1,
                                    callbacks=[model_checkpoint])
best_model.save_weights('ch/last_epoch_.h5')


y_pred = best_model.predict(x_test)
print('Balanced:', balanced_accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred > 0.08, axis=1)))
print('AUC: ', auc(y_test, y_pred))

probs = y_pred[:, 1]
print(probs)
auc = roc_auc_score(np.argmax(y_test, axis=1), probs)
fpr, tpr, thresholds = roc_curve(np.argmax(y_test, axis=1), probs)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr)
plt.show()
print('SKLEARN AUC: ', auc)
print((probs > 0.5).sum())
print('LABEL_AUC: ', roc_auc_score(test_y, (probs > 0.08)))
#  0.7270484305443423

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
