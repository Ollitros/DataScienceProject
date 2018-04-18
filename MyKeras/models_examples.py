import matplotlib.pyplot as plt
import numpy as np
import keras
from tensorflow.examples.tutorials.mnist import input_data
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, Conv2D, MaxPool2D, Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
from keras import backend as K


"""CONV2D model"""


def model_conv2d():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    img_rows, img_cols = 28, 28
    num_classes = 10

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential([Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=input_shape),
                        Activation('relu'),
                        MaxPool2D(pool_size=(2, 2)),
                        Dropout(0.25),
                        Flatten(),
                        Dense(128, activation='relu'),
                        Dropout(0.5),
                        Dense(10, activation='softmax')])
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=250, epochs=5)
    score = model.evaluate(x_test, y_test)
    print(score)


"""SEQUENTIAL MODEL"""


def model_sequential():
    mnist = input_data.read_data_sets('data/MNIST_data', one_hot=True)

    x_train = mnist.train.images
    y_train = mnist.train.labels

    x_test = mnist.test.images
    y_test = mnist.test.labels
    model = Sequential([Dense(124, input_dim=784), Activation('relu'),
                        Dense(64), Activation('relu'),
                        Dense(10), Activation('softmax')])
    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='sgd', loss='mean_squared_error',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=1000, batch_size=100)
    score = model.evaluate(x_test, y_test, batch_size=100)
    print(score)
    print(model.metrics_names)
    print(y_train[0])

    y_pred = model.predict_classes(x_train[0:10])
    print(y_pred)


"""MODEL MODEL"""


def model_model():
    mnist = input_data.read_data_sets('data/MNIST_data', one_hot=True)

    x_train = mnist.train.images
    y_train = mnist.train.labels

    x_test = mnist.test.images
    y_test = mnist.test.labels

    a = Input(shape=(784,))
    b = Dense(10)(a)
    c = Activation('softmax')(b)
    model = Model(inputs=a, outputs=c)
    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=1000, epochs=100)
    score = model.evaluate(x_test, y_test, batch_size=1000)
    print(score)

