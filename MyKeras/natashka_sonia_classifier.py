import matplotlib.pyplot as plt
import numpy as np
import keras
from tensorflow.examples.tutorials.mnist import input_data
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, Activation, Input, Conv2D, MaxPool2D, Flatten, SimpleRNN, RNN, LSTM, Add
from keras.datasets import mnist
from keras import backend as K
from PIL import Image
from scipy import signal
from keras.optimizers import SGD



image = Image.open("data/test-0.JPG")
image = image.convert('L')
image1 = Image.open('data/test-1.JPG')
image1 = image.convert('L')
arr = np.asarray(image)
arr1 = np.asarray(image1)
print(arr.shape)
arr = np.append(arr, arr1)
arr = np.append(arr, arr)
arr = np.append(arr, arr)
arr = np.append(arr, arr)
arr = np.append(arr, arr)
arr = np.append(arr, arr)
arr = np.append(arr, arr)
print(arr.shape)


test = Image.open('data/ttt.JPG')
test = test.convert('L')
test = np.asarray(test)
test = test.reshape(1, 73728)

arr = np.append(arr, test)
arr = arr.reshape((129, 73728))

y_train = np.array(np.ones([128]))
y_train = np.append(y_train, [0])
y_train = keras.utils.to_categorical(y_train, 2)


model = Sequential([Dense(124, input_shape=(73728, )), Activation('relu'),
                        Dense(64), Activation('relu'),
                        Dense(2), Activation('softmax')])
sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='sgd', loss='mean_squared_error',
              metrics=['accuracy'])
model.fit(arr, y_train, epochs=50)

pred = model.predict(test)
score = model.evaluate(arr, y_train, batch_size=100)
print(score)
print(model.metrics_names)
print(pred)



""" How to change image """
""" 256x288 the size
    469 dpi             """

# image = Image.open('1.JPG')
# print(image.bits, image.size, image.format, image.filename, image.mode, image.height, image.info)
#
# image = image.resize((256, 288))
# print(image.size, image.format, image.mode, image.height, image.info)
# image.save("ttt.JPG", dpi=(469, 469))