import tensorflow as tf
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.examples.tutorials.mnist import input_data
import pickle
from sklearn.datasets import make_blobs


mnist = input_data.read_data_sets('data/MNIST_data', one_hot=True)

train_imagies = mnist.train.images
train_labels = mnist.train.labels
test_imagies = mnist.test.images
test_labels = mnist.test.labels

Num_feature_train = train_imagies.shape
Num_labels_train = train_labels.shape

X = tf.placeholder(tf.float32, shape=[None, Num_feature_train[1]])
Y = tf.placeholder(tf.float32, shape=[None, Num_labels_train[1]])

weights = tf.Variable(tf.random_normal([Num_feature_train[1], Num_labels_train[1]]))
bias = tf.Variable(tf.random_normal([1, Num_labels_train[1]]))

w_out = tf.placeholder(tf.float32, shape=[Num_feature_train[1], Num_labels_train[1]])
b_out = tf.placeholder(tf.float32, shape=[1, Num_labels_train[1]])

prediction = tf.nn.softmax(tf.add(tf.matmul(X, weights), bias))
loss = tf.nn.l2_loss(Y - prediction)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.3).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    cost = 0
    diff = 1
    epoch_values = []
    accuracy_values = []
    cost_values = []
    correct_predictions_OP = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))

    for epoch in range(1000):
        batch = mnist.train.next_batch(100)
        sess.run(optimizer, feed_dict={X: batch[0], Y: batch[1]})

        if epoch % 10 == 0:

            epoch_values.append(epoch)
            train_accuracy, newCost = sess.run([accuracy_OP, loss], feed_dict={X: batch[0], Y: batch[1]})
            accuracy_values.append(train_accuracy)
            cost_values.append(newCost)
            diff = abs(newCost - cost)
            cost = newCost

            print("step %d, training accuracy %g, cost %g, change in cost %g" % (epoch, train_accuracy, newCost, diff))

    w_out = sess.run(weights)
    b_out = sess.run(bias)

    pred = sess.run(prediction, feed_dict={X: [train_imagies[5]], weights: w_out, bias: b_out})
    print(pred)
    val = np.max(pred)
    index = np.where(pred == val)
    print(index)

    image = train_imagies[5]
    image = np.reshape(image, (28, 28))

    plt.imshow(image)
    plt.show()