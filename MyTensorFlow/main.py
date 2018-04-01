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

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


sess = tf.InteractiveSession()

width = 28
height = 28
flat = width * height
class_output = 10

X = tf.placeholder(tf.float32, shape=[None, flat])
Y = tf.placeholder(tf.float32, shape=[None, class_output])

X_image = tf.reshape(X, [-1, 28, 28, 1])


# The FIRST layer
w_conv1 = tf.Variable(tf.truncated_normal([5,5,1,32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

convolve1 = tf.nn.conv2d(X_image, w_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1
h_conv1 = tf.nn.relu(convolve1)
conv1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")


# The SECOND layer
w_conv2 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

convolve2 = tf.nn.conv2d(conv1, w_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2
h_conv2 = tf.nn.relu(convolve2)
conv2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# The THIRD layer OR Compiling all this
layer2_matrix = tf.reshape(conv2, [-1, 49*64])

W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

fcl=tf.matmul(layer2_matrix, W_fc1) + b_fc1
h_fc1 = tf.nn.relu(fcl)

keep_prob = tf.placeholder(tf.float32)
layer_drop = tf.nn.dropout(h_fc1, keep_prob)


# Readout Layer (Softmax Layer)
W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

fc=tf.matmul(layer_drop, W_fc2) + b_fc2
y_CNN= tf.nn.softmax(fc)


####################
# Creating the train
####################


loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_CNN), reduction_indices=[1]))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_CNN, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

for i in range(10):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={X:batch[0], Y: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, float(train_accuracy)))
    optimizer.run(feed_dict={X: batch[0], Y: batch[1], keep_prob: 0.5})


print("test accuracy %g" % accuracy.eval(feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0}))
