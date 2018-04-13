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
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.datasets import make_blobs
from PIL import Image
from scipy import signal
from MyTensorFlow.utils import tile_raster_images

def start_RBM():

    mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

    vb = tf.placeholder("float", [784])
    hb = tf.placeholder("float", [500])

    W = tf.placeholder("float", [784, 500])
    X = tf.placeholder("float", [None, 784])

    _h0 = tf.nn.sigmoid(tf.matmul(X, W) + hb)  # probabilities of the hidden units
    h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))  # sample_h_given_X

    _v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb)
    v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))  # sample_v_given_h
    h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)

    """ How this works ?
    
    with  tf.Session() as sess:
        a= tf.constant([0.7, 0.1, 0.8, 0.2])
        print (sess.run(a))
        b=sess.run(tf.random_uniform(tf.shape(a)))
        print (b)
        print (sess.run(a-b))
        print (sess.run(tf.sign( a - b)))
        print (sess.run(tf.nn.relu(tf.sign( a - b))))
    
    """

    alpha = 1.0
    w_pos_grad = tf.matmul(tf.transpose(X), h0)
    w_neg_grad = tf.matmul(tf.transpose(v1), h1)
    CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(X)[0])
    update_w = W + alpha * CD
    update_vb = vb + alpha * tf.reduce_mean(X - v1, 0)
    update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)

    err = tf.reduce_mean(tf.square(X - v1))

    cur_w = np.zeros([784, 500], np.float32)
    cur_vb = np.zeros([784], np.float32)
    cur_hb = np.zeros([500], np.float32)
    prv_w = np.zeros([784, 500], np.float32)
    prv_vb = np.zeros([784], np.float32)
    prv_hb = np.zeros([500], np.float32)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    sess.run(err, feed_dict={X: trX, W: prv_w, vb: prv_vb, hb: prv_hb})

    epochs = 5
    batchsize = 100
    weights = []
    errors = []

    for epoch in range(epochs):
        for start, end in zip(range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
            batch = trX[start:end]
            cur_w = sess.run(update_w, feed_dict={X: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
            cur_vb = sess.run(update_vb, feed_dict={X: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
            cur_hb = sess.run(update_hb, feed_dict={X: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
            prv_w = cur_w
            prv_vb = cur_vb
            prv_hb = cur_hb
            if start % 10000 == 0:
                errors.append(sess.run(err, feed_dict={X: trX, W: cur_w, vb: cur_vb, hb: cur_hb}))
                weights.append(cur_w)
        print('Epoch: %d' % epoch, 'reconstruction error: %f' % errors[-1])
    plt.plot(errors)
    plt.xlabel("Batch Number")
    plt.ylabel("Error")
    plt.show()

    uw = weights[-1].T
    print(uw)  # a weight matrix of shape (500,784)




# # VISUALISATION
#
# tile_raster_images(X=cur_w.T, img_shape=(28, 28), tile_shape=(25, 20), tile_spacing=(1, 1))
# image = Image.fromarray(tile_raster_images(X=cur_w.T, img_shape=(28, 28), tile_shape=(25, 20), tile_spacing=(1, 1)))
# ### Plot image
# plt.rcParams['figure.figsize'] = (18.0, 18.0)
# imgplot = plt.imshow(image)
# imgplot.set_cmap('gray')
# plt.show()
#
# image = Image.fromarray(
#     tile_raster_images(X=cur_w.T[10:11], img_shape=(28, 28), tile_shape=(1, 1), tile_spacing=(1, 1)))
# ### Plot image
# plt.rcParams['figure.figsize'] = (4.0, 4.0)
# imgplot = plt.imshow(image)
# imgplot.set_cmap('gray')
# plt.show()
#
# sample_case = trX[1:2]
# img = Image.fromarray(tile_raster_images(X=sample_case, img_shape=(28, 28), tile_shape=(1, 1), tile_spacing=(1, 1)))
# plt.rcParams['figure.figsize'] = (2.0, 2.0)
# imgplot = plt.imshow(img)
# imgplot.set_cmap('gray')  # you can experiment different colormaps (Greys,winter,autumn)
# plt.show()
#
# hh0 = tf.nn.sigmoid(tf.matmul(X, W) + hb)
# vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
# feed = sess.run(hh0, feed_dict={X: sample_case, W: prv_w, hb: prv_hb})
# rec = sess.run(vv1, feed_dict={hh0: feed, W: prv_w, vb: prv_vb})
# img = Image.fromarray(tile_raster_images(X=rec, img_shape=(28, 28), tile_shape=(1, 1), tile_spacing=(1, 1)))
# plt.rcParams['figure.figsize'] = (2.0, 2.0)
# imgplot = plt.imshow(img)
# imgplot.set_cmap('gray')
# plt.show()



# Class that defines the behavior of the RBM
class RBM(object):

    def __init__(self, input_size, output_size):
        # Defining the hyperparameters
        self._input_size = input_size  # Size of input
        self._output_size = output_size  # Size of output
        self.epochs = 5  # Amount of training iterations
        self.learning_rate = 1.0  # The step used in gradient descent
        self.batchsize = 100  # The size of how much data will be used for training per sub iteration

        # Initializing weights and biases as matrices full of zeroes
        self.w = np.zeros([input_size, output_size], np.float32)  # Creates and initializes the weights with 0
        self.hb = np.zeros([output_size], np.float32)  # Creates and initializes the hidden biases with 0
        self.vb = np.zeros([input_size], np.float32)  # Creates and initializes the visible biases with 0

    # Fits the result from the weighted visible layer plus the bias into a sigmoid curve
    def prob_h_given_v(self, visible, w, hb):
        # Sigmoid
        return tf.nn.sigmoid(tf.matmul(visible, w) + hb)

    # Fits the result from the weighted hidden layer plus the bias into a sigmoid curve
    def prob_v_given_h(self, hidden, w, vb):
        return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(w)) + vb)

    # Generate the sample probability
    def sample_prob(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

    # Training method for the model
    def train(self, X):
        # Create the placeholders for our parameters
        _w = tf.placeholder("float", [self._input_size, self._output_size])
        _hb = tf.placeholder("float", [self._output_size])
        _vb = tf.placeholder("float", [self._input_size])

        prv_w = np.zeros([self._input_size, self._output_size],
                         np.float32)  # Creates and initializes the weights with 0
        prv_hb = np.zeros([self._output_size], np.float32)  # Creates and initializes the hidden biases with 0
        prv_vb = np.zeros([self._input_size], np.float32)  # Creates and initializes the visible biases with 0

        cur_w = np.zeros([self._input_size, self._output_size], np.float32)
        cur_hb = np.zeros([self._output_size], np.float32)
        cur_vb = np.zeros([self._input_size], np.float32)
        v0 = tf.placeholder("float", [None, self._input_size])

        # Initialize with sample probabilities
        h0 = self.sample_prob(self.prob_h_given_v(v0, _w, _hb))
        v1 = self.sample_prob(self.prob_v_given_h(h0, _w, _vb))
        h1 = self.prob_h_given_v(v1, _w, _hb)

        # Create the Gradients
        positive_grad = tf.matmul(tf.transpose(v0), h0)
        negative_grad = tf.matmul(tf.transpose(v1), h1)

        # Update learning rates for the layers
        update_w = _w + self.learning_rate * (positive_grad - negative_grad) / tf.to_float(tf.shape(v0)[0])
        update_vb = _vb + self.learning_rate * tf.reduce_mean(v0 - v1, 0)
        update_hb = _hb + self.learning_rate * tf.reduce_mean(h0 - h1, 0)

        # Find the error rate
        err = tf.reduce_mean(tf.square(v0 - v1))

        # Training loop
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            # For each epoch
            for epoch in range(self.epochs):
                # For each step/batch
                for start, end in zip(range(0, len(X), self.batchsize), range(self.batchsize, len(X), self.batchsize)):
                    batch = X[start:end]
                    # Update the rates
                    cur_w = sess.run(update_w, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    cur_hb = sess.run(update_hb, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    cur_vb = sess.run(update_vb, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    prv_w = cur_w
                    prv_hb = cur_hb
                    prv_vb = cur_vb
                error = sess.run(err, feed_dict={v0: X, _w: cur_w, _vb: cur_vb, _hb: cur_hb})
                print ('Epoch: %d' % epoch, 'reconstruction error: %f' % error)
            self.w = prv_w
            self.hb = prv_hb
            self.vb = prv_vb

    # Create expected output for our DBN
    def rbm_outpt(self, X):
        input_X = tf.constant(X)
        _w = tf.constant(self.w)
        _hb = tf.constant(self.hb)
        out = tf.nn.sigmoid(tf.matmul(input_X, _w) + _hb)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(out)

def main():
    start_RBM()


if __name__ == '__main__':
    main()