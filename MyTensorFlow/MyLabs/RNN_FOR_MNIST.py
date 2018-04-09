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


iris = datasets.load_iris()
x = iris.data
y = iris.target

iris_y = pd.get_dummies(y).values

trainX, testX, trainY, testY = train_test_split(x, iris_y, test_size=0.2)
print(x[111, :], y[111])
NumFeature = trainX.shape[1]
NumLabel = trainY.shape[1]

X = tf.placeholder(tf.float32, [None, NumFeature])
Y = tf.placeholder(tf.float32, [None, NumLabel])

W = tf.Variable(tf.zeros([4, 3]))
b = tf.Variable(tf.zeros([3]))
inpt = tf.Variable(x[111, :], tf.float32)
inpt = tf.cast(inpt, tf.float32)

weights = tf.Variable(tf.random_normal([NumFeature, NumLabel], mean=0, stddev=0.01, name='weights'))
bias = tf.Variable(tf.random_normal([1, NumLabel], mean=0, stddev=0.01, name='bias'))

mul_weights = tf.matmul(X, weights, name='mul_weights')
add_bias = tf.add(mul_weights, bias, name='add_bias')
prediction = tf.nn.sigmoid(add_bias, name='prediction')

learningRate = tf.train.exponential_decay(learning_rate=0.0008, global_step=1, decay_steps=trainX.shape[0],
                                          decay_rate=0.95, staircase=True)
loss = tf.nn.l2_loss(Y - prediction, name='loss')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(loss)


def predict():
    prediction_2 = dict()
    for i in range(NumLabel):
        mul_weights_2 = tf.multiply(inpt, weights[:, i])
        mul_weights_2 = tf.reduce_sum(mul_weights_2)
        add_bias_2 = tf.add(mul_weights_2, bias[:, i], name='add_bias_custom')
        temp = tf.nn.sigmoid(add_bias_2, name='prediction_custom')
        prediction_2[i] = temp
    return prediction_2


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    correct_predictions_OP = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))

    # If every false prediction is 0 and every true prediction is 1, the average returns us the accuracy
    accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))

    # Summary op for regression output
    activation_summary_OP = tf.summary.histogram("output", prediction)
    # Summary op for accuracy
    accuracy_summary_OP = tf.summary.scalar("accuracy", accuracy_OP)
    # Summary op for cost
    cost_summary_OP = tf.summary.scalar("cost", loss)
    # Summary ops to check how variables (W, b) are updating after each iteration
    weightSummary = tf.summary.histogram("weights", weights.eval(session=sess))
    biasSummary = tf.summary.histogram("biases", bias.eval(session=sess))
    # Merge all summaries
    merged = tf.summary.merge([activation_summary_OP, accuracy_summary_OP, cost_summary_OP, weightSummary, biasSummary])

    writer = tf.summary.FileWriter("summary_logs", sess.graph)

    cost = 0
    diff = 1
    epoch_values = []
    accuracy_values = []
    cost_values = []

    for i in range(700):
        if i > 1 and diff < .0001:
            print("change in cost %g; convergence." % diff)
            break
        else:
            # Run training step
            step = sess.run(optimizer, feed_dict={X: trainX, Y: trainY})
            # Report occasional stats
            if i % 10 == 0:
                # Add epoch to epoch_values
                epoch_values.append(i)
                # Generate accuracy stats on test data
                train_accuracy, newCost = sess.run([accuracy_OP, loss], feed_dict={X: trainX, Y: trainY})
                # Add accuracy to live graphing variable
                accuracy_values.append(train_accuracy)
                # Add cost to live graphing variable
                cost_values.append(newCost)
                # Re-assign values for variables
                diff = abs(newCost - cost)
                cost = newCost

                # generate print statements
                print("step %d, training accuracy %g, cost %g, change in cost %g" % (i, train_accuracy, newCost, diff))

        # How well do we perform on held-out test data?
        print("final accuracy on test set: %s" % str(sess.run(accuracy_OP,
                                                              feed_dict={X: testX,
                                                                         Y: testY})))
    writer.close()

    W = sess.run(weights)
    b = sess.run(bias)

    print(sess.run(predict()))