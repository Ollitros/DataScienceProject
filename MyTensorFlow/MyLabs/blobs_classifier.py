import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

x, y = make_blobs(n_samples=900, centers=3, n_features=2, random_state=0, cluster_std=0.7)
plt.scatter(x[:, 0], x[:, 1])
plt.show()

y = pd.get_dummies(y).values
NumFeatures = x.shape
NumLabels = y.shape

X = tf.placeholder(tf.float32, shape=[None, NumFeatures[1]])
Y = tf.placeholder(tf.float32, shape=[None, NumLabels[1]])

w = tf.Variable(tf.random_normal([NumFeatures[1], NumLabels[1]]))
b = tf.Variable(tf.random_normal([1, NumLabels[1]]))

prediction = tf.nn.softmax(tf.add(tf.matmul(X, w), b))

loss = tf.nn.l2_loss(Y - prediction)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00008).minimize(loss)

weights = tf.Variable(tf.zeros([NumFeatures[1], NumLabels[1]]))
bias = tf.Variable(tf.zeros([1, NumLabels[1]]))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    cost = 0
    diff = 1
    epoch_values = []
    accuracy_values = []
    cost_values = []
    correct_predictions_OP = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))

    # If every false prediction is 0 and every true prediction is 1, the average returns us the accuracy
    accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))

    for step in range(1000):
        sess.run(optimizer, feed_dict={X: x, Y: y})

        if step % 10 == 0:
            # Add epoch to epoch_values
            epoch_values.append(step)
            # Generate accuracy stats on test data
            train_accuracy, newCost = sess.run([accuracy_OP, loss], feed_dict={X: x, Y: y})
            # Add accuracy to live graphing variable
            accuracy_values.append(train_accuracy)
            # Add cost to live graphing variable
            cost_values.append(newCost)
            # Re-assign values for variables
            diff = abs(newCost - cost)
            cost = newCost

            # generate print statements
            print("step %d, training accuracy %g, cost %g, change in cost %g" % (step, train_accuracy, newCost, diff))

    weights = sess.run(w)
    bias = sess.run(b)

    print(sess.run(prediction, feed_dict={X: [[1, 2]], w: weights, b: bias}))