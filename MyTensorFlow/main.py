import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

width = 28
height = 28
flat = width * height
class_output = 10

X = tf.placeholder(tf.float32, shape=[None, flat])
Y = tf.placeholder(tf.float32, shape=[None, class_output])

X_image = tf.reshape(X, [-1, 28, 28, 1])

w_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

convolve1 = tf.nn.conv2d(X_image, w_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1
h_conv1 = tf.nn.relu(convolve1)
conv1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

layer_matrix = tf.reshape(conv1, [-1, 14*14*32])

W_together = tf.Variable(tf.truncated_normal([14 * 14 * 32, 1024], stddev=0.1))
b_together = tf.Variable(tf.constant(0.1, shape=[1024]))

layer_out = tf.matmul(layer_matrix, W_together) + b_together
layer_h = tf.nn.relu(layer_out)

# Readout Layer (Softmax Layer)
W_readout = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_readout = tf.Variable(tf.constant(0.1, shape=[10]))

readout_layer = tf.matmul(layer_h, W_readout) + b_readout
y_CNN = tf.nn.softmax(readout_layer)

loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_CNN), reduction_indices=[1]))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_CNN, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

for i in range(500):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={X: batch[0], Y: batch[1]})
        print("step %d, training accuracy %g" % (i, float(train_accuracy)))
    optimizer.run(feed_dict={X: batch[0], Y: batch[1]})


print("test accuracy %g" % accuracy.eval(feed_dict={X: mnist.test.images, Y: mnist.test.labels}))