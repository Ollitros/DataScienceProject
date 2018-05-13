import tensorflow as tf
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sess = tf.Session()
LSTM_CELL_SIZE = 4  # output size (dimension), which is same as hidden size in the cell

lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_CELL_SIZE, state_is_tuple=True)
state = (tf.zeros([2,LSTM_CELL_SIZE]),)*2

# Let define a sample input. In this example, batch_size =2,and seq_len= 6:
sample_input = tf.constant([[1,2,3,4,3,2],[3,2,2,2,2,2]],dtype=tf.float32)
print (sess.run(sample_input))

#Now, we can pass the input to lstm_cell, and check the new state:
with tf.variable_scope("LSTM_sample1"):
    output, state_new = lstm_cell(sample_input, state)
sess.run(tf.global_variables_initializer())
print (sess.run(state_new))

# As we can see, the states has 2 parts, the new state, c, and also the output,
#  h. Lets check the output again:
print (sess.run(output))


#What about if we want to have a RNN with stacked LSTM?
# For example, a 2-layer LSTM. In this case, the output of the first
# layer will become the input of the second.

sess = tf.Session()

LSTM_CELL_SIZE = 4  #4 hidden nodes = state_dim = the output_dim
input_dim = 6
num_layers = 2

#Lets create the stacked LSTM cell:
cells = []
for _ in range(num_layers):
    cell = tf.contrib.rnn.LSTMCell(LSTM_CELL_SIZE)
    cells.append(cell)
stacked_lstm = tf.contrib.rnn.MultiRNNCell(cells)

#Now we can create the RNN:
# Batch size x time steps x features.
data = tf.placeholder(tf.float32, [None, None, input_dim])
output, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

#Lets say the input sequence length is 3, and the dimensionality
# of the inputs is 6. The input should be a Tensor of shape:
# [batch_size, max_time, dimension], in our case it would be (2, 3, 6)
#Batch size x time steps x features.
sample_input = [[[1,2,3,4,3,2], [1,2,1,1,1,2],[1,2,2,2,2,2]],
                [[1,2,3,4,3,2],[3,2,2,1,1,2],[0,0,0,0,3,2]]]

#we can now send our input to network:
sess.run(tf.global_variables_initializer())
sess.run(output, feed_dict={data: sample_input})