import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from keras.applications import VGG19
from matplotlib.pyplot import imshow
from PIL import Image
from MyKeras.NeuralStyleTransfer.nst_utils import *
from MyKeras.NeuralStyleTransfer.nst_functions import *
import numpy as np
import tensorflow as tf


# Reset the graph
tf.reset_default_graph()

# Start interactive session
sess = tf.InteractiveSession()

model = load_vgg_model("data/pretrained-model/imagenet-vgg-verydeep-19.mat")

(content_image, style_image, generated_image) = get_images()

# Assign the content image to be the input of the VGG model.
sess.run(model['input'].assign(content_image))

# Select the output tensor of layer conv4_2
out = model['conv4_2']

# Set a_C to be the hidden layer activation from the layer we have selected
a_C = sess.run(out)

# Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2']
# and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
# when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
a_G = out

# Compute the content cost
J_content = compute_content_cost(a_C, a_G)

# Assign the input of the model to be the "style" image
sess.run(model['input'].assign(style_image))

# Compute the style cost
J_style = compute_style_cost(sess, model, STYLE_LAYERS)

J = total_cost(J_content, J_style,  alpha = 10, beta = 40)
optimizer = tf.train.AdamOptimizer(2.0)
train_step = optimizer.minimize(J)


def model_nn(sess, input_image, num_iterations=200):
    # Initialize global variables (you need to run the session on the initializer)
    ### START CODE HERE ### (1 line)
    sess.run(tf.global_variables_initializer())
    ### END CODE HERE ###

    # Run the noisy input image (initial generated image) through the model. Use assign().
    ### START CODE HERE ### (1 line)
    sess.run(model['input'].assign(input_image))
    ### END CODE HERE ###

    for i in range(num_iterations):

        # Run the session on the train_step to minimize the total cost
        ### START CODE HERE ### (1 line)
        _ = sess.run(train_step)
        ### END CODE HERE ###

        # Compute the generated image by running the session on the current model['input']
        ### START CODE HERE ### (1 line)
        generated_image = sess.run(model['input'])
        ### END CODE HERE ###

        # Print every 20 iteration.
        if i % 20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))

            # save current generated image in the "/output" directory
            save_image("data/output_img" + str(i) + ".jpg", generated_image)

    # save last generated image
    save_image('data/output_img/generated_image.jpg', generated_image)

    return generated_image

model_nn(sess, generated_image)