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


image = Image.open('data/bird.jpg')
print(image.bits, image.size, image.format, image.filename, image.mode, image.height, image.info )

im = image.convert('L')
print(im)

arr = np.asarray(im)
print(arr)

imgplot = plt.imshow(arr)
imgplot.set_cmap('gray')
plt.show()

kernel = np.array([[0,  1, 0],
                   [1, -4, 1],
                   [0,  1, 0], ])

grad = signal.convolve2d(arr, kernel, mode='same', boundary='symm')

fig, aux = plt.subplots(figsize=(10, 10))
aux.imshow(np.absolute(grad), cmap='gray')
plt.show()