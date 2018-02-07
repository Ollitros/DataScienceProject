from scipy import special, optimize
import numpy as np
import os
import matplotlib  as plt
from MachineLearning.lib import *
from mpl_toolkits.mplot3d import Axes3D
from sklearn import  *
import math

data = np.matrix('[10, 30, 14, 16, 32, 9, 24, 20, 22, 20, 19, 26, 27; '
                     '1,8,2,2,9,1,6,4,5,4,4,6,7]')
size = data.shape

y = np.array([])
y = np.append(y, data[0])
x = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1],
              [1,8,2,2,9,1,6,4,5,4,4,6,7],
              [2,4,2,2,3,2,3,2,3,2,2,4,4],
              [1,2,1,1,2,1,2,1,2,1,1,3,3]])
teta0, teta1 = 0, 0
teta2, teta3 = 0, 0
grown = 0.05

def cost_funct(xi):
    sum = 0
    hypo = 0
    for i in range(13):
        hypo = teta0 * x[0][i] + teta1 * x[1][i] + teta2 * (x[2][i]) + teta3 * (x[3][i])
        hypo = (hypo - y[i]) * x[xi][i]
        sum = sum + hypo

    sum = (1/13) * sum
    return sum


for i in range(10000):
    temp0 = teta0 - grown * (cost_funct(0))
    temp1 = teta1 - grown * (cost_funct(1))
    temp2 = teta2 - grown * (cost_funct(2))
    temp3 = teta3 - grown * (cost_funct(3))
    teta0 = temp0
    teta1 = temp1
    teta2 = temp2
    teta3 = temp3

teta = np.array([teta0, teta1, teta2, teta3])
print(teta)

plt.title('Regression')
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)

for i in range(size[1]):
    plt.scatter(data[1, i], data[0, i])

z = np.array([0, 2, 3, 4, 5, 6, 7, 8, 10])
z1 = np.array([2,4,2,2,3,2,3,2,3,2,2,4,4])
z2 = np.array([1,2,1,1,2,1,2,1,2,1,1,3,3])
hypo = np.array([])
for i in range(9):
    hypo = np.append(hypo, [(teta[0] + teta[1] * z[i] + teta2 * z1[i] + teta3 * z2[i])])
plt.plot(z, hypo)
plt.show()