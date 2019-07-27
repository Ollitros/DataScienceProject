import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


a = np.array([1, 2, 3, 4, 5])
b = np.array([1, 4, 9, 16, 25])
plt.plot(a, b, label='plot')
plt.plot(b, a, label='pppplooot')
plt.legend()
plt.xlim(0, 35)
plt.ylim(0, 35)
plt.xlabel('X')
plt.ylabel('Y')
plt.text(5, 16, 'lol')
plt.annotate('Interestings', xy=(25, 5), xytext=(15, 15), arrowprops=dict(arrowstyle="->",
                             connectionstyle="angle3,angleA=0,angleB=-90"))
plt.show()
