from scipy import special, optimize
import numpy as np
import os
import matplotlib.pyplot  as plt

def ml_regresion_normal_eq():

    """
        This is linear regression with normal equation
        The algorithm works not properly
        Be careful by using this
    """
    data = np.matrix('[45, 85, 65, 55, 115, 46, 85, 75, 75, 70, 70, 90, 113; '
                     '1,8,2,2,9,1,6,4,5,4,4,6,7]')
    size = data.shape

    y = np.array([])
    x0 = np.array([])
    x1 = np.array([])

    for i in range(13):
        x0 = np.append(x0, 1)

    y = np.append(y, data[0])
    x1 = np.append(x1, data[1])

    teta0 = np.dot(((float(np.dot(np.transpose(x0), x0))) ** (-1)), (np.dot(np.transpose(x0), y)))
    teta1 = np.dot(((float(np.dot(np.transpose(x1), x1))) ** (-1)), (np.dot(np.transpose(x1), y)))
    print(teta0)
    print(teta1)

    teta = np.transpose([teta0, teta1])
    print(teta)

    z = [0, 2, 3, 4, 5, 6, 7, 8, 10]

    hypo = [(teta[1] * z[0]), (teta[1] * z[1]),
            (teta[1] * z[2]), (teta[1] * z[3]),
            (teta[1] * z[4]), (teta[1] * z[5]),
            (teta[1] * z[6]), (teta[1] * z[7]),
            (teta[1] * z[8])]

    print(hypo)

    fig1 = plt.figure()

    plt.title('Regression')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)

    for i in range(size[1]):
        plt.scatter(data[1, i], data[0, i])

    plt.plot(z, hypo)
    plt.show()


def ml_regression_gradient_descent():
    data = np.matrix('[45, 95, 65, 65, 119, 30, 80, 62, 75, 60, 61, 85, 93; '
                     '1,8,2,2,9,1,6,4,5,4,4,6,7]')
    size = data.shape

    y = np.array([])
    y = np.append(y, data[0])
    x = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 8, 2, 2, 9, 1, 6, 4, 5, 4, 4, 6, 7]])
    teta0, teta1 = 0, 0
    grown = 0.02

    def cost_funct(xi):
        sum = 0
        hypo = 0
        for i in range(13):
            hypo = teta0 * x[0][i] + teta1 * x[1][i]
            hypo = (hypo - y[i]) * x[xi][i]
            sum = sum + hypo

        sum = (1 / 13) * sum
        return sum

    for i in range(10000):
        temp0 = teta0 - grown * (cost_funct(0))
        temp1 = teta1 - grown * (cost_funct(1))
        teta0 = temp0
        teta1 = temp1

    teta = np.array([teta0, teta1])
    print(teta)

    plt.title('Regression')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)

    for i in range(size[1]):
        plt.scatter(data[1, i], data[0, i])

    z = np.array([0, 2, 3, 4, 5, 6, 7, 8, 10])
    hypo = np.array([])
    for i in range(9):
        hypo = np.append(hypo, [(teta[0] + teta[1] * z[i])])
    plt.plot(z, hypo)
    plt.show()


def ml_polynominal_regression():
    data = np.matrix('[10, 30, 14, 16, 32, 9, 24, 20, 22, 20, 19, 26, 27; '
                     '1,8,2,2,9,1,6,4,5,4,4,6,7]')
    size = data.shape

    y = np.array([])
    y = np.append(y, data[0])
    x = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 8, 2, 2, 9, 1, 6, 4, 5, 4, 4, 6, 7]])
    teta0, teta1 = 0, 0
    teta2 = 0
    grown = 0.05

    def cost_funct(xi):
        sum = 0
        hypo = 0
        for i in range(13):
            hypo = teta0 * x[0][i] + teta1 * x[1][i] + teta2 * (math.sqrt(x[1][i]))
            hypo = (hypo - y[i]) * x[xi][i]
            sum = sum + hypo

        sum = (1 / 13) * sum
        return sum

    for i in range(10000):
        temp0 = teta0 - grown * (cost_funct(0))
        temp1 = teta1 - grown * (cost_funct(1))
        temp2 = teta2 - grown * (cost_funct(1))
        teta0 = temp0
        teta1 = temp1
        teta2 = temp2

    teta = np.array([teta0, teta1, teta2])
    print(teta)

    plt.title('Regression')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)

    for i in range(size[1]):
        plt.scatter(data[1, i], data[0, i])

    z = np.array([0, 2, 3, 4, 5, 6, 7, 8, 10])
    hypo = np.array([])
    for i in range(9):
        hypo = np.append(hypo, [(teta[0] + teta[1] * z[i] + teta2 * (math.sqrt(z[i])))])
    plt.plot(z, hypo)
    plt.show()


def ml_multiclass_regression():
    data = np.matrix('[10, 30, 14, 16, 32, 9, 24, 20, 22, 20, 19, 26, 27; '
                     '1,8,2,2,9,1,6,4,5,4,4,6,7]')
    size = data.shape

    y = np.array([])
    y = np.append(y, data[0])
    x = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 8, 2, 2, 9, 1, 6, 4, 5, 4, 4, 6, 7],
                  [2, 4, 2, 2, 3, 2, 3, 2, 3, 2, 2, 4, 4],
                  [1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 1, 3, 3]])
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

        sum = (1 / 13) * sum
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
    z1 = np.array([2, 4, 2, 2, 3, 2, 3, 2, 3, 2, 2, 4, 4])
    z2 = np.array([1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 1, 3, 3])
    hypo = np.array([])
    for i in range(9):
        hypo = np.append(hypo, [(teta[0] + teta[1] * z[i] + teta2 * z1[i] + teta3 * z2[i])])
    plt.plot(z, hypo)
    plt.show()



def sklearn_multiple_LR():
    x = np.array(np.random.rand(2, 100))
    X = x.T
    y = 3 * x[0] + 2 * x[1] - 1.5 + np.random.rand(100)
    plt.scatter(x[0], y, s=40, cmap='viridis')

    model = LinearRegression()
    model.fit(X, y)
    xfit = np.array([np.linspace(-1, 2), np.linspace(-1, 2)])
    yfit = model.predict(xfit.T)
    plt.plot(xfit[0], yfit)
    plt.show()


def sklearn_grid_polimenal_LR():
    def PolynomialRegression(degree=2, **kwargs):
        return make_pipeline(PolynomialFeatures(degree),
                             LinearRegression(**kwargs))

    x = np.array(np.random.rand(1, 100))
    X = x.T
    y = 2.5 * (X.ravel()) ** 2 - 1.5 + np.random.rand(100)
    plt.scatter(x, y, s=40, cmap='viridis')

    param_grid = {'polynomialfeatures__degree': np.arange(21),
                  'linearregression__fit_intercept': [True, False],
                  'linearregression__normalize': [True, False]}

    grid = GridSearchCV(PolynomialRegression(), param_grid)
    grid.fit(X, y)
    print(grid.best_params_)
    model = grid.best_estimator_
    Xfit = np.linspace(-0.1, 1.1, 500)[:, None]
    model.fit(X, y)
    yfit = model.predict(Xfit)
    plt.plot(Xfit.ravel(), yfit, hold=True)
    plt.show()



def PERCEPTRON():
    class Perceptron(object):

        def __init__(self, eta=0.01, n_iter=10):
            self.eta = eta
            self.n_iter = n_iter

        def fit(self, X, y):
            self.w_ = np.zeros(1 + X.shape[1])
            self.errors_ = []
            print(self.w_)

            for _ in range(self.n_iter):
                errors = 0
                for xi, target in zip(X, y):
                    update = self.eta * (target - self.predict(xi))
                    self.w_[1:] += update * xi
                    self.w_[0] += update
                    errors += int(update != 0.0)
                self.errors_.append(errors)
            print(self.w_)
            return self

        def net_input(self, X):
            return np.dot(X, self.w_[1:]) + self.w_[0]

        def predict(self, X):
            return np.where(self.net_input(X) >= 0.0, 1, -1)

    def plot_decision_regions(X, y, classifier, resolution=0.02):
        # setup marker generator and color map
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])
        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        # plot class samples
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    plt.xlabel('sepal length')
    plt.ylabel('petal length')
    plt.legend(loc='upper left')
    plt.show()

    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.show()

    plot_decision_regions(X, y, classifier=ppn)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()



def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot all samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='',
                alpha=1.0, linewidths=1, marker='o',
                s=55, label='test set')


""" PROBLEMS WITH THE DOWNLOAD DATA AND STACK INTO THE DATAFRAME !!!!!!!!!!!!!!
        wine = datasets.load_wine()
        x = np.array(wine.data)
        y = np.array(wine.target)
        y = y[:, np.newaxis]
        x = np.hstack((x,y))
        df = pd.DataFrame(x)
        print(df)
"""




""" HOW to LOAD DIGITS FOR NEURAL NETWORK !!!!!!!!!!!!
    #Code from Python ML by Rashka on In chapter 12

    # http://yann.lecun.com/exdb/mnist/    ---------- Download page
    
    def load_mnist(path, kind='train'):
        labels_path = os.path.join(path,'%s-labels.idx1-ubyte' % kind)
        images_path = os.path.join(path,'%s-images.idx3-ubyte' % kind)
    
        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)
    
    
        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    
    
        return images, labels
    
    
    X_train, y_train = load_mnist('D:\DOWNLOADS\DOWNLOADS')
    
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
    ax = ax.flatten()
    for i in range(10):
        img = X_train[y_train == 2][i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()
    print(X_train[y_train==0][0].reshape(28,28))
    
    np.savetxt('train_img.csv', X_train,fmt='%i', delimiter=',')
    np.savetxt('train_labels.csv', y_train,fmt='%i', delimiter=',')
    np.savetxt('test_img.csv', X_test,fmt='%i', delimiter=',')
    np.savetxt('test_labels.csv', y_test, fmt='%i', delimiter=',')

    
    X_train = np.genfromtxt('train_img.csv',dtype=int, delimiter=',')
    y_train = np.genfromtxt('train_labels.csv',dtype=int, delimiter=',')
    X_test = np.genfromtxt('test_img.csv',dtype=int, delimiter=',')
    y_test = np.genfromtxt('test_labels.csv',dtype=int, delimiter=',')

"""



"""

   1) Open cmd
   2) Go to dir where is main.py (not dir where lies graph)
   3) Execute command:
   python -m tensorboard.main --logdir=[PATH_TO_LOGDIR]
   4) Open url which shown
   

   #I have a http://DESKTOP-28DHEA5:6006 url now
   #If tensorboard cant display data - try do it: ' python -m tensorboard.main --logdir summary_logs ' 

"""