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