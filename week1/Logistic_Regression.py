import math
import csv
import numpy as np


def loaddata(filename):
    label = []
    with open(filename, 'r') as f:
        lines = csv.reader(f)
        data = list(lines)
    for i in range(len(data)):
        del(data[i][0])
        for j in range(len(data[i])):
            data[i][j] = float(data[i][j])
        label.append(data[i][-1])
        del(data[i][-1])
    return np.array(data), np.array(label)


def sigmoid(x):
    return 1.0/(1 + np.exp(-x))


def J(theta, X, Y, theLambda=0):
    m, n = X.shape
    h = sigmoid(np.dot(X, theta))
    J = (-1.0/m)*(np.dot(np.log(h).T, Y)+np.dot(np.log(1-h).T, 1-Y)) + (theLambda/(2.0*m))*np.sum(np.square(theta[1:]))

    return J.flatten()[0]


def gradient_sgd(X, Y, alpha=0.01, epsilon=0.00001, maxloop=1000, theLambda=0.0):

    m, n = X.shape

    theta = np.zeros((n, 1))

    cost = J(theta, X, Y)
    costs = [cost]
    thetas = [theta]

    # 随机梯度下降
    count = 0
    flag = False
    while count < maxloop:
        if flag:
            break

        for i in range(m):
            h = sigmoid(np.dot(X[i].reshape((1, n)), theta))

            theta = theta - alpha * (
                        (1.0 / m) * X[i].reshape((n, 1)) * (h - Y[i]) + (theLambda / m) * np.r_[[[0]], theta[1:]])
            thetas.append(theta)
            cost = J(theta, X, Y, theLambda)
            costs.append(cost)
            if abs(costs[-1] - costs[-2]) < epsilon:
                flag = True
                break
        count += 1

        if count % 100 == 0:
            print("cost:", cost)

    return thetas, costs, count


def gradient_newton(X, Y, epsilon=0.00001, maxloop=1000, theLambda=0.0):

    m, n = X.shape

    theta = np.zeros((n, 1))

    cost = J(theta, X, Y)
    costs = [cost]
    thetas = [theta]

    count = 0

    while count < maxloop:

        delta_J = 0.0
        H = 0.0

        for i in range(m):
            h = sigmoid(np.dot(X[i].reshape((1, n)), theta))

            delta_J += X[i] * (h - Y[i])

            H += h.T * (1 - h) * X[i] * X[i].T


        delta_J /= m
        H /= m

        print(H, delta_J)

        theta = theta - 1.0 / H * delta_J

        thetas.append(theta)
        cost = J(theta, X, Y, theLambda)
        costs.append(cost)

        if abs(costs[-1] - costs[-2]) < epsilon:
            break
        count += 1

        if count % 100 == 0:
            print("cost:", cost)

    return thetas, costs, count


def gradient_adagrad(X, Y, alpha=0.01, sigma=1e-7, epsilon=0.00001, maxloop=1000, theLambda=0.0):

    m, n = X.shape

    theta = np.zeros((n, 1))

    r = [[0.0] for _ in range(n)]


    cost = J(theta, X, Y)
    costs = [cost]
    thetas = [theta]

    count = 0
    flag = False
    while count < maxloop:
        if flag:
            break

        for i in range(m):
            h = sigmoid(np.dot(X[i].reshape((1, n)), theta))

            grad = (1.0 / m) * X[i].reshape((n, 1)) * (h - Y[i])

            for j in range(n):
                r[j].append(grad[j]**2+r[j][-1])
                theta[j] = theta[j] - alpha * grad[j] / (sigma + math.sqrt(r[j][-1]))

            thetas.append(theta)
            cost = J(theta, X, Y, theLambda)
            costs.append(cost)
            if abs(costs[-1] - costs[-2]) < epsilon:
                flag = True
                break
        count += 1

        if count % 100 == 0:
            print("cost:", cost)
    return thetas, costs, count


def gradient_adadelta(X, Y, rho=0.01, alpha=0.01, sigma=1e-7, epsilon=0.00001, maxloop=1000, theLambda=0.0):

    m, n = X.shape

    theta = np.zeros((n, 1))

    r = [[0.0] for _ in range(n)]
    deltax = [[0.0] for _ in range(n)]
    deltax_ = [[1.0] for _ in range(n)]


    cost = J(theta, X, Y)
    costs = [cost]
    thetas = [theta]

    count = 0
    flag = False
    while count < maxloop:
        if flag:
            break

        for i in range(m):
            h = sigmoid(np.dot(X[i].reshape((1, n)), theta))

            grad = (1.0 / m) * X[i].reshape((n, 1)) * (h - Y[i])

            for j in range(n):
                r[j].append((1-rho) * grad[j]**2 + rho * r[j][-1])

                deltax[j].append(- (math.sqrt(deltax_[j][-1] / sigma + r[j][-1]))*alpha)

                theta[j] = theta[j] + deltax[j][-1]

                deltax_[j].append((1-rho)*deltax[j][-1]**2+rho*deltax_[j][-1])
                # print(deltax)
                # print(deltax_)

            thetas.append(theta)
            cost = J(theta, X, Y, theLambda)
            costs.append(cost)
            if abs(costs[-1] - costs[-2]) < epsilon:
                flag = True
                break
        count += 1

        if count % 100 == 0:
            print("cost:", cost)

    return thetas, costs, count


def gradient_RMSProp(X, Y, rho=0.01, alpha=0.01, sigma=1e-7, epsilon=0.00001, maxloop=1000, theLambda=0.0):

    m, n = X.shape

    theta = np.zeros((n, 1))

    r = [[0.0] for _ in range(n)]


    cost = J(theta, X, Y)
    costs = [cost]
    thetas = [theta]

    count = 0
    flag = False
    while count < maxloop:
        if flag:
            break

        for i in range(m):
            h = sigmoid(np.dot(X[i].reshape((1, n)), theta))

            grad = (1.0 / m) * X[i].reshape((n, 1)) * (h - Y[i])

            for j in range(n):
                r[j].append((1 - rho)*grad[j]**2+rho*r[j][-1])
                theta[j] = theta[j] - alpha * grad[j] / (sigma + math.sqrt(r[j][-1]))

            thetas.append(theta)
            cost = J(theta, X, Y, theLambda)
            costs.append(cost)
            if abs(costs[-1] - costs[-2]) < epsilon:
                print(costs)
                flag = True
                break
        count += 1

        if count % 100 == 0:
            print("cost:", cost)
    return thetas, costs, count



def gradient_adam(X, Y, rho1=0.9, rho2=0.999, alpha=0.01, sigma=1e-7, epsilon=0.00001, maxloop=1000, theLambda=0.0):

    m, n = X.shape

    theta = np.zeros((n, 1))

    s = [[0.0] for _ in range(n)]
    r = [[0.0] for _ in range(n)]

    cost = J(theta, X, Y)
    costs = [cost]
    thetas = [theta]

    count = 0
    flag = False
    while count < maxloop:
        if flag:
            break

        for i in range(m):
            h = sigmoid(np.dot(X[i].reshape((1, n)), theta))

            grad = (1.0 / m) * X[i].reshape((n, 1)) * (h - Y[i])

            for j in range(n):
                r[j].append((1 - rho2)*grad[j]**2+rho2*r[j][-1])
                s[j].append((1 - rho1)*grad[j]+rho1*r[j][-1])

                theta[j] = theta[j] - alpha * (s[j][-1]/(1-rho1**2))/(math.sqrt(r[j][-1]/(1-rho2**2))+sigma)

            thetas.append(theta)
            cost = J(theta, X, Y, theLambda)
            costs.append(cost)
            if abs(costs[-1] - costs[-2]) < epsilon:
                print(costs)
                flag = True
                break
        count += 1

        if count % 100 == 0:
            print("cost:", cost)
    return thetas, costs, count


if __name__ == '__main__':
    train_data, train_label = loaddata('pima_train.csv')
    test_data, test_label = loaddata('pima_test.csv')

    # print(train_data)
    # print(train_label)

    m = train_data.shape[0]
    # print(m)
    X = np.concatenate((np.ones((m, 1)), train_data), axis=1)
    # print(X)

    # thetas, costs, iterationCount = gradient_sgd(X, train_label, 0.05, 0.00000001, 10000)
    # thetas, costs, iterationCount = gradient_newton(X, train_label, 0.00000001, 10000)
    # thetas, costs, iterationCount = gradient_adagrad(X, train_label, 0.05, 1e-7, 0.000000000000001, 10000)
    # thetas, costs, iterationCount = gradient_adadelta(X, train_label, 0.9999999999, 1e-5, 1e-7, 0.000000000000001, 10000)
    # thetas, costs, iterationCount = gradient_RMSProp(X, train_label, 0.9, 1e-5, 1e-7, 0.000000000000001, 10000)
    thetas, costs, iterationCount = gradient_adam(X, train_label, 0.9, 0.99, 0.001, 1e-7, 0.000000000000001, 10000)
    print(costs[-1], iterationCount)

