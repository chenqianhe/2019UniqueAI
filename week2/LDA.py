import numpy as np
import csv
import matplotlib.pyplot as plt


def loadDataset(filename):
    data1 ,data2 = [], []
    with open(filename, 'r') as f:
        lines = csv.reader(f)
        data = list(lines)
    for i in range(len(data)):
        del(data[i][0])
        for j in range(len(data[i])):
            data[i][j] = float(data[i][j])
        if data[i][-1]:
            del(data[i][-1])
            data1.append(data[i])
        else:
            del(data[i][-1])
            data2.append(data[i])

    return np.array(data1), np.array(data2)



def lda_num2(data1,  data2,  n=2):
    mu0 = data2.mean(0)
    mu1 = data1.mean(0)
    print(mu0)
    print(mu1)

    sum0 = np.zeros((mu0.shape[0], mu0.shape[0]))
    for i in range(len(data2)):
        sum0 += np.dot((data2[i] - mu0).T, (data2[i] - mu0))
    sum1 = np.zeros(mu1.shape[0])
    for i in range(len(data1)):
        sum1 += np.dot((data1[i] - mu1).T, (data1[i] - mu1))

    s_w = sum0 + sum1
    print(s_w)
    w = np.linalg.pinv(s_w) * (mu0 - mu1)

    new_w = w[:n].T

    new_data1 = np.dot(data1, new_w)
    new_data2 = np.dot(data2, new_w)

    return new_data1, new_data2


def view(data):
    x = [i[0] for i in data]
    y = [i[1] for i in data]

    plt.figure()
    plt.scatter(x, y)
    plt.show()


if __name__ == '__main__':
    data1, data2 = loadDataset("Pima.csv")

    newdata1, newdata2 = lda_num2(data1, data2, 2)

    print(newdata1)
    print(newdata2)
    view(np.concatenate((newdata1, newdata2))*10**7)
    view(newdata1 * 10 ** 7)
    view(newdata2 * 10 ** 7)