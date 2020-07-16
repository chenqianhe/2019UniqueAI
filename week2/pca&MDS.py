import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import metrics


def loadDataset(filename):
    label = []
    with open(filename, 'r') as f:
        lines = csv.reader(f)
        data = list(lines)
    for i in range(len(data)):
        del(data[i][0])
        for j in range(len(data[i])):
            data[i][j] = float(data[i][j])
        if data[i][-1]:
            label.append(data[i][-1])
        else:
            label.append(-1)
        del(data[i][-1])
    return data, label


def calculate_distance(x, y):
    d = np.sqrt(np.sum((x - y) ** 2))
    return d


# 计算矩阵各行之间的欧式距离；x矩阵的第i行与y矩阵的第0-j行继续欧式距离计算，构成新矩阵第i行[i0、i1...ij]
def calculate_distance_matrix(x, y):
    d = metrics.pairwise_distances(x, y)
    return d


def cal_B(D):
    (n1, n2) = D.shape
    DD = np.square(D)                    # 矩阵D 所有元素平方
    Di = np.sum(DD, axis=1) / n1         # 计算dist(i.)^2
    Dj = np.sum(DD, axis=0) / n1         # 计算dist(.j)^2
    Dij = np.sum(DD) / (n1 ** 2)         # 计算dist(ij)^2
    B = np.zeros((n1, n1))
    for i in range(n1):
        for j in range(n2):
            B[i, j] = (Dij + DD[i, j] - Di[i] - Dj[j]) / (-2)   # 计算b(ij)
    return B


def MDS(data, n=2):
    D = calculate_distance_matrix(data, data)
    # print(D)
    B = cal_B(D)
    Be, Bv = np.linalg.eigh(B)             # Be矩阵B的特征值，Bv归一化的特征向量
    # print numpy.sum(B-numpy.dot(numpy.dot(Bv,numpy.diag(Be)),Bv.T))
    Be_sort = np.argsort(-Be)
    Be = Be[Be_sort]                          # 特征值从大到小排序
    Bv = Bv[:, Be_sort]                       # 归一化特征向量
    Bez = np.diag(Be[0:n])                 # 前n个特征值对角矩阵
    # print Bez
    Bvz = Bv[:, 0:n]                          # 前n个归一化特征向量
    Z = np.dot(np.sqrt(Bez), Bvz.T).T
    # print(Z)
    return Z


def pca(data, n):
    data = np.array(data)

    # 均值
    mean_vector = np.mean(data, axis=0)

    # 协方差
    cov_mat = np.cov(data - mean_vector, rowvar=0)

    # 特征值 特征向量
    fvalue, fvector = np.linalg.eig(cov_mat)

    # 排序
    fvaluesort = np.argsort(-fvalue)

    # 取前几大的序号
    fValueTopN = fvaluesort[:n]

    # 保留前几大的数值
    newdata = fvector[:, fValueTopN]

    new = np.dot(data, newdata)

    return new


def view(data):
    x = [i[0] for i in data]
    y = [i[1] for i in data]

    plt.figure()
    plt.scatter(x, y)
    plt.show()




if __name__ == '__main__':
    data, labels = loadDataset("Pima.csv")

    newdata = pca(data, 2)

    newdata2 = MDS(data, 2)

    view(newdata)
    view(newdata2)
