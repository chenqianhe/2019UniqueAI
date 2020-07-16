import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, datasets


def floyd(D, n_neighbors=15):
    Max = np.max(D) * 1000
    n1, n2 = D.shape
    k = n_neighbors
    D1 = np.ones((n1, n1)) * Max
    D_arg = np.argsort(D, axis=1)
    for i in range(n1):
        D1[i, D_arg[i, 0:k + 1]] = D[i, D_arg[i, 0:k + 1]]
    for k in range(n1):
        for i in range(n1):
            for j in range(n1):
                if D1[i, k] + D1[k, j] < D1[i, j]:
                    D1[i, j] = D1[i, k] + D1[k, j]

    return D1


def calculate_distance(x, y):
    d = np.sqrt(np.sum((x - y) ** 2))
    return d


# 计算矩阵各行之间的欧式距离；x矩阵的第i行与y矩阵的第0-j行继续欧式距离计算，构成新矩阵第i行[i0、i1...ij]
def calculate_distance_matrix(x, y):
    d = metrics.pairwise_distances(x, y)
    return d


def cal_B(D):
    (n1, n2) = D.shape
    DD = np.square(D)  # 矩阵D 所有元素平方
    Di = np.sum(DD, axis=1) / n1  # 计算dist(i.)^2
    Dj = np.sum(DD, axis=0) / n1  # 计算dist(.j)^2
    Dij = np.sum(DD) / (n1 ** 2)  # 计算dist(ij)^2
    B = np.zeros((n1, n1))
    for i in range(n1):
        for j in range(n2):
            B[i, j] = (Dij + DD[i, j] - Di[i] - Dj[j]) / (-2)  # 计算b(ij)
    return B


def MDS(data, n=2):
    D = calculate_distance_matrix(data, data)
    # print(D)
    B = cal_B(D)
    Be, Bv = np.linalg.eigh(B)  # Be矩阵B的特征值，Bv归一化的特征向量
    # print numpy.sum(B-numpy.dot(numpy.dot(Bv,numpy.diag(Be)),Bv.T))
    Be_sort = np.argsort(-Be)
    Be = Be[Be_sort]  # 特征值从大到小排序
    Bv = Bv[:, Be_sort]  # 归一化特征向量
    Bez = np.diag(Be[0:n])  # 前n个特征值对角矩阵
    # print Bez
    Bvz = Bv[:, 0:n]  # 前n个归一化特征向量
    Z = np.dot(np.sqrt(Bez), Bvz.T).T
    # print(Z)
    return Z


def Isomap(data, n=2, n_neighbors=30):
    D = calculate_distance_matrix(data, data)
    D_floyd = floyd(D)
    B = cal_B(D_floyd)
    Be, Bv = np.linalg.eigh(B)
    Be_sort = np.argsort(-Be)
    Be = Be[Be_sort]
    Bv = Bv[:, Be_sort]
    Bez = np.diag(Be[0:n])
    Bvz = Bv[:, 0:n]
    Z = np.dot(np.sqrt(Bez), Bvz.T).T
    return Z


def generate_curve_data():
    xx, target = datasets.samples_generator.make_s_curve(400, random_state=9)
    return xx, target


if __name__ == '__main__':
    data, target = generate_curve_data()
    Z_Isomap = Isomap(data, n=2)
    Z_MDS = MDS(data)
    figure = plt.figure()
    plt.suptitle('ISOMAP COMPARE TO MDS')
    plt.subplot(1, 2, 1)
    plt.title('ISOMAP')
    plt.scatter(Z_Isomap[:, 0], Z_Isomap[:, 1], c=target, s=60)
    plt.subplot(1, 2, 2)
    plt.title('MDS')
    plt.scatter(Z_MDS[:, 0], Z_MDS[:, 1], c=target, s=60)
    plt.show()
