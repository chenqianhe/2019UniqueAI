import csv
import numpy as np
import matplotlib.pyplot as plt
import copy
from time import sleep
import random


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



def select_j_rand(i ,m):
    # 选取alpha
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clip_alptha(aj, H, L):
    # 修剪alpha
    if aj > H:
        aj = H
    if L > aj:
        aj = L

    return aj


def smo(data_mat_In, class_label, C, toler, max_iter):
    # 转化为numpy的mat存储
    data_matrix = np.mat(data_mat_In)
    label_mat = np.mat(class_label).transpose()
    # data_matrix = data_mat_In
    # label_mat = class_label
    # 初始化b，统计data_matrix的纬度
    b = 0
    m, n = np.shape(data_matrix)
    # 初始化alpha，设为0
    alphas = np.mat(np.zeros((m, 1)))
    # 初始化迭代次数
    iter_num = 0
    # 最多迭代max_iter次
    while iter_num < max_iter:
        alpha_pairs_changed = 0
        for i in range(m):
            # 计算误差Ei
            fxi = float(np.multiply(alphas, label_mat).T*(data_matrix*data_matrix[i, :].T)) + b
            Ei = fxi - float(label_mat[i])
            # 优化alpha，松弛向量
            if (label_mat[i]*Ei < -toler and alphas[i] < C) or (label_mat[i]*Ei > toler and alphas[i] > 0):
                # 随机选取另一个与alpha_j成对优化的alpha_j
                j = select_j_rand(i, m)
                # 1.计算误差Ej
                fxj = float(np.multiply(alphas, label_mat).T*(data_matrix*data_matrix[j, :].T)) + b
                Ej = fxj - float(label_mat[j])
                # 保存更新前的alpha，deepcopy
                alpha_i_old = copy.deepcopy(alphas[i])
                alpha_j_old = copy.deepcopy(alphas[j])
                # 2.计算上下界L和H
                if label_mat[i] != label_mat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L == H")
                    continue
                # 3.计算eta
                eta = 2.0 * data_matrix[i, :]*data_matrix[j, :].T - data_matrix[i, :]*data_matrix[i, :].T - data_matrix[j, :]*data_matrix[j, :].T
                if eta >= 0:
                    print("eta >= 0")
                    continue
                # 4.更新alpha_j
                alphas[j] -= label_mat[j]*(Ei - Ej)/eta
                # 5.修剪alpha_j
                alphas[j] = clip_alptha(alphas[j], H, L)
                if abs(alphas[j] - alphas[i]) < 0.001:
                    print("alpha_j变化太小")
                    continue
                # 6.更新alpha_i
                alphas[i] += label_mat[j]*label_mat[i]*(alpha_j_old - alphas[j])
                # 7.更新b_1和b_2
                b_1 = b - Ei - label_mat[i]*(alphas[i] - alpha_i_old)*data_matrix[i, :]*data_matrix[i, :].T - label_mat[j]*(alphas[j] - alpha_j_old)*data_matrix[i, :]*data_matrix[j, :].T
                b_2 = b - Ej - label_mat[i]*(alphas[i] - alpha_i_old)*data_matrix[i, :]*data_matrix[j, :].T - label_mat[j]*(alphas[j] - alpha_j_old)*data_matrix[j, :] * data_matrix[j, :].T
                # 8.根据b_1和b_2更新b
                if 0 < alphas[i] and C > alphas[i]:
                    b = b_1
                elif 0 < alphas[j] and C > alphas[j]:
                    b = b_2
                else:
                    b = (b_1 + b_2)/2
                # 统计优化次数
                alpha_pairs_changed += 1
                # 打印统计信息
                print("第%d次迭代 样本：%d , alpha优化次数：%d" % (iter_num, i, alpha_pairs_changed))
        # 更新迭代次数
        if alpha_pairs_changed == 0:
            iter_num += 1
        else:
            iter_num = 0
        print("迭代次数：%d" % iter_num)

    return b, alphas


def caluelate_w(data_mat, label_mat, alphas):
    # 计算w
    alphas = np.array(alphas)
    data_mat = np.array(data_mat)
    label_mat = np.array(label_mat)

    # numpy.tile(A, reps):通过重复A给出的次数来构造数组。

    # numpy中reshape函数的三种常见相关用法
    # reshape(1, -1)转化成1行：
    # reshape(2, -1)转换成两行：
    # reshape(-1, 1)转换成1列：
    # reshape(-1, 2)转化成两列

    w = np.dot((np.tile(label_mat.reshape(1, -1).T, (1, 8))*data_mat).T, alphas)
    return w.tolist()



def prediction(test, w, b):
    test = np.mat(test)
    result = []

    for i in test:
        if i*w+b > 0:
            result.append(1)
        else:
            result.append(-1)

    print(result)

    return result


if __name__ == "__main__":
    test_set, test_labels = loadDataset('pima_test.csv')
    train_set, train_labels = loadDataset('pima_train.csv')


    b, alphas = smo(train_set, train_labels, 0.05, 0.001, 40)
    print(b)
    print(alphas)
    w = caluelate_w(train_set, train_labels, alphas)
    print(w)


    result = prediction(test_set, w, b)


    count = 0
    for i in range(len(result)):
        if result[i] == test_labels[i]:
            count += 1

    print(count)

    print(count/len(result))