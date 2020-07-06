import numpy as np
import math
import csv
import numpy as np
import random


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










if __name__ == '__main__':


    train_data, train_label = loaddata('pima_train.csv')
    test_data, test_label = loaddata('pima_test.csv')

    print(train_data)
    print(train_label)