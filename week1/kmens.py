import operator
import csv
import numpy as np
import random

names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
train_lables = []
test_lables = []

def loaddata(filename1, filename2, train_set=[], test_set=[]):

    with open(filename1, 'r') as f:
        lines = csv.reader(f)
        train_set = list(lines)
        for i in range(len(train_set)):
            del(train_set[i][0])
            if train_set[i][-1] == names[0]:
                train_lables.append(0)
            elif train_set[i][-1] == names[1]:
                train_lables.append(1)
            else:
                train_lables.append(2)
            del(train_set[i][-1])
            for j in range(4):
                train_set[i][j] = float(train_set[i][j])
            train_set[i] = np.array(train_set[i])

    with open(filename2, 'r') as f:
        lines = csv.reader(f)
        test_set = list(lines)
        for i in range(len(test_set)):
            del(test_set[i][0])
            if test_set[i][-1] == names[0]:
                test_lables.append(0)
            elif test_set[i][-1] == names[1]:
                test_lables.append(1)
            else:
                test_lables.append(2)
            del(test_set[i][-1])
            for j in range(4):
                test_set[i][j] = float(test_set[i][j])
            test_set[i] = np.array(test_set[i])


    return train_set, test_set


def get_core(core, k):
    newcore = []
    newcoredata = []
    for i in range(k):
        newcoredata.append([])

    for i in range(len(train_set)):
        distance = []
        for j in range(k):
            distance.append(np.linalg.norm(train_set[i] - core[j], ord=None, axis=None, keepdims=False))
        newcoredata[distance.index(min(distance))].append(i)

    for i in range(k):
        temp = np.zeros(4)
        for j in newcoredata[i]:
            temp += train_set[j]
        newcore.append(temp / len(newcoredata[i]))

    return newcore, newcoredata


def kmeans(k):

    core = []

    while len(train_set) < 3:
        x = random.randint(0, len(train_set))
        if x not in core:
            core.append(x)

    now_core = [train_set[i] for i in core]
    print(now_core)

    for i in range(10):
        new_core, newcoredata = get_core(now_core, k)

        print(new_core, newcoredata)
        dis = 0
        for i in range(k):
            dis += np.linalg.norm(new_core[i] - now_core[i], ord=None, axis=None, keepdims=False)
        if dis >= 0.1:
            now_core = new_core
        else:
            break

    return now_core, newcoredata


def get_response(neighbors):
    class_vote = {}
    for i in range(len(neighbors)):
        response = neighbors[i]
        if response not in class_vote:
            class_vote[response] = 1
        else:
            class_vote[response] += 1
    softed_vote = sorted(class_vote.items(), key=operator.itemgetter(1), reverse=True)
    return softed_vote[0][0]


def predection(test_data, core_label):

    result = []
    for i in range(len(test_data)):
        distance = []
        for j in range(k):
            distance.append(np.linalg.norm(test_data[i] - core[j], ord=None, axis=None, keepdims=False))
        result.append(core_label[distance.index(min(distance))])

    return result


if __name__ == "__main__":
    k = 3
    train_set, test_set = loaddata('iris_train.csv', 'iris_test.csv')

    core, data = kmeans(k)

    core_label = []

    for i in range(k):
        temp = [train_lables[j] for j in data[i]]
        core_label.append(get_response(temp))

    print(core_label)

    result = predection(test_set, core_label)

    print(test_lables)
    print(result)

    acc_num = 0
    for i in range(len(test_lables)):
        if result[i] == test_lables[i]:
            acc_num += 1

    print(acc_num/len(test_lables))
