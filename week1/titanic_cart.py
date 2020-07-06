import csv
import operator
import copy
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph

# 0PassengerId：乘客的ID                               不重要
# 1Survived：乘客是否获救，Key：0=没获救，1=已获救
# 2Pclass：乘客船舱等级（1/2/3三个等级舱位）
# 3Name：乘客姓名                                       不重要
# 4Sex：性别
# 5Age：年龄
# 6SibSp：乘客在船上的兄弟姐妹/配偶数量
# 7Parch：乘客在船上的父母/孩子数量
# 8Ticket：船票号                                         不重要
# 9Fare：船票价
# 10Cabin：客舱号码                                        不重要
# 11Embarked：登船的港口                                   不重要



def loadDataset(filename):
    with open(filename, 'r') as f:
        lines = csv.reader(f)
        data_set = list(lines)
    if filename != 'titanic.csv':
        for i in range(len(data_set)):
            del(data_set[i][0])
    # 整理数据
    for i in range(len(data_set)):
        del(data_set[i][0])
        del(data_set[i][2])
        data_set[i][4] += data_set[i][5]
        del(data_set[i][5])
        del(data_set[i][5])
        del(data_set[i][6])
        del(data_set[i][-1])

    category = data_set[0]

    del (data_set[0])
    # 转换数据格式
    for data in data_set:
        data[0] = int(data[0])
        data[1] = int(data[1])
        if data[3] != '':
            data[3] = float(data[3])
        else:
            data[3] = None
        data[4] = float(data[4])
        data[5] = float(data[5])
    # 补全缺失值 转换记录方式 分类
    for data in data_set:
        if data[3] is None:
            data[3] = 28
        # male : 1, female : 0
        if data[2] == 'male':
            data[2] = 1
        else:
            data[2] = 0
        # age <25 为0, 25<=age<31为1，age>=31为2
        if data[3] < 60: # 但是测试得60分界准确率最高？？？！！！
            data[3] = 0
        else:
            data[3] = 1
        # sibsp&parcg以2为界限，小于为0，大于为1
        if data[4] < 2:
            data[4] = 0
        else:
            data[4] = 1
        # fare以64为界限
        if data[-1] < 64:
            data[-1] = 0
        else:
            data[-1] = 1
    return data_set, category


def gini(data, i):

    num = len(data)
    label_counts = [0, 0, 0, 0]

    p_count = [0, 0, 0, 0]

    gini_count = [0, 0, 0, 0]

    for d in data:
        label_counts[d[i]] += 1

    for l in range(len(label_counts)):
        for d in data:
            if label_counts[l] != 0 and d[0] == 1 and d[i] == l:
                p_count[l] += 1

    print(label_counts)
    print(p_count)

    for l in range(len(label_counts)):
        if label_counts[l] != 0:
            gini_count[l] = 2*(p_count[l]/label_counts[l])*(1 - p_count[l]/label_counts[l])

    gini_p = 0
    for l in range(len(gini_count)):
        gini_p += (label_counts[l]/num)*gini_count[l]

    print(gini_p)

    return gini_p


def get_best_feature(data, category):
    if len(category) == 2:
        return 1, category[1]

    feature_num = len(category) - 1
    data_num = len(data)

    feature_gini = []

    for i in range(1, feature_num+1):
        feature_gini.append(gini(data, i))

    min = 0

    for i in range(len(feature_gini)):
        if feature_gini[i] < feature_gini[min]:
            min = i

    print(feature_gini)
    print(category)
    print(min+1)
    print(category[min+1])

    return min+1, category[min + 1]


def majority_cnt(class_list):
    class_count = {}
    # 统计class_list中每个元素出现的次数
    for vote in class_list:
        if vote not in class_count:
            class_count[vote] = 0
        class_count[vote] += 1
        # 根据字典的值降序排列
        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


class Node(object):
    def __init__(self, item):
        self.name = item
        self.lchild = None
        self.rchild = None


def creat_tree(data, labels, feature_labels=[]):
# 三种结束情况
    # 取分类标签(survivor or death)
    class_list = [exampel[0] for exampel in data]

    if class_list == []:
        return Node(0)
    # 如果类别完全相同则停止分类
    if class_list.count(class_list[0]) == len(class_list):
        return Node(class_list[0])
    # 遍历完所有特征时返回出现次数最多的类标签
    if len(data[0]) == 1:
        return Node(majority_cnt(class_list))

    # 最优特征的标签
    best_feature_num, best_feature_label = get_best_feature(data, labels)

    feature_labels.append(best_feature_label)

    node = Node(best_feature_label)

    ldata = []
    rdata = []

    for d in data:
        if d[best_feature_num] == 1:
            del(d[best_feature_num])
            ldata.append(d)
        else:
            del(d[best_feature_num])
            rdata.append(d)

    labels2 = copy.deepcopy(labels)
    del(labels2[best_feature_num])

    tree = node
    tree.lchild = creat_tree(ldata, labels2, feature_labels)
    tree.rchild = creat_tree(rdata, labels2, feature_labels)

    return tree


def breadth_travel(tree):
    """广度遍历"""
    queue = [tree]
    while queue:
        cur_node = queue.pop(0)
        print(cur_node.name, end=" ")
        if cur_node.lchild is not None:
            queue.append(cur_node.lchild)
        if cur_node.rchild is not None:
            queue.append(cur_node.rchild)
    print()

def prediction(t_tree, test, labels):
    result = []
    for data in test:
        l = copy.deepcopy(labels)
        tree = t_tree
        for i in range(len(labels)):
            if tree.name == 1 or tree.name == 0:
                result.append(tree.name)
                break
            if len(data) == 1:
                result.append(0)
                break
            j = 1
            while j < len(data)-1:
                if tree.name == l[j]:
                    break
                j += 1

            if data[j] == 1:
                tree = tree.lchild
            else:
                tree = tree.rchild
            del(l[j])
            del(data[j])
    return result


def prune(tree):

    while tree.name != 1 and tree.name != 0:

        data_set, category = loadDataset('titanic.csv')
        bootstrapping = []
        for i in range(len(data_set)):
            bootstrapping.append(np.floor(np.random.random()*len(data_set)))
        test = []
        for i in range(len(data_set)):
            test.append(data_set[int(bootstrapping[i])])

        test2 = copy.deepcopy(test)
        label = copy.deepcopy(category)

        accurancy = 0

        result = prediction(tree, test2, label)
        counts = 0
        for i in range(len(test)):
            if test[i][0] == result[i]:
                counts += 1

        accurancy = counts/len(test)
        print(accurancy)


        breadth_travel(tree)
        l = copy.deepcopy(tree.name)
        l1 = copy.deepcopy(tree.lchild)
        l2 = copy.deepcopy(tree.rchild)
        tree.name = 1
        tree.lchild = None
        tree.rchild = None
        now_test = copy.deepcopy(test)
        label = copy.deepcopy(category)
        result = prediction(tree, now_test, label)
        counts = 0
        for i in range(len(test_set)):
            if test_set[i][0] == result[i]:
                counts += 1
        print(counts/len(test))
        if counts/len(test) > accurancy:
            print(counts/len(test))
            return tree
        else:
            tree.name = l
            tree.lchild = l1
            tree.rchild = l2

        tree.name = 0
        tree.lchild = None
        tree.rchild = None
        now_test = copy.deepcopy(test)
        label = copy.deepcopy(category)
        result = prediction(tree, now_test, label)
        counts = 0
        for i in range(len(test_set)):
            if test_set[i][0] == result[i]:
                counts += 1
        print(counts/len(test))
        if counts / len(test) > accurancy:
            return tree
        else:
            tree.name = l
            tree.lchild = l1
            tree.rchild = l2

        if (tree.lchild != None and tree.rchild != None) or (tree.lchild.name != 1 and tree.lchild.name != 0):
            tree.lchild = prune(tree.lchild)
            tree.rchild = prune(tree.rchild)
            return tree
        else:
            return tree

    return tree


if __name__ == "__main__":

    test_set, category = loadDataset('titanic_test.csv')
    train_set, category = loadDataset('titanic_train.csv')

    print(category)
    print(train_set)
    print()
    print(test_set)


    my_tree = creat_tree(train_set, category)
    print(my_tree)
    breadth_travel(my_tree)
    print(category)
    print(test_set)

    test = copy.deepcopy(test_set)

    result = prediction(my_tree, test_set, category)
    print(len(test_set))

    print(result)

    counts = 0

    for i in range(len(test_set)):
        if test_set[i][0] == result[i]:
            counts += 1

    print(counts)
    accurancy = counts/len(test_set)
    print(accurancy)

    # my_tree2 = copy.deepcopy(my_tree)
    #
    # new = prune(my_tree2)
    #
    # breadth_travel(new)





