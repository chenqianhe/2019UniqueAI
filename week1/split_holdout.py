import random
import csv
import pandas as pd

def loadDataset(filename, split, trainingSet = [], testSet = []):
     with open(filename, 'r') as f:
         lines = csv.reader(f)
         dataset = list(lines)
         for x in range(len(dataset)-1):
             if random.random() < split:  #将数据集随机划分
                 trainingSet.append(dataset[x])
             else:
                 testSet.append(dataset[x])


if __name__ == "__main__":
    train = []
    test = []
    loadDataset('Pima.csv', 0.7, train, test)
    print(train)
    print(test)

    train2 = pd.DataFrame(data=train)
    train2.to_csv('pima_train.csv')

    test2 = pd.DataFrame(data=test)
    test2.to_csv('pima_test.csv')