import random
import pandas as pd
import numpy as np
import jieba
from snownlp import normal
import pickle

'''
划分训练集 'train_set.csv' 和测试集 'test_set.csv'
划分前可设置以下两个参数
numberOfSamples : 总的样本数
ratioOfTestSet : 测试集的占比
'''

def partitionDataset(nos = 311603, ros = 0.0001):
    numberOfSamples = nos
    ratioOfTestSet = ros
    numberOfTrainSet = int(numberOfSamples * (1 - ratioOfTestSet))

    print(f"numberOfSamples: {numberOfSamples}")
    print(f"numberOfTrainSet: {numberOfTrainSet}")
    print(f"numberOfTestSet: {int(numberOfSamples * ratioOfTestSet)}")

    # '大作业-文本情感分析-训练集.csv' 包括标题共有 311604 行
    randomList = random.sample(range(1, 311604), numberOfSamples)
    fileSrc = open("../dataset/大作业-文本情感分析-训练集.csv", encoding="utf8")
    linesSrc = fileSrc.readlines()
    fileTrain = open("../dataset/train_set.csv", mode="w", encoding="utf8")
    fileTest = open("../dataset/test_set.csv", mode="w", encoding="utf8")

    fileTrain.write(linesSrc[0])
    fileTest.write(linesSrc[0])
    linesTrain = []
    linesTest = []
    for index in randomList[0: numberOfTrainSet]:
        linesTrain.append(linesSrc[index])
    for index in randomList[numberOfTrainSet: numberOfSamples]:
        linesTest.append(linesSrc[index])
    linesTrain.sort(key=lambda k: (k[0]))
    linesTest.sort(key=lambda k: (k[0]))

    fileTrain.writelines(linesTrain)
    fileTrain.flush()
    fileTest.writelines(linesTest)
    fileTest.flush()

    fileSrc.close()
    fileTrain.close()
    fileTest.close()


if __name__ == "__main__":
    # MAX numberOfSamples 311603
    partitionDataset(311603, 0.0001)