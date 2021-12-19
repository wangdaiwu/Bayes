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

获得训练集 'train_set.csv' 和测试集 'test_set.csv' 的 label 和 review
将 label 分别放入 'train_label.dat' 和 'test_review.dat'
将 review 分词后分别放入 'train_review.dat' 和 'test_review.dat'
'''


def seg(reviews):
    reviewsSeg = []
    for review in reviews:
        review = normal.zh2hans(review)
        review = jieba.lcut(review)
        review = [x for x in review if x != ' ']
        review = normal.filter_stop(review)
        reviewsSeg.append(review)
    return reviewsSeg


def buildLabels():
    trainLabels = pd.read_csv("../dataset/train_set.csv", usecols=["label"], low_memory=False)
    trainLabels = np.array(trainLabels).reshape(-1)
    trainLabels = list(trainLabels)

    testLabels = pd.read_csv("../dataset/test_set.csv", usecols=["label"], low_memory=False)
    testLabels = np.array(testLabels).reshape(-1)
    testLabels = list(testLabels)

    with open("../dataset/train_label.dat", mode="wb") as f:
        pickle.dump(trainLabels, f)

    with open("../dataset/test_label.dat", mode="wb") as f:
        pickle.dump(testLabels, f)


def buildReviewsSeg():
    trainReviews = pd.read_csv("../dataset/train_set.csv", usecols=["review"], low_memory=False)
    trainReviews = np.array(trainReviews).reshape(-1)
    trainReviews = list(trainReviews)

    testReviews = pd.read_csv("../dataset/test_set.csv", usecols=["review"], low_memory=False)
    testReviews = np.array(testReviews).reshape(-1)
    testReviews = list(testReviews)

    trainReviews = seg(trainReviews)
    testReviews = seg(testReviews)

    with open("../dataset/train_review.dat", mode="wb") as f:
        pickle.dump(trainReviews, f)

    with open("../dataset/test_review.dat", mode="wb") as f:
        pickle.dump(testReviews, f)


if __name__ == "__main__":
    # MAX numberOfSamples 311603
    numberOfSamples = 311603
    ratioOfTestSet = 0.0001
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

    # 分词前要导入词典
    jieba.load_userdict("initVocabulary.txt")
    buildLabels()
    buildReviewsSeg()
