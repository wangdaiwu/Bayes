import random
import pandas as pd
import numpy as np
import jieba
from snownlp import normal
import pickle

'''
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


def buildTrainReviewsSeg():
    trainReviews = pd.read_csv("../dataset/train_set.csv", usecols=["review"], low_memory=False)
    trainReviews = np.array(trainReviews).reshape(-1)
    trainReviews = list(trainReviews)

    trainReviews = seg(trainReviews)

    with open("../dataset/train_review.dat", mode="wb") as f:
        pickle.dump(trainReviews, f)


def buildTestReviewsSeg(path = "../dataset/test_set.csv"):
    testReviews = pd.read_csv(path, usecols=["review"], low_memory=False)
    testReviews = np.array(testReviews).reshape(-1)
    testReviews = list(testReviews)

    testReviews = seg(testReviews)

    with open("../dataset/test_review.dat", mode="wb") as f:
        pickle.dump(testReviews, f)


if __name__ == "__main__":
    # 分词前要导入词典
    jieba.load_userdict("initVocabulary.txt")
    buildLabels()
    buildTrainReviewsSeg()

    buildTestReviewsSeg()
    # buildTestReviewsSeg("../dataset/testshuffle_nolabel.csv")
