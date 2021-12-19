import math

import pandas as pd
import numpy as np
import jieba
from snownlp import normal
import pickle


def loadData():
    with open("../dataset/train_label.dat", mode="rb") as f:
        trainLabels = pickle.load(f)

    with open("../dataset/train_review.dat", mode="rb") as f:
        trainReviews = pickle.load(f)

    with open("../dataset/test_label.dat", mode="rb") as f:
        testLabels = pickle.load(f)

    with open("../dataset/test_review.dat", mode="rb") as f:
        testReviews = pickle.load(f)

    vocabulary = np.loadtxt(open("bayesVocabulary.txt", encoding="utf8"), dtype="str")
    vocabulary = list(vocabulary)

    print(len(trainLabels))
    print(len(trainReviews))
    print(len(testLabels))
    print(len(testReviews))
    print(len(vocabulary))

    return trainLabels, trainReviews, testLabels, testReviews, vocabulary


class Bayes(object):
    def __init__(self):
        self.trainLabels = []
        self.trainReviews = []
        self.trainDocLen = 0  # 训练集文本数，训练文本长度
        self.numOfDataInOneBatch = 50000
        self.trainBatch = 0

        self.vocabulary = []
        self.vocabLen = 0  # 词典词长,self.vocabulary长度

        # self.TF = 0
        # self.IDF = 0
        # self.TF_IDF = 0
        self.prioProb = {}  # P(yi)
        self.condProb = {}  # P(x|yi)

        self.testReviews = []
        self.testReviewsBOW = []
        self.testDocLen = 0

    # P(yi)
    def calcPrioProb(self):
        print("(1)[Begin-function] calcPrioProb")
        for label in set(self.trainLabels):
            self.prioProb[label] = self.trainLabels.count(label) / len(self.trainLabels)
        print("(1)[End-function] calcPrioProb")

    # tf_idf
    def calcTF_IDF(self):
        print("(2)[Begin-function] calcTF_IDF")
        TF = np.zeros([self.trainDocLen, self.vocabLen], dtype="float16")
        IDF = np.zeros([1, self.vocabLen])

        print("(2.1)[Begin-calculation] TF and IDF")
        for index in range(self.trainDocLen):
            trainReviewBow = 0
            for word in self.trainReviews[index]:
                if word in self.vocabulary:
                    trainReviewBow += 1
                    TF[index, self.vocabulary.index(word)] += 1
            if trainReviewBow > 0:
                TF[index] /= trainReviewBow

            for word in set(self.trainReviews[index]):
                if word in self.vocabulary:
                    IDF[0, self.vocabulary.index(word)] += 1

            if index % 1000 == 0:
                print(f"     progress: {index} / {self.trainDocLen}")

        IDF = np.log(self.trainDocLen / (IDF + 1))
        print("(2.1)[End-calculation] TF and IDF")

        print("(2.2)[Begin-calculation] TF_IDF")
        numOfRemainingData = self.trainDocLen
        startingIndex = 0
        endingIndex = 0

        for batch in range(self.trainBatch):
            # 设置当前批的 numOfCurrentBatchData 和 numOfRemainingData
            if numOfRemainingData >= self.numOfDataInOneBatch:
                numOfCurrentBatchData = self.numOfDataInOneBatch
            else:
                numOfCurrentBatchData = numOfRemainingData
            numOfRemainingData -= numOfCurrentBatchData

            # 设置当前批的 endingIndex
            endingIndex += numOfCurrentBatchData

            print(f"     batch: {batch + 1} / {self.trainBatch} ---> index range: [{startingIndex}, {endingIndex}]")
            print(f"       Start calculating ...")
            TF_IDF = np.multiply(TF[startingIndex:endingIndex], IDF)
            print(f"       Complete the calculation of current batch, dump ../tf_idf/tf_idf{batch + 1}.dat")
            with open(f"../tf_idf/tf_idf{batch + 1}.dat", mode="wb") as f:
                pickle.dump(TF_IDF, f)
            # self.TF_IDF = np.multiply(self.TF, self.IDF)

            # 设置下一批的 startingIndex
            startingIndex = endingIndex

        print("(2.2)[End-calculation] TF_IDF")
        print("(2)[End-function] calcTF_IDF")

    def calcCondProb(self):
        print("(3)[Begin-function] calcCondProb")
        self.condProb = np.zeros([len(self.prioProb), self.vocabLen])
        sumList = np.zeros([len(self.prioProb), 1])

        numOfRemainingData = self.trainDocLen
        startingIndex = 0
        endingIndex = 0

        for batch in range(self.trainBatch):
            # 设置当前批的 numOfCurrentBatchData 和 numOfRemainingData
            if numOfRemainingData >= self.numOfDataInOneBatch:
                numOfCurrentBatchData = self.numOfDataInOneBatch
            else:
                numOfCurrentBatchData = numOfRemainingData
            numOfRemainingData -= numOfCurrentBatchData

            # 设置当前批的 endingIndex
            endingIndex += numOfCurrentBatchData

            print(f"   batch: {batch + 1} / {self.trainBatch} ---> index range: [{startingIndex}, {endingIndex}]")
            print(f"     load ../tf_idf/tf_idf{batch + 1}.dat")
            with open(f"../tf_idf/tf_idf{batch + 1}.dat", mode="rb") as f:
                TF_IDF = pickle.load(f)

            print(f"     Start calculating ...")
            for index in range(startingIndex, endingIndex):
                self.condProb[int(self.trainLabels[index])] += TF_IDF[int(index - startingIndex)]
                sumList[int(self.trainLabels[index])] += np.sum(int(self.trainLabels[index]))
            print(f"     Complete the calculation of current batch")

            # 设置下一批的 startingIndex
            startingIndex = endingIndex

        with np.errstate(divide='ignore', invalid='ignore'):
            self.condProb = self.condProb / sumList
            self.condProb[~np.isfinite(self.condProb)] = 0
        print("(3)[End-function] calcCondProb")

    # train
    def train(self, labels, reviews, vocabulary):
        self.trainLabels = labels
        self.trainReviews = reviews
        self.trainDocLen = len(reviews)
        self.trainBatch = math.ceil(self.trainDocLen / self.numOfDataInOneBatch)
        self.vocabulary = vocabulary
        self.vocabLen = len(vocabulary)

        bayes.calcPrioProb()
        bayes.calcTF_IDF()
        bayes.calcCondProb()

    def transform(self):
        self.testReviewsBOW = np.zeros([self.testDocLen, self.vocabLen])
        for index in range(self.testDocLen):
            for word in testReviews[index]:
                if word in self.vocabulary:
                    self.testReviewsBOW[index, self.vocabulary.index(word)] += 1

    def predict(self, testReviews):
        self.testReviews = testReviews
        self.testDocLen = len(testReviews)
        self.transform()
        predictResultList = []
        for index in range(self.testDocLen):
            postProb = 0
            predictResult = None
            for cond_prob_label, label in zip(self.condProb, self.prioProb):
                post_Prob_label = np.sum(self.testReviewsBOW[index] * cond_prob_label * self.prioProb[label])
                if post_Prob_label > postProb:
                    postProb = post_Prob_label
                    predictResult = label
            predictResultList.append(predictResult)
        return predictResultList


if __name__ == '__main__':
    bayes = Bayes()
    trainLabels, trainReviews, testLabels, testReviews, vocabulary = loadData()
    bayes.train(trainLabels, trainReviews, vocabulary)
    # s = []
    # s.append("看着真挺让人难过的。我100岁了。图为谭孝珍老人背着沉重的废纸壳走在贵阳市文会巷。")
    # s = seg(s)
    # print(s[0])
    print(bayes.predict(testReviews))
