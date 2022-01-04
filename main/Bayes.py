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

    print(f"trainLabels len: {len(trainLabels)}")
    print(f"trainReviews len: {len(trainReviews)}")
    print(f"testLabels len: {len(testLabels)}")
    print(f"testReviews len: {len(testReviews)}")
    print(f"vocabulary len: {len(vocabulary)}")

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
        print("(1)[begin-function] calcPrioProb")
        for label in set(self.trainLabels):
            self.prioProb[label] = self.trainLabels.count(label) / len(self.trainLabels)

        print(f"   dump ../model/prioProb.dat")
        with open(f"../model/prioProb.dat", mode="wb") as f:
            pickle.dump(self.prioProb, f)

        print("(1)[end-function] calcPrioProb")

    # tf_idf
    def calcTF_IDF(self):
        print("(2)[begin-function] calcTF_IDF")
        TF = np.zeros([self.trainDocLen, self.vocabLen], dtype="float16")
        IDF = np.zeros([1, self.vocabLen])

        print("(2.1)[begin-calculation] TF and IDF")
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
        print("(2.1)[end-calculation] TF and IDF")

        print("(2.2)[begin-calculation] TF_IDF")
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
            with open(f"../model/tf_idf{batch + 1}.dat", mode="wb") as f:
                pickle.dump(TF_IDF, f)
            # self.TF_IDF = np.multiply(self.TF, self.IDF)

            # 设置下一批的 startingIndex
            startingIndex = endingIndex

        print("(2.2)[end-calculation] TF_IDF")
        print("(2)[end-function] calcTF_IDF")

    def calcCondProb(self):
        print("(3)[begin-function] calcCondProb")
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
            with open(f"../model/tf_idf{batch + 1}.dat", mode="rb") as f:
                TF_IDF = pickle.load(f)

            print(f"     Start calculating ...")
            for index in range(startingIndex, endingIndex):
                label = int(self.trainLabels[index])
                self.condProb[label] += TF_IDF[int(index - startingIndex)]
                sumList[label] += np.sum(self.condProb[label])
            print(f"     Complete the calculation of current batch")

            # 设置下一批的 startingIndex
            startingIndex = endingIndex

        with np.errstate(divide='ignore', invalid='ignore'):
            self.condProb = self.condProb / sumList
            self.condProb[~np.isfinite(self.condProb)] = 0

        print(f"   dump ../model/condProb.dat")
        with open(f"../model/condProb.dat", mode="wb") as f:
            pickle.dump(self.condProb, f)

        print("(3)[end-function] calcCondProb")

    # init
    def init(self, labels, reviews, vocabulary):
        self.trainLabels = labels
        self.trainReviews = reviews
        self.trainDocLen = len(reviews)
        self.trainBatch = math.ceil(self.trainDocLen / self.numOfDataInOneBatch)
        self.vocabulary = vocabulary
        self.vocabLen = len(vocabulary)

    # train
    def train(self):
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
        print("(4)[begin-function] predict")
        self.testReviews = testReviews
        self.testDocLen = len(testReviews)
        self.transform()

        print(f"   load ../model/prioProb.dat")
        with open(f"../model/prioProb.dat", mode="rb") as f:
            self.prioProb = pickle.load(f)


        print(f"   load ../model/condProb.dat")
        with open(f"../model/condProb.dat", mode="rb") as f:
            self.condProb = pickle.load(f)

        predictResultList = []
        for index in range(self.testDocLen):
            postProb = 0
            predictResult = 0
            for cond_prob_label, label in zip(self.condProb, self.prioProb):
                post_Prob_label = np.sum(self.testReviewsBOW[index] * cond_prob_label * self.prioProb[label])
                if post_Prob_label > postProb:
                    postProb = post_Prob_label
                    predictResult = int(label)
            predictResultList.append(predictResult)

        print(f"   dump ../model/result.csv")
        predictResultDF = pd.DataFrame(data=predictResultList)
        predictResultDF.to_csv("../model/result.csv", encoding="utf8", index=False, header=None)

        print("(4)[end-function] predict")
        return predictResultList


if __name__ == '__main__':
    bayes = Bayes()
    trainLabels, trainReviews, testLabels, testReviews, vocabulary = loadData()
    bayes.init(trainLabels, trainReviews, vocabulary)
    # bayes.train()
    print(bayes.predict(testReviews))
