import pandas as pd
import numpy as np
import jieba
from snownlp import normal

'''
由 'initVocabulary.txt' 生成 'bayesVocabulary.txt'
生成的 'bayesVocabulary.txt' 中仅含有 'initVocabulary.txt' 在 '大作业-文本情感分析-训练集.csv' 的 review 中出现过的词
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


if __name__ == "__main__":
    data = pd.read_csv("../dataset/大作业-文本情感分析-训练集.csv")
    data = np.array(data)
    data = data[:, 1]
    docLen = len(data)

    # 分词前要导入词典
    jieba.load_userdict("initVocabulary.txt")
    data = seg(data)

    print(f"len of data: {docLen}")

    initVocabulary = np.loadtxt(open("initVocabulary.txt", encoding="utf8"), dtype="str")
    initVocabulary = list(initVocabulary)
    vocabLen = len(initVocabulary)

    bayesVocabulary = set()
    for index in range(docLen):
        for word in data[index]:
            if word in initVocabulary:
                # print(word)
                bayesVocabulary.add(word)
        # print("")
        if index % 1000 == 0:
            print(f"{index} / {docLen}")

    print(f"len of bayesVocabulary: {len(bayesVocabulary)}")

    file = open("bayesVocabulary.txt", mode="w", encoding="utf8")
    for word in bayesVocabulary:
        file.write(f"{word}\n")
    file.flush()