import pandas as pd
import numpy as np
import jieba
from snownlp import normal


def seg(reviews):
    reviewsSeg = []
    for review in reviews:
        review = normal.zh2hans(review)
        review = jieba.lcut(review)
        review = [x for x in review if x != ' ']
        review = normal.filter_stop(review)
        reviewsSeg.append(review)
    return reviewsSeg


def f():
    fileSrc = open("../dataset/大作业-文本情感分析-训练集.csv", encoding="utf8")
    linesSrc = fileSrc.readlines()


if __name__ == "__main__":
    jieba.load_userdict("initVocabulary.txt")
