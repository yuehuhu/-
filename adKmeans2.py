import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy import sparse
from sklearn import metrics
from sklearn.cluster import KMeans
import csv

data_path = "adFeature.csv"


def load_data():
    df = pd.read_csv("adFeature.csv")
    data = df.values
    return data


def main():
    dataSet = load_data()
    print(len(dataSet[1]))
    for k in range(2, 10):
        clf = KMeans(n_clusters=k)  # 设定k  ！！！！！！！！！！这里就是调用KMeans算法
        s = clf.fit(dataSet)  # 加载数据集合
        numSamples = len(dataSet)
        print(numSamples)
        centroids = clf.labels_
        # print(centroids, type(centroids))  # 显示中心点
        # print(clf.inertia_ ) # 显示聚类效果
        mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
        # 画出所有样例点 属于同一分类的绘制同样的颜色
        for i in range(numSamples):
            plt.plot(dataSet[i][0], dataSet[i][1], mark[clf.labels_[i]])  # mark[markIndex])
        mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
        # 画出质点，用特殊图型
        # centroids = clf.cluster_centers_
        # for i in range(k):
        #     plt.plot(centroids[i][0], centroids[i][1], mark[i], markersize=12)
        #     # print centroids[i, 0], centroids[i, 1]
        # # name = "./广告聚类/" + str(k) + "_2.png"
        # # plt.savefig(name)
        # plt.show()



if __name__ == '__main__':
    main()
