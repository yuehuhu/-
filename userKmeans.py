import pandas as pd
import numpy as np
import random
import math
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data_path = "userFeature_1.csv"

def simpleNum(user_feature):  # 对取值单一的特征进行处理
    Age = user_feature['age']
    Age.isna().sum()  # age值无缺省

    Gender = user_feature['gender']
    Gender.isna().sum()  # gender值无缺省

    MarriageStatus = user_feature['marriageStatus']
    MarriageStatus.isna().sum()  # marriageStatus值无缺省

    ConsumptionAbility = user_feature['consumptionAbility']
    ConsumptionAbility.isna().sum()  # consumptionAbility值无缺省

    LBS = user_feature['LBS']
    LBS = LBS.fillna(LBS.mode().iloc[0])  # 众数对缺失值进行补全
    user_feature['LBS'] = LBS

    Ct = user_feature['ct']
    Ct.isna().sum()  # ct值无缺省

    Os = user_feature['os']
    Os.isna().sum()  # os值无缺省

    House = user_feature['house']
    ConsumptionAbility = user_feature['consumptionAbility']
    for i in range(len(House)):
        if House[i] == 1:
            continue
        else:
            if ConsumptionAbility[i] == 1:
                House[i] = 1
            else:
                House[i] = 0
    user_feature['House'] = House  # house值用consumptionAbility值填补

    Carrier = user_feature['carrier']
    # carrier值无缺省

    c_features = ['marriageStatus', 'appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4',
                  'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
    for f in c_features:
        user_feature[f] = user_feature[f].fillna('-1')

    return user_feature


def hardNum(user_feature):
    onehot_feature = ['age', 'gender', 'education', 'consumptionAbility', 'LBS', 'house', 'os', 'ct', 'carrier']
    for f in onehot_feature:
        try:
            user_feature[f] = LabelEncoder().fit_transform(user_feature[f].apply(int))
        except:
            user_feature[f] = LabelEncoder().fit_transform(user_feature[f])

    X_onehot = pd.DataFrame()
    for f in onehot_feature:
        fd = user_feature[f]
        encode_matrix = OneHotEncoder(categories='auto').fit_transform(fd.values.reshape(-1, 1))
        X_onehot = sparse.hstack((X_onehot, encode_matrix))

    X_count = pd.DataFrame()
    count_feature = ['marriageStatus', 'appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3',
                     'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
    for f in count_feature:
        fd = user_feature[f]
        encode_matrix = CountVectorizer().fit_transform(fd)
        X_count = sparse.hstack((X_count, encode_matrix))
    X_train = sparse.hstack((X_onehot, X_count)).tocsr()[:1000]
    return X_train


def cal_dis(data, clu, k):
    """
    计算质点与数据点的距离
    :param data: 样本点
    :param clu:  质点集合
    :param k: 类别个数
    :return: 质心与样本点距离矩阵
    """
    dis = []
    for i in range(len(data)):
        dis.append([])
        for j in range(k):
            dis[i].append(math.sqrt((data[i, 0] - clu[j, 0]) ** 2 + (data[i, 1] - clu[j, 1]) ** 2))
    return np.asarray(dis)


def divide(data, dis):
    """
    对数据点分组
    :param data: 样本集合
    :param dis: 质心与所有样本的距离
    :param k: 类别个数
    :return: 分割后样本
    """
    clusterRes = [0] * len(data)
    for i in range(len(data)):
        seq = np.argsort(dis[i])
        clusterRes[i] = seq[0]

    return np.asarray(clusterRes)


def center(data, clusterRes, k):
    """
    计算质心
    :param group: 分组后样本
    :param k: 类别个数
    :return: 计算得到的质心
    """
    clunew = []
    for i in range(k):
        # 计算每个组的新质心
        idx = np.where(clusterRes == i)
        sum = data[idx].sum(axis=0)
        avg_sum = sum / len(data[idx])
        clunew.append(avg_sum)
    clunew = np.asarray(clunew)
    return clunew[:, 0: 7]


def classfy(data, clu, k):
    """
    迭代收敛更新质心
    :param data: 样本集合
    :param clu: 质心集合
    :param k: 类别个数
    :return: 误差， 新质心
    """
    clulist = cal_dis(data, clu, k)
    clusterRes = divide(data, clulist)
    clunew = center(data, clusterRes, k)
    err = clunew - clu
    return err, clunew, k, clusterRes


def plotRes(data, clusterRes, clusterNum):
    """
    结果可视化
    :param data:样本集
    :param clusterRes:聚类结果
    :param clusterNum: 类个数
    :return:
    """
    nPoints = len(data)
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    for i in range(clusterNum):
        color = scatterColors[i % len(scatterColors)]
        x1 = []
        y1 = []
        for j in range(nPoints):
            if clusterRes[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])
        plt.scatter(x1, y1, c=color, alpha=1, marker='+')
    name = "./用户聚类/" + str(clusterNum) + "_1.png"
    plt.savefig(name)
    plt.show()


def kMeans(data):
    dataSet = data.todense()
    pca_sk = PCA(n_components=7)
    data = pca_sk.fit_transform(dataSet)
    for k in range(2, 10):
        clu = random.sample(data[:, 0:7].tolist(), k)  # 随机取质心
        clu = np.asarray(clu)
        err, clunew, k, clusterRes = classfy(data, clu, k)
        while np.any(abs(err) > 0):
            print(clunew)
            err, clunew, k, clusterRes = classfy(data, clunew, k)

        clulist = cal_dis(data, clunew, k)
        clusterResult = divide(data, clulist)
        plotRes(data, clusterResult, k)


def fill(user_feature):
    con_house = pd.crosstab(user_feature['consumptionAbility'], user_feature['house'])
    con_house.head()
    print(con_house)

def main():
    user_feature = pd.read_csv(data_path)
    user_feature = simpleNum(user_feature)
    X_train = hardNum(user_feature)
    kMeans(X_train)


if __name__ == '__main__':
    main()
