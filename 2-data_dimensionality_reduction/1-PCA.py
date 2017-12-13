import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, decomposition, manifold


# 加载数据
# Iris也称鸢尾花卉数据集，是一类多重变量分析的数据集。
# 数据集包含150个数据集，分为3类，每类50个数据，每个数据包含4个属性。
# 可通过花萼长度，花萼宽度，花瓣长度，花瓣宽度4个属性
# 预测鸢尾花卉属于（Setosa，Versicolour，Virginica）三个种类中的哪一类。
def load_data():
    iris = datasets.load_iris()
    return iris.data, iris.target


# 打印Iris
def print_iris():
    data, target = load_data()
    print(data)
    print(target)


def run_pca(*data):
    x, y = data
    pca = decomposition.PCA(n_components=None)
    # pca = decomposition.IncrementalPCA(n_components=None, batch_size=20)
    print("训练模型")
    # 训练模型，得到投影矩阵W
    #   1.去中心化
    #   2.计算协方差矩阵
    #   3.对协方差特征分解
    #   4.取d个特征值对应的特征向量，构造投影矩阵W
    pca.fit(x)
    print("主成分数组")
    print(str(pca.components_))
    print("主成分的explained variance")
    print(str(pca.explained_variance_))
    print("主成分的explained variance的比例")
    print(str(pca.explained_variance_ratio_))
    print("统计平均值")
    print(str(pca.mean_))
    print("主成分元素数")
    print(str(pca.n_components_))

X, Y = load_data()
run_pca(X, Y)


def plot_pca(*data):
    x, y = data
    pca = decomposition.PCA(n_components=2)
    # pca = decomposition.IncrementalPCA(n_components=2, batch_size=20)
    print("训练模型")
    # 训练模型，得到投影矩阵W
    #   1.去中心化
    #   2.计算协方差矩阵
    #   3.对协方差特征分解
    #   4.取d个特征值对应的特征向量，构造投影矩阵W
    pca.fit(x)
    print("降维")
    # 计算 z = W * x
    x_r = pca.transform(x)
    # print(str(x_r))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    cur = {}

    colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    for label, color in zip(np.unique(y), colors):
        print(label, " ", color)
        position = y == label
        ax.scatter(x_r[position, 0], x_r[position, 1], label="target=%d" % label, color=color)

        cur[label] = x_r[position]
        print(cur[label].shape)

    ax.set_xlabel("x[0]")
    ax.set_ylabel("y[0]")
    ax.legend(loc="best")
    ax.set_title("PCA")

    plt.show()

plot_pca(X, Y)
