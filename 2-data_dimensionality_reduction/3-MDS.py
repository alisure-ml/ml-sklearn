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


def run_mds(*data):
    x, y = data
    for n in [4, 3, 2, 1]:
        print("-" * 50)
        mds = manifold.MDS(n_components=n)
        print("训练模型")
        mds.fit(x)
        print("不一致的距离总和：", (n, str(mds.stress_)))

X, Y = load_data()
run_mds(X, Y)


def plot_mds(*data):
    x, y = data
    print("-" * 50)
    mds = manifold.MDS(n_components=2)
    x_r = mds.fit_transform(x)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    for label, color in zip(np.unique(y), colors):
        print(label, " ", color)
        position = y == label
        ax.scatter(x_r[position, 0], x_r[position, 1], label="target=%d" % label, color=color)

    ax.set_xlabel("x[0]")
    ax.set_ylabel("y[0]")
    ax.legend(loc="best")
    ax.set_title("MDS")

    plt.show()

plot_mds(X, Y)
