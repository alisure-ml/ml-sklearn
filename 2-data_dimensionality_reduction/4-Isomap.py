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


def run_isomap(*data):
    x, y = data
    for n in [4, 3, 2, 1]:
        print("-" * 50)
        isomap = manifold.Isomap(n_components=n)
        print("训练模型")
        isomap.fit(x)
        print("重构误差：", (n, str(isomap.reconstruction_error())))

X, Y = load_data()
run_isomap(X, Y)


def plot_isomap(*data):
    x, y = data
    print("-" * 50)
    fig = plt.figure()
    colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    ks = [1, 2, 4, 8, 16, 32, 64, 128, y.size - 1]
    for i, k in enumerate(ks):
        print("-" * 50)
        print(k)
        isomap = manifold.Isomap(n_components=2, n_neighbors=k)
        x_r = isomap.fit_transform(x)

        # 展示
        ax = fig.add_subplot(3, 3, i + 1)
        for label, color in zip(np.unique(y), colors):
            print(label, " ", color)
            position = y == label
            ax.scatter(x_r[position, 0], x_r[position, 1], label="target=%d" % label, color=color)
        ax.set_xlabel("x[0]")
        ax.set_ylabel("x[1]")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="best")
        ax.set_title("k=%d" % k)

    plt.suptitle("Isomap")
    plt.show()

plot_isomap(X, Y)


def plot_isomap_k_d1(*data):
    x, y = data
    print("-" * 50)
    fig = plt.figure()
    colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    for k in range(1, y.size + 1, 1):
        print("-" * 50)
        print(k)
        isomap = manifold.Isomap(n_components=1, n_neighbors=k)
        x_r = isomap.fit_transform(x)

        # 展示
        ax = fig.add_subplot(15, 10, (k - 1) / 1 + 1)
        for label, color in zip(np.unique(y), colors):
            print(label, " ", color)
            position = y == label
            ax.scatter(x_r[position], np.zeros_like(x_r[position]), label="target=%d" % label, color=color)
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.legend(loc="best")
        # ax.set_title("k=%d" % k)

    plt.suptitle("Isomap")
    plt.show()

plot_isomap_k_d1(X, Y)
