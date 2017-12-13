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


def run_kpca_sigmoid(*data):
    x, y = data
    params = [(0.01, 0.1), (0.01, 0.2), (0.1, 0.1),
              (0.1, 0.2), (0.2, 0.1), (0.2, 0.2)]
    for i, (gamma, r) in enumerate(params):
        print("-" * 50)
        print((gamma, r))
        kpca = decomposition.KernelPCA(n_components=None, kernel="sigmoid", gamma=gamma, coef0=r)
        print("训练模型")
        kpca.fit(x)
        print("核化矩阵的特征值")
        print(str(kpca.lambdas_))
        print("核化矩阵的特征向量")
        print(str(kpca.alphas_))

X, Y = load_data()
run_kpca_sigmoid(X, Y)


def plot_kpca(*data):
    x, y = data
    fig = plt.figure()
    colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    params = [(0.01, 0.1), (0.01, 0.2), (0.1, 0.1),
              (0.1, 0.2), (0.2, 0.1), (0.2, 0.2)]
    for i, (gamma, r) in enumerate(params):
        print("-" * 50)
        print((gamma, r))
        kpca = decomposition.KernelPCA(n_components=2, kernel="sigmoid", gamma=gamma, coef0=r)
        kpca.fit(x)
        x_r = kpca.transform(x)

        # 展示
        ax = fig.add_subplot(3, 2, i + 1)
        for label, color in zip(np.unique(y), colors):
            print(label, " ", color)
            position = y == label
            ax.scatter(x_r[position, 0], x_r[position, 1], label="target=%d" % label, color=color)
        ax.set_xlabel("x[0]")
        ax.set_ylabel("x[1]")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="best")
        ax.set_title(r"$\tanh(%s(x\cdot z)+%s)$" % (gamma, r))

    plt.suptitle("KPCA-Sigmoid")
    plt.show()

plot_kpca(X, Y)
