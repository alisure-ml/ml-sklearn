# 多层感知器算法（多层神经网络）
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier


# 生成线性不可分数据集: (x, y, label)
def create_data_no_linear_2d(n):
    np.random.seed(1)

    # 正类
    x_11 = np.random.randint(0, 100, (n, 1))
    x_12 = 10 + np.random.randint(-5, 25, (n, 1))
    # 负类
    x_21 = np.random.randint(0, 100, (n, 1))
    x_22 = 40 + np.random.randint(0, 30, (n, 1))
    # 正类:和负类混在一起的正类
    x_31 = np.random.randint(0, 100, (int(n/10), 1))
    x_32 = 35 + np.random.randint(0, 10, (int(n/10), 1))

    new_x_11 = x_11 * np.sqrt(2) / 2 - x_12 * np.sqrt(2) / 2
    new_x_12 = x_11 * np.sqrt(2) / 2 + x_12 * np.sqrt(2) / 2
    new_x_21 = x_21 * np.sqrt(2) / 2 - x_22 * np.sqrt(2) / 2
    new_x_22 = x_21 * np.sqrt(2) / 2 + x_22 * np.sqrt(2) / 2
    new_x_31 = x_31 * np.sqrt(2) / 2 - x_32 * np.sqrt(2) / 2
    new_x_32 = x_31 * np.sqrt(2) / 2 + x_32 * np.sqrt(2) / 2

    # 组装样本点
    plus_samples = np.hstack([new_x_11, new_x_12, np.ones((n, 1))])
    minus_samples = np.hstack([new_x_21, new_x_22, -np.ones((n, 1))])
    error_samples = np.hstack([new_x_31, new_x_32, np.ones((int(n/10), 1))])
    samples = np.vstack([plus_samples, minus_samples, error_samples])
    # 打乱
    np.random.shuffle(samples)

    return samples


# 绘制不可分数据集
def plot_samples_2d(ax, samples):
    y = samples[:, -1]
    position_p = y == 1
    position_m = y == -1

    ax.scatter(samples[position_p, 0], samples[position_p, 1], marker="+", color="r")
    ax.scatter(samples[position_m, 0], samples[position_m, 1], marker="_", color="b")

    pass


# 显示
def show_plot():

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    data = create_data_no_linear_2d(n=100)
    plot_samples_2d(ax, data)
    ax.legend(loc="best")
    plt.show()

    pass


# 训练多层神经网络分类器
def mlp(n=1000):
    # (x, y, label)
    data = create_data_no_linear_2d(n)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    train_x = data[:, :-1]  # (x, y)
    train_y = data[:, -1]  # (label)
    clf = MLPClassifier(hidden_layer_sizes=(100, 100, 15), activation="logistic", max_iter=10000, tol=1e-10)
    clf.fit(train_x, train_y)
    print(clf.n_iter_)
    print(clf.loss_)
    print(clf.score(train_x, train_y))

    # 预测平面上的每一个点的输出
    x_min, x_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 2
    y_min, y_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 2
    plot_step = 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    ax.contourf(xx, yy, z, cmap=plt.cm.Paired)

    plot_samples_2d(ax, data)
    plt.show()

    pass


if __name__ == "__main__":

    # show_plot()
    mlp(5000)
