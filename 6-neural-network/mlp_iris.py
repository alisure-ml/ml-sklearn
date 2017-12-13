# 多层神经网络的应用
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import datasets


# 预处理 iris 数据
def load_data_iris():
    iris = datasets.load_iris()
    x = iris.data[:, 0: 2]
    y = iris.target

    data = np.hstack((x, y.reshape(y.size, 1)))
    np.random.seed(0)
    np.random.shuffle(data)

    x = data[:, :-1]
    y = data[:, -1]
    train_x = x[:-30]
    test_x = x[-30:]
    train_y = y[:-30]
    test_y = y[-30:]

    return iris, train_x, test_x, train_y, test_y


# common
def _common(ax, classifier, text):
    iris, train_x, test_x, train_y, test_y = load_data_iris()

    classifier.fit(train_x, train_y)
    train_score = classifier.score(train_x, train_y)
    test_score = classifier.score(test_x, test_y)

    # 预测平面上的每一个点的输出
    x_min, x_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 2
    y_min, y_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 2
    plot_step = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
    z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    ax.contourf(xx, yy, z, cmap=plt.cm.Paired)

    # 绘图
    n_classes = 3
    plot_colors = "bry"
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(train_y == i)
        ax.scatter(train_x[idx, 0], train_x[idx, 1], c=color, label=iris.target_names[i], cmap=plt.cm.Paired)

    ax.set_xlabel(iris.feature_names[0])
    ax.set_ylabel(iris.feature_names[1])
    ax.set_title("%s, train score:%f, test score:%f" % (text, train_score, test_score))

    pass


# 默认参数的多层感知机
def mlp_iris():
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)
    classifier = MLPClassifier()

    _common(ax, classifier, "mlp iris")

    plt.show()


# 隐层数量对多层感知机的影响
def mlp_iris_hidden_layer_sizes():
    fig = plt.figure()
    hidden_layer_sizes = [(10,), (30,), (100,), (5, 5), (10, 10), (30, 30)]
    for itx, size in enumerate(hidden_layer_sizes):
        ax = fig.add_subplot(2, 3, itx + 1)

        classifier = MLPClassifier(activation="logistic", max_iter=10000, hidden_layer_sizes=size)
        _common(ax, classifier, "layer size:%s" % str(size))

    plt.show()

    pass


# 激活函数对多层感知机的影响
def mlp_iris_activations():
    fig = plt.figure()
    activations = ['logistic', 'tanh', 'relu']
    for itx, act in enumerate(activations):
        ax = fig.add_subplot(1, 3, itx + 1)

        classifier = MLPClassifier(activation=act, max_iter=10000, hidden_layer_sizes=(30,))
        _common(ax, classifier, "activation:%s" % act)

    plt.show()

    pass


# 优化算法对多层感知机的影响
def mlp_iris_solver():
    fig = plt.figure()
    solvers = ['lbfgs', 'sgd', 'adam']
    for itx, solver in enumerate(solvers):
        ax = fig.add_subplot(1, 3, itx + 1)

        classifier = MLPClassifier(solver=solver, max_iter=10000, hidden_layer_sizes=(30,))
        _common(ax, classifier, "solver:%s" % solver)

    plt.show()

    pass


# 学习率对多层感知机的影响
def mlp_iris_etas():
    fig = plt.figure()
    etas = [0.1, 0.01, 0.001, 0.0001]
    for itx, eta in enumerate(etas):
        ax = fig.add_subplot(2, 2, itx + 1)
        classifier = MLPClassifier(learning_rate_init=eta, max_iter=10000, hidden_layer_sizes=(30,))

        _common(ax, classifier, "eta:%s" % eta)

    plt.show()

    pass


if __name__ == "__main__":

    mlp_iris()
    mlp_iris_hidden_layer_sizes()
    mlp_iris_activations()
    mlp_iris_solver()
    mlp_iris_etas()

    pass
