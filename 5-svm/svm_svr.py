import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, model_selection, svm


# 载入糖尿病数据集
def load_data_regression():
    diabetes = datasets.load_diabetes()
    return model_selection.train_test_split(diabetes.data, diabetes.target, test_size=0.25, random_state=0)


# 打印糖尿病数据集
def print_diabetes():
    np.set_printoptions(threshold=np.nan, linewidth=322)
    x_train, x_test, y_train, y_test = load_data_regression()
    print(x_train)
    print(y_train)
    print(x_test)
    print(y_test)


# 线性回归SVR
def linear_svr(*data):
    x_train, x_test, y_train, y_test = data
    svr = svm.LinearSVR()
    svr.fit(x_train, y_train)
    print(svr.coef_)
    print(svr.intercept_)
    print(svr.score(x_train, y_train))
    print(svr.score(x_test, y_test))


# 损失函数类型对线性回归SVR的影响
def linear_svr_loss(*data):
    x_train, x_test, y_train, y_test = data
    losses = ['epsilon_insensitive', 'squared_epsilon_insensitive']
    for loss in losses:
        svr = svm.LinearSVR(loss=loss)
        svr.fit(x_train, y_train)
        print(loss)
        print(svr.coef_)
        print(svr.intercept_)
        print(svr.score(x_train, y_train))
        print(svr.score(x_test, y_test))


# 偏差边界值对线性回归SVR的影响
def linear_svr_epsilon(*data):
    x_train, x_test, y_train, y_test = data
    epsilons = np.logspace(-2, 2)
    train_scores = []
    test_scores = []
    for epsilon in epsilons:
        svr = svm.LinearSVR(epsilon=epsilon, loss="squared_epsilon_insensitive")
        svr.fit(x_train, y_train)
        train_scores.append(svr.score(x_train, y_train))
        test_scores.append(svr.score(x_test, y_test))

    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epsilons, train_scores, label="training score", marker="+")
    ax.plot(epsilons, test_scores, label="testing score", marker="o")

    ax.set_title("svr epsilon")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel("score")
    ax.set_ylim(-1, 1.05)
    ax.legend(loc="best", framealpha=0.5)

    plt.show()


# 惩罚项系数对线性回归SVR的影响
def linear_svr_c(*data):
    x_train, x_test, y_train, y_test = data
    cs = np.logspace(-1, 3)
    train_scores = []
    test_scores = []
    for c in cs:
        svr = svm.LinearSVR(epsilon=0.1, loss="squared_epsilon_insensitive", C=c)
        svr.fit(x_train, y_train)
        train_scores.append(svr.score(x_train, y_train))
        test_scores.append(svr.score(x_test, y_test))

    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(cs, train_scores, label="training score", marker="+")
    ax.plot(cs, test_scores, label="testing score", marker="o")

    ax.set_title("svr c")
    ax.set_xscale("log")
    ax.set_xlabel(r"C")
    ax.set_ylabel("score")
    ax.set_ylim(-1, 1.05)
    ax.legend(loc="best", framealpha=0.5)

    plt.show()


# 执行线性分类支持向量机
def run_linear_svr():
    x_train, x_test, y_train, y_test = load_data_regression()
    linear_svr(x_train, x_test, y_train, y_test)
    linear_svr_loss(x_train, x_test, y_train, y_test)
    linear_svr_epsilon(x_train, x_test, y_train, y_test)
    linear_svr_c(x_train, x_test, y_train, y_test)


# 非线性回归SVR
#  线性核函数
def no_linear_svr_linear(*data):
    x_train, x_test, y_train, y_test = data
    svr = svm.SVR(kernel="linear")
    svr.fit(x_train, y_train)
    print(svr.score(x_train, y_train))
    print(svr.score(x_test, y_test))


# 多项式核函数
def no_linear_svr_poly(*data):
    x_train, x_test, y_train, y_test = data

    fig = plt.figure()

    degrees = range(1, 20)
    train_scores = []
    test_scores = []
    for degree in degrees:
        cls = svm.SVR(kernel="poly", degree=degree, coef0=1)
        cls.fit(x_train, y_train)
        train_scores.append(cls.score(x_train, y_train))
        test_scores.append(cls.score(x_test, y_test))
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(degrees, train_scores, label="training score ", marker="+")
    ax.plot(degrees, test_scores, label="testing score ", marker="o")
    ax.set_xlabel("p")
    ax.set_ylabel("score")
    ax.set_title("svr ploy degree")
    ax.set_ylim(-1, 1.)
    ax.legend(loc="best", framealpha=0.5)

    gammas = range(1, 40)
    train_scores = []
    test_scores = []
    for gamma in gammas:
        cls = svm.SVR(kernel="poly", gamma=gamma, degree=3, coef0=1)
        cls.fit(x_train, y_train)
        train_scores.append(cls.score(x_train, y_train))
        test_scores.append(cls.score(x_test, y_test))
    ax = fig.add_subplot(1, 3, 2)
    ax.plot(gammas, train_scores, label="training score ", marker="+")
    ax.plot(gammas, test_scores, label="testing score ", marker="o")
    ax.set_xlabel("$\gamma$")
    ax.set_ylabel("score")
    ax.set_title("svr ploy gamma")
    ax.set_ylim(-1, 1.)
    ax.legend(loc="best", framealpha=0.5)

    rs = range(0, 20)
    train_scores = []
    test_scores = []
    for r in rs:
        cls = svm.SVR(kernel="poly", gamma=20, degree=3, coef0=r)
        cls.fit(x_train, y_train)
        train_scores.append(cls.score(x_train, y_train))
        test_scores.append(cls.score(x_test, y_test))
    ax = fig.add_subplot(1, 3, 3)
    ax.plot(rs, train_scores, label="training score ", marker="+")
    ax.plot(rs, test_scores, label="testing score ", marker="o")
    ax.set_xlabel(r"r")
    ax.set_ylabel("score")
    ax.set_title("svr ploy r")
    ax.set_ylim(-1, 1.)
    ax.legend(loc="best", framealpha=0.5)

    plt.show()


# 高斯径向基核函数
def no_linear_svr_rbf(*data):
    x_train, x_test, y_train, y_test = data

    fig = plt.figure()

    gammas = range(1, 20)
    train_scores = []
    test_scores = []
    for gamma in gammas:
        cls = svm.SVR(kernel="rbf", gamma=gamma)
        cls.fit(x_train, y_train)
        train_scores.append(cls.score(x_train, y_train))
        test_scores.append(cls.score(x_test, y_test))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(gammas, train_scores, label="training score ", marker="+")
    ax.plot(gammas, test_scores, label="testing score ", marker="o")
    ax.set_xlabel("$\gamma$")
    ax.set_ylabel("score")
    ax.set_title("svr rbf gamma")
    ax.set_ylim(-1, 1.)
    ax.legend(loc="best", framealpha=0.5)

    plt.show()


# sigmoid核函数
def no_linear_svr_sigmoid(*data):
    x_train, x_test, y_train, y_test = data

    fig = plt.figure()

    gammas = np.logspace(-1, 3)
    train_scores = []
    test_scores = []
    for gamma in gammas:
        cls = svm.SVC(kernel="sigmoid", gamma=gamma, coef0=0.01)
        cls.fit(x_train, y_train)
        train_scores.append(cls.score(x_train, y_train))
        test_scores.append(cls.score(x_test, y_test))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(gammas, train_scores, label="training score ", marker="+")
    ax.plot(gammas, test_scores, label="testing score ", marker="o")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("score")
    ax.set_ylim(-1, 1.)
    ax.set_title("svr sigmoid gamma")
    ax.legend(loc="best", framealpha=0.5)

    rs = np.linspace(0, 5)
    train_scores = []
    test_scores = []
    for r in rs:
        cls = svm.SVC(kernel="sigmoid", gamma=10, coef0=r)
        cls.fit(x_train, y_train)
        train_scores.append(cls.score(x_train, y_train))
        test_scores.append(cls.score(x_test, y_test))
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(rs, train_scores, label="training score ", marker="+")
    ax.plot(rs, test_scores, label="testing score ", marker="o")
    ax.set_xlabel(r"r")
    ax.set_ylabel("score")
    ax.set_ylim(-1, 1.)
    ax.set_title("svr sigmoid r")
    ax.legend(loc="best", framealpha=0.5)

    plt.show()


# 执行非线性分类支持向量机
def run_no_linear_svr():
    x_train, x_test, y_train, y_test = load_data_regression()
    no_linear_svr_linear(x_train, x_test, y_train, y_test)
    no_linear_svr_poly(x_train, x_test, y_train, y_test)
    no_linear_svr_rbf(x_train, x_test, y_train, y_test)
    no_linear_svr_sigmoid(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    run_linear_svr()
    run_no_linear_svr()
