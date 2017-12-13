import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, model_selection, svm


# 载入iris数据集
def load_data_classfication():
    iris = datasets.load_iris()
    x_train = iris.data
    y_train = iris.target
    # stratify 分层采样
    return model_selection.train_test_split(x_train, y_train, test_size=0.25, random_state=0, stratify=y_train)


# 打印iris数据集
def print_iris():
    np.set_printoptions(threshold=np.nan, linewidth=322)
    x_train, x_test, y_train, y_test = load_data_classfication()
    print(x_train)
    print(y_train)
    print(x_test)
    print(y_test)


# 线性分类支持向量机
def linear_svc(*data):
    x_train, x_test, y_train, y_test = data
    cls = svm.LinearSVC()
    cls.fit(x_train, y_train)
    print(cls.coef_)
    print(cls.intercept_)
    print(cls.score(x_test, y_test))


def linear_svc_loss(*data):
    x_train, x_test, y_train, y_test = data
    losses = ['hinge', 'squared_hinge']
    for loss in losses:
        cls = svm.LinearSVC(loss=loss)
        cls.fit(x_train, y_train)
        print("loss=", loss)
        print(cls.coef_)
        print(cls.intercept_)
        print(cls.score(x_test, y_test))


def linear_svc_penalty(*data):
    x_train, x_test, y_train, y_test = data
    penalties = ['l1', 'l2']
    for penalty in penalties:
        cls = svm.LinearSVC(penalty=penalty, dual=False)
        cls.fit(x_train, y_train)
        print("penalty=", penalty)
        print(cls.coef_)
        print(cls.intercept_)
        print(cls.score(x_test, y_test))


def linear_svc_c(*data):
    x_train, x_test, y_train, y_test = data
    cs = np.logspace(-2, 1)
    train_scores = []
    test_scores = []
    for c in cs:
        cls = svm.LinearSVC(C=c)
        cls.fit(x_train, y_train)
        train_scores.append(cls.score(x_train, y_train))
        test_scores.append(cls.score(x_test, y_test))

    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(cs, train_scores, label="training score")
    ax.plot(cs, test_scores, label="testing score")
    ax.set_xlabel("c")
    ax.set_ylabel("score")
    ax.set_title("LinearSVC")
    ax.legend(loc="best")

    plt.show()


# 执行线性分类支持向量机
def run_linear_svc():
    x_train, x_test, y_train, y_test = load_data_classfication()
    linear_svc(x_train, x_test, y_train, y_test)
    linear_svc_loss(x_train, x_test, y_train, y_test)
    linear_svc_penalty(x_train, x_test, y_train, y_test)
    linear_svc_c(x_train, x_test, y_train, y_test)


# 非线性分类支持向量机
# 线性核函数
def no_linear_svc_linear(*data):
    x_train, x_test, y_train, y_test = data
    cls = svm.SVC(kernel="linear")
    cls.fit(x_train, y_train)
    print(cls.coef_)
    print(cls.intercept_)
    print(cls.score(x_test, y_test))


# 多项式核函数
def no_linear_svc_poly(*data):
    x_train, x_test, y_train, y_test = data

    fig = plt.figure()

    degrees = range(1, 20)
    train_scores = []
    test_scores = []
    for degree in degrees:
        cls = svm.SVC(kernel="poly", degree=degree)
        cls.fit(x_train, y_train)
        train_scores.append(cls.score(x_train, y_train))
        test_scores.append(cls.score(x_test, y_test))
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(degrees, train_scores, label="training score ", marker="+")
    ax.plot(degrees, test_scores, label="testing score ", marker="o")
    ax.set_xlabel("p")
    ax.set_ylabel("score")
    ax.set_title("svc ploy degree")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", framealpha=0.5)

    gammas = range(1, 20)
    train_scores = []
    test_scores = []
    for gamma in gammas:
        cls = svm.SVC(kernel="poly", gamma=gamma, degree=3)
        cls.fit(x_train, y_train)
        train_scores.append(cls.score(x_train, y_train))
        test_scores.append(cls.score(x_test, y_test))
    ax = fig.add_subplot(1, 3, 2)
    ax.plot(gammas, train_scores, label="training score ", marker="+")
    ax.plot(gammas, test_scores, label="testing score ", marker="o")
    ax.set_xlabel("$\gamma$")
    ax.set_ylabel("score")
    ax.set_title("svc ploy gamma")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", framealpha=0.5)

    rs = range(1, 20)
    train_scores = []
    test_scores = []
    for r in rs:
        cls = svm.SVC(kernel="poly", gamma=10, degree=3, coef0=r)
        cls.fit(x_train, y_train)
        train_scores.append(cls.score(x_train, y_train))
        test_scores.append(cls.score(x_test, y_test))
    ax = fig.add_subplot(1, 3, 3)
    ax.plot(rs, train_scores, label="training score ", marker="+")
    ax.plot(rs, test_scores, label="testing score ", marker="o")
    ax.set_xlabel("r")
    ax.set_ylabel("score")
    ax.set_title("svc ploy r")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", framealpha=0.5)

    plt.show()


# 高斯径向基核函数
def no_linear_svc_rbf(*data):
    x_train, x_test, y_train, y_test = data

    fig = plt.figure()

    gammas = range(1, 20)
    train_scores = []
    test_scores = []
    for gamma in gammas:
        cls = svm.SVC(kernel="rbf", gamma=gamma)
        cls.fit(x_train, y_train)
        train_scores.append(cls.score(x_train, y_train))
        test_scores.append(cls.score(x_test, y_test))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(gammas, train_scores, label="training score ", marker="+")
    ax.plot(gammas, test_scores, label="testing score ", marker="o")
    ax.set_xlabel("$\gamma$")
    ax.set_ylabel("score")
    ax.set_title("svc rbf gamma")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", framealpha=0.5)

    plt.show()


# sigmoid核函数
def no_linear_svc_sigmoid(*data):
    x_train, x_test, y_train, y_test = data

    fig = plt.figure()

    gammas = np.logspace(-2, 1)
    train_scores = []
    test_scores = []
    for gamma in gammas:
        cls = svm.SVC(kernel="sigmoid", gamma=gamma, coef0=0)
        cls.fit(x_train, y_train)
        train_scores.append(cls.score(x_train, y_train))
        test_scores.append(cls.score(x_test, y_test))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(gammas, train_scores, label="training score ", marker="+")
    ax.plot(gammas, test_scores, label="testing score ", marker="o")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title("svc sigmoid gamma")
    ax.legend(loc="best", framealpha=0.5)

    rs = np.linspace(0, 5)
    train_scores = []
    test_scores = []
    for r in rs:
        cls = svm.SVC(kernel="sigmoid", gamma=0.01, coef0=r)
        cls.fit(x_train, y_train)
        train_scores.append(cls.score(x_train, y_train))
        test_scores.append(cls.score(x_test, y_test))
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(rs, train_scores, label="training score ", marker="+")
    ax.plot(rs, test_scores, label="testing score ", marker="o")
    ax.set_xlabel("r")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title("svc sigmoid r")
    ax.legend(loc="best", framealpha=0.5)

    plt.show()


# 执行非线性分类支持向量机
def run_no_linear_svc():
    x_train, x_test, y_train, y_test = load_data_classfication()
    no_linear_svc_linear(x_train, x_test, y_train, y_test)
    no_linear_svc_poly(x_train, x_test, y_train, y_test)
    no_linear_svc_rbf(x_train, x_test, y_train, y_test)
    no_linear_svc_sigmoid(x_train, x_test, y_train, y_test)


if __name__ == "__main__":

    run_linear_svc()
    run_no_linear_svc()
