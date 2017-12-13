import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets, model_selection


# 载入手写体识别数据集
def load_classification_data():
    digits = datasets.load_digits()
    x_train = digits.data
    y_train = digits.target

    return model_selection.train_test_split(x_train, y_train, test_size=0.25, random_state=0, stratify=y_train)


# 在sin(x)的基础上添加噪声
def create_regression_data(n):
    x = 5 * np.random.rand(n, 1)
    # numpy.ravel() vs numpy.flatten() 将多维数组降为一维
    # numpy.flatten() 返回拷贝(copy),numpy.ravel()返回视图(view)
    y = np.sin(x).ravel()
    y[::5] += 1 * (0.5 - np.random.rand(int(n / 5)))

    return model_selection.train_test_split(x, y, test_size=0.25, random_state=0)


# 模型
def k_neighbors_classifier(*data):
    x_train, x_test, y_train, y_test = data
    clf = neighbors.KNeighborsClassifier()
    clf.fit(x_train, y_train)
    print("Training Score:", clf.score(x_train, y_train))
    print("Testing  Score:", clf.score(x_test, y_test))


def k_neighbors_classifier_k_w(*data):
    x_train, x_test, y_train, y_test = data
    ks = np.linspace(1, y_train.size, num=100, endpoint=False, dtype=int)
    weights = ["uniform", "distance"]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for weight in weights:
        training_scores = []
        testing_scores = []
        for k in ks:
            clf = neighbors.KNeighborsClassifier(weights=weight, n_neighbors=k)
            clf.fit(x_train, y_train)
            testing_scores.append(clf.score(x_test, y_test))
            training_scores.append(clf.score(x_train, y_train))
            print("weight=", weight, ", k=", k, " is OK")

        ax.plot(ks, testing_scores, label="testing score:weight=%s" % weight)
        ax.plot(ks, training_scores, label="training score:weight=%s" % weight)

    ax.legend(loc="best")
    ax.set_xlabel("k")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title("KNeighborsClassifier")
    plt.show()
    pass


def k_neighbors_classifier_k_p(*data):
    x_train, x_test, y_train, y_test = data
    ks = np.linspace(1, y_train.size, endpoint=False, dtype=int)
    ps = [1, 2, 10]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for p in ps:
        training_scores = []
        testing_scores = []
        for k in ks:
            clf = neighbors.KNeighborsClassifier(p=p, n_neighbors=k)
            clf.fit(x_train, y_train)
            testing_scores.append(clf.score(x_test, y_test))
            training_scores.append(clf.score(x_train, y_train))
            print("p=", p, ", k=", k, " is OK")

        ax.plot(ks, testing_scores, label="testing score:p=%s" % p)
        ax.plot(ks, training_scores, label="training score:p=%s" % p)

    ax.legend(loc="best")
    ax.set_xlabel("p")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title("KNeighborsClassifier")
    plt.show()
    pass


# 模型
def k_neighbors_regressor(*data):
    x_train, x_test, y_train, y_test = data
    clf = neighbors.KNeighborsRegressor()
    clf.fit(x_train, y_train)
    print("Training Score:", clf.score(x_train, y_train))
    print("Testing  Score:", clf.score(x_test, y_test))


def k_neighbors_regressor_k_w(*data):
    x_train, x_test, y_train, y_test = data
    ks = np.linspace(1, y_train.size, num=100, endpoint=False, dtype=int)
    weights = ["uniform", "distance"]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for weight in weights:
        testing_scores = []
        training_scores = []
        for k in ks:
            clf = neighbors.KNeighborsRegressor(weights=weight, n_neighbors=k)
            clf.fit(x_train, y_train)
            testing_scores.append(clf.score(x_test, y_test))
            training_scores.append(clf.score(x_train, y_train))
            print("weight=", weight, ", k=", k, " is OK")

        ax.plot(ks, testing_scores, label="testing score:weight=%s" % weight)
        ax.plot(ks, training_scores, label="training score:weight=%s" % weight)

    ax.legend(loc="best")
    ax.set_xlabel("weight")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title("KNeighborsRegressor")
    plt.show()
    pass


def k_neighbors_regressor_k_p(*data):
    x_train, x_test, y_train, y_test = data
    ks = np.linspace(1, y_train.size, endpoint=False, dtype=int)
    ps = [1, 2, 10]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for p in ps:
        training_scores = []
        testing_scores = []
        for k in ks:
            clf = neighbors.KNeighborsRegressor(p=p, n_neighbors=k)
            clf.fit(x_train, y_train)
            testing_scores.append(clf.score(x_test, y_test))
            training_scores.append(clf.score(x_train, y_train))
            print("p=", p, ", k=", k, " is OK")

        ax.plot(ks, testing_scores, label="testing score:p=%s" % p)
        ax.plot(ks, training_scores, label="training score:p=%s" % p)
        print(p)

    ax.legend(loc="best")
    ax.set_xlabel("p")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title("KNeighborsRegressor")
    plt.show()
    pass


if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test = load_classification_data()
    k_neighbors_classifier(X_train, X_test, Y_train, Y_test)
    k_neighbors_classifier_k_w(X_train, X_test, Y_train, Y_test)
    k_neighbors_classifier_k_p(X_train, X_test, Y_train, Y_test)

    X_train, X_test, Y_train, Y_test = create_regression_data(100)
    k_neighbors_regressor(X_train, X_test, Y_train, Y_test)
    k_neighbors_regressor_k_w(X_train, X_test, Y_train, Y_test)
    k_neighbors_regressor_k_p(X_train, X_test, Y_train, Y_test)


