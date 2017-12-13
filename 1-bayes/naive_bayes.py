from sklearn import datasets, model_selection, naive_bayes
import numpy as np
import matplotlib.pyplot as plt


def show_digits():
    digits = datasets.load_digits()
    fig = plt.figure()
    print("vector from images 0:", digits.data[0])
    for i in range(25):
        ax = fig.add_subplot(5, 5, i + 1)
        ax.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation="nearest")
    plt.show()


def load_data():
    digits = datasets.load_digits()
    return model_selection.train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)


x_train, x_test, y_train, y_test = load_data()

print("1")
print("高斯贝叶斯分类器")
# 高斯贝叶斯分类器
def my_GaussianNB(*data):
    train_x, test_x, train_y, test_y = data
    cls = naive_bayes.GaussianNB()
    # 训练模型
    cls.fit(train_x, train_y)
    # 返回预测的准确率
    result_train = cls.score(train_x, train_y)
    result_test = cls.score(test_x, test_y)

    print("Training Score:", result_train)
    print("Training Score:", result_test)

my_GaussianNB(x_train, x_test, y_train, y_test)


print("2")
print("多项式贝叶斯分类器")
# 多项式贝叶斯分类器
def my_MultinomialNB(*data):
    train_x, test_x, train_y, test_y = data
    cls = naive_bayes.MultinomialNB()
    # 训练模型
    cls.fit(train_x, train_y)
    # 返回预测的准确率
    result_train = cls.score(train_x, train_y)
    result_test = cls.score(test_x, test_y)

    print("Training Score:", result_train)
    print("Training Score:", result_test)

my_MultinomialNB(x_train, x_test, y_train, y_test)

def my_MultinomialNB_alpha(*data):
    train_x, test_x, train_y, test_y = data
    alphas = np.logspace(-2, 5, num=200)
    train_scores = []
    test_scores = []
    for alpha in alphas:
        cls = naive_bayes.MultinomialNB(alpha=alpha)
        cls.fit(train_x, train_y)
        train_scores.append(cls.score(X=train_x, y=train_y))
        test_scores.append(cls.score(X=test_x, y=test_y))

    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alphas, train_scores, label="Training Score")
    ax.plot(alphas, test_scores, label="Tesing Score")
    ax.set_xlabel(r"$\alpha$")
    ax.set_xscale("log")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)
    ax.set_title("MultinomialNB")
    ax.legend(loc="best")
    plt.show()

my_MultinomialNB_alpha(x_train, x_test, y_train, y_test)


print("3")
print("伯努利贝叶斯分类器")
# 伯努利贝叶斯分类器
def my_BernoulliNB(*data):
    train_x, test_x, train_y, test_y = data
    cls = naive_bayes.BernoulliNB()
    # 训练模型
    cls.fit(train_x, train_y)
    # 返回预测的准确率
    result_train = cls.score(train_x, train_y)
    result_test = cls.score(test_x, test_y)

    print("Training Score:", result_train)
    print("Training Score:", result_test)

my_BernoulliNB(x_train, x_test, y_train, y_test)

def my_BernoulliNB_alpha(*data):
    train_x, test_x, train_y, test_y = data
    alphas = np.logspace(-2, 5, num=200)
    train_scores = []
    test_scores = []
    for alpha in alphas:
        cls = naive_bayes.BernoulliNB(alpha=alpha)
        cls.fit(train_x, train_y)
        train_scores.append(cls.score(X=train_x, y=train_y))
        test_scores.append(cls.score(X=test_x, y=test_y))

    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alphas, train_scores, label="Training Score")
    ax.plot(alphas, test_scores, label="Tesing Score")
    ax.set_xlabel(r"$\alpha$")
    ax.set_xscale("log")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)
    ax.set_title("BernoulliNB")
    ax.legend(loc="best")
    plt.show()

my_BernoulliNB_alpha(x_train, x_test, y_train, y_test)

def my_BernoulliNB_binarize(*data):
    train_x, test_x, train_y, test_y = data
    min_x = min(np.min(train_x.ravel()), np.min(test_x.ravel())) - 0.1
    max_x = max(np.max(train_x.ravel()), np.max(test_x.ravel())) + 0.1
    binarizes = np.linspace(min_x, max_x, endpoint=True, num=100)
    train_scores = []
    test_scores = []
    for binarize in binarizes:
        cls = naive_bayes.BernoulliNB(binarize=binarize)
        cls.fit(train_x, train_y)
        train_scores.append(cls.score(X=train_x, y=train_y))
        test_scores.append(cls.score(X=test_x, y=test_y))

    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(binarizes, train_scores, label="Training Score")
    ax.plot(binarizes, test_scores, label="Tesing Score")
    ax.set_xlabel("binarize")
    ax.set_xlim(min_x - 1, max_x + 1)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)
    ax.set_title("BernoulliNB")
    ax.legend(loc="best")
    plt.show()

my_BernoulliNB_binarize(x_train, x_test, y_train, y_test)

