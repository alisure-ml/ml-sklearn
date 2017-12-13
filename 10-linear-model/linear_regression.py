"""
线性回归
"""
from sklearn import datasets, linear_model, discriminant_analysis, model_selection
import matplotlib.pyplot as plt
import numpy as np


# 加载数据
def load_data():

    data = datasets.load_iris()
    return model_selection.train_test_split(data.data, data.target, test_size=0.25, random_state=0)


# 线性回归模型
def linear_regression(*data):
    _x_train, _x_test, _y_train, _y_test = data

    regression = linear_model.LinearRegression(normalize=False)
    regression.fit(_x_train, _y_train)
    print(regression.score(_x_train, _y_train))
    print(regression.score(_x_test, _y_test))

    # 归一化 the regressors X will be normalized before regression.
    regression = linear_model.LinearRegression(normalize=True)
    regression.fit(_x_train, _y_train)
    print(regression.score(_x_train, _y_train))
    print(regression.score(_x_test, _y_test))

    pass


# 逻辑回归模型
def logistic_regression(*data):
    _x_train, _x_test, _y_train, _y_test = data

    regression = linear_model.LogisticRegression()
    regression.fit(_x_train, _y_train)
    print(regression.score(_x_train, _y_train))
    print(regression.score(_x_test, _y_test))

    pass


# 线性判别分析
def linear_discriminant_analysis(*data):
    _x_train, _x_test, _y_train, _y_test = data

    regression = discriminant_analysis.LinearDiscriminantAnalysis()
    regression.fit(_x_train, _y_train)
    print(regression.score(_x_train, _y_train))
    print(regression.score(_x_test, _y_test))

    pass


if __name__ == "__main__":

    x_train, x_test, y_train, y_test = load_data()
    linear_regression(x_train, x_test, y_train, y_test)
    logistic_regression(x_train, x_test, y_train, y_test)
    linear_discriminant_analysis(x_train, x_test, y_train, y_test)

    pass
