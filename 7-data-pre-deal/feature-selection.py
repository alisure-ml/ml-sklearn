from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectPercentile, f_classif, f_regression
from sklearn.feature_selection import RFE, RFECV, SelectFromModel
from sklearn import model_selection
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris


# 过滤式特征选取：根据方差阈值
def variance_threshold():
    x = [[1, -2, 3, 4, 5.],
         [3, 4, -5, 6, 7],
         [1, 7, 2, -6, 2],
         [3, 8, 6, 2, -8]]
    print(x)

    selector = VarianceThreshold(threshold=2)
    selector.fit(x)

    print()
    print(selector.variances_)
    print()
    print(selector.transform(x))
    print()
    print(selector.get_support(True))
    print()
    print(selector.inverse_transform(selector.transform(x)))

    pass


# 过滤式特征选取：单变量特征选择
def select_k_best():
    x = [[1, 2, 3],
         [2.1, 4, 6.5],
         [3, 6, 9.4]]
    y = [1, 2, 3]
    print(x)

    selector = SelectKBest(score_func=f_regression, k=2)
    selector.fit(x, y)

    print(selector.scores_)
    print(selector.pvalues_)
    print(selector.get_support(True))
    print(selector.transform(x))
    pass


# 过滤式特征选取：单变量特征选择
def select_percentile():
    x = [[0.6, 2, 3],
         [2.5, 4, 6],
         [3.4, 6.2, 9.4]]
    y = [1, 2, 3]
    print(x)

    selector = SelectPercentile(score_func=f_regression, percentile=100)
    selector.fit(x, y)

    print(selector.scores_)
    print(selector.pvalues_)
    print(selector.get_support(True))
    print(selector.transform(x))
    pass


# 包裹式特征选取：RFE
def ref():
    iris = load_iris()
    x = iris.data
    y = iris.target

    estimator = LinearSVC()
    selector = RFE(estimator=estimator, n_features_to_select=2)
    selector.fit(x, y)

    print(selector.n_features_)
    print(selector.support_)
    print(selector.ranking_)

    pass


def test_ref():
    # 加载数据
    iris = load_iris()
    x, y = iris.data, iris.target
    # 特征提取
    estimator = LinearSVC()
    selector = RFE(estimator, n_features_to_select=2)
    x_t = selector.fit_transform(x, y)
    # 切分测试集和验证集
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.25, random_state=0, stratify=y)
    x_train_t, x_test_t, y_train_t, y_test_t = model_selection.train_test_split(x_t, y, test_size=0.25, random_state=0, stratify=y)
    # 测试和验证
    clf = LinearSVC()
    clf_t = LinearSVC()
    clf.fit(x_train, y_train)
    clf_t.fit(x_train_t, y_train_t)
    print(clf.score(x_test, y_test))
    print(clf_t.score(x_test_t, y_test_t))

    pass


# 包裹式特征选取：REFCV
def refcv():
    iris = load_iris()
    x = iris.data
    y = iris.target

    estimator = LinearSVC()
    selector = RFECV(estimator=estimator, cv=5)
    selector.fit(x, y)

    print(selector.n_features_)
    print(selector.support_)
    print(selector.ranking_)
    print(selector.grid_scores_)

    pass


def test_refcv():
    # 加载数据
    iris = load_iris()
    x, y = iris.data, iris.target
    # 特征提取
    estimator = LinearSVC()
    selector = RFECV(estimator, cv=5)
    x_t = selector.fit_transform(x, y)
    # 切分测试集和验证集
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.25, random_state=0, stratify=y)
    x_train_t, x_test_t, y_train_t, y_test_t = model_selection.train_test_split(x_t, y, test_size=0.25, random_state=0, stratify=y)
    # 测试和验证
    clf = LinearSVC()
    clf_t = LinearSVC()
    clf.fit(x_train, y_train)
    clf_t.fit(x_train_t, y_train_t)
    print(clf.score(x_test, y_test))
    print(clf_t.score(x_test_t, y_test_t))

    pass


# 嵌入式特征选择
def select_from_model():
    iris = load_iris()
    x = iris.data
    y = iris.target

    estimator = LinearSVC(penalty="l1", dual=False)
    selector = SelectFromModel(estimator=estimator, threshold="mean")
    selector.fit(x, y)
    selector.transform(x)

    print(selector.threshold_)
    print(selector.get_support(indices=True))
    print(selector.get_support(indices=False))
    print(selector.inverse_transform(selector.transform(x)))

    pass


if __name__ == "__main__":
    # variance_threshold()
    # select_k_best()
    # select_percentile()
    # ref()
    # test_ref()
    # refcv()
    #test_refcv()
    select_from_model()
    pass