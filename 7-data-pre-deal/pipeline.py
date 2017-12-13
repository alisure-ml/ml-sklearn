# 学习器流水线
from sklearn.svm import LinearSVC
from sklearn.datasets import load_digits
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def test_pipe_line(data):
    x_train, x_test, y_train, y_test = data
    steps = [("linear svm", LinearSVC(C=1, penalty="l1", dual=False)),
             ("logistic regression", LogisticRegression(C=1))]

    pipeline = Pipeline(steps=steps)
    pipeline.fit(x_train, y_train)

    print(pipeline.named_steps)
    print(pipeline.score(x_test, y_test))

    pass


if __name__ == "__main__":
    data = load_digits()
    x = data.data
    y = data.target
    test_pipe_line(model_selection.train_test_split(x, y, test_size=0.25, random_state=0, stratify=y))

    pass