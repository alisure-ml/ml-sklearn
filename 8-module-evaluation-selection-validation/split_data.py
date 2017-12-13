"""
 split train and test
"""
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut


# Split arrays or matrices into random train and test subsets
def split():

    x = [
        [11, 12, 13, 14],
        [21, 22, 23, 24],
        [31, 32, 33, 34],
        [41, 42, 43, 44],
        [51, 52, 53, 54],
        [61, 62, 63, 64],
        [71, 72, 73, 74],
        [81, 82, 83, 84],
        [91, 92, 93, 94]
    ]
    y = [1, 1, 1, 1, 1, 1, 0, 1, 0]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    print(x_train)
    print(x_test)
    print(y_train)
    print(y_test)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0, stratify=y)
    print(x_train)
    print(x_test)
    print(y_train)
    print(y_test)

    pass


# K-Folds cross-validator
def split_k_fold():

    x = np.array([
        [11, 12, 13, 14],
        [21, 22, 23, 24],
        [31, 32, 33, 34],
        [41, 42, 43, 44],
        [51, 52, 53, 54],
        [61, 62, 63, 64],
        [71, 72, 73, 74],
        [81, 82, 83, 84],
        [91, 92, 93, 94]
    ])
    y = np.array([1, 1, 1, 1, 1, 1, 0, 1, 0])

    folder = KFold(n_splits=3, shuffle=False, random_state=0)
    for train_index, test_index in folder.split(x, y):
        print(train_index)
        print(test_index)
        print(x[train_index])
        print(x[test_index])
        print()

    print()

    shuffle_folder = KFold(n_splits=3, random_state=0, shuffle=True)
    for train_index, test_index in shuffle_folder.split(x, y):
        print(train_index)
        print(test_index)
        print(x[train_index])
        print(x[test_index])
        print()

    pass


# Stratified K-Folds cross-validator
def split_stratified_k_fold():

    x = np.array([
        [11, 12, 13, 14],
        [21, 22, 23, 24],
        [31, 32, 33, 34],
        [41, 42, 43, 44],
        [51, 52, 53, 54],
        [61, 62, 63, 64],
        [71, 72, 73, 74],
        [81, 82, 83, 84],
        [91, 92, 93, 94]
    ])
    y = np.array([1, 0, 0, 1, 1, 1, 1, 0, 1])

    folder = StratifiedKFold(n_splits=3, shuffle=False, random_state=0)
    for train_index, test_index in folder.split(x, y):
        print(train_index)
        print(test_index)
        print(x[train_index])
        print(x[test_index])
        print()

    print()

    shuffle_folder = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
    for train_index, test_index in shuffle_folder.split(x, y):
        print(train_index)
        print(test_index)
        print(x[train_index])
        print(x[test_index])
        print()

    pass


# Leave-One-Out cross-validator
def split_leave_one_out():

    x = np.array([
        [11, 12, 13, 14],
        [21, 22, 23, 24],
        [31, 32, 33, 34],
        [41, 42, 43, 44],
        [51, 52, 53, 54],
        [61, 62, 63, 64],
        [71, 72, 73, 74],
        [81, 82, 83, 84],
        [91, 92, 93, 94]
    ])
    y = np.array([1, 0, 0, 1, 1, 1, 1, 0, 1])

    loo = LeaveOneOut()

    for train_index, test_index in loo.split(x, y):
        print(train_index)
        print(test_index)
        print(x[train_index])
        print(x[test_index])
        print()

    pass


if __name__ == "__main__":

    split_leave_one_out()

    pass
