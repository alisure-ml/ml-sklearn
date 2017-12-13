from sklearn.preprocessing import OneHotEncoder


def one_hot():
    x = [[1, 2, 3, 4, 5, 6, 7],
         [3, 4, 5, 6, 7, 8, 9],
         [1, 7, 2, 6, 2, 7, 2],
         [3, 8, 6, 2, 8, 3, 8]]
    print(x)

    encoder = OneHotEncoder(sparse=False)
    encoder.fit(x)

    print(encoder.active_features_)
    print(encoder.feature_indices_)
    print(encoder.n_values_)

    # [[ 1.  0.  1.  0.  0.  0.  1.  0.  0.  0.  1.  0.  0.  1.  0.  0.  0.  1.  0.  0.  0.  1.  0.  0.  0.]]
    print(encoder.transform([[1, 2, 2, 2, 2, 3, 2]]))

    pass


if __name__ == "__main__":

    one_hot()

    pass
