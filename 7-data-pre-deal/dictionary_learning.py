from sklearn.decomposition import DictionaryLearning


# 字典学习
def dictionary_learn():
    x = [[1, -2, 3, 4, 5.],
         [3, 4, -5, 6, 7],
         [1, 7, 2, -6, 2],
         [3, 8, 6, 2, -8]]
    print(x)

    dct = DictionaryLearning(n_components=5)
    dct.fit(x)

    print(dct.components_)
    print(dct.transform(x))

    pass


if __name__ == "__main__":
    dictionary_learn()
    pass