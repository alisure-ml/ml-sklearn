from sklearn.preprocessing import Normalizer


# 正则化
# 将每一个样本正则化为范数等于单位1
def norm():
    x = [[1, -2, 3, 4, 5.],
         [3, 4, -5, 6, 7],
         [1, 7, 2, -6, 2],
         [3, 8, 6, 2, -8]]
    print(x)

    normalizer = Normalizer(norm="l1")
    print()
    print(normalizer.transform(x))

    normalizer = Normalizer(norm="l2")
    print()
    print(normalizer.transform(x))

    normalizer = Normalizer(norm="max")
    print()
    print(normalizer.transform(x))

    pass

if __name__ == "__main__":

    norm()

    pass
