from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler


# min-max标准化
def min_max():
    x = [[1, 2, 3, 4, 5],
         [3, 4, 5, 6, 7],
         [1, 7, 2, 6, 2],
         [3, 8, 6, 2, 8]]
    print(x)

    scaler = MinMaxScaler(feature_range=(0, 2))
    scaler.fit(x)
    print(scaler.data_max_)
    print(scaler.data_min_)
    print(scaler.data_range_)
    print(scaler.feature_range)

    # feature_range[0] - data_min * self.scale_
    print(scaler.min_)
    print(scaler.scale_)
    print(scaler.transform(x))

    pass


# 每个属性值除以该属性的绝对值的最大值
def max_abs():
    x = [[1, -2, 3, 4, 5],
         [3, 4, -5, 6, 7],
         [1, 7, 2, -6, 2],
         [3, 8, 6, 2, -8]]
    print(x)

    scaler = MaxAbsScaler()
    scaler.fit(x)

    print(scaler.scale_)
    print(scaler.max_abs_)
    print(scaler.transform(x))

    pass


# z-score标准化
# 标准化后，每个特征的均值为0， 方差为1
def standard_scaler():
    x = [[1, -2, 3, 4, 5],
         [3, 4, -5, 6, 7],
         [1, 7, 2, -6, 2],
         [3, 8, 6, 2, -8]]
    print(x)

    scaler = StandardScaler()
    scaler.fit(x)

    print(scaler.scale_)
    print(scaler.mean_)
    print(scaler.var_)
    print(scaler.transform(x))

    pass


if __name__ == "__main__":

    min_max()
    max_abs()
    standard_scaler()

    pass
