from sklearn.preprocessing import Binarizer


# 二元化
def run_binarizer():

    x = [[1, 2, 3, 4, 5, 6, 7],
         [3, 4, 5, 6, 7, 8, 9],
         [1, 7, 2, 6, 2, 7, 2],
         [3, 8, 6, 2, 8, 3, 8]]

    print(x)

    binarizer = Binarizer(threshold=4)
    print(binarizer.transform(x))

    pass


if __name__ == "__main__":

    run_binarizer()
