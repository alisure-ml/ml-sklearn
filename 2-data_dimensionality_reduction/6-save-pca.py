from sklearn.externals import joblib
from sklearn import datasets, decomposition
import pickle


# 加载数据
# Iris也称鸢尾花卉数据集，是一类多重变量分析的数据集。
# 数据集包含150个数据集，分为3类，每类50个数据，每个数据包含4个属性。
# 可通过花萼长度，花萼宽度，花瓣长度，花瓣宽度4个属性
# 预测鸢尾花卉属于（Setosa，Versicolour，Virginica）三个种类中的哪一类。
def load_data():
    iris = datasets.load_iris()
    return iris.data, iris.target


# 打印Iris
def print_iris():
    data, target = load_data()
    print(data)
    print(target)


# 运行PCA并保存训练结果
def run_pca(*data):
    x, y, path = data
    pca = decomposition.PCA(n_components=2)
    # 训练模型，得到投影矩阵W
    #   1.去中心化
    #   2.计算协方差矩阵
    #   3.对协方差特征分解
    #   4.取d个特征值对应的特征向量，构造投影矩阵W
    pca.fit(x)

    # 保存训练结果
    joblib.dump(pca, path)


# 读取保存的训练结果、执行降维并保存降维结果
def run_transform(*data):
    x, y, m_path, p_path = data
    pca = joblib.load(m_path)

    # 计算 z = W * x
    x_t = pca.transform(x)

    # 保存计算结果
    with open(p_path, "wb") as f:
        pickle.dump(x_t, f)


# 读取降维结果
def read_pkl(path):
    with open(path, "rb") as f:
        x_r = pickle.load(f)
    return x_r


M_path = "6-save-pca_model.m"
P_path = "pca_wdref2.pkl"
X, Y = load_data()
run_pca(X, Y, M_path)
run_transform(X, Y, M_path, P_path)
print(read_pkl(P_path))