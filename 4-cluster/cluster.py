import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
from sklearn import mixture


# 数据
def create_data(centers, num=100, std=0.7):
    x, labels_true = make_blobs(n_samples=num, centers=centers, cluster_std=std)
    return x, labels_true


# 查看生成的数据
def plot_data(*data):
    x, labels_true = data
    labels = np.unique(labels_true)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = "rgbyckm"
    for i, label in enumerate(labels):
        position = labels_true == label
        ax.scatter(x[position, 0], x[position, 1], label="cluster %d" % label, color=colors[i % len(colors)])

    ax.legend(loc="best", framealpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("data")
    plt.show()

    pass


# X, Labels_true = create_data([[1, 1], [2, 2], [3, 3], [4, 4]], 100, 0.3)
# plot_data(X, Labels_true)


# KMean
def k_means(*data):
    x, _labels_true, n_clusters = data
    clu = cluster.KMeans(n_clusters=n_clusters, max_iter=100)
    clu.fit(x)
    predicted_labels = clu.predict(x)
    print(adjusted_rand_score(_labels_true, predicted_labels))
    print(clu.inertia_)
    print(clu.labels_)
    print(clu.cluster_centers_)
    print(clu)

# centers = [[1, 1], [5, 5], [10, 10], [10, 20]]
# X, labels_true = create_data(centers, 1000, 0.5)
# k_means(X, labels_true, len(centers))


# 簇个数对KMeans的影响
def k_means_n_clusters(*data):
    x, _labels_true = data
    nums = range(1, 50)
    aris = []
    distances = []
    for num in nums:
        clu = cluster.KMeans(n_clusters=num)
        clu.fit(x)
        predicted_labels = clu.predict(x)
        aris.append(adjusted_rand_score(_labels_true, predicted_labels))
        distances.append(clu.inertia_)

    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 1)
    ax.plot(nums, aris, marker="+")
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("ARI")

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(nums, distances, marker="o")
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("inertia_")

    fig.suptitle("K-Means")
    plt.show()


# centers = [[1, 1], [5, 5], [10, 10], [10, 20]]
# X, labels_true = create_data(centers, 1000, 0.5)
# k_means_n_clusters(X, labels_true)


# 迭代次数对KMeans的影响
def k_means_n_init(*data):
    x, _labels_true, n_clusters = data
    nums = range(1, 50)
    aris = []
    distances = []
    for num in nums:
        clu = cluster.KMeans(n_clusters=n_clusters, n_init=num, init="random")
        clu.fit(x)
        predicted_labels = clu.predict(x)
        aris.append(adjusted_rand_score(_labels_true, predicted_labels))
        distances.append(clu.inertia_)

    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 1)
    ax.plot(nums, aris, marker="+")
    ax.set_xlabel("n_init")
    ax.set_ylabel("ARI")

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(nums, distances, marker="o")
    ax.set_xlabel("n_init")
    ax.set_ylabel("inertia_")

    fig.suptitle("K-Means")
    plt.show()


# centers = [[1, 1], [2, 2], [1, 2], [2, 1]]
# X, labels_true = create_data(centers, 1000, 0.5)
# k_means_n_init(X, labels_true, len(centers))


# 密度聚类
def db_scan(*data):
    x, _labels_true = data
    cls = cluster.DBSCAN(min_samples=50)
    predicted_labels = cls.fit_predict(x)
    print(adjusted_rand_score(_labels_true, predicted_labels))
    print(cls.core_sample_indices_)
    print(cls.labels_)


# centers = [[0, 0], [0, 20], [20, 0], [20, 20]]
# X, labels_true = create_data(centers, 1000, 0.5)
# db_scan(X, labels_true)


# 邻域大小对密度聚类的影响
def db_scan_epsilon(*data):
    x, _labels_true = data
    epsilons = np.logspace(-1, 1.5)
    aris = []
    core_nums = []
    for epsilon in epsilons:
        cls = cluster.DBSCAN(eps=epsilon)
        predicted_labels = cls.fit_predict(x)
        aris.append(adjusted_rand_score(_labels_true, predicted_labels))
        core_nums.append(len(cls.core_sample_indices_))

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(epsilons, aris, marker="+")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylim(0, 1)
    ax.set_ylabel("ARI")

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(epsilons, core_nums, marker="o")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel("core_nums")

    fig.suptitle("DBSCAN")
    plt.show()


# centers = [[0, 0], [0, 20], [20, 0], [20, 20]]
# X, labels_true = create_data(centers, 1000, 0.5)
# db_scan_epsilon(X, labels_true)


# 确定核心对象所需的样本个数对密度聚类的影响
def db_scan_min_pts(*data):
    x, _labels_true = data
    min_pts = range(1, 100)
    aris = []
    core_nums = []
    for num in min_pts:
        cls = cluster.DBSCAN(min_samples=num)
        predicted_labels = cls.fit_predict(x)
        aris.append(adjusted_rand_score(_labels_true, predicted_labels))
        core_nums.append(len(cls.core_sample_indices_))

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(min_pts, aris, marker="+")
    ax.set_xlabel("min_pts")
    ax.set_ylim(0, 1)
    ax.set_ylabel("ARI")

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(min_pts, core_nums, marker="o")
    ax.set_xlabel("min_pts")
    ax.set_ylabel("core_nums")

    fig.suptitle("DBSCAN")
    plt.show()


# centers = [[0, 0], [0, 20], [20, 0], [20, 20]]
# X, labels_true = create_data(centers, 1000, 0.5)
# db_scan_min_pts(X, labels_true)


# 层次聚类
def agglomerative_clustering(*data):
    x, _labels_true = data
    cls = cluster.AgglomerativeClustering(n_clusters=4)
    predicted_labels = cls.fit_predict(x)
    print(adjusted_rand_score(_labels_true, predicted_labels))
    print(cls.labels_)
    print(cls.n_leaves_)
    print(cls.n_components_)
    print(cls.children_)


# centers = [[0, 0], [0, 20], [20, 0], [20, 20]]
# X, labels_true = create_data(centers, 1000, 0.5)
# agglomerative_clustering(X, labels_true)


# 簇的个数对聚类效果的影响
def agglomerative_clustering_n_clusters(*data):
    x, _labels_true = data
    n_clusters = range(1, 100)
    linkages = ["ward", "complete", "average"]
    markers = "+o*"
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i, linkage in enumerate(linkages):
        aris = []
        for n_cluster in n_clusters:
            cls = cluster.AgglomerativeClustering(n_clusters=n_cluster, linkage=linkage)
            predicted_labels = cls.fit_predict(x)
            aris.append(adjusted_rand_score(_labels_true, predicted_labels))
        ax.plot(n_clusters, aris, marker=markers[i], label="linkage:%s" % linkage)

    ax.set_xlabel("n_clusters")
    ax.set_ylim(0, 1)
    ax.set_ylabel("ARI")
    ax.legend(loc="best")

    fig.suptitle("AgglomerativeClustering")
    plt.show()


# centers = [[0, 0], [0, 20], [20, 0], [20, 20]]
# X, labels_true = create_data(centers, 1000, 0.5)
# agglomerative_clustering_n_clusters(X, labels_true)


# 混合高斯模型
def gmm(*data):
    x, _labels_true = data
    cls = mixture.GaussianMixture(n_components=4)
    cls.fit(x)
    predicted_labels = cls.predict(x)
    print(adjusted_rand_score(_labels_true, predicted_labels))


# centers = [[1, 1], [2, 2], [1, 2], [10, 20]]
# X, labels_true = create_data(centers, 1000, 0.5)
# gmm(X, labels_true)


# 簇的个数对混合高斯聚类效果的影响
def gmm_n_components(*data):
    x, _labels_true = data
    n_clusters = range(1, 50)
    aris = []
    for n_cluster in n_clusters:
        cls = mixture.GaussianMixture(n_components=n_cluster)
        cls.fit(x)
        predicted_labels = cls.predict(x)
        aris.append(adjusted_rand_score(_labels_true, predicted_labels))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(n_clusters, aris, marker="+")
    ax.set_xlabel("n_clusters")
    ax.set_ylim(0, 1)
    ax.set_ylabel("ARI")

    fig.suptitle("GaussianMixture")
    plt.show()


# centers = [[0, 0], [0, 20], [20, 0], [20, 20]]
# X, labels_true = create_data(centers, 1000, 0.5)
# gmm_n_components(X, labels_true)

