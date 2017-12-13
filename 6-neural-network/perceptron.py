# 感知机学习算法的原始形式
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def create_data(n):
    np.random.seed(1)
    # 正类样本
    x_11 = np.random.randint(0, 100, (n, 1))
    x_12 = np.random.randint(0, 100, (n, 1))
    x_13 = 20 + np.random.randint(0, 10, (n, 1))
    # 负类样本
    x_21 = np.random.randint(0, 100, (n, 1))
    x_22 = np.random.randint(0, 100, (n, 1))
    x_23 = 10 - np.random.randint(0, 10, (n, 1))
    # 沿X轴旋转45°
    new_x_12 = x_12 * np.sqrt(2) / 2 - x_13 * np.sqrt(2) / 2
    new_x_13 = x_12 * np.sqrt(2) / 2 + x_13 * np.sqrt(2) / 2
    new_x_22 = x_22 * np.sqrt(2) / 2 - x_23 * np.sqrt(2) / 2
    new_x_23 = x_22 * np.sqrt(2) / 2 + x_23 * np.sqrt(2) / 2
    # 组装样本点
    plus_samples = np.hstack([x_11, new_x_12, new_x_13, np.ones((n, 1))])
    minus_samples = np.hstack([x_21, new_x_22, new_x_23, -np.ones((n, 1))])
    samples = np.vstack([plus_samples, minus_samples])
    # 打乱
    np.random.shuffle(samples)
    return samples


# 绘制
def plot_samples(ax, samples):
    y = samples[:, -1]
    # 正类位置
    position_p = y == 1
    # 负类位置
    position_m = y == -1
    # 绘制
    ax.scatter(samples[position_p, 0], samples[position_p, 1], samples[position_p, 2], marker="+", label="+", color="b")
    ax.scatter(samples[position_m, 0], samples[position_m, 1], samples[position_m, 2], marker="^", label="-", color="r")


# 展示
def show_plot():
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    data = create_data(100)
    plot_samples(ax, data)
    ax.legend(loc="best")
    plt.show()


# 感知机学习算法的原始形式:迭代和更新策略1
def perceptron_1(train_data, eta, w_0, b_0):
    x = train_data[:, :-1]
    y = train_data[:, -1]
    length = train_data.shape[0]
    # 参数初始化
    w, b = w_0, b_0
    step_num = 0
    while True:
        i = 0
        while i < length:
            step_num += 1
            # 选取数据
            x_i, y_i = x[i].reshape((x.shape[1], 1)), y[i]
            # 判别是否分类正确
            z = y_i * (np.dot(np.transpose(w), x_i) + b)
            if z <= 0:
                # 若分类错误， 更新参数
                w, b = w + eta * y_i * x_i, b + eta * y_i
                break
            else:
                i += 1
        if i == length:
            break

    return w, b, step_num


# 感知机学习算法的原始形式:迭代和更新策略2
def perceptron_2(train_data, eta, w_0, b_0):
    x = train_data[:, :-1]
    y = train_data[:, -1]
    length = train_data.shape[0]
    # 参数初始化
    w, b = w_0, b_0
    step_num = 0
    while True:
        i = 0
        is_ok = True
        while i < length:
            step_num += 1
            # 选取数据
            x_i, y_i = x[i].reshape((x.shape[1], 1)), y[i]
            # 判别是否分类正确
            z = y_i * (np.dot(np.transpose(w), x_i) + b)
            if z <= 0:
                # 若分类错误， 更新参数
                w, b = w + eta * y_i * x_i, b + eta * y_i
                is_ok = False
            i += 1
        if is_ok:
            break

    return w, b, step_num


def run_perceptron():
    data = create_data(100)
    eta, w_0, b_0 = 0.1, np.ones((3, 1), dtype=float), 1
    w, b, num = perceptron_1(data, eta, w_0, b_0)

    fig = plt.figure()
    ax = Axes3D(fig)

    # 绘制样本
    plot_samples(ax, data)

    # 绘制分离超平面
    x = np.linspace(-30, 100, 100)
    y = np.linspace(-30, 100, 100)
    x, y = np.meshgrid(x, y)
    z = (-w[0][0] * x - w[1][0] * y - b) / w[2][0]
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color="g", alpha=0.2)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="best")

    plt.suptitle("perceptron")
    plt.show()

    pass


# 对比两种迭代和更新策略：perceptron_1()/perceptron_2()
def compare():
    data = create_data(100)
    eta, w_0, b_0 = 0.1, np.ones((3, 1), dtype=float), 1

    w, b, num = perceptron_1(data, eta, w_0, b_0)
    print(w, b, num)
    w, b, num = perceptron_1(data, eta, w_0, b_0)
    print(w, b, num)
    w, b, num = perceptron_2(data, eta, w_0, b_0)
    print(w, b, num)
    w, b, num = perceptron_2(data, eta, w_0, b_0)
    print(w, b, num)


# 感知机学习算法的对偶形式
def create_w(train_data, alpha):
    x = train_data[:, :-1]
    y = train_data[:, -1]
    n = train_data.shape[0]
    w = np.zeros((x.shape[1], 1))
    # 计算W
    for i in range(0, n):
        w = w + alpha[i][0] * y[i] * (x[i].reshape(x[i].size, 1))

    return w


# 感知机学习算法的对偶形式
def perceptron_dual(train_data, eta, alpha_0, b_0):
    x = train_data[:, :-1]
    y = train_data[:, -1]
    length = train_data.shape[0]
    # 参数初始化
    alpha, b = alpha_0, b_0
    step_num = 0
    while True:
        is_ok = True
        for i in range(length):
            step_num += 1
            # 选取数据
            x_i, y_i = x[i].reshape((x.shape[1], 1)), y[i]
            # 计算W
            w = create_w(train_data, alpha)
            # 判别是否分类正确
            z = y_i * (np.dot(np.transpose(w), x_i) + b)
            if z <= 0:
                # 若分类错误， 更新参数
                alpha[i][0], b = alpha[i][0] + eta, b + eta * y_i
                is_ok = False
        if is_ok:
            break

    return alpha, b, step_num


def run_perceptron_dual():
    data = create_data(50)
    alpha, b, num = perceptron_dual(data, eta=0.1, alpha_0=np.zeros((data.shape[0] * 2, 1)), b_0=0)
    w = create_w(data, alpha)

    fig = plt.figure()
    ax = Axes3D(fig)

    # 绘制样本
    plot_samples(ax, data)

    # 绘制分离超平面
    x = np.linspace(-30, 100, 100)
    y = np.linspace(-30, 100, 100)
    x, y = np.meshgrid(x, y)
    z = (-w[0][0] * x - w[1][0] * y - b) / w[2][0]
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color="g", alpha=0.2)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="best")

    plt.suptitle("perceptron")
    plt.show()

    pass


# eta 参数对模型学习收敛速度的影响
def learn_rating(data, ax, etas, w_0, alpha_0, b_0):

    nums1 = []
    nums2 = []
    print("sum length is ", len(etas))
    for i, eta in enumerate(etas):
        print("i=", i, " eta=", eta)
        _, _, num_1 = perceptron_2(data, eta, w_0=w_0, b_0=b_0)
        print("i=", i, " perceptron step number =", num_1)
        _, _, num_2 = perceptron_dual(data, eta=0.1, alpha_0=alpha_0, b_0=b_0)
        print("i=", i, " perceptron dual step number =", num_2)
        nums1.append(num_1)
        nums2.append(num_2)

    ax.plot(etas, np.array(nums1), label="orignal")
    ax.plot(etas, np.array(nums2), label="dual")

    pass


# eta 参数对模型学习收敛速度的影响
def run_learn_rating():
    fig = plt.figure()
    fig.suptitle("percetron")
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(r"$\eta$")

    data = create_data(10)
    etas = np.linspace(0.01, 1, num=25, endpoint=False)
    w_0, b_0, alpha_0 = np.ones((3, 1)), 0, np.zeros((data.shape[0], 1))
    learn_rating(data, ax, etas, w_0=w_0, b_0=b_0, alpha_0=alpha_0)

    ax.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    # show_plot()
    # run_perceptron()
    # compare()
    # run_perceptron_dual()
    run_learn_rating()

    pass
