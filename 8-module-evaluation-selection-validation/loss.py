"""
loss function
"""
from sklearn.metrics import zero_one_loss
from sklearn.metrics import log_loss


def loss():

    y_true = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    y_pred = [0, 0, 1, 1, 0, 1, 0, 1, 1, 0]

    # Zero-one classification loss.
    print(zero_one_loss(y_true=y_true, y_pred=y_pred, normalize=True))
    print(zero_one_loss(y_true=y_true, y_pred=y_pred, normalize=False))

    pass


def loss_log():

    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [
        [0.2, 0.8],
        [0.4, 0.6],
        [0.1, 0.9],
        [0.6, 0.4],
        [0.8, 0.2],
        [0.9, 0.1]
    ]

    # Log loss, aka logistic loss or cross-entropy loss.
    #
    # The log loss is only defined for two or more labels.
    # For a single sample with true label yt in {0,1} and
    # estimated probability yp that yt = 1, the log loss is:
    #   -log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))
    print(log_loss(y_true=y_true, y_pred=y_pred, normalize=True))
    print(log_loss(y_true=y_true, y_pred=y_pred, normalize=False))

    pass


if __name__ == "__main__":

    loss()
    loss_log()

    pass
