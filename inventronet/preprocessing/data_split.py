import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    data_size = len(X)
    indices = np.arange(data_size)
    np.random.shuffle(indices)

    test_count = int(test_size * data_size)
    train_indices = indices[:-test_count]
    test_indices = indices[-test_count:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test
