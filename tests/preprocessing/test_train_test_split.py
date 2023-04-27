import numpy as np
import pytest

from inventronet.preprocessing import train_test_split


@pytest.fixture(scope="module")
def data():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, size=(100,))
    return X, y


@pytest.fixture(scope="module")
def splitted_data(data):
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def test_X_train_shape(splitted_data):
    X_train, _, _, _ = splitted_data
    assert X_train.shape == (80, 5)


def test_X_test_shape(splitted_data):
    _, X_test, _, _ = splitted_data
    assert X_test.shape == (20, 5)


def test_y_train_shape(splitted_data):
    _, _, y_train, _ = splitted_data
    assert y_train.shape == (80,)


def test_y_test_shape(splitted_data):
    _, _, _, y_test = splitted_data
    assert y_test.shape == (20,)


def test_indices_shuffled(data, splitted_data):
    X, _ = data
    X_train, X_test, _, _ = splitted_data
    assert not np.array_equal(np.arange(100), np.concatenate([X_train, X_test]))
