import numpy as np
import pytest

from inventronet.metrics import Accuracy


@pytest.fixture
def accuracy() -> Accuracy:
    yield Accuracy()


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        (
            np.array([1, 0, 0, 1, 1]),
            np.array([0.9, 0.1, 0.2, 0.8, 0.7]),
            1.0,
        ),  # exact match
        (
            np.array([1, 0, 0, 1, 1]),
            np.array([0.6, 0.4, 0.3, 0.7, 0.5]),
            0.8,
        ),  # small difference
        (
            np.array([1, 0, 0, 1, 1]),
            np.array([0.1, 0.9, 0.8, 0.2, 0.3]),
            0.0,
        ),  # large difference
        (
            [1, 0, 0, 1, 1],
            [0.9, 0.1, 0.2, 0.8, 0.7],
            1.0,
        ),  # Python list input
        (
            np.array([1, 0, 1, 0, 1]),
            np.array([0.6, 0.4, 0.6, 0.4, 0.6]),
            1.0,  # Update the expected value to 1.0
        ),  # equal number of correct and incorrect predictions
    ],
)
def test_accuracy_call(y_true, y_pred, expected, accuracy: Accuracy):
    np.testing.assert_almost_equal(accuracy.call(y_true, y_pred), expected, decimal=7)
