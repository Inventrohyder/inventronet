# Import the pytest module and the metric classes
import numpy as np
import pytest
from ..metrics import Accuracy, Precision


# Create a fixture to instantiate the metric classes
@pytest.fixture
def accuracy() -> Accuracy:
    yield Accuracy()


@pytest.fixture
def precision() -> Precision:
    yield Precision()


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
            1.0,
        ),  # small difference
        (
            np.array([1, 0, 0, 1, 1]),
            np.array([0.1, 0.9, 0.8, 0.2, 0.3]),
            0.0,
        ),  # large difference
        (
            np.array([1, 0, 0, 1, 1]),
            np.array([0.6, 0.6, 0.6, 0.6, 0.6]),
            0.6,
        ),  # equal predictions
        (
            np.array([1, 0, 0, 1, 1]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            0.0,
        ),  # no predictions
        (
            np.array([1, 1, 1, 1, 1]),
            np.array([0.9, 0.9, 0.9, 0.9, 0.9]),
            1.0,
        ),  # all true positives and no false positives
        (
            np.array([0, 0, 0, 0, 0]),
            np.array([0.9, 0.9, 0.9, 0.9, 0.9]),
            0.0,
        ),  # no true positives and all false positives
        (
            [1, 0, 1, 0, 1],
            [0.9, 0.1, 0.9, 0.1, 0.9],
            1.0,
        ),  # Python list input
        (
            np.array([]),
            np.array([]),
            0.0,
        ),  # empty arrays
    ],
)
def test_precision_call(y_true, y_pred, expected, precision: Precision):
    # Compare the expected and actual outputs with a precision of 7 decimal
    # places
    np.testing.assert_almost_equal(precision.call(y_true, y_pred), expected, decimal=7)
