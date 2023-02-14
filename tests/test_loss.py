import pytest
import numpy as np
from ..loss import MAE, MSE


# Create an instance of the MAE loss function
@pytest.fixture
def mae() -> MAE:
    yield MAE()


# Create an instance of the MSE loss function
@pytest.fixture
def mse() -> MSE:
    yield MSE()


# Test the function method of the MSE loss function with different inputs and outputs
@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        (np.array([1, 2, 3]), np.array([1, 2, 3]), 0.0),  # exact match
        (np.array([1, 2, 3]), np.array([2, 3, 4]), 1.0),  # one unit difference
        (
            np.array([1, 2, 3]),
            np.array([0, 0, 0]),
            2,
        ),  # large difference
    ],
)
def test_mae_function(y_true, y_pred, expected, mae: MAE):
    # Compare the expected and actual outputs with a precision of 7 decimal places
    np.testing.assert_almost_equal(mae.function(y_true, y_pred), expected, decimal=7)


# Test the derivative method of the MAE loss function with different inputs and outputs
@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        (np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([0, 0, 0])),  # exact match
        (
            np.array([1, 2, 3]),
            np.array([2, 3, 4]),
            np.array([0.33333333, 0.33333333, 0.33333333]),
        ),  # one unit difference
        (
            np.array([1, 2, 3]),
            np.array([0, 0, 0]),
            np.array([-0.33333333, -0.33333333, -0.33333333]),
        ),  # large difference
    ],
)
def test_mae_derivative(y_true, y_pred, expected, mae: MAE):
    # Compare the expected and actual outputs with a precision of 7 decimal places
    np.testing.assert_almost_equal(mae.gradient(y_true, y_pred), expected, decimal=7)


# Test the function method of the MSE loss function with different inputs and outputs
@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        (np.array([1, 2, 3]), np.array([1, 2, 3]), 0.0),  # exact match
        (np.array([1, 2, 3]), np.array([2, 3, 4]), 1.0),  # one unit difference
        (
            np.array([1, 2, 3]),
            np.array([0, 0, 0]),
            4.666666666666667,
        ),  # large difference
    ],
)
def test_mse_function(y_true, y_pred, expected, mse: MSE):
    # Compare the expected and actual outputs with a precision of 7 decimal places
    np.testing.assert_almost_equal(mse.function(y_true, y_pred), expected, decimal=7)


# Test the derivative method of the MSE loss function with different inputs and outputs
@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        (np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([0, 0, 0])),  # exact match
        (
            np.array([1, 2, 3]),
            np.array([2, 3, 4]),
            np.array([0.66666667, 0.66666667, 0.66666667]),
        ),  # one unit difference
        (
            np.array([1, 2, 3]),
            np.array([0, 0, 0]),
            np.array([-0.66666667, -1.33333333, -2.0]),
        ),  # large difference
    ],
)
def test_mse_derivative(y_true, y_pred, expected, mse: MSE):
    # Compare the expected and actual outputs with a precision of 7 decimal places
    np.testing.assert_almost_equal(mse.gradient(y_true, y_pred), expected, decimal=7)
