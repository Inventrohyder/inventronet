import numpy as np
import pytest

from inventronet.losses import MeanAbsoluteError


class TestMAE:
    # Create an instance of the MAE loss function
    @pytest.fixture
    def mae(self) -> MeanAbsoluteError:
        yield MeanAbsoluteError()

    # Test the function method of the MSE loss function with different inputs
    # and outputs
    @pytest.mark.parametrize(
        "y_true, y_pred, expected",
        [
            (np.array([1, 2, 3]), np.array([1, 2, 3]), 0.0),  # exact match
            # one unit difference
            (np.array([1, 2, 3]), np.array([2, 3, 4]), 1.0),
            (
                    np.array([1, 2, 3]),
                    np.array([0, 0, 0]),
                    2,
            ),  # large difference
        ],
    )
    def test_mae_function(self, y_true, y_pred, expected, mae: MeanAbsoluteError):
        # Compare the expected and actual outputs with a precision of 7
        # decimal places
        np.testing.assert_almost_equal(
            mae.function(y_true, y_pred), expected, decimal=7
        )

    # Test the derivative method of the MAE loss function with different
    # inputs and outputs
    @pytest.mark.parametrize(
        "y_true, y_pred, expected",
        [
            (
                    np.array([1, 2, 3]),
                    np.array([1, 2, 3]),
                    np.array([0, 0, 0]),
            ),  # exact match
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
    def test_mae_derivative(self, y_true, y_pred, expected, mae: MeanAbsoluteError):
        # Compare the expected and actual outputs with a precision of 7
        # decimal places
        np.testing.assert_almost_equal(
            mae.gradient(y_true, y_pred), expected, decimal=7
        )
