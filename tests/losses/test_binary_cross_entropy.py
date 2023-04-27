import numpy as np
import pytest

from inventronet.losses import BinaryCrossEntropy


class TestBCE:
    # Create an instance of the BCE loss function
    @pytest.fixture
    def bce(self) -> BinaryCrossEntropy:
        yield BinaryCrossEntropy()

    # Test the function method of the BCE loss function with different
    # inputs and outputs
    @pytest.mark.parametrize(
        "y_true, y_pred, expected",
        [
            (
                    np.array([1, 0, 1]),
                    np.array([0.9, 0.1, 0.9]),
                    0.10536052,
            ),  # high confidence
            (
                    np.array([1, 0, 1]),
                    np.array([0.6, 0.4, 0.6]),
                    0.51082562,
            ),  # medium confidence
            (
                    np.array([1, 0, 1]),
                    np.array([0.5, 0.5, 0.5]),
                    0.69314718,
            ),  # low confidence
        ],
    )
    def test_bce_function(self, y_true, y_pred, expected, bce: BinaryCrossEntropy):
        # Compare the expected and actual outputs with a precision o
        # f 7 decimal places
        np.testing.assert_almost_equal(
            bce.function(y_true, y_pred), expected, decimal=7
        )

    # Test the derivative method of the BCE loss function with different
    # inputs and outputs
    @pytest.mark.parametrize(
        "y_true, y_pred, expected",
        [
            (
                    np.array([1, 0, 1]),
                    np.array([0.9, 0.1, 0.9]),
                    np.array([-0.37037037, 0.3703704, -0.37037037]),
            ),  # high confidence
            (
                    np.array([1, 0, 1]),
                    np.array([0.6, 0.4, 0.6]),
                    np.array([-0.55555556, 0.55555556, -0.55555556]),
            ),  # medium confidence
            (
                    np.array([1, 0, 1]),
                    np.array([0.5, 0.5, 0.5]),
                    np.array([-0.66666667, 0.66666667, -0.66666667]),
            ),  # low confidence
        ],
    )
    def test_bce_derivative(self, y_true, y_pred, expected, bce: BinaryCrossEntropy):
        # Compare the expected and actual outputs with a precision of 7
        # decimal places
        np.testing.assert_almost_equal(
            bce.gradient(y_true, y_pred), expected, decimal=7
        )
