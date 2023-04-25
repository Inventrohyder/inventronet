import numpy as np
import pytest

from inventronet.losses import (
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    MeanAbsoluteError,
    MeanSquaredError,
)


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


class TestMSE:
    # Create an instance of the MSE loss function
    @pytest.fixture
    def mse(self) -> MeanSquaredError:
        yield MeanSquaredError()

    # Test the function method of the MSE loss function with different
    # inputs and outputs
    @pytest.mark.parametrize(
        "y_true, y_pred, expected",
        [
            (np.array([1, 2, 3]), np.array([1, 2, 3]), 0.0),  # exact match
            # one unit difference
            (np.array([1, 2, 3]), np.array([2, 3, 4]), 1.0),
            (
                    np.array([1, 2, 3]),
                    np.array([0, 0, 0]),
                    4.666666666666667,
            ),  # large difference
        ],
    )
    def test_mse_function(self, y_true, y_pred, expected, mse: MeanSquaredError):
        # Compare the expected and actual outputs with a precision
        # of 7 decimal places
        np.testing.assert_almost_equal(
            mse.function(y_true, y_pred), expected, decimal=7
        )

    # Test the derivative method of the MSE loss function with different
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
                    np.array([0.66666667, 0.66666667, 0.66666667]),
            ),  # one unit difference
            (
                    np.array([1, 2, 3]),
                    np.array([0, 0, 0]),
                    np.array([-0.66666667, -1.33333333, -2.0]),
            ),  # large difference
        ],
    )
    def test_mse_derivative(self, y_true, y_pred, expected, mse: MeanSquaredError):
        # Compare the expected and actual outputs with a precision of
        # 7 decimal places
        np.testing.assert_almost_equal(
            mse.gradient(y_true, y_pred), expected, decimal=7
        )


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


class TestCCE:
    @pytest.fixture
    def cce(self) -> CategoricalCrossEntropy:
        yield CategoricalCrossEntropy()

    @pytest.mark.parametrize(
        "y_true, y_pred, expected",
        [
            (
                    np.array([[1, 0], [0, 1], [1, 0]]),
                    np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1]]),
                    0.10536052,
            ),  # high confidence
            (
                    np.array([[1, 0], [0, 1], [1, 0]]),
                    np.array([[0.6, 0.4], [0.4, 0.6], [0.6, 0.4]]),
                    0.51082562,
            ),  # medium confidence
            (
                    np.array([[1, 0], [0, 1], [1, 0]]),
                    np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]),
                    0.69314718,
            ),  # low confidence
        ],
    )
    def test_cce_function(self, y_true, y_pred, expected, cce: CategoricalCrossEntropy):
        # Compare the expected and actual outputs with a precision of
        # 7 decimal places
        np.testing.assert_almost_equal(
            cce.function(y_true, y_pred), expected, decimal=7
        )

    @pytest.mark.parametrize(
        "y_true, y_pred, expected",
        [
            (
                    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                    np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]),
                    np.array([[-0.1, 0.05, 0.05], [0.05, -0.1, 0.05], [0.05, 0.05, -0.1]]),
            ),  # high confidence
            (
                    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                    np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]]),
                    np.array([[-0.4, 0.2, 0.2], [0.2, -0.4, 0.2], [0.2, 0.2, -0.4]]),
            ),  # medium confidence
            (
                    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                    np.array([[0.33, 0.33, 0.33], [0.33, 0.33, 0.33], [0.33, 0.33, 0.33]]),
                    np.array(
                        [[-0.67, 0.33, 0.33], [0.33, -0.67, 0.33], [0.33, 0.33, -0.67]]
                    ),
            ),  # low confidence
        ],
    )
    def test_cce_derivative(
            self, y_true, y_pred, expected, cce: CategoricalCrossEntropy
    ):
        # Compare the expected and actual outputs with a precision of 7
        # decimal places
        np.testing.assert_almost_equal(
            cce.gradient(y_true, y_pred), expected, decimal=7
        )

    def test_cce_non_negativity(self, cce: CategoricalCrossEntropy):
        """
        Test that the CCE loss is non-negative.
        """
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.random.rand(3, 3)
        y_pred /= y_pred.sum(axis=1, keepdims=True)

        loss = cce(y_true, y_pred)
        assert loss >= 0, f"CCE loss should be non-negative, got {loss}"

    def test_cce_lower_bound(self, cce: CategoricalCrossEntropy):
        """
        Test that the CCE loss is 0 when the true and predicted labels are
        the same.
        """
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = y_true.copy()

        loss = cce(y_true, y_pred)
        assert np.isclose(loss, 0), f"CCE loss should be 0, got {loss}"

    def test_cce_gradient_property(self, cce: CategoricalCrossEntropy):
        """
        Test that the gradient of the CCE loss is 0 when the true and
        predicted labels are the same.
        """
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = y_true.copy()

        gradient = cce.gradient(y_true, y_pred)
        assert np.allclose(gradient, 0), f"Gradient should be 0, got {gradient}"

    def test_non_negative(self, cce: CategoricalCrossEntropy) -> None:
        """
        Test that the categorical cross entropy loss is non-negative for any
          true and predicted labels.
        """
        # Generate some random true and predicted labels
        y_true = np.random.randint(0, 3, size=(10, 1))
        y_pred = np.random.rand(10, 3)
        # Compute the loss
        loss = cce.function(y_true, y_pred)
        # Assert that the loss is non-negative
        assert loss >= 0
