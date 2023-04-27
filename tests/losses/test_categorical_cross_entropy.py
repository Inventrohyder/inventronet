import numpy as np
import pytest

from inventronet.losses import CategoricalCrossEntropy


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
