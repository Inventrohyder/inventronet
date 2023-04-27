import numpy as np
import pytest

from inventronet.activations import ReLU


class TestReLU:
    # Test the ReLU activation function

    @pytest.fixture
    def relu(self) -> ReLU:
        # Fixture for the ReLU function
        yield ReLU()

    @pytest.mark.parametrize(
        "x, y",
        [
            (np.array([0, 1, -1]), np.array([0, 1, 0])),
            (np.array([2, -2, 0.5]), np.array([2, 0, 0.5])),
        ],
    )
    def test_relu_function(self, relu: ReLU, x, y):
        # Test the function method of the ReLU function with different
        # inputs and outputs
        assert np.allclose(relu.function(x), y)

    @pytest.mark.parametrize(
        "x, y",
        [
            (np.array([0, 1, -1]), np.array([0, 1, 0])),
            (np.array([2, -2, 0.5]), np.array([1, 0, 1])),
        ],
    )
    def test_relu_derivative(self, relu: ReLU, x, y):
        # Test the derivative method of the ReLU function with different
        # inputs and outputs
        assert np.allclose(relu.derivative(x), y)
