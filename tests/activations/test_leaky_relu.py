import numpy as np
import pytest

from inventronet.activations import LeakyReLU


class TestLeakyReLU:
    # Test the LeakyReLU activation function

    @pytest.fixture
    def leaky_relu(self) -> LeakyReLU:
        # Fixture for the LeakyReLU function with default alpha
        yield LeakyReLU()

    @pytest.fixture
    def leaky_relu_01(self):
        # Fixture for the LeakyReLU function with alpha = 0.1
        return LeakyReLU(alpha=0.1)

    @pytest.mark.parametrize(
        "x, y",
        [
            (np.array([0, 1, -1]), np.array([0, 1, -0.01])),
            (np.array([2, -2, 0.5]), np.array([2, -0.02, 0.5])),
        ],
    )
    def test_leaky_relu_function(self, leaky_relu: LeakyReLU, x, y):
        # Test the function method of the LeakyReLU function with different
        # inputs and outputs
        assert np.allclose(leaky_relu.function(x), y)

    @pytest.mark.parametrize(
        "x, y",
        [
            (np.array([0, 1, -1]), np.array([0.01, 1, 0.01])),
            (np.array([2, -2, 0.5]), np.array([1, 0.01, 1])),
        ],
    )
    def test_leaky_relu_derivative(self, leaky_relu: LeakyReLU, x, y):
        # Test the derivative method of the LeakyReLU function with different
        # inputs and outputs
        assert np.allclose(leaky_relu.derivative(x), y)

    @pytest.mark.parametrize(
        "x, y",
        [
            (np.array([0, 1, -1]), np.array([0, 1, -0.1])),
            (np.array([2, -2, 0.5]), np.array([2, -0.2, 0.5])),
        ],
    )
    def test_leaky_relu_function_01(self, leaky_relu_01: LeakyReLU, x, y):
        # Test the function method of the LeakyReLU function with different
        # inputs and outputs
        assert np.allclose(leaky_relu_01.function(x), y)

    @pytest.mark.parametrize(
        "x, y",
        [
            (np.array([0, 1, -1]), np.array([0.1, 1, 0.1])),
            (np.array([2, -2, 0.5]), np.array([1, 0.1, 1])),
        ],
    )
    def test_leaky_relu_derivative_01(self, leaky_relu_01: LeakyReLU, x, y):
        # Test the derivative method of the LeakyReLU function with different
        # inputs and outputs
        assert np.allclose(leaky_relu_01.derivative(x), y)
