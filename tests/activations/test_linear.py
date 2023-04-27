import numpy as np
import pytest

from inventronet.activations import Linear


class TestLinear:
    # Test the Linear activation function

    @pytest.fixture
    def linear(self) -> Linear:
        # Fixture for the Linear function
        yield Linear()

    @pytest.mark.parametrize(
        "x, y",
        [
            (np.array([0, 1, -1]), np.array([0, 1, -1])),
            (np.array([2, -2, 0.5]), np.array([2, -2, 0.5])),
        ],
    )
    def test_linear_function(self, linear: Linear, x, y):
        # Test the function method of the Linear function with different
        # inputs and outputs
        assert np.allclose(linear.function(x), y)

    @pytest.mark.parametrize(
        "x, y",
        [
            (np.array([0, 1, -1]), np.array([1, 1, 1])),
            (np.array([2, -2, 0.5]), np.array([1, 1, 1])),
        ],
    )
    def test_linear_derivative(self, linear: Linear, x, y):
        # Test the derivative method of the Linear function with different
        # inputs and outputs
        assert np.allclose(linear.derivative(x), y)
