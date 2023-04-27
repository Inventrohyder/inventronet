import numpy as np
import pytest

from inventronet.activations import Tanh


class TestTanh:
    # Test the Tanh activation function

    @pytest.fixture
    def tanh(self) -> Tanh:
        # Fixture for the Tanh function
        yield Tanh()

    @pytest.mark.parametrize(
        "x, y",
        [
            (np.array([0, 1, -1]), np.array([0, 0.7615942, -0.7615942])),
            (np.array([2, -2, 0.5]), np.array([0.9640276, -0.9640276, 0.4621172])),
        ],
    )
    def test_tanh_function(self, tanh: Tanh, x, y):
        # Test the function method of the Tanh function with different
        # inputs and outputs
        assert np.allclose(tanh.function(x), y, atol=1e-4)

    @pytest.mark.parametrize(
        "x, y",
        [
            (np.array([0, 1, -1]), np.array([1, 0.4199743, 0.4199743])),
            (np.array([2, -2, 0.5]), np.array([0.0706508, 0.0706508, 0.7864477])),
        ],
    )
    def test_tanh_derivative(self, tanh: Tanh, x, y):
        # Test the derivative method of the Tanh function with different
        # inputs and outputs
        assert np.allclose(tanh.derivative(x), y, atol=1e-4)
