import numpy as np
import pytest

from inventronet.activations import Sigmoid


class TestSigmoid:
    # Test the Sigmoid activation function

    @pytest.fixture
    def sigmoid(self) -> Sigmoid:
        # Fixture for the Sigmoid function
        yield Sigmoid()

    @pytest.mark.parametrize(
        "x, y",
        [
            (np.array([0, 1, -1]), np.array([0.5, 0.7310586, 0.2689414])),
            (np.array([2, -2, 0.5]), np.array([0.8807971, 0.1192029, 0.6224593])),
        ],
    )
    def test_sigmoid_function(self, sigmoid: Sigmoid, x, y):
        # Test the function method of the Sigmoid function with different
        # inputs and outputs
        assert np.allclose(sigmoid.function(x), y, atol=1e-4)
