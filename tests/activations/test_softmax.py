import numpy as np
import pytest

from inventronet.activations import SoftMax


class TestSoftMax:
    # Test the SoftMax activation function

    @pytest.fixture
    def softmax(self) -> SoftMax:
        # Fixture for the SoftMax function
        yield SoftMax()

    @pytest.mark.parametrize(
        "x, y",
        [
            (
                np.array([0, 1, -1]),
                np.array([0.24472847, 0.66524096, 0.09003057]),
            ),
            (
                np.array([2, -2, 0.5]),
                np.array([0.80551241, 0.01475347, 0.17973411]),
            ),
        ],
    )
    def test_softmax_function(self, softmax: SoftMax, x, y):
        # Test the function method of the SoftMax function with different
        # inputs and outputs
        assert np.allclose(softmax.function(x), y, atol=1e-4)

    @pytest.mark.parametrize(
        "x, y",
        [
            (
                np.array([0, 1, -1]),
                np.array(
                    [
                        [0.18483646, -0.1628034, -0.02203305],
                        [-0.1628034, 0.22269507, -0.05989167],
                        [-0.02203305, -0.05989167, 0.08192472],
                    ]
                ),
            ),
            (
                np.array([2, -2, 0.5]),
                np.array(
                    [
                        [0.15666217, -0.01188411, -0.14477806],
                        [-0.01188411, 0.01453581, -0.0026517],
                        [-0.14477806, -0.0026517, 0.14742976],
                    ]
                ),
            ),
        ],
    )
    def test_softmax_derivative(self, softmax: SoftMax, x, y):
        # Test the derivative method of the SoftMax function with different
        # inputs and outputs
        jacobian = softmax.derivative(x)
        assert np.allclose(jacobian, y, atol=1e-4)
