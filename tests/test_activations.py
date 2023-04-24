import pytest
import numpy as np
from inventronet.activations import Sigmoid, ReLU, LeakyReLU, Tanh, Linear, SoftMax


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
