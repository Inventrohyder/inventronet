import pytest
import numpy as np
from inventronet.activations import Sigmoid, ReLU, LeakyReLU, Tanh, Linear, SoftMax


class TestSigmoid:
    # create a fixture for the sigmoid function
    @pytest.fixture
    def sigmoid(self) -> Sigmoid:
        yield Sigmoid()

    # test the function method of the sigmoid function with different inputs and outputs
    @pytest.mark.parametrize(
        "x, y",
        [
            (np.array([0, 1, -1]), np.array([0.5, 0.731, 0.269])),
            (np.array([2, -2, 0.5]), np.array([0.881, 0.119, 0.622])),
        ],
    )
    def test_sigmoid_function(self, sigmoid: Sigmoid, x, y):
        assert np.allclose(sigmoid.function(x), y, atol=0.1)


class TestReLU:
    # create a fixture for the ReLU function
    @pytest.fixture
    def relu(self) -> ReLU:
        yield ReLU()

    # test the function method of the ReLU function with different inputs and outputs
    @pytest.mark.parametrize(
        "x, y",
        [
            (np.array([0, 1, -1]), np.array([0, 1, 0])),
            (np.array([2, -2, 0.5]), np.array([2, 0, 0.5])),
        ],
    )
    def test_relu_function(self, relu: ReLU, x, y):
        assert np.allclose(relu.function(x), y)

    # test the derivative method of the ReLU function with different inputs and outputs
    @pytest.mark.parametrize(
        "x, y",
        [
            (np.array([0, 1, -1]), np.array([0, 1, 0])),
            (np.array([2, -2, 0.5]), np.array([1, 0, 1])),
        ],
    )
    def test_relu_derivative(self, relu: ReLU, x, y):
        assert np.allclose(relu.derivative(x), y)


class TestLeakyReLU:
    # create a fixture for the LeakyReLU function
    @pytest.fixture
    def leaky_relu(self) -> LeakyReLU:
        yield LeakyReLU()

    # create a LeakyReLU object with alpha = 0.1
    @pytest.fixture
    def leaky_relu_01(self):
        return LeakyReLU(alpha=0.1)

    # test the function method of the LeakyReLU function with different inputs and outputs
    @pytest.mark.parametrize(
        "x, y",
        [
            (np.array([0, 1, -1]), np.array([0, 1, -0.01])),
            (np.array([2, -2, 0.5]), np.array([2, -0.02, 0.5])),
        ],
    )
    def test_leaky_relu_function(self, leaky_relu: LeakyReLU, x, y):
        assert np.allclose(leaky_relu.function(x), y)

    # test the derivative method of the LeakyReLU function with different inputs and outputs
    @pytest.mark.parametrize(
        "x, y",
        [
            (np.array([0, 1, -1]), np.array([0.01, 1, 0.01])),
            (np.array([2, -2, 0.5]), np.array([1, 0.01, 1])),
        ],
    )
    def test_leaky_relu_derivative(self, leaky_relu: LeakyReLU, x, y):
        assert np.allclose(leaky_relu.derivative(x), y)

    # test the function method of the LeakyReLU function with different inputs and outputs
    @pytest.mark.parametrize(
        "x, y",
        [
            # test the function with a zero input, which should return zero
            (np.array([0, 1, -1]), np.array([0, 1, -0.1])),
            # test the function with a positive input, which should return the same input
            (np.array([2, -2, 0.5]), np.array([2, -0.2, 0.5])),
        ],
    )
    def test_leaky_relu_function_01(self, leaky_relu_01: LeakyReLU, x, y):
        assert np.allclose(leaky_relu_01.function(x), y)

    # test the derivative method of the LeakyReLU function with different inputs and outputs
    @pytest.mark.parametrize(
        "x, y",
        [
            # test the derivative with a zero input, which should return the alpha parameter
            (np.array([0, 1, -1]), np.array([0.1, 1, 0.1])),
            # test the derivative with a positive input, which should return one
            (np.array([2, -2, 0.5]), np.array([1, 0.1, 1])),
        ],
    )
    def test_leaky_relu_derivative_01(self, leaky_relu_01: LeakyReLU, x, y):
        assert np.allclose(leaky_relu_01.derivative(x), y)


class TestTanh:
    # create a fixture for the tanh function
    @pytest.fixture
    def tanh(self) -> Tanh:
        yield Tanh()

    # test the function method of the tanh function with different inputs and outputs
    @pytest.mark.parametrize(
        "x, y",
        [
            (np.array([0, 1, -1]), np.array([0, 0.762, -0.762])),
            (np.array([2, -2, 0.5]), np.array([0.964, -0.964, 0.462])),
        ],
    )
    def test_tanh_function(self, tanh: Tanh, x, y):
        assert np.allclose(tanh.function(x), y, atol=0.1)

    # test the derivative method of the tanh function with different inputs and outputs
    @pytest.mark.parametrize(
        "x, y",
        [
            (np.array([0, 1, -1]), np.array([1, 0.42, 0.42])),
            (np.array([2, -2, 0.5]), np.array([0.071, 0.071, 0.786])),
        ],
    )
    def test_tanh_derivative(self, tanh: Tanh, x, y):
        assert np.allclose(tanh.derivative(x), y, atol=0.1)


class TestLinear:
    # create a fixture for the linear function
    @pytest.fixture
    def linear(self) -> Linear:
        yield Linear()

    # test the function method of the linear function with different inputs and outputs
    @pytest.mark.parametrize(
        "x, y",
        [
            (np.array([0, 1, -1]), np.array([0, 1, -1])),
            (np.array([2, -2, 0.5]), np.array([2, -2, 0.5])),
        ],
    )
    def test_linear_function(self, linear: Linear, x, y):
        assert np.allclose(linear.function(x), y)

    # test the derivative method of the linear function with different inputs and outputs
    @pytest.mark.parametrize(
        "x, y",
        [
            (np.array([0, 1, -1]), np.array([1, 1, 1])),
            (np.array([2, -2, 0.5]), np.array([1, 1, 1])),
        ],
    )
    def test_linear_derivative(self, linear: Linear, x, y):
        assert np.allclose(linear.derivative(x), y)


class TestSoftMax:
    # create a fixture for the softmax function
    @pytest.fixture
    def softmax(self) -> SoftMax:
        yield SoftMax()

    # test the function method of the softmax function with different inputs and outputs
    @pytest.mark.parametrize(
        "x, y",
        [
            (np.array([0, 1, -1]), np.array([0.244, 0.665, 0.091])),
            (np.array([2, -2, 0.5]), np.array([0.785, 0.036, 0.179])),
        ],
    )
    def test_softmax_function(self, softmax: SoftMax, x, y):
        assert np.allclose(softmax.function(x), y, atol=0.1)

    # test the derivative method of the softmax function with different inputs and outputs
    @pytest.mark.parametrize(
        "x, y",
        [
            (np.array([0, 1, -1]), np.array([0.185, 0.222, 0.083])),
            (np.array([2, -2, 0.5]), np.array([0.168, 0.035, 0.147])),
        ],
    )
    def test_softmax_derivative(self, softmax: SoftMax, x, y):
        # compare the diagonal elements of the softmax derivative matrix with the vector y
        assert np.allclose(softmax.derivative(x), np.diag(y), atol=0.2)
