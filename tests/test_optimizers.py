from typing import Dict

import numpy as np
import pytest

from inventronet.optimizers import Adam, StochasticGradientDescent


@pytest.fixture
def params() -> Dict[str, np.ndarray]:
    yield {
        "weights": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "biases": np.array([0.5, 0.5]),
    }


@pytest.fixture
def gradients() -> Dict[str, np.ndarray]:
    yield {
        "weights": np.array([[0.1, 0.2], [0.3, 0.4]]),
        "biases": np.array([0.1, -0.1]),
    }


@pytest.fixture
def zero_params() -> Dict[str, np.ndarray]:
    yield {
        "weights": np.zeros((2, 2)),
        "biases": np.zeros(2),
    }


@pytest.fixture
def zero_gradients() -> Dict[str, np.ndarray]:
    yield {
        "weights": np.zeros((2, 2)),
        "biases": np.zeros(2),
    }


class TestGradientDescent:
    def test_gradient_descent_init_default(self):
        optimizer = StochasticGradientDescent()
        assert optimizer.learning_rate == 0.01

    def test_gradient_descent_init_custom(self):
        optimizer = StochasticGradientDescent(learning_rate=0.05)
        assert optimizer.learning_rate == 0.05

    def test_gradient_descent_update(
            self, params: Dict[str, np.ndarray], gradients: Dict[str, np.ndarray]
    ):
        optimizer = StochasticGradientDescent(learning_rate=0.1)

        optimizer.update(0, params, gradients)

        # Expected updated parameters after applying gradient descent
        expected_params = {
            "weights": np.array([[0.99, 1.98], [2.97, 3.96]]),
            "biases": np.array([0.49, 0.51]),
        }

        # Check if the updated parameters are equal to the expected values
        assert np.allclose(params["weights"], expected_params["weights"])
        assert np.allclose(params["biases"], expected_params["biases"])

    def test_gradient_descent_update_with_zeros(
            self, zero_params: Dict[str, np.ndarray], zero_gradients: Dict[str, np.ndarray]
    ):
        optimizer = StochasticGradientDescent(learning_rate=0.1)

        optimizer.update(0, zero_params, zero_gradients)

        # Expected updated parameters after applying gradient descent
        # with zero gradients
        expected_params = {
            "weights": np.zeros((2, 2)),
            "biases": np.zeros(2),
        }

        # Check if the updated parameters are equal to the expected values
        assert np.allclose(zero_params["weights"], expected_params["weights"])
        assert np.allclose(zero_params["biases"], expected_params["biases"])


class TestAdam:
    def test_adam_init_default(self):
        optimizer = Adam()
        assert optimizer.learning_rate == 0.001
        assert optimizer.beta1 == 0.9
        assert optimizer.beta2 == 0.999
        assert optimizer.epsilon == 1e-8

    def test_adam_init_custom(self):
        optimizer = Adam(learning_rate=0.01, beta1=0.8, beta2=0.99, epsilon=1e-7)
        assert optimizer.learning_rate == 0.01
        assert optimizer.beta1 == 0.8
        assert optimizer.beta2 == 0.99
        assert optimizer.epsilon == 1e-7

    def test_adam_update(
            self, params: Dict[str, np.ndarray], gradients: Dict[str, np.ndarray]
    ):
        optimizer = Adam(learning_rate=0.001)

        optimizer.update(0, params, gradients)

        # Corrected expected updated parameters after applying Adam optimizer (calculated manually)
        expected_params = {
            "weights": np.array([[0.999, 1.999], [2.999, 3.999]]),
            "biases": np.array([0.499, 0.501]),
        }

        # Check if the updated parameters are equal to the expected values
        assert np.allclose(
            params["weights"], expected_params["weights"], rtol=1e-7, atol=1e-7
        )
        assert np.allclose(
            params["biases"], expected_params["biases"], rtol=1e-7, atol=1e-7
        )

    def test_adam_update_with_zeros(
            self, zero_params: Dict[str, np.ndarray], zero_gradients: Dict[str, np.ndarray]
    ):
        optimizer = Adam(learning_rate=0.01)

        optimizer.update(0, zero_params, zero_gradients)

        # Expected updated parameters after applying Adam optimizer with
        # zero gradients
        expected_params = {
            "weights": np.zeros((2, 2)),
            "biases": np.zeros(2),
        }

        # Check if the updated parameters are equal to the expected values
        assert np.allclose(zero_params["weights"], expected_params["weights"], rtol=1e-7)
        assert np.allclose(zero_params["biases"], expected_params["biases"], rtol=1e-7)
