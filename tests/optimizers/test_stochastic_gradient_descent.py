from typing import Dict

import numpy as np

from inventronet.optimizers import StochasticGradientDescent


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
