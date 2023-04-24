import numpy as np

from inventronet.optimizers import Adam, StochasticGradientDescent


class TestGradientDescent:
    def test_gradient_descent_init_default(self):
        optimizer = StochasticGradientDescent()
        assert optimizer.learning_rate == 0.01

    def test_gradient_descent_init_custom(self):
        optimizer = StochasticGradientDescent(learning_rate=0.05)
        assert optimizer.learning_rate == 0.05

    def test_gradient_descent_update(self):
        optimizer = StochasticGradientDescent(learning_rate=0.1)

        # Initialize parameters and gradients with some sample values
        params = {
            "w": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "b": np.array([0.5, 0.5]),
        }
        gradients = {
            "w": np.array([[0.1, 0.2], [0.3, 0.4]]),
            "b": np.array([0.1, -0.1]),
        }

        optimizer.update(params, gradients)

        # Expected updated parameters after applying gradient descent
        expected_params = {
            "w": np.array([[0.99, 1.98], [2.97, 3.96]]),
            "b": np.array([0.49, 0.51]),
        }

        # Check if the updated parameters are equal to the expected values
        assert np.allclose(params["w"], expected_params["w"])
        assert np.allclose(params["b"], expected_params["b"])

    def test_gradient_descent_update_with_zeros(self):
        optimizer = StochasticGradientDescent(learning_rate=0.1)

        # Initialize parameters and gradients with zeros
        params = {
            "w": np.zeros((2, 2)),
            "b": np.zeros(2),
        }
        gradients = {
            "w": np.zeros((2, 2)),
            "b": np.zeros(2),
        }

        optimizer.update(params, gradients)

        # Expected updated parameters after applying gradient descent with zero gradients
        expected_params = {
            "w": np.zeros((2, 2)),
            "b": np.zeros(2),
        }

        # Check if the updated parameters are equal to the expected values
        assert np.allclose(params["w"], expected_params["w"])
        assert np.allclose(params["b"], expected_params["b"])


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

    def test_adam_update(self):
        optimizer = Adam(learning_rate=0.001)

        # Initialize parameters and gradients with some sample values
        params = {
            "w": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "b": np.array([0.5, 0.5]),
        }
        gradients = {
            "w": np.array([[0.1, 0.2], [0.3, 0.4]]),
            "b": np.array([0.1, -0.1]),
        }

        optimizer.update(params, gradients)

        # Corrected expected updated parameters after applying Adam optimizer (calculated manually)
        expected_params = {
            "w": np.array([[0.999, 1.999], [2.999, 3.999]]),
            "b": np.array([0.499, 0.501]),
        }

        # Check if the updated parameters are equal to the expected values
        assert np.allclose(params["w"], expected_params["w"], rtol=1e-7, atol=1e-7)
        assert np.allclose(params["b"], expected_params["b"], rtol=1e-7, atol=1e-7)

    def test_adam_update_with_zeros(self):
        optimizer = Adam(learning_rate=0.01)

        # Initialize parameters and gradients with zeros
        params = {
            "w": np.zeros((2, 2)),
            "b": np.zeros(2),
        }
        gradients = {
            "w": np.zeros((2, 2)),
            "b": np.zeros(2),
        }

        optimizer.update(params, gradients)

        # Expected updated parameters after applying Adam optimizer with
        # zero gradients
        expected_params = {
            "w": np.zeros((2, 2)),
            "b": np.zeros(2),
        }

        # Check if the updated parameters are equal to the expected values
        assert np.allclose(params["w"], expected_params["w"], rtol=1e-7)
        assert np.allclose(params["b"], expected_params["b"], rtol=1e-7)
