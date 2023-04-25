from typing import Tuple
import numpy as np


from .layer import Layer


class BatchNormalization(Layer):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        momentum: float = 0.9,
        epsilon: float = 1e-5,
    ) -> None:
        """Initialize the BatchNormalization layer with the given arguments.

        Args:
            input_dim: The dimension of the input features.
            output_dim: The dimension of the output features.
            momentum: The momentum for the moving average of the running
            statistics. Defaults to 0.9.
            epsilon: The small constant for numerical stability.
              Defaults to 1e-5.
        """
        super().__init__(input_dim, output_dim, None)
        self.normalized_input = None
        self.batch_var = None
        self.batch_mean = None
        self.gamma = np.ones((output_dim,))
        self.beta = np.zeros((output_dim,))
        self.running_mean = np.zeros((output_dim,))
        self.running_var = np.ones((output_dim,))
        self.momentum = momentum
        self.epsilon = epsilon
        self.parameters = {"gamma": self.gamma, "beta": self.beta}
        self.gradients = {
            "gamma": np.zeros_like(self.gamma),
            "beta": np.zeros_like(self.beta),
        }

    def forward(self, inputs: np.ndarray, training: bool = False) -> np.ndarray:
        """Perform the layer operation on the inputs.

        Args:
            inputs: The input array of shape (batch_size, input_dim).
            training: The training flag, True or False.

        Returns:
            The output array of shape (batch_size, output_dim).
        """
        self.previous_layer_output = inputs
        if training:
            batch_mean = np.mean(inputs, axis=0)
            batch_var = np.var(inputs, axis=0)
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * batch_var
            )
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var
        normalized_input = (inputs - batch_mean) / np.sqrt(batch_var + self.epsilon)
        self.batch_mean = batch_mean
        self.batch_var = batch_var
        self.normalized_input = normalized_input
        output = self.gamma * normalized_input + self.beta
        return output

    def backward(
        self,
        error: np.ndarray,
        prev_output: np.ndarray = None,
        training: bool = False,
    ) -> np.ndarray:
        """Perform the backward propagation on the layer.

        Args:
            training: A boolean indicating whether the layer is in training
            error: The error array of shape (batch_size, output_dim).
            prev_output: The previous layer output, not used in this layer.

        Returns:
            The gradient array of shape (batch_size, input_dim).
        """
        batch_size = error.shape[0]
        grad_gamma = np.sum(error * self.normalized_input, axis=0)
        grad_beta = np.sum(error, axis=0)

        self.gradients["gamma"] = grad_gamma
        self.gradients["beta"] = grad_beta

        grad_normalized_input = error * self.gamma
        grad_input = (
            batch_size * grad_normalized_input
            - np.sum(grad_normalized_input, axis=0)
            - self.normalized_input
            * np.sum(grad_normalized_input * self.normalized_input, axis=0)
        ) / (batch_size * np.sqrt(self.batch_var + self.epsilon))
        grad_prev_output = grad_input
        return grad_prev_output

    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the weights and biases of the layer as numpy arrays."""
        return self.gamma, self.beta
