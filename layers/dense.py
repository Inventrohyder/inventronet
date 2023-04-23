from typing import Callable, Dict, Tuple
import numpy as np

from ..activations.activation import Activation
from .layer import Layer


class Dense(Layer):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: Activation = None,
        use_bias: bool = True,
        weight_initializer: Callable = np.random.uniform,
        bias_initializer: Callable = np.zeros,
    ) -> None:
        """Initialize the dense layer with the given arguments.

        Args:
            input_dim: The dimension of the input features.
            output_dim: The dimension of the output features.
            activation: The activation function for the layer.
                Defaults to None.
            use_bias: Whether to use a bias term for the layer.
                Defaults to True.
            weight_initializer: The function to initialize the weight matrix.
                Defaults to np.random.uniform.
            bias_initializer: The function to initialize the bias vector.
                Defaults to np.zeros.
        """
        super().__init__(input_dim, output_dim, activation)
        self.use_bias: bool = use_bias
        self.weight_initializer: Callable = weight_initializer
        self.bias_initializer: Callable = bias_initializer

        self.weights: np.ndarray = 0.01 * self.weight_initializer(
            size=(self.input_dim, self.output_dim)
        )

        if self.use_bias:
            self.biases: np.ndarray = self.bias_initializer(shape=(self.output_dim,))
        self.parameters = {"weights": self.weights}
        if self.use_bias:
            self.parameters["biases"] = self.biases

        self.gradients: Dict[str, np.ndarray] = {"weights": None, "biases": None}

    def forward(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        """Perform the layer operation on the inputs.

        Args:
            inputs: The input array of shape (batch_size, input_dim).

        Returns:
            The output array of shape (batch_size, output_dim).
        """
        self.previous_layer_output: np.ndarray = inputs

        output = inputs @ self.weights

        if self.use_bias:
            output += self.biases

        if self.activation is not None:
            self.pre_activation_output = output
            return self.activation(output)
        else:
            return output

    def backward(
        self,
        error: np.ndarray,
        learning_rate: float,
        prev_output: np.ndarray = None,
        **kwargs,
    ) -> np.ndarray:
        if prev_output is None:
            prev_output = self.previous_layer_output
        else:
            self.previous_layer_output = prev_output

        if self.activation:
            activation_derivative = self.activation.derivative(
                self.pre_activation_output
            )
            if error.shape == activation_derivative.shape:
                error = error * activation_derivative
            elif len(error.shape) == 2 and len(activation_derivative.shape) == 3:
                error = np.einsum("ij,jk->ik", error, activation_derivative)
            elif len(error.shape) == 1 and len(activation_derivative.shape) == 2:
                error = np.dot(error, activation_derivative)
            else:
                raise ValueError(
                    "The shapes of the error and the activation derivative "
                    + "are incompatible."
                )

        weight_gradient = np.dot(prev_output.T, error)

        bias_gradient = np.sum(error, axis=0)

        if len(error.shape) == 2 and len(self.weights.shape) == 2:
            input_gradient = np.dot(error, self.weights.T)
        elif len(error.shape) == 2 and len(self.weights.shape) == 3:
            input_gradient = np.einsum("ij,jkl->ikl", error, self.weights)
        elif len(error.shape) == 1 and len(self.weights.shape) == 2:
            input_gradient = np.dot(error, self.weights.T).T
        else:
            raise ValueError(
                "The shapes of the error and the weights are incompatible."
            )

        self.weights -= learning_rate * weight_gradient
        if self.use_bias:
            if bias_gradient.shape == self.biases.shape:
                self.biases -= learning_rate * bias_gradient
            elif len(bias_gradient.shape) == 2 and len(self.biases.shape) == 1:
                self.biases -= learning_rate * np.broadcast_to(
                    bias_gradient, self.biases.shape
                )
            elif len(bias_gradient.shape) == 1 and len(self.biases.shape) == 2:
                self.biases -= learning_rate * bias_gradient.reshape(-1)
            else:
                raise ValueError(
                    "The shapes of the bias gradient and the biases "
                    + "are incompatible."
                )
        self.gradients["weights"] = weight_gradient
        self.gradients["biases"] = bias_gradient

        return input_gradient

    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the weights and biases of the dense layer as numpy arrays."""
        return self.weights.copy(), self.biases.copy()
