from abc import ABC, abstractmethod

from typing import Tuple
import numpy as np

from ..activations.activation import Activation


class Layer(ABC):
    @abstractmethod
    def __init__(
        self, input_dim: int, output_dim: int, activation: Activation, **kwargs
    ) -> None:
        """Initialize the layer with the given arguments.

        Args:
            input_dim: The dimension of the input features.
            output_dim: The dimension of the output features.
            activation: The activation function for the layer.
            **kwargs: The keyword arguments for the layer.
        """
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.activation: Activation = activation
        self.kwargs = kwargs
        self.parameters = None
        self.gradients = None

        self.previous_layer_output = None

    @abstractmethod
    def forward(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        """Perform the layer operation on the inputs.

        Args:
            inputs: The input array of shape (batch_size, input_dim).

        Returns:
            The output array of shape (batch_size, output_dim).
        """
        raise NotImplementedError("Layer.forward is not implemented.")

    @abstractmethod
    def backward(
        self,
        grad: np.ndarray,
        learning_rate: float,
        prev_output: np.ndarray = None,
        **kwargs,
    ) -> np.ndarray:
        """Perform the backward propagation on the layer.

        Args:
            error: The error array of shape (batch_size, output_dim).
            learning_rate: The learning rate for the gradient update.
            prev_output: The previous output array of
                shape (batch_size, input_dim).

        Returns:
            The propagated error array of shape (batch_size, input_dim).
        """
        raise NotImplementedError("Layer.backward is not implemented.")

    @abstractmethod
    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the weights and biases of the layer as numpy arrays."""
        raise NotImplementedError("Layer.get_parameters is not implemented.")

    @property
    def name(self) -> str:
        """Return the name of the layer.

        Returns:
            The name of the layer as a string.
        """
        return self.__class__.__name__

    @property
    def input_shape(self) -> tuple:
        """Return the input shape of the layer.

        Returns:
            The input shape of the layer as a tuple of integers.
        """
        return (self.input_dim,)

    @property
    def output_shape(self) -> tuple:
        """Return the output shape of the layer.

        Returns:
            The output shape of the layer as a tuple of integers.
        """
        return (self.output_dim,)

    def summary(self) -> None:
        """Print the summary of the layer.

        Prints the layer name, input shape, output shape, and number of
             parameters.
        """
        print(f"Layer: {self.name}")
        print(f"Input shape: {self.input_shape}")
        print(f"Output shape: {self.output_shape}")
        if self.parameters is not None:
            print(
                "Number of parameters: "
                + f"{sum(param.size for param in self.parameters.values())}"
            )
