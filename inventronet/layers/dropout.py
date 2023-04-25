import numpy as np

from .layer import Layer
from .shape_error import ShapeError


class Dropout(Layer):
    def __init__(self, dropout_rate: float, input_dim: int) -> None:
        """Initialize the dropout layer with the given dropout rate.

        Args:
            dropout_rate: The probability of dropping out each neuron.
        """
        super().__init__(input_dim=input_dim, output_dim=input_dim, activation=None)
        self.dropout_rate: float = dropout_rate
        self.mask: np.ndarray = None

    def forward(self, inputs: np.ndarray, training: bool = False) -> np.ndarray:
        """Perform the layer operation on the inputs.

        Args:
            training: A boolean indicating whether the layer is in training
            inputs: The input array of any shape.

        Returns:
            The output array of the same shape as the input.
        """
        self.previous_layer_output: np.ndarray = inputs

        if not isinstance(inputs, np.ndarray):
            raise ValueError("The input must be a numpy array.")

        if training:
            self.mask = np.random.rand(*inputs.shape) >= self.dropout_rate
            output = inputs * self.mask
            if self.dropout_rate != 1:
                output = output / (1 - self.dropout_rate)
            return output
        else:
            return inputs

    def backward(
        self,
        error: np.ndarray,
        prev_output: np.ndarray = None,
        training: bool = False,
    ) -> np.ndarray:
        """Perform the backpropagation on the error.

        Args:
            training:
            prev_output:
            error: The error array of the same shape as the output.

        Returns:
            The output error of the same shape as the input.
        """
        if not isinstance(error, np.ndarray):
            raise ValueError("The error must be a numpy array.")

        if error.shape != self.previous_layer_output.shape:
            raise ShapeError(
                f"The shape of the error {error.shape} does not match the "
                + "shape of the input {self.previous_layer_output.shape}."
            )

        if training:
            output_error = error * self.mask
            if self.dropout_rate != 1:
                output_error = output_error / (1 - self.dropout_rate)
            return output_error
        else:
            return error

    def get_parameters(self):
        return []
