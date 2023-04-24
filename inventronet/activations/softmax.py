from typing import Optional
import numpy as np
from .activation import Activation


class SoftMax(Activation):
    """
    A class for the softmax activation function.

    The softmax activation function is defined as:
    softmax(x)_i = exp(x_i) / sum(exp(x_j)) for j = 1, ..., n

    It is a vector-wise function that normalizes the input vector to a
    probability distribution, where the sum of the output elements is 1.
    It is often used as the output layer of a neural network for
    multi-class classification problems.

    Attributes:
        None
    """

    # Define the function method
    def function(self, x: np.ndarray) -> np.ndarray:
        """
        A method for the function.

        This method implements the softmax function using the np.exp and
        np.sum functions:
        softmax(x)_i = exp(x_i) / sum(exp(x_j)) for j = 1, ..., n

        Args:
            x: A numpy array of shape (n,) or (m, n), representing the
            input to the softmax function.

        Returns:
            A numpy array of the same shape as x, representing the
            output of the softmax function.
        """
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
        return exp_x / sum_exp_x

    # Define the derivative method
    def derivative(
        self, x: np.ndarray, output: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        A method for the derivative.

        This method implements the derivative of the softmax function using
        the np.outer and np.exp functions:
        softmax(x)_i * (delta_i_j - softmax(x)_j)

        Args:
            x: A numpy array of shape (n,) or (m, n), representing the input
            to the softmax function.

        Returns:
            A numpy array of shape (n, n) or (m, n, n), representing the
            output of the derivative of the softmax function.
        """
        if output is None:
            output = self.function(x)

        outer_product = np.outer(output, output)
        outer_product = outer_product.reshape(*x.shape, *x.shape[-1:])
        diag_output = np.identity(x.shape[-1]) * output[..., np.newaxis]
        return diag_output - outer_product
