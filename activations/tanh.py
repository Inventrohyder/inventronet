from typing import Optional
import numpy as np
from .activation import Activation


class Tanh(Activation):
    def function(self, x: np.ndarray) -> np.ndarray:
        """
        A method for the tanh function.

        This method implements the tanh function using the np.tanh function:
        (exp(x) - exp(-x)) / (exp(x) + exp(-x))

        Args:
            x: A numpy array of any shape, representing the input to the tanh
            function.

        Returns:
            A numpy array of the same shape as x, representing the output of
            the tanh function.
        """
        return np.tanh(x)

    def derivative(
        self, x: np.ndarray, output: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        A method for the derivative of the tanh function.

        This method implements the derivative of the tanh function using the
        np.tanh function: 1 - tanh(x)^2

        Args:
            x: A numpy array of any shape, representing the input to the
            tanh function.

        Returns:
            A numpy array of the same shape as x, representing the output
            of the derivative of the tanh function.
        """
        return 1 - np.tanh(x) ** 2
