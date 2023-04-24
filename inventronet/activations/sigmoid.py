from typing import Optional
import numpy as np
from .activation import Activation


class Sigmoid(Activation):
    # Implement the abstract methods
    def function(self, x: np.ndarray) -> np.ndarray:
        """
        A method for the sigmoid function.

        This method implements the sigmoid function: 1 / (1 + exp(-x))

        Args:
            x: A numpy array of any shape, representing the input to
            the sigmoid function.

        Returns:
            A numpy array of the same shape as x, representing the output
            of the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))

    def derivative(
        self, x: np.ndarray, output: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        A method for the derivative of the sigmoid function.

        This method implements the derivative of the sigmoid function:
        sigmoid(x) * (1 - sigmoid(x))

        Args:
            x: A numpy array of any shape, representing the input to the
            sigmoid function.

        Returns:
            A numpy array of the same shape as x, representing the output of
            the derivative of the sigmoid function.
        """
        if output is None:
            output = self.function(x)
        return output * (1 - output)
