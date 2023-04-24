from typing import Optional
import numpy as np
from .activation import Activation


class ReLU(Activation):
    def function(self, x: np.ndarray) -> np.ndarray:
        """
        A method for the ReLU function.

        This method implements the ReLU function using the np.maximum
        function.

        Args:
            x: A numpy array of any shape, representing the input to the
            ReLU function.

        Returns:
            A numpy array of the same shape as x, representing the output of
            the ReLU function.
        """
        return np.maximum(x, 0)

    def derivative(
        self, x: np.ndarray, output: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        A method for the derivative of the ReLU function.

        This method implements the derivative of the ReLU function using
        the np.where function.

        Args:
            x: A numpy array of any shape, representing the input to the
            ReLU function.

        Returns:
            A numpy array of the same shape as x, representing the output of
            the derivative of the ReLU function.
        """
        return np.where(x > 0, 1, 0)
