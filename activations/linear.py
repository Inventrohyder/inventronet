from typing import Optional
import numpy as np
from .activation import Activation


class Linear(Activation):
    """
    A class for the linear activation function.

    This class implements the linear activation function and its derivative,
    which are both simply the identity function.
    """

    def function(self, x: np.ndarray) -> np.ndarray:
        """
        The linear activation function.

        This method implements the linear activation function: `x`.

        Args:
            x: A numpy array of any shape, representing the input to the
            activation function.

        Returns:
            A numpy array of the same shape as x, representing the output of
            the activation function.
        """
        return x

    def derivative(
        self, x: np.ndarray, output: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        The derivative of the linear activation function.

        This method implements the derivative of the linear activation
        function: `1`.

        Args:
            x: A numpy array of any shape, representing the input to
            the activation function.

        Returns:
            A numpy array of the same shape as x, representing the output
            of the derivative of the activation function.
        """
        return np.ones_like(x)
