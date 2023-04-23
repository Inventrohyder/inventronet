from typing import Optional
import numpy as np
from .activation import Activation


class LeakyReLU(Activation):
    def __init__(self, alpha: float = 0.01) -> None:
        """
        A method to initialize the LeakyReLU object with a given alpha
        parameter.

        Args:
            alpha: A float value, representing the slope of the negative part
            of the LeakyReLU function. Default is 0.01.

        Returns:
            None
        """
        super().__init__()
        self.alpha = alpha

    def function(self, x: np.ndarray) -> np.ndarray:
        """
        A method for the LeakyReLU function.

        This method implements the LeakyReLU function using the np.where
        function: max(alpha * x, x)

        Args:
            x: A numpy array of any shape, representing the input to the
            LeakyReLU function.

        Returns:
            A numpy array of the same shape as x, representing the output
            of the LeakyReLU function.
        """
        return np.where(x > 0, x, self.alpha * x)

    def derivative(
        self, x: np.ndarray, output: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        A method for the derivative of the LeakyReLU function.

        This method implements the derivative of the LeakyReLU function using
        the np.where function: 1 if x > 0 else alpha

        Args:
            x: A numpy array of any shape, representing the input to the
            LeakyReLU function.

        Returns:
            A numpy array of the same shape as x, representing the output of
            the derivative of the LeakyReLU function.
        """
        return np.where(x > 0, 1, self.alpha)
