from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class Activation(ABC):
    """
    An abstract class for an activation function.

    An activation function is a function that maps an input to an output,
    usually in the range (0, 1) or (-1, 1).
    It is used to introduce non-linearity to a neural network and to control
    the output of a neuron.

    Attributes:
        None
    """

    @abstractmethod
    def function(self, x: np.ndarray) -> np.ndarray:
        """
        An abstract method for the function.

        This method should take an input x and return the output of the
        activation function.

        Args:
            x: A numpy array of shape (n,) or (m, n), representing the input
            to the activation function.
            output: Optional precomputed output of the activation function
            to avoid redundant computation in the derivative calculation.

        Returns:
            A numpy array of the same shape as x, representing the output of
            the activation function.
        """
        raise NotImplementedError("Activation.function is not implemented.")

    @abstractmethod
    def derivative(
        self, x: np.ndarray, output: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        An abstract method for the derivative.

        This method should take an input x and return the output of the
        derivative of the activation function.

        Args:
            x: A numpy array of shape (n,) or (m, n), representing the input
            to the activation function.

        Returns:
            A numpy array of shape (n, n) or (m, n, n), representing the
            output of the derivative of the activation function.
        """
        raise NotImplementedError("Activation.derivative is not implemented.")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        A method to make the Activation object callable like a function.

        This method overrides the __call__ method of the Object class, and
        returns the output of the function method.

        Args:
            x: A numpy array of any shape, representing the input to the
            activation function.

        Returns:
            A numpy array of the same shape as x, representing the output
            of the activation function.
        """
        return self.function(x)
