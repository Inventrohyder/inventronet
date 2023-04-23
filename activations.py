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


# Define a subclass that inherits from the Activation abstract class
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
