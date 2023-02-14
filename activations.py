from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    """An abstract class for an activation function.

    An activation function is a function that maps an input to an output, usually in the range [0, 1] or [-1, 1].
    It is used to introduce non-linearity to a neural network and to control the output of a neuron.

    Attributes:
        None
    """

    # Define an abstract method for the function
    @abstractmethod
    def function(self, x: np.ndarray) -> np.ndarray:
        """An abstract method for the function.

        This method should take an input x and return the output of the activation function.

        Args:
            x: A numpy array of any shape, representing the input to the activation function.

        Returns:
            A numpy array of the same shape as x, representing the output of the activation function.
        """
        pass

    # Define an abstract method for the derivative
    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """An abstract method for the derivative.

        This method should take an input x and return the output of the derivative of the activation function.

        Args:
            x: A numpy array of any shape, representing the input to the activation function.

        Returns:
            A numpy array of the same shape as x, representing the output of the derivative of the activation function.
        """
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """A method to make the Activation object callable like a function.

        This method overrides the __call__ method of the Object class, and returns the output of the function method.

        Args:
            x: A numpy array of any shape, representing the input to the activation function.

        Returns:
            A numpy array of the same shape as x, representing the output of the activation function.
        """
        return self.function(x)


class ReLU(Activation):
    def function(self, x: np.ndarray) -> np.ndarray:
        """A method for the ReLU function.

        This method implements the ReLU function using the np.maximum function.

        Args:
            x: A numpy array of any shape, representing the input to the ReLU function.

        Returns:
            A numpy array of the same shape as x, representing the output of the ReLU function.
        """
        return np.maximum(x, 0)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """A method for the derivative of the ReLU function.

        This method implements the derivative of the ReLU function using the np.where function.

        Args:
            x: A numpy array of any shape, representing the input to the ReLU function.

        Returns:
            A numpy array of the same shape as x, representing the output of the derivative of the ReLU function.
        """
        return np.where(x > 0, 1, 0)


class LeakyReLU(Activation):
    def __init__(self, alpha: float = 0.01) -> None:
        """A method to initialize the LeakyReLU object with a given alpha parameter.

        Args:
            alpha: A float value, representing the slope of the negative part of the LeakyReLU function. Default is 0.01.

        Returns:
            None
        """
        super().__init__()
        self.alpha = alpha

    def function(self, x: np.ndarray) -> np.ndarray:
        """A method for the LeakyReLU function.

        This method implements the LeakyReLU function using the np.where function: max(alpha * x, x)

        Args:
            x: A numpy array of any shape, representing the input to the LeakyReLU function.

        Returns:
            A numpy array of the same shape as x, representing the output of the LeakyReLU function.
        """
        return np.where(x > 0, x, self.alpha * x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """A method for the derivative of the LeakyReLU function.

        This method implements the derivative of the LeakyReLU function using the np.where function: 1 if x > 0 else alpha

        Args:
            x: A numpy array of any shape, representing the input to the LeakyReLU function.

        Returns:
            A numpy array of the same shape as x, representing the output of the derivative of the LeakyReLU function.
        """
        return np.where(x > 0, 1, self.alpha)


class Tanh(Activation):
    def function(self, x: np.ndarray) -> np.ndarray:
        """A method for the tanh function.

        This method implements the tanh function using the np.tanh function: (exp(x) - exp(-x)) / (exp(x) + exp(-x))

        Args:
            x: A numpy array of any shape, representing the input to the tanh function.

        Returns:
            A numpy array of the same shape as x, representing the output of the tanh function.
        """
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """A method for the derivative of the tanh function.

        This method implements the derivative of the tanh function using the np.tanh function: 1 - tanh(x)^2

        Args:
            x: A numpy array of any shape, representing the input to the tanh function.

        Returns:
            A numpy array of the same shape as x, representing the output of the derivative of the tanh function.
        """
        return 1 - np.tanh(x) ** 2


class Linear(Activation):
    """A class for the linear activation function.

    This class implements the linear activation function and its derivative,
    which are both simply the identity function.
    """

    def function(self, x: np.ndarray) -> np.ndarray:
        """The linear activation function.

        This method implements the linear activation function: `x`.

        Args:
            x: A numpy array of any shape, representing the input to the activation function.

        Returns:
            A numpy array of the same shape as x, representing the output of the activation function.
        """
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """The derivative of the linear activation function.

        This method implements the derivative of the linear activation function: `1`.

        Args:
            x: A numpy array of any shape, representing the input to the activation function.

        Returns:
            A numpy array of the same shape as x, representing the output of the derivative of the activation function.
        """
        return np.ones_like(x)


# Define a subclass that inherits from the Activation abstract class
class Sigmoid(Activation):
    # Implement the abstract methods
    def function(self, x):
        """A method for the sigmoid function.

        This method implements the sigmoid function: 1 / (1 + exp(-x))

        Args:
            x: A numpy array of any shape, representing the input to the sigmoid function.

        Returns:
            A numpy array of the same shape as x, representing the output of the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        """A method for the derivative of the sigmoid function.

        This method implements the derivative of the sigmoid function: sigmoid(x) * (1 - sigmoid(x))

        Args:
            x: A numpy array of any shape, representing the input to the sigmoid function.

        Returns:
            A numpy array of the same shape as x, representing the output of the derivative of the sigmoid function.
        """
        return self.function(x) * (1 - self.function(x))


class SoftMax(Activation):
    """A subclass that implements the softmax activation function and its derivative."""

    def function(self, x: np.ndarray) -> np.ndarray:
        """A method for the softmax function.

        This method implements the softmax function using the np.exp and np.sum functions: exp(x) / sum(exp(x))

        Args:
            x: A numpy array of any shape, representing the input to the softmax function.

        Returns:
            A numpy array of the same shape as x, representing the output of the softmax function.
        """
        # Compute the exponential of each element of x
        exp_x = np.exp(x)
        # Compute the sum of the exponential of x along the last axis
        sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
        # Divide each element of exp_x by the sum of exp_x
        return exp_x / sum_exp_x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """A method for the derivative of the softmax function.

        This method implements the derivative of the softmax function using the np.diag and np.exp functions:
        softmax(x) * (1 - softmax(x))

        Args:
            x: A numpy array of any shape, representing the input to the softmax function.

        Returns:
            A numpy array of the same shape as x, representing the output of the derivative of the softmax function.
        """
        # Compute the softmax of x
        s = self.function(x)
        # Initialize the Jacobian matrix
        jacobian = np.diag(s)
        # Loop over the rows and columns of the Jacobian matrix
        for i in range(len(jacobian)):
            for j in range(len(jacobian)):
                # Apply the formula for the derivative of the softmax function
                if i == j:
                    jacobian[i][j] = s[i] * (1 - s[i])
                else:
                    jacobian[i][j] = -s[i] * s[j]
        # Return the Jacobian matrix
        return jacobian
