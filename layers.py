from abc import ABC, abstractmethod

from typing import Callable, Dict, Tuple
import numpy as np

from .activations import Activation


# Define a custom exception class for shape mismatch
class ShapeError(Exception):
    """An exception class for shape mismatch."""

    def __init__(self, message: str):
        """Initialize the exception with a message."""
        super().__init__(message)


# Define an abstract base class for a layer
class Layer(ABC):
    # Define an abstract method for the initialization
    @abstractmethod
    def __init__(
        self, input_dim: int, output_dim: int, activation: Activation, **kwargs
    ) -> None:
        """Initialize the layer with the given arguments.

        Args:
            input_dim: The dimension of the input features.
            output_dim: The dimension of the output features.
            activation: The activation function for the layer.
            **kwargs: The keyword arguments for the layer.
        """
        # Store the arguments as attributes
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.activation: Activation = activation
        # Store the keyword arguments as a dictionary
        self.kwargs = kwargs
        # Initialize the parameters and gradients as None
        # They will be set later by the subclasses
        self.parameters = None
        self.gradients = None

        # Initialize the previous layer output
        self.previous_layer_output = None

    # Define an abstract method for the forward propagation
    @abstractmethod
    def forward(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        """Perform the layer operation on the inputs.

        Args:
            inputs: The input array of shape (batch_size, input_dim).

        Returns:
            The output array of shape (batch_size, output_dim).
        """
        # Perform the layer operation on the inputs
        pass

    # Define the backward abstract method
    @abstractmethod
    def backward(
        self,
        grad: np.ndarray,
        learning_rate: float,
        prev_output: np.ndarray = None,
        **kwargs,
    ) -> np.ndarray:
        """Perform the backward propagation on the layer.

        Args:
            error: The error array of shape (batch_size, output_dim).
            learning_rate: The learning rate for the gradient update.
            prev_output: The previous output array of
                shape (batch_size, input_dim).

        Returns:
            The propagated error array of shape (batch_size, input_dim).
        """
        # Perform the backward propagation on the layer
        # This will be implemented by the subclasses
        pass

    @abstractmethod
    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the weights and biases of the layer as numpy arrays."""
        pass

    @property
    def name(self) -> str:
        """Return the name of the layer.

        Returns:
            The name of the layer as a string.
        """
        return self.__class__.__name__

    @property
    def input_shape(self) -> tuple:
        """Return the input shape of the layer.

        Returns:
            The input shape of the layer as a tuple of integers.
        """
        return (self.input_dim,)

    @property
    def output_shape(self) -> tuple:
        """Return the output shape of the layer.

        Returns:
            The output shape of the layer as a tuple of integers.
        """
        return (self.output_dim,)

    # Define a concrete method for the summary
    def summary(self) -> None:
        """Print the summary of the layer.

        Prints the layer name, input shape, output shape, and number of
             parameters.
        """
        # Print the layer name, input shape, and output shape
        print(f"Layer: {self.name}")
        print(f"Input shape: {self.input_shape}")
        print(f"Output shape: {self.output_shape}")
        # Print the number of parameters if they are set
        if self.parameters is not None:
            print(
                f"Number of parameters: {sum(param.size for param in self.parameters.values())}"
            )


# Define a class for a dense layer
class Dense(Layer):
    # Define the initialization method
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: Activation = None,
        use_bias: bool = True,
        weight_initializer: Callable = np.random.uniform,
        bias_initializer: Callable = np.zeros,
    ) -> None:
        """Initialize the dense layer with the given arguments.

        Args:
            input_dim: The dimension of the input features.
            output_dim: The dimension of the output features.
            activation: The activation function for the layer.
                Defaults to None.
            use_bias: Whether to use a bias term for the layer.
                Defaults to True.
            weight_initializer: The function to initialize the weight matrix.
                Defaults to np.random.uniform.
            bias_initializer: The function to initialize the bias vector.
                Defaults to np.zeros.
        """
        # Call the superclass constructor with the input and output dimensions
        super().__init__(input_dim, output_dim, activation)
        # Store the arguments as attributes
        self.use_bias: bool = use_bias
        self.weight_initializer: Callable = weight_initializer
        self.bias_initializer: Callable = bias_initializer

        # Initialize the weight and bias matrices with the initializer functions
        # Initialize the weight matrix with a normal distribution and a small factor
        # to avoid large values that could cause numerical instability
        self.weights: np.ndarray = 0.01 * self.weight_initializer(
            size=(self.input_dim, self.output_dim)
        )

        # Initialize the bias vector only if use_bias is True
        if self.use_bias:
            self.biases: np.ndarray = self.bias_initializer(shape=(self.output_dim,))
        # Store the weight and bias matrices as parameters in a dictionary
        self.parameters = {"weights": self.weights}
        if self.use_bias:
            self.parameters["biases"] = self.biases

        # Initialize the gradients for the weight and bias matrices as None
        # They will be updated during the backward propagation
        self.gradients: Dict[str, np.ndarray] = {"weights": None, "biases": None}

    def forward(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        """Perform the layer operation on the inputs.

        Args:
            inputs: The input array of shape (batch_size, input_dim).

        Returns:
            The output array of shape (batch_size, output_dim).
        """
        # Store the input as an attribute
        self.previous_layer_output: np.ndarray = inputs

        # Compute the linear combination of the inputs and the weights
        output = inputs @ self.weights

        # Add the bias if needed
        if self.use_bias:
            output += self.biases

        # Apply the activation function if any
        if self.activation is not None:
            # Store the output before activation as an attribute
            self.pre_activation_output = output
            # Return the output after activation
            return self.activation(output)
        else:
            # Return the output without activation
            return output

    def backward(
        self,
        error: np.ndarray,
        learning_rate: float,
        prev_output: np.ndarray = None,
        **kwargs,
    ) -> np.ndarray:
        """Perform the backward propagation on the layer.

        Args:
            error: The error array of shape (batch_size, output_dim).
            learning_rate: The learning rate for the gradient update.
            prev_output: The previous output array of
                shape (batch_size, input_dim).

        Returns:
            The propagated error array of shape (batch_size, input_dim).
        """
        # Check if the prev_output argument is given
        if prev_output is None:
            # Use the stored previous layer output as the default value
            prev_output = self.previous_layer_output
        else:
            # Store the given previous layer output as an attribute
            self.previous_layer_output = prev_output

        # Compute the error with respect to the activation function if any
        if self.activation:
            # Use the stored output before activation to compute the derivative
            error = error * self.activation.derivative(self.pre_activation_output)

        # Calculate the gradient of the error with respect to the weights
        weight_gradient = np.dot(prev_output.T, error)

        # Calculate the gradient of the error with respect to the biases
        bias_gradient = np.sum(error, axis=0, keepdims=True)

        # Calculate the gradient of the error with respect to the inputs
        input_gradient = np.dot(error, self.weights.T)

        # Update the weights and biases using the gradient descent algorithm
        self.weights -= learning_rate * weight_gradient
        if self.use_bias:
            self.biases -= learning_rate * bias_gradient.reshape(-1)

        # Store the gradients in the dictionary
        self.gradients["weights"] = weight_gradient
        self.gradients["biases"] = bias_gradient

        # Return the error of the previous layer
        return input_gradient

    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the weights and biases of the dense layer as numpy arrays."""
        return self.weights.copy(), self.biases.copy()


# Define the Dropout class that inherits from the Layer class
class Dropout(Layer):
    # Define the initialization method
    def __init__(self, dropout_rate: float, input_dim: int) -> None:
        """Initialize the dropout layer with the given dropout rate.

        Args:
            dropout_rate: The probability of dropping out each neuron.
        """
        # Call the superclass constructor with the input and output dimensions
        # equal and activation as None
        super().__init__(input_dim=input_dim, output_dim=input_dim, activation=None)
        # Store the dropout rate as an attribute
        self.dropout_rate: float = dropout_rate
        # Initialize the dropout mask as None
        self.mask: np.ndarray = None

    # Define the forward method
    def forward(self, inputs: np.ndarray, training: bool = None) -> np.ndarray:
        """Perform the layer operation on the inputs.

        Args:
            inputs: The input array of any shape.

        Returns:
            The output array of the same shape as the input.
        """
        # Store the input as an attribute
        self.previous_layer_output: np.ndarray = inputs

        # Check the type of the input
        if not isinstance(inputs, np.ndarray):
            raise ValueError("The input must be a numpy array.")

        # Check the mode of the layer
        if training:
            # Generate a random dropout mask of the same shape as the input
            self.mask = np.random.rand(*inputs.shape) >= self.dropout_rate
            output = inputs * self.mask
            # Check if the dropout rate is 1
            if self.dropout_rate != 1:
                # Apply the dropout mask to the input and scale up by the inverse of the dropout rate
                output = output / (1 - self.dropout_rate)
            # Return the output
            return output
        else:
            # Return the input as it is
            return inputs

    # Define the backward method
    def backward(
        self,
        error: np.ndarray,
        learning_rate: float,
        prev_output: np.ndarray = None,
        training: bool = None,
    ) -> np.ndarray:
        """Perform the backpropagation on the error.

        Args:
            error: The error array of the same shape as the output.
            learning_rate: The learning rate for the weight update.

        Returns:
            The output error of the same shape as the input.
        """
        # Check the type of the error
        if not isinstance(error, np.ndarray):
            raise ValueError("The error must be a numpy array.")

        # Check the shape of the error
        if error.shape != self.previous_layer_output.shape:
            raise ShapeError(
                f"The shape of the error {error.shape} does not match the shape of the input {self.previous_layer_output.shape}."
            )

        # Check the mode of the layer
        if training:
            # Apply the dropout mask to the error and scale up by the inverse of the dropout rate
            output_error = error * self.mask
            if self.dropout_rate != 1:
                output_error = output_error / (1 - self.dropout_rate)
            # Return the output error
            return output_error
        else:
            # Return the error as it is
            return error

    # Define the get_parameters method with no arguments
    def get_parameters(self):
        # Return an empty list, since the dropout layer has no parameters
        return []


class BatchNormalization(Layer):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        momentum: float = 0.9,
        epsilon: float = 1e-5,
    ) -> None:
        """Initialize the BatchNormalization layer with the given arguments.

        Args:
            input_dim: The dimension of the input features.
            output_dim: The dimension of the output features.
            momentum: The momentum for the moving average of the running statistics. Defaults to 0.9.
            epsilon: The small constant for numerical stability. Defaults to 1e-5.
        """
        # Call the initialization method of the layer class
        super().__init__(input_dim, output_dim, None)
        # Initialize the gamma parameter with ones
        self.gamma = np.ones((output_dim,))
        # Initialize the beta parameter with zeros
        self.beta = np.zeros((output_dim,))
        # Initialize the running_mean parameter with zeros
        self.running_mean = np.zeros((output_dim,))
        # Initialize the running_var parameter with ones
        self.running_var = np.ones((output_dim,))
        # Store the momentum and epsilon arguments as attributes
        self.momentum = momentum
        self.epsilon = epsilon
        # Set the parameters and gradients as dictionaries
        self.parameters = {"gamma": self.gamma, "beta": self.beta}
        self.gradients = {
            "gamma": np.zeros_like(self.gamma),
            "beta": np.zeros_like(self.beta),
        }

    def forward(self, inputs: np.ndarray, training: bool = None) -> np.ndarray:
        """Perform the layer operation on the inputs.

        Args:
            inputs: The input array of shape (batch_size, input_dim).
            training: The training flag, True or False.

        Returns:
            The output array of shape (batch_size, output_dim).
        """
        # Store the previous layer output as an attribute
        self.previous_layer_output = inputs
        # Check the training flag
        if training:
            # Compute the mean and variance of the input across the batch dimension
            batch_mean = np.mean(inputs, axis=0)
            batch_var = np.var(inputs, axis=0)
            # Update the running_mean and running_var parameters with a moving average
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * batch_var
            )
        else:
            # Use the running_mean and running_var parameters to normalize the input
            batch_mean = self.running_mean
            batch_var = self.running_var
        # Normalize the input using the batch mean, batch variance, and epsilon
        normalized_input = (inputs - batch_mean) / np.sqrt(batch_var + self.epsilon)
        # Store the batch mean, batch variance, and normalized input as attributes
        self.batch_mean = batch_mean
        self.batch_var = batch_var
        self.normalized_input = normalized_input
        # Apply the gamma and beta parameters to the normalized input, to produce the output
        output = self.gamma * normalized_input + self.beta
        # Return the output array
        return output

    def backward(
        self,
        error: np.ndarray,
        learning_rate: float,
        prev_output: np.ndarray = None,
        **kwargs,
    ) -> np.ndarray:
        """Perform the backward propagation on the layer.

        Args:
            error: The error array of shape (batch_size, output_dim).
            learning_rate: The learning rate for the parameter update.
            prev_output: The previous layer output, not used in this layer.

        Returns:
            The gradient array of shape (batch_size, input_dim).
        """
        # Get the batch size from the error array
        batch_size = error.shape[0]
        # Compute the gradient of the output with respect to the gamma parameter
        grad_gamma = np.sum(error * self.normalized_input, axis=0)
        # Compute the gradient of the output with respect to the beta parameter
        grad_beta = np.sum(error, axis=0)
        # Update the gamma and beta parameters with the learning rate
        self.gamma -= learning_rate * grad_gamma
        self.beta -= learning_rate * grad_beta
        # Update the gradients dictionary with the grad_gamma and grad_beta arrays
        self.gradients["gamma"] = grad_gamma
        self.gradients["beta"] = grad_beta
        # Compute the gradient of the output with respect to the normalized input
        grad_normalized_input = error * self.gamma
        # Compute the gradient of the normalized input with respect to the input
        grad_input = (
            batch_size * grad_normalized_input
            - np.sum(grad_normalized_input, axis=0)
            - self.normalized_input
            * np.sum(grad_normalized_input * self.normalized_input, axis=0)
        ) / (batch_size * np.sqrt(self.batch_var + self.epsilon))
        # Compute the gradient of the input with respect to the previous layer output
        grad_prev_output = grad_input
        # Return the gradient array
        return grad_prev_output

    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the weights and biases of the layer as numpy arrays."""
        # Return the gamma and beta parameters as the weights and biases
        return self.gamma, self.beta
