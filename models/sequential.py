# Define a class for a sequential model
import numpy as np
from ..losses.loss import Loss
from ..metrics.metric import Metric
from ..layers.layer import Layer
from .model import Model


class Sequential(Model):
    # Define the initialization method
    def __init__(self) -> None:
        """Call the superclass constructor."""
        # Call the superclass constructor
        super().__init__()

    # Define a method for adding a layer to the model
    def add(self, layer: Layer) -> None:
        """Add a layer to the model.

        Args:
            layer (Layer): The layer to be added.

        Raises:
            AssertionError: If the layer input dimension is not None and
            does not match the previous layer output dimension.
        """
        # Check if the layer is compatible with the previous layer
        if self.layers:
            previous_layer = self.layers[-1]
            # If the layer input dimension is None, infer it from the
            # previous layer output dimension
            if layer.input_dim is None:
                layer.input_dim = previous_layer.output_dim
            # Otherwise, check if the layer input dimension matches the
            # previous layer output dimension
            else:
                assert (
                    layer.input_dim == previous_layer.output_dim
                ), "Layer input dimension does not match previous layer output dimension"
        # Append the layer to the list of layers
        self.layers.append(layer)

    # Define a method for compiling the model with a loss function and a metric
    def compile(self, loss: Loss, metric: Metric) -> None:
        """Compile the model with a loss function and a metric.

        Args:
            loss (Loss): The loss function to be used.
            metric (Metric): The metric to be used.
        """
        # Store the loss function and the metric as attributes
        self.loss = loss
        self.metric = metric

    # Define a method for fitting the model on training data
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
        learning_rate: float,
    ) -> None:
        """Fit the model on training data.

        Args:
            x_train (np.ndarray): The input data for training.
            y_train (np.ndarray): The output data for training.
            epochs (int): The number of epochs to train the model.
            learning_rate (float): The learning rate for updating the
            weights and biases.
        """
        # Loop over the epochs
        for epoch in range(epochs):
            # Forward pass the input data through the network
            layer_output = x_train
            for layer in self.layers:
                layer_output = layer.forward(layer_output, training=True)

            # Calculate the loss and the metric
            loss_value = self.loss.function(y_train, layer_output)
            metric_value = self.metric.call(y_train, layer_output)

            # Print the loss and the metric
            print(
                f"Epoch {epoch + 1}, Loss: {loss_value:.4f}, Metric: {metric_value:.4f}"
            )

            # Backward pass the error through the network
            layer_error = self.loss.gradient(y_train, layer_output)
            for layer in reversed(self.layers):
                layer_input = (
                    layer.previous_layer_output
                    if layer.previous_layer_output is not None
                    else x_train
                )
                layer_error = layer.backward(
                    layer_error, learning_rate, prev_output=layer_input, training=True
                )

    # Define a method for predicting the output for new data
    def predict(self, x_test):
        # Forward pass the input data through the network
        layer_output = x_test
        for layer in self.layers:
            layer_output = layer.forward(layer_output)
        # Return the final output
        return layer_output

    # Define a method for evaluating the model on test data
    def evaluate(self, x_test, y_test):
        # Predict the output for the test data
        y_pred = self.predict(x_test)
        # Calculate the loss and the metric
        loss_value = self.loss.function(y_test, y_pred)
        metric_value = self.metric.call(y_test, y_pred)
        # Return the loss and the metric
        return loss_value, metric_value
