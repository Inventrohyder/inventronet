# Define a class for a sequential model
from typing import List, Tuple, Type
import numpy as np

from ..optimizers.optimizer import Optimizer
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

    def compile(
        self, loss: Loss, optimizer: Optimizer, metrics: List[Type[Metric]]
    ) -> None:
        """Compile the model with a loss function, an optimizer, and metrics.

        Args:
            loss (Loss): The loss function to be used.
            optimizer (Optimizer): The optimizer to be used.
            metrics (List[Metric]): The metrics to be used.
        """
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
    ) -> None:
        """Fit the model on training data.

        Args:
            x_train (np.ndarray): The input data for training.
            y_train (np.ndarray): The output data for training.
            epochs (int): The number of epochs to train the model.
        """
        for epoch in range(epochs):
            # Forward pass the input data through the network
            layer_output = x_train
            for layer in self.layers:
                layer_output = layer.forward(layer_output, training=True)

            # Calculate the loss and the metrics
            loss_value = self.loss.function(y_train, layer_output)
            metric_values = [
                metric.call(y_train, layer_output) for metric in self.metrics
            ]

            # Print the loss and the metrics
            metrics_str = ", ".join(
                [
                    f"{metric.__class__.__name__}: {m:.4f}"
                    for metric, m in zip(self.metrics, metric_values)
                ]
            )
            print(f"Epoch {epoch + 1}, Loss: {loss_value:.4f}, {metrics_str}")

            # Backward pass the error through the network
            layer_error = self.loss.gradient(y_train, layer_output)
            for layer in reversed(self.layers):
                layer_input = (
                    layer.previous_layer_output
                    if layer.previous_layer_output is not None
                    else x_train
                )
                layer_error = layer.backward(
                    layer_error, self.optimizer.learning_rate, prev_output=layer_input
                )
                self.optimizer.update(layer.parameters, layer.gradients)

    # Define a method for predicting the output for new data
    def predict(self, x_test: np.ndarray) -> np.ndarray:
        # Forward pass the input data through the network
        layer_output = x_test
        for layer in self.layers:
            layer_output = layer.forward(layer_output)
        # Return the final output
        return layer_output

    # Define a method for evaluating the model on test data
    def evaluate(
        self, x_test: np.ndarray, y_test: np.ndarray
    ) -> Tuple[float, List[float]]:
        # Predict the output for the test data
        y_pred = self.predict(x_test)
        # Calculate the loss and the metrics
        loss_value = self.loss.function(y_test, y_pred)
        metric_values = [metric.call(y_test, y_pred) for metric in self.metrics]
        # Return the loss and the metrics
        return loss_value, metric_values
