# Define a class for a sequential model
from typing import List, Tuple, Type

import numpy as np
from tqdm import tqdm

from .model import Model
from ..layers.layer import Layer
from ..losses.loss import Loss
from ..metrics.metric import Metric
from ..optimizers.optimizer import Optimizer
from ..preprocessing import train_test_split


class Sequential(Model):
    # Define the initialization method
    def __init__(self) -> None:
        """Call the superclass constructor."""
        # Call the superclass constructor
        super().__init__()
        self.metrics: List[Type[Metric]] = None
        self.optimizer: Type[Optimizer] = None
        self.loss: Type[Loss] = None
        # Attributes for early stopping
        self.patience: int = None
        self.min_delta: float = None
        self.wait: int = 0
        self.best_loss: float = np.inf

        # Attribute for history
        self.history = {
            "loss": [],
        }

    # Add a method to set early stopping parameters
    def set_early_stopping(self, patience: int, min_delta: float) -> None:
        self.patience = patience
        self.min_delta = min_delta

    # Define a method for adding a layer to the model
    def add(self, layer: Type[Layer]) -> None:
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
            self, loss: Type[Loss], optimizer: Type[Optimizer], metrics: List[Type[Metric]]
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
        # Initialize metric history
        for metric in metrics:
            self.history[f"{metric.__class__.__name__}"] = []

    def fit(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            epochs: int,
            validation_split: float = None,
    ) -> None:
        """Fit the model on training data.

        Args:
            x_train (np.ndarray): The input data for training.
            y_train (np.ndarray): The output data for training.
            epochs (int): The number of epochs to train the model.
            validation_split (float, optional): The fraction of the data to use for validation. Default is None.
        """
        if validation_split is not None:
            x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train, test_size=validation_split, random_state=42
            )
            # Initialize validation history
            self.history["val_loss"] = []
            for metric in self.metrics:
                self.history[f"val_{metric.__class__.__name__}"] = []

        progress_bar = tqdm(range(epochs), desc="Training progress")
        for epoch in progress_bar:
            # Forward pass the input data through the network
            layer_output = x_train
            for layer in self.layers:
                layer_output = layer.forward(layer_output, training=True)

            # Calculate the loss and the metrics
            loss_value = self.loss.function(y_train, layer_output)
            metric_values = [
                metric.call(y_train, layer_output) for metric in self.metrics
            ]
            # Update the history
            self.history["loss"].append(loss_value)
            for metric, m_value in zip(self.metrics, metric_values):
                self.history[f"{metric.__class__.__name__}"].append(m_value)

            # Update the progress bar description with the loss and metrics
            metrics_str = ", ".join(
                [
                    f"{metric.__class__.__name__}: {m:.4f}"
                    for metric, m in zip(self.metrics, metric_values)
                ]
            )

            if validation_split is not None:
                # Calculate the loss and metrics for the validation data
                y_val_pred = self.predict(x_val)
                val_loss_value = self.loss.function(y_val, y_val_pred)
                val_metric_values = [
                    metric.call(y_val, y_val_pred) for metric in self.metrics
                ]
                # Update the validation history
                self.history["val_loss"].append(val_loss_value)
                for metric, m_value in zip(self.metrics, val_metric_values):
                    self.history[f"val_{metric.__class__.__name__}"].append(m_value)

                # Update the progress bar description with the validation loss and metrics
                val_metrics_str = ", ".join(
                    [
                        f"Val {metric.__class__.__name__}: {m:.4f}"
                        for metric, m in zip(self.metrics, val_metric_values)
                    ]
                )
                progress_bar.set_description(
                    f"Epoch {epoch + 1}, Loss: {loss_value:.4f}, {metrics_str}, Val Loss: {val_loss_value:.4f}, {val_metrics_str}"
                )

            else:
                progress_bar.set_description(f"Epoch {epoch + 1}, Loss: {loss_value:.4f}, {metrics_str}")

            # Backward pass the error through the network
            layer_error = self.loss.gradient(y_train, layer_output)
            for layer_index, layer in enumerate(reversed(self.layers)):
                layer_input = (
                    layer.previous_layer_output
                    if layer.previous_layer_output is not None
                    else x_train
                )
                layer_error = layer.backward(layer_error, prev_output=layer_input)

                # Check shapes before updating the optimizer
                for key in layer.parameters.keys():
                    assert layer.parameters[key].shape == layer.gradients[
                        key].shape, f"Parameter shape {layer.parameters[key].shape} does not match gradient shape {layer.gradients[key].shape}"

                self.optimizer.update(len(self.layers) - 1 - layer_index, layer.parameters, layer.gradients)

            # Check for early stopping
            if self.patience is not None and self.min_delta is not None:
                if loss_value < self.best_loss - self.min_delta:
                    self.best_loss = loss_value
                    self.wait = 0
                else:
                    self.wait += 1

                if self.wait >= self.patience:
                    progress_bar.close()
                    print(f"Early stopping on epoch {epoch + 1}")
                    break

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
