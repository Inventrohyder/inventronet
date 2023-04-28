# Define a class for a sequential model
import math
from typing import List, Tuple, Type

import numpy as np
from tqdm import tqdm

from .model import Model
from ..layers.layer import Layer
from ..losses.loss import Loss
from ..metrics.metric import Metric
from ..optimizers.optimizer import Optimizer
from ..preprocessing import train_test_split


def _generate_batches(x: np.ndarray, y: np.ndarray, batch_size: int):
    num_batches = math.ceil(len(x) / batch_size)
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        yield x[start:end], y[start:end]


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
            batch_size: int = 32,
    ) -> None:
        if validation_split is not None:
            x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train, test_size=validation_split, random_state=42
            )

            if x_train.shape[0] < 1:
                raise ValueError("Not enough samples in the training set after applying the validation split.")

            # Initialize validation history
            self.history["val_loss"] = []
            for metric in self.metrics:
                self.history[f"val_{metric.__class__.__name__}"] = []

        progress_bar = tqdm(range(epochs), desc="Training progress")
        for epoch in progress_bar:
            # Initialize loss and metric accumulators for the current epoch
            epoch_loss = 0
            epoch_metric_values = [0] * len(self.metrics)
            batch_count = 0

            # Iterate through mini-batches
            for x_batch, y_batch in _generate_batches(x_train, y_train, batch_size):
                batch_count += 1
                # Forward pass the input data through the network
                layer_output = x_batch
                for layer in self.layers:
                    layer_output = layer.forward(layer_output, training=True)

                # Calculate the loss and the metrics
                loss_value = self.loss.function(y_batch, layer_output)
                metric_values = [
                    metric.call(y_batch, layer_output) for metric in self.metrics
                ]

                # Accumulate the loss and metric values for the current epoch
                epoch_loss += loss_value
                epoch_metric_values = [
                    m_accum + m_value
                    for m_accum, m_value in zip(epoch_metric_values, metric_values)
                ]

                # Backward pass the error through the network
                layer_error = self.loss.gradient(y_batch, layer_output)
                for layer_index, layer in enumerate(reversed(self.layers)):
                    layer_input = (
                        layer.previous_layer_output
                        if layer.previous_layer_output is not None
                        else x_batch
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

            # Calculate the average loss and metrics for the current epoch
            avg_epoch_loss = epoch_loss / batch_count
            avg_epoch_metric_values = [
                m_value / batch_count for m_value in epoch_metric_values
            ]

            # Calculate metrics_str for the current epoch
            metrics_str = ", ".join(
                [
                    f"{metric.__class__.__name__}: {m:.4f}"
                    for metric, m in zip(self.metrics, avg_epoch_metric_values)
                ]
            )

            # Update the history
            self.history["loss"].append(avg_epoch_loss)
            for metric, m_value in zip(self.metrics, avg_epoch_metric_values):
                self.history[f"{metric.__class__.__name__}"].append(m_value)

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
                    f"Epoch {epoch + 1}, Loss: {avg_epoch_loss:.4f}, {metrics_str}, Val Loss: {val_loss_value:.4f}, {val_metrics_str}"
                )

            else:
                progress_bar.set_description(f"Epoch {epoch + 1}, Loss: {avg_epoch_loss:.4f}, {metrics_str}")

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
