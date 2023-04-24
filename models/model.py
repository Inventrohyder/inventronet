from abc import ABC, abstractmethod
from typing import List, Tuple, Type

import numpy as np

from ..layers.layer import Layer
from ..losses.loss import Loss
from ..metrics.metric import Metric


# Define an abstract base class for a model
class Model(ABC):
    # Define the initialization method
    def __init__(self) -> None:
        """Initialize an empty list of layers."""
        # Initialize an empty list of layers
        self.layers: List[Type[Layer]] = []

    # Define an abstract method for adding a layer to the model
    @abstractmethod
    def add(self, layer: Layer) -> None:
        """Add a layer to the model.

        Args:
            layer (Layer): The layer to be added.
        """
        raise NotImplementedError("The add method must be implemented.")

    # Define an abstract method for compiling the model with a loss
    # function and a metric
    @abstractmethod
    def compile(self, loss: Loss, metric: Metric) -> None:
        """Compile the model with a loss function and a metric.

        Args:
            loss (Loss): The loss function to be used.
            metric (Metric): The metric to be used.
        """
        raise NotImplementedError("The compile method must be implemented.")

    # Define an abstract method for fitting the model on training data
    @abstractmethod
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
        raise NotImplementedError("The fit method must be implemented.")

    # Define an abstract method for predicting the output for new data
    @abstractmethod
    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """Predict the output for new data.

        Args:
            x_test (np.ndarray): The input data for testing.

        Returns:
            np.ndarray: The output data for testing.
        """
        raise NotImplementedError("The predict method must be implemented.")

    # Define an abstract method for evaluating the model on test data
    @abstractmethod
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        """Evaluate the model on test data.

        Args:
            x_test (np.ndarray): The input data for testing.
            y_test (np.ndarray): The output data for testing.

        Returns:
            Tuple[float, float]: The loss and the metric values for the test data.
        """
        raise NotImplementedError("The evaluate method must be implemented.")
