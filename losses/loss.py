from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    @abstractmethod
    def function(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the loss between the true labels and the predicted values.

        Parameters
        ----------
        y_true : np.ndarray
            The true labels of the data, encoded as one-hot vectors or as
                scalars.
            Shape: (n_samples, n_classes) or (n_samples,)
        y_pred : np.ndarray
            The predicted values of the data, output of the activation
                function.
            Shape: (n_samples, n_classes) or (n_samples,)

        Returns
        -------
        float
            The loss, averaged over all samples and classes or samples.
        """
        raise NotImplementedError("The function method must be implemented.")

    @abstractmethod
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the loss with respect to the predictions.

        Parameters
        ----------
        y_true : np.ndarray
            The true labels of the data, encoded as one-hot vectors or as
                scalars.
            Shape: (n_samples, n_classes) or (n_samples,)
        y_pred : np.ndarray
            The predicted values of the data, output of the activation
                function.
            Shape: (n_samples, n_classes) or (n_samples,)

        Returns
        -------
        np.ndarray
            The derivative of the loss, scaled by the number of samples.
            Shape: (n_samples, n_classes) or (n_samples,)
        """
        raise NotImplementedError("The gradient method must be implemented.")

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self.function(y_true, y_pred)
