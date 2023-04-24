import numpy as np
from .loss import Loss


class CategoricalCrossEntropy(Loss):
    def function(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the categorical cross-entropy loss between the true labels
            and the predicted values.

        Parameters
        ----------
        y_true : np.ndarray
            The true labels of the data, encoded as one-hot vectors.
            Shape: (n_samples, n_classes)
        y_pred : np.ndarray
            The predicted values of the data, output of the softmax function.
            Shape: (n_samples, n_classes)

        Returns
        -------
        float
            The loss, averaged over all samples and classes.
        """
        # Clip the predictions to avoid log(0) errors
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        # Compute the categorical cross-entropy loss
        loss = -np.sum(y_true * np.log(y_pred)) / len(y_true)
        return loss

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the categorical cross-entropy loss with
            respect to the predictions.

        Parameters
        ----------
        y_true : np.ndarray
            The true labels of the data, encoded as one-hot vectors.
            Shape: (n_samples, n_classes)
        y_pred : np.ndarray
            The predicted values of the data, output of the softmax function.
            Shape: (n_samples, n_classes)

        Returns
        -------
        np.ndarray
            The derivative of the loss, scaled by the number of samples.
            Shape: (n_samples, n_classes)
        """
        # Clip the predictions to avoid division by zero errors in the
        # gradient calculation
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        # Compute the derivative of the categorical cross-entropy loss
        grad = -(y_true - y_pred)
        return grad
