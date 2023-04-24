import numpy as np
from .loss import Loss


class BinaryCrossEntropy(Loss):
    def function(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the binary cross-entropy loss between the true labels and the
            predicted values.

        Parameters
        ----------
        y_true : np.ndarray
            The true labels of the data, encoded as one-hot vectors or as
                scalars.
            Shape: (n_samples, n_classes) or (n_samples,)
        y_pred : np.ndarray
            The predicted values of the data, output of the sigmoid function.
            Shape: (n_samples, n_classes) or (n_samples,)

        Returns
        -------
        float
            The loss, averaged over all samples and classes or samples.
        """
        # Clip the predictions to avoid log(0) or log(1) errors
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        # Compute the binary cross-entropy loss
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the binary cross-entropy loss with respect
            to the predictions.

        Parameters
        ----------
        y_true : np.ndarray
            The true labels of the data, encoded as one-hot vectors or as
                scalars.
            Shape: (n_samples, n_classes) or (n_samples,)
        y_pred : np.ndarray
            The predicted values of the data, output of the sigmoid function.
            Shape: (n_samples, n_classes) or (n_samples,)

        Returns
        -------
        np.ndarray
            The derivative of the loss, scaled by the number of samples.
            Shape: (n_samples, n_classes) or (n_samples,)
        """
        # Clip the predictions to avoid division by 0 errors
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        # Compute the derivative of the binary cross-entropy loss
        grad = -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / len(y_true)
        return grad
