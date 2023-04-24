import numpy as np
from .loss import Loss


class MeanSquaredError(Loss):
    def function(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the mean squared error loss between the true labels
        and the predicted values.

        Parameters
        ----------
        y_true : np.ndarray
            The true labels of the data, encoded as one-hot vectors or
              as scalars.
            Shape: (n_samples, n_classes) or (n_samples,)
        y_pred : np.ndarray
            The predicted values of the data, output of the linear function
            or the softmax function.
            Shape: (n_samples, n_classes) or (n_samples,)

        Returns
        -------
        float
            The mean squared error loss, averaged over all samples and
            classes or samples.
        """
        return np.mean((y_true - y_pred) ** 2)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the mean squared error loss with respect
        to the predictions.

        Parameters
        ----------
        y_true : np.ndarray
            The true labels of the data, encoded as one-hot vectors or
            as scalars.
            Shape: (n_samples, n_classes) or (n_samples,)
        y_pred : np.ndarray
            The predicted values of the data, output of the linear
            function or the softmax function.
            Shape: (n_samples, n_classes) or (n_samples,)

        Returns
        -------
        np.ndarray
            The derivative of the mean squared error loss, scaled
            by the number of samples.
            Shape: (n_samples, n_classes) or (n_samples,)
        """
        return 2 * (y_pred - y_true) / y_true.size
