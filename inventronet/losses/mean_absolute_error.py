import numpy as np

from .loss import Loss


class MeanAbsoluteError(Loss):
    def function(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the mean absolute error loss between the true labels and the
        predicted values.

        Parameters
        ----------
        y_true : np.ndarray
            The true labels of the data, encoded as scalars.
            Shape: (n_samples,)
        y_pred : np.ndarray
            The predicted values of the data, output of the activation
            function.
            Shape: (n_samples,)

        Returns
        -------
        float
            The loss, averaged over all samples.
        """
        # Compute the absolute error
        error = np.abs(y_true - y_pred)
        # Compute the mean absolute error
        loss = np.mean(error)
        return loss

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the mean absolute error loss with respect to
        the predictions.

        Parameters
        ----------
        y_true : np.ndarray
            The true labels of the data, encoded as scalars.
            Shape: (n_samples,)
        y_pred : np.ndarray
            The predicted values of the data, output of the
            activation function.
            Shape: (n_samples,)

        Returns
        -------
        np.ndarray
            The derivative of the loss, scaled by the number of samples.
            Shape: (n_samples,)
        """
        # Compute the sign of the error
        sign = np.sign(y_true - y_pred)
        # Compute the derivative of the mean absolute error
        derivative = -sign / y_true.size
        print(f"Gradient: {derivative}")
        return derivative
