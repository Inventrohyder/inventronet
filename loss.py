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
            The true labels of the data, encoded as one-hot vectors or as scalars.
            Shape: (n_samples, n_classes) or (n_samples,)
        y_pred : np.ndarray
            The predicted values of the data, output of the activation function.
            Shape: (n_samples, n_classes) or (n_samples,)

        Returns
        -------
        float
            The loss, averaged over all samples and classes or samples.
        """
        pass

    @abstractmethod
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the loss with respect to the predictions.

        Parameters
        ----------
        y_true : np.ndarray
            The true labels of the data, encoded as one-hot vectors or as scalars.
            Shape: (n_samples, n_classes) or (n_samples,)
        y_pred : np.ndarray
            The predicted values of the data, output of the activation function.
            Shape: (n_samples, n_classes) or (n_samples,)

        Returns
        -------
        np.ndarray
            The derivative of the loss, scaled by the number of samples.
            Shape: (n_samples, n_classes) or (n_samples,)
        """
        pass


# Define the mean absolute error loss function and its derivative
class MeanAbsoluteError(Loss):
    def function(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the mean absolute error loss between the true labels and the predicted values.

        Parameters
        ----------
        y_true : np.ndarray
            The true labels of the data, encoded as scalars.
            Shape: (n_samples,)
        y_pred : np.ndarray
            The predicted values of the data, output of the activation function.
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
        Computes the derivative of the mean absolute error loss with respect to the predictions.

        Parameters
        ----------
        y_true : np.ndarray
            The true labels of the data, encoded as scalars.
            Shape: (n_samples,)
        y_pred : np.ndarray
            The predicted values of the data, output of the activation function.
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
        print("Gradient: {derivative}")
        return derivative


# Define the mean squared error loss function and its derivative
class MeanSquaredError(Loss):
    def function(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the mean squared error loss between the true labels and the predicted values.

        Parameters
        ----------
        y_true : np.ndarray
            The true labels of the data, encoded as one-hot vectors or as scalars.
            Shape: (n_samples, n_classes) or (n_samples,)
        y_pred : np.ndarray
            The predicted values of the data, output of the linear function or the softmax function.
            Shape: (n_samples, n_classes) or (n_samples,)

        Returns
        -------
        float
            The mean squared error loss, averaged over all samples and classes or samples.
        """
        return np.mean((y_true - y_pred) ** 2)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the mean squared error loss with respect to the predictions.

        Parameters
        ----------
        y_true : np.ndarray
            The true labels of the data, encoded as one-hot vectors or as scalars.
            Shape: (n_samples, n_classes) or (n_samples,)
        y_pred : np.ndarray
            The predicted values of the data, output of the linear function or the softmax function.
            Shape: (n_samples, n_classes) or (n_samples,)

        Returns
        -------
        np.ndarray
            The derivative of the mean squared error loss, scaled by the number of samples.
            Shape: (n_samples, n_classes) or (n_samples,)
        """
        return 2 * (y_pred - y_true) / y_true.size


class BinaryCrossEntropy(Loss):
    def function(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the binary cross-entropy loss between the true labels and the predicted values.

        Parameters
        ----------
        y_true : np.ndarray
            The true labels of the data, encoded as one-hot vectors or as scalars.
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
        Computes the derivative of the binary cross-entropy loss with respect to the predictions.

        Parameters
        ----------
        y_true : np.ndarray
            The true labels of the data, encoded as one-hot vectors or as scalars.
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


class CategoricalCrossEntropy(Loss):
    def function(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the categorical cross-entropy loss between the true labels and the predicted values.

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
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return loss

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the categorical cross-entropy loss with respect to the predictions.

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
        # Compute the derivative of the categorical cross-entropy loss
        grad = -y_true / np.exp(np.logaddexp(0, -y_pred))
        return grad / len(y_true)
