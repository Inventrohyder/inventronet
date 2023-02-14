from abc import ABC, abstractmethod
import numpy as np


class Metric(ABC):
    @abstractmethod
    def call(self, y_true, y_pred):
        pass


# Define the accuracy metric
class Accuracy(Metric):
    def call(self, y_true, y_pred):
        return np.mean(y_true == np.round(y_pred))


# Define the precision metric
class Precision(Metric):
    def call(self, y_true, y_pred):
        # Round the predictions to get the binary labels
        y_pred = np.round(y_pred)
        # Count the true positives and the predicted positives
        tp = np.sum(y_true * y_pred)
        pp = np.sum(y_pred)
        # Compute the precision
        return tp / pp if pp > 0 else 0
