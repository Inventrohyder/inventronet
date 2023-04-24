# Define the precision metric
import numpy as np
from .metric import Metric


class Precision(Metric):
    def call(self, y_true, y_pred) -> float:
        # Round the predictions to get the binary labels
        y_pred = np.round(y_pred)
        # Count the true positives and the predicted positives
        tp = np.sum(y_true * y_pred)
        pp = np.sum(y_pred)
        # Compute the precision
        return tp / pp if pp > 0.0 else 0.0
