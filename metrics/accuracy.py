import numpy as np
from .metric import Metric


class Accuracy(Metric):
    def call(self, y_true, y_pred) -> float:
        return np.mean(y_true == np.round(y_pred))
