from abc import ABC, abstractmethod


class Metric(ABC):
    @abstractmethod
    def call(self, y_true, y_pred):
        raise NotImplementedError("The call method must be implemented.")

    def __call__(self, y_true, y_pred):
        return self.call(y_true, y_pred)
