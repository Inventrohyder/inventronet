from abc import ABC, abstractmethod


class Metric(ABC):
    @abstractmethod
    def call(self, y_true, y_pred) -> float:
        raise NotImplementedError("The call method must be implemented.")

    def __call__(self, y_true, y_pred) -> float:
        return self.call(y_true, y_pred)
