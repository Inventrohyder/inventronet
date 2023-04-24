from abc import ABC, abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def update(self, params: dict, gradients: dict):
        raise NotImplementedError("The update method must be implemented.")
