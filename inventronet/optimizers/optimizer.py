from abc import ABC, abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def update(self, layer_id: int, params: dict, gradients: dict):
        raise NotImplementedError("The update method must be implemented.")
