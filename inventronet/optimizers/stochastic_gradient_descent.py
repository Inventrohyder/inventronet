from .optimizer import Optimizer


class StochasticGradientDescent(Optimizer):
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def update(self, layer_id: int, params: dict, gradients: dict):
        for key in params.keys():
            params[key] -= self.learning_rate * gradients[key]
