import numpy as np
from .optimizer import Optimizer


class Adam(Optimizer):
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # Initialize m as an empty dictionary
        self.v = {}  # Initialize v as an empty dictionary
        self.t = 0

    def update(self, layer_id: int, params: dict, gradients: dict):
        if layer_id not in self.m:
            self.m[layer_id] = {key: np.zeros_like(value) for key, value in params.items()}
            self.v[layer_id] = {key: np.zeros_like(value) for key, value in params.items()}

        self.t += 1

        for key in params.keys():
            self.m[layer_id][key] = self.beta1 * self.m[layer_id][key] + (1 - self.beta1) * gradients[key]
            self.v[layer_id][key] = self.beta2 * self.v[layer_id][key] + (1 - self.beta2) * np.square(gradients[key])

            m_hat = self.m[layer_id][key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[layer_id][key] / (1 - self.beta2 ** self.t)

            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
