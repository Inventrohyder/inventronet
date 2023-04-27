import inspect
from typing import Type

import numpy as np
import pytest
from _pytest.fixtures import SubRequest

from inventronet import optimizers as optimizers_module
from inventronet.activations import Sigmoid
from inventronet.activations.activation import Activation
from inventronet.optimizers.optimizer import Optimizer


def generate_random_array(shape):
    return np.random.randn(*shape)


@pytest.fixture
def random_input() -> np.ndarray:
    yield generate_random_array((10, 5))


@pytest.fixture
def input_dim(random_input: np.ndarray) -> int:
    yield random_input.shape[1]


@pytest.fixture
def random_error(random_input: np.ndarray) -> np.ndarray:
    yield generate_random_array(random_input.shape) / 2


@pytest.fixture
def random_biases(random_error: np.ndarray) -> np.ndarray:
    yield generate_random_array((random_error.shape[1],)) * 2


all_optimizers = [
    optimizer
    for _, optimizer in inspect.getmembers(optimizers_module, inspect.isclass)
    if issubclass(optimizer, Optimizer) and optimizer != Optimizer
]


@pytest.fixture(params=all_optimizers)
def optimizer(request: SubRequest) -> Type[Optimizer]:
    opt: Type[Optimizer] = request.param()
    # Increase the learning rate
    opt.learning_rate = 0.1
    yield opt


@pytest.fixture
def activation() -> Type[Activation]:
    yield Sigmoid()
