from typing import Dict

import numpy as np
import pytest


@pytest.fixture
def params() -> Dict[str, np.ndarray]:
    yield {
        "weights": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "biases": np.array([0.5, 0.5]),
    }


@pytest.fixture
def gradients() -> Dict[str, np.ndarray]:
    yield {
        "weights": np.array([[0.1, 0.2], [0.3, 0.4]]),
        "biases": np.array([0.1, -0.1]),
    }


@pytest.fixture
def zero_params() -> Dict[str, np.ndarray]:
    yield {
        "weights": np.zeros((2, 2)),
        "biases": np.zeros(2),
    }


@pytest.fixture
def zero_gradients() -> Dict[str, np.ndarray]:
    yield {
        "weights": np.zeros((2, 2)),
        "biases": np.zeros(2),
    }
