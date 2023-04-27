from typing import Tuple

import numpy as np

from inventronet.activations import Sigmoid, ReLU
from inventronet.layers import Dense
from inventronet.losses import BinaryCrossEntropy as BCE
from inventronet.metrics import Accuracy, Precision
from inventronet.models import Sequential
from inventronet.optimizers import StochasticGradientDescent

epochs = 10000


def glorot_uniform(size: Tuple[int, int]) -> np.ndarray:
    input_dim, output_dim = size
    limit = np.sqrt(6 / (input_dim + output_dim))
    return np.random.uniform(low=-limit, high=limit, size=size)


# Define the input and output data
input_data = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
output_data = np.array([[0], [1], [1], [0]])

# Define the neural network with two dense layers
model = Sequential()
model.add(Dense(input_dim=3, output_dim=4, activation=ReLU(), weight_initializer=glorot_uniform))
model.add(Dense(input_dim=4, output_dim=1, activation=Sigmoid(), weight_initializer=glorot_uniform))

# Define the loss function and the metric
loss = BCE()
optimizer = StochasticGradientDescent(learning_rate=0.1)

# Compile the model with the loss function, optimizer and the metrics
model.compile(loss, optimizer, metrics=[Precision(), Accuracy()])

# Set early stopping parameters
model.set_early_stopping(patience=500, min_delta=1e-4)

# Fit the model on the training data
model.fit(input_data, output_data, epochs)

# Evaluate the model on the test data
loss_value, metric_value = model.evaluate(input_data, output_data)
print(f"Test Loss: {loss_value:.4f}, Test metrics: {metric_value}")
