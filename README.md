# inventronet

inventronet is a package for building and testing neural networks in Python. 
It provides a simple and intuitive API for creating, training, 
and evaluating various types of neural network models. 
It also includes some common loss functions, activation functions, 
and metrics for neural network problems.

## Installation
You can install inventronet using pip:

```bash
pip install inventronet
```

## Usage

To use inventronet, you need to import the package and create a 
neural network object. You can then add layers, loss functions, activation 
functions, and metrics to the network. You can also specify the learning rate, 
batch size, and number of epochs for training. 
Here is an example of creating a simple feed forward neural network for a 
binary classification problem:

```python
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from inventronet.activations import Sigmoid, ReLU
from inventronet.layers import Dense
from inventronet.losses import BinaryCrossEntropy as BCE
from inventronet.metrics import Accuracy, Precision
from inventronet.models import Sequential
from inventronet.optimizers import StochasticGradientDescent


def plot_history(history):
    fig, axs = plt.subplots(1, len(history), figsize=(12, 4), sharex=True)

    for idx, (label, values) in enumerate(history.items()):
        axs[idx].plot(range(1, len(values) + 1), values)
        axs[idx].set_title(label)
        axs[idx].set_xlabel("Epoch")
        axs[idx].set_ylabel(label)
        axs[idx].grid(True)

    plt.tight_layout()
    plt.show()


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

# Plot the training history
plot_history(model.history)
```


```python
# Example of validation splitting
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from inventronet.activations import Sigmoid, ReLU
from inventronet.layers import Dense
from inventronet.losses import BinaryCrossEntropy as BCE
from inventronet.metrics import Accuracy, Precision
from inventronet.models import Sequential
from inventronet.optimizers import StochasticGradientDescent


def plot_history(history):
    # Get the keys for training and validation metrics
    train_keys = [key for key in history.keys() if not key.startswith("val_")]
    val_keys = [f"val_{key}" for key in train_keys]

    fig, axs = plt.subplots(1, len(train_keys), figsize=(12, 4), sharex=True)

    for idx, (train_key, val_key) in enumerate(zip(train_keys, val_keys)):
        axs[idx].plot(range(1, len(history[train_key]) + 1), history[train_key], label="Training")
        if val_key in history:
            axs[idx].plot(range(1, len(history[val_key]) + 1), history[val_key], label="Validation")
        axs[idx].set_title(train_key)
        axs[idx].set_xlabel("Epoch")
        axs[idx].set_ylabel(train_key)
        axs[idx].legend()
        axs[idx].grid(True)

    plt.tight_layout()
    plt.show()



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

# Specify the validation_split parameter (e.g., 0.2 for using 20% of the data for validation)
validation_split = 0.5

# Fit the model on the training data, with validation
model.fit(input_data, output_data, epochs, validation_split=validation_split)

# Evaluate the model on the test data
loss_value, metric_values = model.evaluate(input_data, output_data)
metric_names = [metric.__class__.__name__ for metric in model.metrics]
metric_str = ', '.join([f"{name}: {value:.4f}" for name, value in zip(metric_names, metric_values)])
print(f"Test Loss: {loss_value:.4f}, Test metrics: {metric_str}")


# Plot the training history
plot_history(model.history)
```

## Documentation

You can find the full documentation of inventronet at https://github.com/inventrohyder/inventronet.

## License

inventronet is licensed under the MIT License. See the LICENSE file for more details.