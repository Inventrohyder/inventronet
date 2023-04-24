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
# Define the number of epochs and the learning rate
import numpy as np
from inventronet.activations import Sigmoid
from inventronet.layers import Dense
from inventronet.losses import MSE
from inventronet.metrics import Accuracy, Precision
from inventronet.models import Sequential
from inventronet.optimizers import StochasticGradientDescent



epochs = 10000
learning_rate = 0.1

# Define the input and output data
input_data = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
output_data = np.array([[0], [1], [1], [0]])

# Define the neural network with two dense layers
model = Sequential()
model.add(Dense(input_dim=3, output_dim=4, activation=Sigmoid()))
model.add(Dense(input_dim=4, output_dim=1, activation=Sigmoid()))

# Define the loss function and the metric
loss = MSE()
metric = Precision()

# Compile the model with the loss function, optimizer and the metric
model.compile(loss, StochasticGradientDescent(), metric)

# Fit the model on the training data
model.fit(input_data, output_data, epochs, learning_rate)

# Evaluate the model on the test data
loss_value, metric_value = model.evaluate(input_data, output_data)
print(f"Test Loss: {loss_value:.4f}, Test Precision: {metric_value:.4f}")
```

## Documentation

You can find the full documentation of inventronet at https://inventronet.readthedocs.io/

## License

inventronet is licensed under the MIT License. See the LICENSE file for more details.