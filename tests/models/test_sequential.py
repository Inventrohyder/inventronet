import inspect
from typing import Type, Tuple

import numpy as np
import pytest
from _pytest.fixtures import SubRequest

import inventronet.losses as losses_module
import inventronet.metrics as metrics_module
import inventronet.optimizers as optimizers_module
from inventronet.activations import ReLU, Sigmoid
from inventronet.layers import Dense, Dropout
from inventronet.losses.loss import Loss
from inventronet.metrics.metric import Metric
from inventronet.models import Sequential
from inventronet.optimizers import StochasticGradientDescent
from inventronet.optimizers.optimizer import Optimizer

# Get all classes in the metrics module that are subclasses of Metric
all_metrics = [
    metric
    for _, metric in inspect.getmembers(metrics_module, inspect.isclass)
    if issubclass(metric, Metric) and metric != Metric
]

# Get all classes in the losses module that are subclasses of Loss
all_losses = [
    loss
    for _, loss in inspect.getmembers(losses_module, inspect.isclass)
    if issubclass(loss, Loss) and loss != Loss
]

all_optimizers = [
    optimizer
    for _, optimizer in inspect.getmembers(optimizers_module, inspect.isclass)
    if issubclass(optimizer, Optimizer) and optimizer != Optimizer
]


@pytest.fixture(params=all_metrics)
def metric(request: SubRequest) -> Type[Metric]:
    yield request.param()


@pytest.fixture(params=all_losses)
def loss(request: SubRequest) -> Type[Loss]:
    yield request.param()


@pytest.fixture(params=all_optimizers)
def optimizer(request: SubRequest) -> Type[Optimizer]:
    opt: Type[Optimizer] = request.param()
    # Increase the learning rate
    opt.learning_rate = 0.1
    yield opt


@pytest.fixture
def dummy_sequential_model() -> Sequential:
    """Create a dummy sequential model with two layers"""
    # Create a sequential model
    model = Sequential()
    # Add a dense layer with input dimension 3 and output dimension 2
    model.add(Dense(3, 2))
    # Add a dense layer with input dimension 2 and output dimension 1
    model.add(Dense(2, 1))
    # Return the model
    return model


@pytest.fixture
def dummy_compiled_sequential_model(
        dummy_sequential_model: Sequential,
        loss: Type[Loss],
        optimizer: Type[Optimizer],
        metric: Type[Metric],
) -> Sequential:
    """Compile the sequential model with a loss function, and a metric."""
    dummy_sequential_model.compile(loss, optimizer, [metric])
    # Return the model
    return dummy_sequential_model


# Set different dataset sizes
dataset_sizes = [2, 4, 10, 100]


@pytest.fixture(scope="module", params=dataset_sizes)
def xy(request: SubRequest) -> tuple:
    np.random.seed(42)  # Set a random seed for reproducibility
    size = request.param
    x = np.random.rand(size, 3)  # `size` rows and 3 columns
    y = np.random.rand(size, 1)  # `size` rows and 1 column
    yield x, y


@pytest.fixture(scope="module")
def x(xy: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    yield xy[0]


@pytest.fixture(scope="module")
def y(xy: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    yield xy[1]


@pytest.fixture
def sequential_model_with_dropout(x: np.ndarray, y: np.ndarray, loss: Loss, optimizer: Optimizer,
                                  metric: Metric) -> Sequential:
    # Define the input and output data
    input_data = x
    output_data = y

    # Define the neural network with two dense layers and a dropout layer
    model = Sequential()
    model.add(Dense(input_dim=3, output_dim=4, activation=ReLU()))
    model.add(Dropout(0.5, input_dim=4))
    model.add(Dense(input_dim=4, output_dim=1, activation=Sigmoid()))

    # Compile the model with the loss function, optimizer, and metric
    model.compile(loss, optimizer, metrics=[metric])

    # Fit the model on the training data
    model.fit(input_data, output_data, epochs=10)

    return model


def test_sequential_model_with_dropout_loss_history_length(sequential_model_with_dropout):
    # Assert that the training didn't raise any errors and completed successfully
    assert len(sequential_model_with_dropout.history["loss"]) == 10


def test_sequential_model_with_dropout_accuracy_history_length(sequential_model_with_dropout, metric: Metric):
    # Assert that the training didn't raise any errors and completed successfully
    assert len(sequential_model_with_dropout.history[metric.__class__.__name__]) == 10


# Define a simple model
def create_model():
    model = Sequential()
    model.add(Dense(input_dim=2, output_dim=2, activation=Sigmoid()))
    model.add(Dense(input_dim=2, output_dim=1, activation=Sigmoid()))
    return model


# Test early stopping
@pytest.mark.parametrize("patience, min_delta, expected_epochs", [
    (2, 0.1, 3),
    (4, 0.05, 5)
])
def test_early_stopping(patience, min_delta, expected_epochs, loss: Loss, optimizer: Optimizer, metric: Metric):
    model = create_model()
    optimizer = StochasticGradientDescent(learning_rate=0.1)

    model.compile(loss, optimizer, metrics=[metric])
    model.set_early_stopping(patience=patience, min_delta=min_delta)

    # Define a simple dataset
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    output_data = np.array([[0], [1], [1], [0]])

    # Train the model
    model.fit(input_data, output_data, epochs=1000)

    # Check if the training stopped at the expected epoch
    assert len(model.history["loss"]) == expected_epochs


def test_add(dummy_sequential_model: Sequential):
    """Test the add method of the sequential model."""
    # Try to add a layer with input dimension 4 and output dimension 2
    with pytest.raises(AssertionError):
        dummy_sequential_model.add(Dense(4, 2))
    # Check that the model still has two layers
    assert len(dummy_sequential_model.layers) == 2


def test_fit_raises_value_error_with_insufficient_samples_after_validation_split(
        dummy_compiled_sequential_model: Sequential,
        dataset_size: int = 1,
):
    np.random.seed(42)  # Set a random seed for reproducibility
    x = np.random.rand(dataset_size, 3)  # `dataset_size` rows and 3 columns
    y = np.random.rand(dataset_size, 1)  # `dataset_size` rows and 1 column

    if dataset_size < 2:  # Check if there are not enough samples for the validation split
        with pytest.raises(ValueError,
                           match="Not enough samples in the training set after applying the validation split."):
            dummy_compiled_sequential_model.fit(x, y, 5, validation_split=0.5, batch_size=1)
    else:
        dummy_compiled_sequential_model.fit(x, y, 5, validation_split=0.5, batch_size=1)


class TestCompile:

    def test_model_loss(
            self,
            dummy_sequential_model: Sequential,
            loss: Type[Loss],
            optimizer: Type[Optimizer],
            metric: Type[Metric],
    ):
        # Compile the model with a loss function and a metric
        dummy_sequential_model.compile(loss, optimizer, [metric])
        # Check that the model has the loss function
        assert dummy_sequential_model.loss == loss

    def test_model_metric_before(self, dummy_sequential_model: Sequential):
        with pytest.raises(AttributeError):
            dummy_sequential_model.metric

    def test_model_metric(
            self,
            dummy_sequential_model: Sequential,
            loss: Type[Loss],
            optimizer: Type[Optimizer],
            metric: Type[Metric],
    ):
        # Compile the model with a loss function and a metric
        dummy_sequential_model.compile(loss, optimizer, [metric])
        # Check that the model has the metrics
        assert dummy_sequential_model.metrics == [metric]


class TestFit:
    def test_fit_history_is_dict(
            self,
            dummy_compiled_sequential_model: Sequential,
            x: np.ndarray,
            y: np.ndarray,
    ):
        dummy_compiled_sequential_model.fit(x, y, 5)
        history = dummy_compiled_sequential_model.history
        assert isinstance(history, dict)

    def test_fit_history_has_loss_and_metric(
            self,
            dummy_compiled_sequential_model: Sequential,
            x: np.ndarray,
            y: np.ndarray,
    ):
        dummy_compiled_sequential_model.fit(x, y, 5)
        history = dummy_compiled_sequential_model.history
        metric_class_name = dummy_compiled_sequential_model.metrics[0].__class__.__name__
        assert "loss" in history and metric_class_name in history.keys()

    def test_fit_history_correct_length(
            self,
            dummy_compiled_sequential_model: Sequential,
            x: np.ndarray,
            y: np.ndarray,
    ):
        dummy_compiled_sequential_model.fit(x, y, 5)
        history = dummy_compiled_sequential_model.history
        assert len(history["loss"]) == 5 and all(len(history[key]) == 5 for key in history.keys() if "metric" in key)

    def test_fit_history_values_are_floats(
            self,
            dummy_compiled_sequential_model: Sequential,
            x: np.ndarray,
            y: np.ndarray,
    ):
        dummy_compiled_sequential_model.fit(x, y, 5)
        history = dummy_compiled_sequential_model.history
        assert all(isinstance(value, float) for value in history["loss"]) and all(
            all(isinstance(value, float) for value in history[key]) for key in history.keys() if "metric" in key)

    def test_fit_history_metric_increases(
            self,
            dummy_compiled_sequential_model: Sequential,
            x: np.ndarray,
            y: np.ndarray,
    ):
        dummy_compiled_sequential_model.fit(x, y, 5)
        history = dummy_compiled_sequential_model.history
        for key in history.keys():
            if "metric" in key:
                assert all(
                    history[key][i] <= history[key][i + 1] for i in range(len(history[key]) - 1)
                )

    def test_fit_history_has_val_loss_and_val_metric_when_validation_split_provided(
            self,
            dummy_compiled_sequential_model: Sequential,
            x: np.ndarray,
            y: np.ndarray,
    ):
        dummy_compiled_sequential_model.fit(x, y, 5, validation_split=0.5)
        history = dummy_compiled_sequential_model.history
        loss_class_name = dummy_compiled_sequential_model.loss.__class__.__name__
        metric_class_name = dummy_compiled_sequential_model.metrics[0].__class__.__name__
        assert f"val_loss" in history and f"val_{metric_class_name}" in history.keys()

    def test_fit_history_val_correct_length_when_validation_split_provided(
            self,
            dummy_compiled_sequential_model: Sequential,
            x: np.ndarray,
            y: np.ndarray,
    ):
        dummy_compiled_sequential_model.fit(x, y, 5, validation_split=0.5)
        history = dummy_compiled_sequential_model.history
        assert len(history["val_loss"]) == 5 and all(
            len(history[key]) == 5 for key in history.keys() if "val_metric" in key)

    def test_fit_history_val_values_are_floats_when_validation_split_provided(
            self,
            dummy_compiled_sequential_model: Sequential,
            x: np.ndarray,
            y: np.ndarray,
    ):
        dummy_compiled_sequential_model.fit(x, y, 5, validation_split=0.5)
        history = dummy_compiled_sequential_model.history
        assert all(isinstance(value, float) for value in history["val_loss"]) and all(
            all(isinstance(value, float) for value in history[key]) for key in history.keys() if "val_metric" in key)

    def test_fit_history_val_metric_increases_when_validation_split_provided(
            self,
            dummy_compiled_sequential_model: Sequential,
            x: np.ndarray,
            y: np.ndarray,
    ):
        dummy_compiled_sequential_model.fit(x, y, 5, validation_split=0.5)
        history = dummy_compiled_sequential_model.history
        for key in history.keys():
            if "val_metric" in key:
                assert all(
                    history[key][i] <= history[key][i + 1] for i in range(len(history[key]) - 1)
                )

    class TestFitUpdatesWeightsAndBiases:
        def test_layer_1_weights(
                self,
                dummy_compiled_sequential_model: Sequential,
                x: np.ndarray,
                y: np.ndarray,
        ):
            # Get the initial weights and biases of the layers
            w1, _ = dummy_compiled_sequential_model.layers[0].get_parameters()
            print("Initial weights:", w1)

            # Fit the model for one epoch
            dummy_compiled_sequential_model.fit(x, y, 1)

            # Get the updated weights and biases of the layers
            w1_new, _ = dummy_compiled_sequential_model.layers[0].get_parameters()
            print("Updated weights:", w1_new)

            # Check that the weights have changed
            assert not np.array_equal(w1, w1_new)

        def test_layer_1_biases(
                self,
                dummy_compiled_sequential_model: Sequential,
                x: np.ndarray,
                y: np.ndarray,
        ):
            # Get the initial weights and biases of the layers
            _, b1 = dummy_compiled_sequential_model.layers[0].get_parameters()
            # Fit the model for one epoch
            dummy_compiled_sequential_model.fit(x, y, 1)
            # Get the updated weights and biases of the layers
            _, b1_new = dummy_compiled_sequential_model.layers[0].get_parameters()
            # Check that the biases have changed
            assert not np.array_equal(b1, b1_new)

        def test_layer_2_weights(
                self,
                dummy_compiled_sequential_model: Sequential,
                x: np.ndarray,
                y: np.ndarray,
        ):
            # Get the initial weights and biases of the layers
            w2, _ = dummy_compiled_sequential_model.layers[1].get_parameters()
            # Fit the model for one epoch
            dummy_compiled_sequential_model.fit(x, y, 1)
            # Get the updated weights and biases of the layers
            w2_new, _ = dummy_compiled_sequential_model.layers[1].get_parameters()
            # Check that the weights have changed
            assert not np.array_equal(w2, w2_new)

        def test_layer_2_biases(
                self,
                dummy_compiled_sequential_model: Sequential,
                x: np.ndarray,
                y: np.ndarray,
        ):
            # Get the initial weights and biases of the layers
            _, b2 = dummy_compiled_sequential_model.layers[1].get_parameters()
            # Fit the model for one epoch
            dummy_compiled_sequential_model.fit(x, y, 1)
            # Get the updated weights and biases of the layers
            _, b2_new = dummy_compiled_sequential_model.layers[1].get_parameters()
            # Check that the biases have changed
            assert not np.array_equal(b2, b2_new)


class TestFitBatchSizes:
    @pytest.mark.parametrize("batch_size", [1, 2, 10, 32, 50])
    def test_fit_history_correct_length_with_batch_size(
            self,
            dummy_compiled_sequential_model: Sequential,
            x: np.ndarray,
            y: np.ndarray,
            batch_size: int,
    ):
        dummy_compiled_sequential_model.fit(x, y, 5, batch_size=batch_size)
        history = dummy_compiled_sequential_model.history
        assert len(history["loss"]) == 5 and all(
            len(history[key]) == 5 for key in history.keys() if "metric" in key)

    @pytest.mark.parametrize("batch_size", [1, 2, 10, 32, 50])
    def test_fit_history_val_correct_length_when_validation_split_provided_and_batch_size(
            self,
            dummy_compiled_sequential_model: Sequential,
            x: np.ndarray,
            y: np.ndarray,
            batch_size: int,
    ):
        dummy_compiled_sequential_model.fit(x, y, 5, validation_split=0.5, batch_size=batch_size)
        history = dummy_compiled_sequential_model.history
        assert len(history["val_loss"]) == 5 and all(
            len(history[key]) == 5 for key in history.keys() if "val_metric" in key)


class TestSequentialModelProperties:
    def test_layers_property(self, dummy_sequential_model: Sequential):
        assert hasattr(dummy_sequential_model, "layers")
        assert len(dummy_sequential_model.layers) == 2
        assert all(isinstance(layer, Dense) for layer in dummy_sequential_model.layers)

    def test_loss_property(
            self, dummy_compiled_sequential_model: Sequential, loss: Type[Loss]
    ):
        assert hasattr(dummy_compiled_sequential_model, "loss")
        assert dummy_compiled_sequential_model.loss == loss

    def test_optimizer_property(
            self, dummy_compiled_sequential_model: Sequential, optimizer: Type[Optimizer]
    ):
        assert hasattr(dummy_compiled_sequential_model, "optimizer")
        assert dummy_compiled_sequential_model.optimizer == optimizer

    def test_metrics_property(
            self, dummy_compiled_sequential_model: Sequential, metric: Type[Metric]
    ):
        assert hasattr(dummy_compiled_sequential_model, "metrics")
        assert len(dummy_compiled_sequential_model.metrics) == 1
        assert dummy_compiled_sequential_model.metrics == [metric]


def test_layer_compatibility(dummy_sequential_model: Sequential):
    # Try to add a layer with input dimension 4 and output dimension 2
    with pytest.raises(AssertionError):
        dummy_sequential_model.add(Dense(4, 2))


def test_layer_ordering(dummy_sequential_model: Sequential):
    first_layer = dummy_sequential_model.layers[0]
    second_layer = dummy_sequential_model.layers[1]

    assert isinstance(first_layer, Dense)
    assert isinstance(second_layer, Dense)

    assert first_layer.output_dim == 2
    assert second_layer.input_dim == 2


class TestPredict:
    def test_predict_output_shape(
            self, dummy_compiled_sequential_model: Sequential, x: np.ndarray
    ):
        predictions = dummy_compiled_sequential_model.predict(x)
        assert predictions.shape == (x.shape[0], 1)

    def test_predict_output_values(
            self, dummy_compiled_sequential_model: Sequential, x: np.ndarray
    ):
        predictions = dummy_compiled_sequential_model.predict(x)
        assert np.all((predictions >= 0) & (predictions <= 1))


class TestEvaluate:
    def test_evaluate_loss(
            self,
            dummy_compiled_sequential_model: Sequential,
            x: np.ndarray,
            y: np.ndarray,
    ):
        loss_value, _ = dummy_compiled_sequential_model.evaluate(x, y)
        assert isinstance(loss_value, float)

    def test_evaluate_metric(
            self,
            dummy_compiled_sequential_model: Sequential,
            x: np.ndarray,
            y: np.ndarray,
    ):
        _, metric_value = dummy_compiled_sequential_model.evaluate(x, y)
        assert isinstance(metric_value, list)
