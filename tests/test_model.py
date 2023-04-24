from typing import Type
import pytest
import numpy as np
from inventronet.optimizers import StochasticGradientDescent

from inventronet.optimizers.optimizer import Optimizer


from inventronet.models import Sequential
from inventronet.layers import Dense
from inventronet.losses import MeanSquaredError
from inventronet.losses.loss import Loss
from inventronet.metrics import Accuracy
from inventronet.metrics.metric import Metric


@pytest.fixture
def loss() -> Type[Loss]:
    yield MeanSquaredError()


@pytest.fixture
def metric() -> Type[Metric]:
    yield Accuracy()


@pytest.fixture
def optimizer() -> Type[Optimizer]:
    yield StochasticGradientDescent()


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


@pytest.fixture
def x() -> np.ndarray:
    # Create some dummy input data
    yield np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])


@pytest.fixture
def y() -> np.ndarray:
    # Create some dummy output data
    yield np.array([[0], [1], [1], [0]])


def test_add(dummy_sequential_model: Sequential):
    """Test the add method of the sequential model."""
    # Try to add a layer with input dimension 4 and output dimension 2
    with pytest.raises(AssertionError):
        dummy_sequential_model.add(Dense(4, 2))
    # Check that the model still has two layers
    assert len(dummy_sequential_model.layers) == 2


class TestCompile:
    def test_model_loss_before(self, dummy_sequential_model: Sequential):
        with pytest.raises(AttributeError):
            dummy_sequential_model.loss

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
    class TestFitUpdatesWeightsAndBiases:
        def test_layer_1_weights(
            self,
            dummy_compiled_sequential_model: Sequential,
            x: np.ndarray,
            y: np.ndarray,
        ):
            # Get the initial weights and biases of the layers
            w1, _ = dummy_compiled_sequential_model.layers[0].get_parameters()
            # Fit the model for one epoch
            dummy_compiled_sequential_model.fit(x, y, 1)
            # Get the updated weights and biases of the layers
            w1_new, _ = dummy_compiled_sequential_model.layers[0].get_parameters()
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

    def test_fit_prints_loss(
        self,
        dummy_compiled_sequential_model: Sequential,
        capsys: pytest.CaptureFixture,
        x: np.ndarray,
        y: np.ndarray,
    ):
        # Fit the model for one epoch
        dummy_compiled_sequential_model.fit(x, y, 1)
        # Capture the standard output
        captured = capsys.readouterr()
        # Check that the loss value is printed
        assert "Loss" in captured.out

    def test_fit_prints_metric(
        self,
        dummy_compiled_sequential_model: Sequential,
        capsys: pytest.CaptureFixture,
        x: np.ndarray,
        y: np.ndarray,
        metric: Type[Metric],
    ):
        """Test the fit method prints the metric value."""
        # Fit the model for one epoch
        dummy_compiled_sequential_model.fit(x, y, 1)
        # Capture the standard output
        captured = capsys.readouterr()
        # Check that the metric value is printed
        assert f"{metric.__class__.__name__}" in captured.out


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


def test_model_compilation(dummy_sequential_model: Sequential):
    with pytest.raises(AttributeError):
        dummy_sequential_model.loss

    with pytest.raises(AttributeError):
        dummy_sequential_model.optimizer

    with pytest.raises(AttributeError):
        dummy_sequential_model.metrics


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
