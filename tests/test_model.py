import pytest
import numpy as np


from ..models import Sequential
from ..layers import Dense
from ..losses import MeanSquaredError
from ..losses.loss import Loss
from ..metrics import Accuracy
from ..metrics.metric import Metric


@pytest.fixture
def mse_loss() -> MeanSquaredError:
    yield MeanSquaredError()


@pytest.fixture
def accuracy_metric() -> Accuracy:
    yield Accuracy()


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
    dummy_sequential_model: Sequential, mse_loss: Loss, accuracy_metric: Metric
) -> Sequential:
    """Compile the sequential model with a loss function, and a metric."""
    dummy_sequential_model.compile(mse_loss, accuracy_metric)
    # Return the model
    return dummy_sequential_model


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
        mse_loss: Loss,
        accuracy_metric: Metric,
    ):
        # Compile the model with a loss function and a metric
        dummy_sequential_model.compile(mse_loss, accuracy_metric)
        # Check that the model has the loss function
        assert dummy_sequential_model.loss == mse_loss

    def test_model_metric_before(self, dummy_sequential_model: Sequential):
        with pytest.raises(AttributeError):
            dummy_sequential_model.metric

    def test_model_metric(
        self,
        dummy_sequential_model: Sequential,
        mse_loss: Loss,
        accuracy_metric: Metric,
    ):
        # Compile the model with a loss function and a metric
        dummy_sequential_model.compile(mse_loss, accuracy_metric)
        # Check that the model has the metric
        assert dummy_sequential_model.metric == accuracy_metric


class TestFit:
    @pytest.fixture
    def x(self) -> np.ndarray:
        # Create some dummy input data
        yield np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

    @pytest.fixture
    def y(self) -> np.ndarray:
        # Create some dummy output data
        yield np.array([[0], [1], [1], [0]])

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
            dummy_compiled_sequential_model.fit(x, y, 1, learning_rate=0.1)
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
            dummy_compiled_sequential_model.fit(x, y, 1, learning_rate=0.1)
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
            dummy_compiled_sequential_model.fit(x, y, 1, learning_rate=0.1)
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
            dummy_compiled_sequential_model.fit(x, y, 1, learning_rate=0.1)
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
        dummy_compiled_sequential_model.fit(x, y, 1, learning_rate=0.01)
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
    ):
        """Test the fit method prints the metric value."""
        # Fit the model for one epoch
        dummy_compiled_sequential_model.fit(x, y, 1, learning_rate=0.01)
        # Capture the standard output
        captured = capsys.readouterr()
        # Check that the metric value is printed
        assert "Metric" in captured.out


# Define a test function that checks if the compile method of the sequential model assigns the loss function and the metric as attributes
# Define a test function that checks if the fit method of the sequential model updates the weights and biases of the layers and prints the loss and the metric values
# Define a test function that checks if the predict method of the sequential model returns the output for the test data
# Define a test function that checks if the evaluate method of the sequential model returns the loss and the metric values for the test data
