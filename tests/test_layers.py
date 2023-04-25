import inspect
from typing import Any, Type
import pytest


import numpy as np
from _pytest.capture import CaptureResult
from _pytest.fixtures import SubRequest

from inventronet.activations import Sigmoid
from inventronet.activations.activation import Activation

from inventronet.layers import Dense, Dropout, BatchNormalization
from inventronet.layers.shape_error import ShapeError
from hypothesis import given, settings, strategies as st

from inventronet.optimizers.optimizer import Optimizer
import inventronet.optimizers as optimizers_module


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


class TestDense:
    @pytest.fixture
    def dense(
        self, activation: Type[Activation], input_dim: int, random_error: np.ndarray
    ) -> Dense:
        yield Dense(
            input_dim,
            random_error.shape[1],
            activation=activation,
            use_bias=False,
        )

    @pytest.fixture
    def dense_with_bias(self, dense: Dense, random_biases: np.ndarray) -> Dense:
        dense.use_bias = True
        dense.biases = random_biases
        yield dense

    @pytest.fixture
    def forward_output_train(
        self, random_input: np.ndarray[Any, np.float64], dense: Dense
    ) -> np.ndarray:
        yield dense.forward(random_input)

    def test_linearity(self, dense: Dense, random_input: np.ndarray):
        """
        Test the linearity property of the Dense layer without
        activation function. Ensures the Dense layer performs
        a linear transformation when no activation function is used.
        """
        dense.activation = None
        input1, input2 = random_input[:5], random_input[5:]
        alpha, beta = 0.3, 0.7
        linear_combination = alpha * input1 + beta * input2
        output1, output2 = dense.forward(input1), dense.forward(input2)
        expected_output = alpha * output1 + beta * output2
        linear_combination_output = dense.forward(linear_combination)
        assert np.allclose(linear_combination_output, expected_output, atol=1e-6)

    def test_fully_connected(self, dense: Dense, random_input: np.ndarray):
        """
        Test the fully connected property of the Dense layer. Ensures
        that every input node is connected to every output node.
        """
        output = dense.forward(random_input)
        for input_vector, output_vector in zip(random_input, output):
            assert not np.all(output_vector == 0) and not np.all(input_vector == 0)

    def test_differentiability(
        self, dense: Dense, random_input: np.ndarray, random_error: np.ndarray
    ):
        """
        Test the differentiability property of the Dense layer. Ensures
        that forward and backward passes can be performed without errors.
        """
        dense.forward(random_input)
        try:
            dense.backward(random_error)
        except Exception as e:
            pytest.fail(f"Differentiability test failed with exception: {e}")

    def test_weight_gradients(
        self, dense: Dense, random_input: np.ndarray, random_error: np.ndarray
    ):
        """
        Test the weight gradient computation after a backward pass. Ensures
        that the Dense layer's weight gradients are computed after backward propagation.
        """
        dense.forward(random_input)
        dense.backward(random_error)
        assert "weights" in dense.gradients
        assert dense.gradients["weights"].shape == dense.weights.shape

    def test_bias_gradients(
        self, dense_with_bias: Dense, random_input: np.ndarray, random_error: np.ndarray
    ):
        """
        Test the bias gradient computation after a backward pass. Ensures
        that the Dense layer's bias gradients are computed after backward propagation.
        """
        dense_with_bias.forward(random_input)
        dense_with_bias.backward(random_error)
        assert "biases" in dense_with_bias.gradients
        assert dense_with_bias.gradients["biases"].shape == dense_with_bias.biases.shape

    class TestDenseInit:
        def test_name(self, dense: Dense):
            assert dense.name == "Dense"

        def test_input_shape(self, dense: Dense):
            assert dense.input_shape == (5,)

        def test_output_shape(self, dense: Dense, random_error: np.ndarray):
            assert dense.output_shape == (random_error.shape[1],)

        def test_summary(
            self, capsys: pytest.CaptureFixture, dense: Dense, random_error: np.ndarray
        ):
            expected_output = f"Layer: Dense\nInput shape: (5,)\nOutput shape: ({random_error.shape[1]},)\nNumber of parameters: 25\n"
            dense.summary()
            captured: CaptureResult = capsys.readouterr()
            assert captured.out == expected_output

    @pytest.fixture
    def forward_output_with_bias(
        self,
        random_input: np.ndarray[Any, np.float64],
        dense_with_bias: Dense,
    ) -> np.ndarray:
        yield dense_with_bias.forward(random_input)

    def test_forward(
        self,
        random_input: np.ndarray[Any, np.float64],
        activation: Sigmoid,
        forward_output_train: np.ndarray,
        dense: Dense,
    ):
        expected_output = activation(np.dot(random_input, dense.weights))
        assert np.allclose(forward_output_train, expected_output)

    def test_forward_invalid_input(
        self, random_input: np.ndarray[Any, np.float64], dense: Dense
    ):
        invalid_input = random_input[:, :4]
        with pytest.raises(ValueError):
            dense.forward(invalid_input)

    class TestBackward:
        def test_backward_output(
            self,
            random_error: np.ndarray,
            activation: Sigmoid,
            dense: Dense,
            forward_output_train: np.ndarray,
        ):
            expected_output: np.ndarray = np.dot(
                random_error * activation.derivative(forward_output_train),
                dense.weights.T,
            )
            # Assert that the output of the backward method is close to the
            # expected output
            assert np.allclose(dense.backward(random_error), expected_output, atol=1e-3)

    class TestBackwardUpdateWeightAndBiases:
        def test_updated_weights(
            self,
            random_input: np.ndarray,
            random_error: np.ndarray,
            activation: Sigmoid,
            dense_with_bias: Dense,
            forward_output_with_bias: np.ndarray,
            optimizer: Type[Optimizer],
        ):
            original_weights = dense_with_bias.weights.copy()
            dense_with_bias.backward(random_error)
            optimizer.update(0, dense_with_bias.parameters, dense_with_bias.gradients)
            learning_rate = optimizer.learning_rate
            expected_weights: np.ndarray = original_weights - learning_rate * np.dot(
                random_input.T,
                random_error * activation.derivative(forward_output_with_bias),
            )

            assert np.allclose(dense_with_bias.weights, expected_weights, atol=1e-1)

        def test_updated_biases(
            self,
            random_error: np.ndarray,
            activation: Sigmoid,
            dense_with_bias: Dense,
            forward_output_with_bias: np.ndarray,
            optimizer: Type[Optimizer],
        ):
            original_bias = dense_with_bias.biases.copy()
            dense_with_bias.backward(random_error)
            optimizer.update(0, dense_with_bias.parameters, dense_with_bias.gradients)
            expected_bias: np.ndarray = original_bias - 0.01 * np.sum(
                random_error * activation.derivative(forward_output_with_bias), axis=0
            )
            assert np.allclose(dense_with_bias.biases, expected_bias, atol=1e-2)

    @pytest.mark.usefixtures("forward_output_train")
    def test_backward_invalid_error(
        self, random_error: np.ndarray[Any, np.dtype[np.floating[Any]]], dense: Dense
    ):
        invalid_error = random_error[:, :2]
        with pytest.raises(ValueError):
            dense.backward(invalid_error, 0.01)


class TestDropout:
    @pytest.fixture
    def training(self) -> bool:
        yield True

    @pytest.fixture
    def dropout_rate(self) -> float:
        yield 0.2

    @pytest.fixture
    def dropout(self, dropout_rate: float, input_dim: int) -> Dropout:
        yield Dropout(dropout_rate=dropout_rate, input_dim=input_dim)

    @given(
        st.floats(min_value=0.0, max_value=1.0), st.integers(min_value=1, max_value=100)
    )
    def test_dropout_init(self, dropout_rate: float, input_dim: int):
        dropout = Dropout(dropout_rate=dropout_rate, input_dim=input_dim)
        assert dropout.dropout_rate == dropout_rate
        assert dropout.mask is None

    @given(
        st.floats(min_value=0.0, max_value=1.0),
        st.integers(min_value=1, max_value=100),
        st.booleans(),
    )
    @settings(max_examples=20)
    def test_forward(self, dropout_rate: float, input_dim: int, training: bool):
        input_shape = (10, input_dim)
        random_input = generate_random_array(input_shape)

        dropout = Dropout(dropout_rate=dropout_rate, input_dim=input_dim)
        output_array = dropout.forward(random_input, training=training)
        assert output_array.shape == random_input.shape

        if training:
            assert dropout.mask.shape == random_input.shape
            if dropout_rate < 1:
                assert np.allclose(
                    output_array, random_input * dropout.mask / (1 - dropout_rate)
                )

    @given(
        st.floats(min_value=0.0, max_value=1.0),
        st.integers(min_value=1, max_value=100),
        st.booleans(),
    )
    @settings(max_examples=20)
    def test_backward(self, dropout_rate: float, input_dim: int, training: bool):
        input_shape = (10, input_dim)
        random_input = generate_random_array(input_shape)

        dropout = Dropout(dropout_rate=dropout_rate, input_dim=input_dim)
        output_array = dropout.forward(random_input, training=training)
        error = generate_random_array(output_array.shape)
        gradient_array = dropout.backward(error, training=training)

        assert gradient_array.shape == random_input.shape

        if training and dropout_rate < 1:
            assert np.allclose(
                gradient_array, error * dropout.mask / (1 - dropout_rate)
            )

    @given(
        st.floats(min_value=0.0, max_value=1.0),
        st.integers(min_value=1, max_value=100),
        st.booleans(),
    )
    @settings(max_examples=20)
    def test_input_validation(
        self, dropout_rate: float, input_dim: int, training: bool
    ):
        dropout = Dropout(dropout_rate=dropout_rate, input_dim=input_dim)

        with pytest.raises(ValueError):
            dropout.forward(42, training=training)

        with pytest.raises(ValueError):
            dropout.forward([1, 2, 3], training=training)

        with pytest.raises(ValueError):
            dropout.forward("Hello, world!", training=training)

    @given(
        st.floats(min_value=0.0, max_value=1.0),
        st.integers(min_value=1, max_value=100),
        st.booleans(),
    )
    @settings(max_examples=20)
    def test_error_validation(
        self, dropout_rate: float, input_dim: int, training: bool
    ):
        random_input = generate_random_array((5, input_dim))

        dropout = Dropout(dropout_rate=dropout_rate, input_dim=input_dim)
        dropout.forward(random_input, training=True)

        with pytest.raises(ShapeError):
            invalid_error = generate_random_array((2, 3, 4))
            dropout.backward(invalid_error, training=training)

    class TestDroputInit:
        def test_droput_rate(self, dropout: Dropout, dropout_rate: float):
            assert dropout.dropout_rate == dropout_rate

        def test_droput_mask(self, dropout: Dropout):
            assert dropout.mask is None

    class TestForward:
        def test_output_shape(
            self, dropout: Dropout, random_input: np.ndarray, training: bool
        ):
            output_array = dropout.forward(random_input, training=training)
            assert output_array.shape == random_input.shape

        def test_mask_shape(
            self, dropout: Dropout, random_input: np.ndarray, training: bool
        ):
            dropout.forward(random_input, training)
            assert dropout.mask.shape == random_input.shape

        def test_output_value(
            self,
            dropout: Dropout,
            dropout_rate: float,
            random_input: np.ndarray,
            training: bool,
        ):
            output_array = dropout.forward(random_input, training=training)
            assert np.allclose(
                output_array,
                random_input * dropout.mask / (1 - dropout_rate),
            )

    class TestBackward:
        def test_gradient_shape(
            self, dropout: Dropout, random_input: np.ndarray, training: bool
        ):
            output_array = dropout.forward(random_input, training=training)
            error = np.random.randn(*output_array.shape)
            gradient_array = dropout.backward(error, training=training)
            assert gradient_array.shape == random_input.shape

        def test_gradient_value(
            self,
            dropout: Dropout,
            dropout_rate: float,
            random_input: np.ndarray,
            training: bool,
        ):
            output_array = dropout.forward(random_input, training=training)
            error = np.random.randn(*output_array.shape)
            gradient_array = dropout.backward(error, training=training)
            assert np.allclose(
                gradient_array, error * dropout.mask / (1 - dropout_rate)
            )

    @pytest.mark.parametrize("dropout_rate", [0])
    @pytest.mark.parametrize("training", [True])
    def test_dropout_rate_zero_training_true(
        self,
        dropout_rate: float,
        random_input: np.ndarray,
        training: bool,
        input_dim: int,
    ):
        dropout = Dropout(dropout_rate, input_dim=input_dim)
        output_array = dropout.forward(random_input, training=training)
        assert np.allclose(output_array, random_input) and np.all(dropout.mask == 1)

    @pytest.mark.parametrize("dropout_rate", [1])
    @pytest.mark.parametrize("training", [True])
    def test_dropout_rate_one_training_true(
        self,
        dropout_rate: float,
        random_input: np.ndarray,
        training: bool,
        input_dim: int,
    ):
        dropout = Dropout(dropout_rate, input_dim=input_dim)
        output_array = dropout.forward(random_input, training=training)
        print("Output array: ", output_array)
        print("Dropout Mask: ", dropout.mask)
        assert np.all(output_array == 0) and np.all(dropout.mask == 0)

    @pytest.mark.parametrize("dropout_rate", [0])
    @pytest.mark.parametrize("training", [False])
    def test_dropout_rate_zero_training_false(
        self,
        dropout_rate: float,
        random_input: np.ndarray,
        training: bool,
        input_dim: int,
    ):
        dropout = Dropout(dropout_rate, input_dim=input_dim)
        output_array = dropout.forward(random_input, training=training)
        assert np.allclose(output_array, random_input)

    @pytest.mark.parametrize("dropout_rate", [1])
    @pytest.mark.parametrize("training", [False])
    def test_dropout_rate_one_training_false(
        self,
        dropout_rate: float,
        random_input: np.ndarray,
        training: bool,
        input_dim: int,
    ):
        dropout = Dropout(dropout_rate, input_dim=input_dim)
        output_array = dropout.forward(random_input, training=training)
        assert np.allclose(output_array, random_input)


class TestBatchNormalization:
    @pytest.fixture
    def epsilon(self) -> float:
        yield 1e-5

    @pytest.fixture
    def batch_normalization(self, input_dim: int, epsilon: float) -> BatchNormalization:
        yield BatchNormalization(
            input_dim=input_dim,
            output_dim=input_dim,
            epsilon=epsilon,
        )

    @pytest.fixture
    def forward_output_train(
        self, batch_normalization: BatchNormalization, random_input: np.ndarray
    ) -> np.ndarray:
        yield batch_normalization.forward(random_input, training=True)

    @pytest.fixture
    def forward_output_inference(
        self, batch_normalization: BatchNormalization, random_input: np.ndarray
    ) -> np.ndarray:
        yield batch_normalization.forward(random_input, training=False)

    def test_mean_and_variance_normalization(
        self, batch_normalization: BatchNormalization, random_input: np.ndarray
    ):
        output = batch_normalization.forward(random_input, training=True)
        assert np.isclose(np.mean(output, axis=0), 0, atol=1e-6).all()
        assert np.isclose(np.var(output, axis=0), 1, atol=1e-4).all()

    def test_running_mean_and_variance_update(
        self, batch_normalization: BatchNormalization, random_input: np.ndarray
    ):
        initial_running_mean = batch_normalization.running_mean.copy()
        initial_running_var = batch_normalization.running_var.copy()
        batch_normalization.forward(random_input, training=True)
        assert not np.allclose(batch_normalization.running_mean, initial_running_mean)
        assert not np.allclose(batch_normalization.running_var, initial_running_var)

    def test_use_running_statistics_during_inference(
        self, batch_normalization: BatchNormalization, random_input: np.ndarray
    ):
        batch_normalization.forward(random_input, training=True)
        output = batch_normalization.forward(random_input, training=False)
        expected_output = (
            batch_normalization.gamma
            * (random_input - batch_normalization.running_mean)
            / np.sqrt(batch_normalization.running_var + batch_normalization.epsilon)
            + batch_normalization.beta
        )
        assert np.allclose(output, expected_output)

    def test_differentiability(
        self,
        batch_normalization: BatchNormalization,
        random_input: np.ndarray,
        random_error: np.ndarray,
    ):
        batch_normalization.forward(random_input, training=True)
        try:
            batch_normalization.backward(random_error, 0.01)
        except Exception as e:
            pytest.fail(f"Differentiability test failed with exception: {e}")

    def test_parameter_updates(
        self,
        batch_normalization: BatchNormalization,
        random_input: np.ndarray,
        random_error: np.ndarray,
        optimizer: Type[Optimizer],
    ):
        original_gamma = batch_normalization.gamma.copy()
        original_beta = batch_normalization.beta.copy()

        batch_normalization.forward(random_input, training=True)
        batch_normalization.backward(random_error)

        optimizer.update(
            0, batch_normalization.parameters, batch_normalization.gradients
        )

        assert not np.allclose(batch_normalization.gamma, original_gamma, atol=1e-6)
        assert not np.allclose(batch_normalization.beta, original_beta, atol=1e-6)

    def test_output_dimension(
        self, batch_normalization: BatchNormalization, random_input: np.ndarray
    ):
        output = batch_normalization.forward(random_input, training=True)
        assert output.shape == random_input.shape

    class TestForward:
        def test_forward_training_mode_without_activation(
            self,
            random_input: np.ndarray,
            batch_normalization: BatchNormalization,
            forward_output_train: np.ndarray,
        ):
            batch_mean = np.mean(random_input, axis=0)
            batch_var = np.var(random_input, axis=0)
            normalized_input = (random_input - batch_mean) / np.sqrt(
                batch_var + batch_normalization.epsilon
            )
            expected_output = (
                batch_normalization.gamma * normalized_input + batch_normalization.beta
            )
            assert np.allclose(forward_output_train, expected_output)

        def test_forward_inference_mode_without_activation(
            self,
            batch_normalization: BatchNormalization,
            random_input: np.ndarray,
            forward_output_inference: np.ndarray,
        ):
            normalized_input = (
                random_input - batch_normalization.running_mean
            ) / np.sqrt(batch_normalization.running_var + batch_normalization.epsilon)
            expected_output = (
                batch_normalization.gamma * normalized_input + batch_normalization.beta
            )
            assert np.allclose(forward_output_inference, expected_output)

    class TestBackward:
        @pytest.fixture
        def batch_size(self, random_input: np.ndarray) -> int:
            yield random_input.shape[0]

        @pytest.fixture
        def gradients(self, random_input: np.ndarray) -> np.ndarray:
            yield np.random.randn(*random_input.shape)

        def test_backward_pass(
            self,
            batch_normalization: BatchNormalization,
            gradients: np.ndarray,
            random_input: np.ndarray,
            epsilon: float,
        ) -> None:
            batch_mean = np.mean(random_input, axis=0)
            batch_var = np.var(random_input, axis=0)
            normalized_input = (random_input - batch_mean) / np.sqrt(
                batch_var + batch_normalization.epsilon
            )
            expected_output = (
                gradients
                - np.mean(gradients, axis=0)
                - normalized_input * np.mean(gradients * normalized_input, axis=0)
            ) / np.sqrt(batch_var + epsilon)
            batch_normalization.forward(random_input, training=True)
            output = batch_normalization.backward(gradients)
            assert np.allclose(output, expected_output, rtol=0.1)

        def test_backward_gamma_without_activation(
            self,
            batch_normalization: BatchNormalization,
            random_input: np.ndarray,
            gradients: np.ndarray,
        ):
            batch_normalization.forward(random_input, training=True)
            batch_normalization.backward(gradients)

            batch_mean = np.mean(random_input, axis=0)
            batch_var = np.var(random_input, axis=0)
            normalized_input = (random_input - batch_mean) / np.sqrt(
                batch_var + batch_normalization.epsilon
            )
            expected_gradient_gamma = np.sum(gradients * normalized_input, axis=0)
            assert np.allclose(
                batch_normalization.gradients["gamma"], expected_gradient_gamma
            )

        def test_backward_beta_without_activation(
            self,
            batch_normalization: BatchNormalization,
            random_input: np.ndarray,
            gradients: np.ndarray,
        ):
            batch_normalization.forward(random_input, training=True)
            batch_normalization.backward(gradients)

            expected_gradient_beta = np.sum(gradients, axis=0)
            assert np.allclose(
                batch_normalization.gradients["beta"], expected_gradient_beta
            )
