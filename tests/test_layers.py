from typing import Any
import pytest

from _pytest.capture import CaptureResult

import numpy as np

from ..activations import Sigmoid

from ..layers import Dense, Dropout, BatchNormalization
from ..layers.shape_error import ShapeError


@pytest.fixture
def random_input():
    np.random.seed(42)
    yield np.random.randn(10, 5)


@pytest.fixture
def input_dim(random_input: np.ndarray) -> int:
    yield random_input.shape[1]


@pytest.fixture
def random_error():
    np.random.seed(42)
    yield np.random.rand(10, 3) / 2


@pytest.fixture
def random_biases() -> np.ndarray:
    np.random.seed(42)
    yield np.random.rand(3) * 2


@pytest.fixture
def sigmoid():
    yield Sigmoid()


class TestDense:
    @pytest.fixture
    def dense(
        self, sigmoid: Sigmoid, input_dim: int, random_error: np.ndarray
    ) -> Dense:
        yield Dense(
            input_dim,
            random_error.shape[1],
            activation=sigmoid,
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

    class TestDenseInit:
        def test_name(self, dense: Dense):
            assert dense.name == "Dense"

        def test_input_shape(self, dense: Dense):
            assert dense.input_shape == (5,)

        def test_output_shape(self, dense: Dense):
            assert dense.output_shape == (3,)

        def test_summary(self, capsys: pytest.CaptureFixture, dense: Dense):
            expected_output = "Layer: Dense\nInput shape: (5,)\nOutput shape: (3,)\nNumber of parameters: 15\n"
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
        sigmoid: Sigmoid,
        forward_output_train: np.ndarray,
        dense: Dense,
    ):
        expected_output = sigmoid(np.dot(random_input, dense.weights))
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
            sigmoid: Sigmoid,
            dense: Dense,
            forward_output_train: np.ndarray,
        ):
            expected_output: np.ndarray = np.dot(
                random_error * sigmoid.derivative(forward_output_train),
                dense.weights.T,
            )
            # Assert that the output of the backward method is close to the
            # expected output
            assert np.allclose(
                dense.backward(random_error, 0.01), expected_output, atol=1e-3
            )

        def test_backward_update_weights(
            self,
            random_input: np.ndarray,
            random_error: np.ndarray,
            sigmoid: Sigmoid,
            dense: Dense,
            forward_output_train: np.ndarray,
        ):
            expected_weights: np.ndarray = dense.weights - 0.01 * np.dot(
                random_input.T,
                random_error * sigmoid.derivative(forward_output_train),
            )
            assert np.allclose(dense.weights, expected_weights, atol=0.1)

    class TestBackwardUpdateWeightAndBiases:
        def test_updated_weights(
            self,
            random_input: np.ndarray,
            random_error: np.ndarray,
            sigmoid: Sigmoid,
            dense_with_bias: Dense,
            forward_output_with_bias: np.ndarray,
        ):
            original_weights = dense_with_bias.weights.copy()
            dense_with_bias.backward(random_error, 0.01)
            expected_weights: np.ndarray = original_weights - 0.01 * np.dot(
                random_input.T,
                random_error * sigmoid.derivative(forward_output_with_bias),
            )
            assert np.allclose(dense_with_bias.weights, expected_weights, atol=1e-2)

        def test_updated_biases(
            self,
            random_error: np.ndarray,
            sigmoid: Sigmoid,
            dense_with_bias: Dense,
            forward_output_with_bias: np.ndarray,
        ):
            original_bias = dense_with_bias.biases.copy()
            dense_with_bias.backward(random_error, 0.01)
            expected_bias: np.ndarray = original_bias - 0.01 * np.sum(
                random_error * sigmoid.derivative(forward_output_with_bias), axis=0
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

    class TestDroputInit:
        def test_droput_rate(self, dropout: Dropout, dropout_rate: float):
            assert dropout.dropout_rate == dropout_rate

        def test_droput_mask(self, dropout: Dropout):
            assert dropout.mask == None

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
            gradient_array = dropout.backward(
                error, learning_rate=0.01, training=training
            )
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
            gradient_array = dropout.backward(
                error, learning_rate=0.01, training=training
            )
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

    @pytest.mark.parametrize("dropout_rate", [0.1, 0.5, 0.9])
    @pytest.mark.parametrize("training", [True, False])
    @pytest.mark.parametrize("invalid_input", [42, [1, 2, 3], "Hello, world!"])
    def test_input_validation(
        self, dropout: Dropout, training: bool, invalid_input: Any
    ):
        with pytest.raises(ValueError):
            dropout.forward(invalid_input, training=training)

    @pytest.mark.parametrize("dropout_rate", [0.1, 0.5, 0.9])
    @pytest.mark.parametrize("learning_rate", [0.01, 0.1, 0.5])
    @pytest.mark.parametrize("invalid_error", [np.random.randn(2, 3, 4)])
    def test_error_validation(
        self, dropout: Dropout, learning_rate: float, invalid_error: Any, training: bool
    ):
        input_array = np.random.randn(3, 4)
        dropout.forward(input_array, training=True)
        with pytest.raises(ShapeError):
            dropout.backward(
                invalid_error, learning_rate=learning_rate, training=training
            )


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

        @pytest.fixture
        def learning_rate(
            self,
        ) -> float:
            yield 0.01

        def test_backward_pass(
            self,
            batch_normalization: BatchNormalization,
            gradients: np.ndarray,
            learning_rate: float,
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
            output = batch_normalization.backward(
                gradients, learning_rate=learning_rate
            )
            assert np.allclose(output, expected_output, rtol=0.1)

        def test_backward_gamma_without_activation(
            self,
            batch_normalization: BatchNormalization,
            random_input: np.ndarray,
            gradients: np.ndarray,
        ):
            batch_normalization.forward(random_input, training=True)
            batch_normalization.backward(gradients, learning_rate=0.01)

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
            batch_normalization.backward(gradients, learning_rate=0.01)

            expected_gradient_beta = np.sum(gradients, axis=0)
            assert np.allclose(
                batch_normalization.gradients["beta"], expected_gradient_beta
            )
