import pytest


import numpy as np

from ..activations import Sigmoid

from ..layers import Dense, Dropout, BatchNormalization
from ..layers.shape_error import ShapeError
from hypothesis import given, settings, strategies as st


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


@pytest.fixture
def sigmoid() -> Sigmoid:
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
            dense.backward(random_error, 0.01)
        except Exception as e:
            pytest.fail(f"Differentiability test failed with exception: {e}")

    def test_weight_updates(
        self, dense: Dense, random_input: np.ndarray, random_error: np.ndarray
    ):
        """
        Test the weight update process after a backward pass. Ensures
        that the Dense layer's weights are updated after backward propagation.
        """
        learning_rate = 0.01
        original_weights = dense.weights.copy()
        dense.forward(random_input)
        dense.backward(random_error, learning_rate)
        assert not np.allclose(dense.weights, original_weights, atol=1e-6)

    def test_bias_updates(
        self, dense_with_bias: Dense, random_input: np.ndarray, random_error: np.ndarray
    ):
        """
        Test the bias update process after a backward pass. Ensures
        that the Dense layer's biases are updated after backward propagation.
        """
        learning_rate = 0.01
        original_biases = dense_with_bias.biases.copy()
        dense_with_bias.forward(random_input)
        dense_with_bias.backward(random_error, learning_rate)
        assert not np.allclose(dense_with_bias.biases, original_biases, atol=1e-6)


class TestDropout:
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
        gradient_array = dropout.backward(error, learning_rate=0.01, training=training)

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
            dropout.backward(invalid_error, learning_rate=0.01, training=training)


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
    ):
        learning_rate = 0.01
        original_gamma = batch_normalization.gamma.copy()
        original_beta = batch_normalization.beta.copy()
        batch_normalization.forward(random_input, training=True)
        batch_normalization.backward(random_error, learning_rate)
        assert not np.allclose(batch_normalization.gamma, original_gamma, atol=1e-6)
        assert not np.allclose(batch_normalization.beta, original_beta, atol=1e-6)

    def test_output_dimension(
        self, batch_normalization: BatchNormalization, random_input: np.ndarray
    ):
        output = batch_normalization.forward(random_input, training=True)
        assert output.shape == random_input.shape
