from typing import Type, Any

import numpy as np
import pytest
from _pytest.capture import CaptureResult

from inventronet.activations import Sigmoid
from inventronet.activations.activation import Activation
from inventronet.layers import Dense
from inventronet.optimizers.optimizer import Optimizer


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
