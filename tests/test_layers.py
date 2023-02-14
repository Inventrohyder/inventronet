# Import the pytest module
from typing import Any
import pytest

from _pytest.capture import CaptureResult

# Import the numpy module
import numpy as np

# Import the activation module
from ..activations import Sigmoid

# Import the layer module
from ..layers import Dense, Dropout, BatchNormalization, ShapeError


# Define a fixture for a random input array
@pytest.fixture
def random_input():
    # Set the seed of the random number generator
    np.random.seed(42)
    # Return a random array of shape (10, 5)
    yield np.random.randn(10, 5)


@pytest.fixture
def input_dim(random_input: np.ndarray) -> int:
    yield random_input.shape[1]


# Define a fixture for a random error array
@pytest.fixture
def random_error():
    # Set the seed of the random number generator
    np.random.seed(42)
    # Return a random array of shape (10, 3) with values between 0 and 0.5
    yield np.random.rand(10, 3) / 2


@pytest.fixture
def random_biases() -> np.ndarray:
    # Set the seed of the random number generator
    np.random.seed(42)
    # Return a random array of shape (3,) with values between 0 and 2
    yield np.random.rand(3) * 2


# Define a fixture for a sigmoid activation function
@pytest.fixture
def sigmoid():
    # Return an instance of the Sigmoid class
    yield Sigmoid()


class TestDense:
    @pytest.fixture
    def dense(
        self, sigmoid: Sigmoid, input_dim: int, random_error: np.ndarray
    ) -> Dense:
        # Create an instance of the Dense class with input_dim=5, output_dim=3, activation=sigmoid, and use_bias=False
        yield Dense(
            input_dim,
            random_error.shape[1],
            activation=sigmoid,
            use_bias=False,
        )

    @pytest.fixture
    def dense_with_bias(self, dense: Dense, random_biases: np.ndarray) -> Dense:
        # Set the use_bias attribute to True and initialize the bias array
        dense.use_bias = True
        dense.biases = random_biases
        yield dense

    @pytest.fixture
    def forward_output_train(self, random_input, dense: Dense) -> np.ndarray:
        # Call the forward method on the random input and return the output
        yield dense.forward(random_input)

    class TestDenseInit:
        def test_name(self, dense: Dense):
            # Check that the name property returns "Dense"
            assert dense.name == "Dense"

        def test_input_shape(self, dense: Dense):
            # Check that the input_shape property returns (5,)
            assert dense.input_shape == (5,)

        def test_output_shape(self, dense: Dense):
            # Check that the output_shape property returns (3,)
            assert dense.output_shape == (3,)

        # Capture the output of the summary method using the capsys fixture
        def test_summary(self, capsys: pytest.CaptureFixture, dense: Dense):
            # Check that the summary method prints the expected output
            expected_output = "Layer: Dense\nInput shape: (5,)\nOutput shape: (3,)\nNumber of parameters: 15\n"

            # Call the summary method
            dense.summary()
            # Get the captured output
            captured: CaptureResult = capsys.readouterr()
            # Check that the captured output matches the expected output
            assert captured.out == expected_output

    @pytest.fixture
    def forward_output_with_bias(
        self, random_input, dense_with_bias: Dense
    ) -> np.ndarray:
        # Call the forward method on the random input and return the output
        yield dense_with_bias.forward(random_input)

    def test_forward(
        self,
        random_input,
        sigmoid: Sigmoid,
        forward_output_train: np.ndarray,
        dense: Dense,
    ):
        # Check that the forward method returns the expected output
        expected_output = sigmoid(np.dot(random_input, dense.weights))
        assert np.allclose(forward_output_train, expected_output)

    def test_forward_invalid_input(self, random_input, dense: Dense):
        # Change the shape of the input to make it incompatible with the weights
        invalid_input = random_input[:, :4]
        # Expect a ValueError to be raised when calling the forward method
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
            # Check that the backward method returns the expected output and updates the weights
            # The expected output is the error propagated to the previous layer, calculated by
            # multiplying the error by the derivative of the activation function and the transpose of the weights
            expected_output: np.ndarray = np.dot(
                random_error * sigmoid.derivative(forward_output_train),
                dense.weights.T,
            )
            # Assert that the output of the backward method is close to the expected output
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
            # The expected weights are the updated weights after applying the gradient descent algorithm, calculated by
            # subtracting the product of the learning rate, the transpose of the input, and the gradient of the error
            expected_weights: np.ndarray = dense.weights - 0.01 * np.dot(
                random_input.T,
                random_error * sigmoid.derivative(forward_output_train),
            )
            # Assert that the weights of the dense layer are close to the expected weights
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
            # Save the original weights before calling the backward method
            original_weights = dense_with_bias.weights.copy()
            # Call the backward method with the random error and a learning rate of 0.01
            dense_with_bias.backward(random_error, 0.01)
            # The expected weights are the updated weights after applying the gradient descent algorithm, calculated by
            # subtracting the product of the learning rate, the transpose of the input, and the gradient of the error
            expected_weights: np.ndarray = original_weights - 0.01 * np.dot(
                random_input.T,
                random_error * sigmoid.derivative(forward_output_with_bias),
            )
            # Assert that the weights are close to the expected values
            assert np.allclose(dense_with_bias.weights, expected_weights, atol=1e-2)

        def test_updated_biases(
            self,
            random_error: np.ndarray,
            sigmoid: Sigmoid,
            dense_with_bias: Dense,
            forward_output_with_bias: np.ndarray,
        ):
            # Save the original bias before calling the backward method
            original_bias = dense_with_bias.biases.copy()
            # Call the backward method with the random error and a learning rate of 0.01
            dense_with_bias.backward(random_error, 0.01)
            # The expected bias is the updated bias after applying the gradient descent algorithm, calculated by
            # subtracting the product of the learning rate and the sum of the gradient of the error
            expected_bias: np.ndarray = original_bias - 0.01 * np.sum(
                random_error * sigmoid.derivative(forward_output_with_bias), axis=0
            )
            # Assert that the biases are close to the expected values
            assert np.allclose(dense_with_bias.biases, expected_bias, atol=1e-2)

    @pytest.mark.usefixtures("forward_output_train")
    def test_backward_invalid_error(self, random_error, dense: Dense):
        # Change the shape of the error to make it incompatible with the weights
        invalid_error = random_error[:, :2]
        # Expect a ValueError to be raised when calling the backward method
        with pytest.raises(ValueError):
            dense.backward(invalid_error, 0.01)


class TestDropout:
    # Define a fixture for providing the training flag to test
    @pytest.fixture
    def training(self) -> bool:
        # Create a training flag with a value of True
        yield True

    @pytest.fixture
    def dropout_rate(self) -> float:
        yield 0.2

    @pytest.fixture
    def dropout(self, dropout_rate: float, input_dim: int) -> Dropout:
        # Create an instance of the Dropout class with a dropout rate of 0.2
        yield Dropout(dropout_rate=dropout_rate, input_dim=input_dim)

    class TestDroputInit:
        def test_droput_rate(self, dropout: Dropout, dropout_rate: float):
            # Check that the dropout_rate attribute of the instance is 0.2
            assert dropout.dropout_rate == dropout_rate

        def test_droput_mask(self, dropout: Dropout):
            # Check that the mask attribute of the instance is None
            assert dropout.mask == None

    # Define a class for testing the forward method of the dropout layer
    class TestForward:
        # Define a test function for checking that the output array has the same shape as the input array
        def test_output_shape(
            self, dropout: Dropout, random_input: np.ndarray, training: bool
        ):
            # Call the forward method of the dropout instance with the input array and the training flag as arguments
            output_array = dropout.forward(random_input, training=training)
            # Check that the output array has the same shape as the input array
            assert output_array.shape == random_input.shape

        # Define a test function for checking that the mask attribute of the dropout instance has the same shape as the input array
        def test_mask_shape(
            self, dropout: Dropout, random_input: np.ndarray, training: bool
        ):
            # Call the forward method of the dropout instance with the input array and the training flag as arguments
            dropout.forward(random_input, training)
            # Check that the mask attribute of the dropout instance has the same shape as the input array
            assert dropout.mask.shape == random_input.shape

        # Define a test function for checking that the output array is equal to the input array multiplied by the mask attribute
        def test_output_value(
            self,
            dropout: Dropout,
            dropout_rate: float,
            random_input: np.ndarray,
            training: bool,
        ):
            # Call the forward method of the dropout instance with the input array and the training flag as arguments
            output_array = dropout.forward(random_input, training=training)
            # Check that the output array is equal to the input array multiplied by the mask attribute
            assert np.allclose(
                output_array,
                random_input * dropout.mask / (1 - dropout_rate),
            )

    # Define a class for testing the backward method of the dropout layer
    class TestBackward:
        # Define a test function for checking that the gradient array has the same shape as the input array
        def test_gradient_shape(
            self, dropout: Dropout, random_input: np.ndarray, training: bool
        ):
            # Call the forward method of the dropout instance with the input array and the training flag as arguments
            output_array = dropout.forward(random_input, training=training)
            # Create a random error array of the same shape as the output array
            error = np.random.randn(*output_array.shape)
            # Call the backward method of the dropout instance with the error array and a learning rate of 0.01 as arguments
            gradient_array = dropout.backward(
                error, learning_rate=0.01, training=training
            )
            # Check that the gradient array has the same shape as the input array
            assert gradient_array.shape == random_input.shape

        # Define a test function for checking that the gradient array is equal to the error array multiplied by the mask attribute
        def test_gradient_value(
            self,
            dropout: Dropout,
            dropout_rate: float,
            random_input: np.ndarray,
            training: bool,
        ):
            # Call the forward method of the dropout instance with the input array and the training flag as arguments
            output_array = dropout.forward(random_input, training=training)
            # Create a random error array of the same shape as the output array
            error = np.random.randn(*output_array.shape)
            # Call the backward method of the dropout instance with the error array and a learning rate of 0.01 as arguments
            gradient_array = dropout.backward(
                error, learning_rate=0.01, training=training
            )
            # Check that the gradient array is equal to the error array multiplied by the mask attribute and scaled up by the inverse of the dropout rate
            assert np.allclose(
                gradient_array, error * dropout.mask / (1 - dropout_rate)
            )

    # Define a parametrized test for testing the dropout rate when the training flag is True and the dropout rate is zero
    @pytest.mark.parametrize("dropout_rate", [0])
    @pytest.mark.parametrize("training", [True])
    def test_dropout_rate_zero_training_true(
        self,
        dropout_rate: float,
        random_input: np.ndarray,
        training: bool,
        input_dim: int,
    ):
        # Create a dropout instance with the given dropout rate
        dropout = Dropout(dropout_rate, input_dim=input_dim)
        # Call the forward method of the dropout instance with the input array and the training flag as arguments
        output_array = dropout.forward(random_input, training=training)
        # Check that the output array is equal to the input array and the mask attribute of the dropout instance is an array of ones
        assert np.allclose(output_array, random_input) and np.all(dropout.mask == 1)

    # Define a parametrized test for testing the dropout rate when the training flag is True and the dropout rate is one
    @pytest.mark.parametrize("dropout_rate", [1])
    @pytest.mark.parametrize("training", [True])
    def test_dropout_rate_one_training_true(
        self,
        dropout_rate: float,
        random_input: np.ndarray,
        training: bool,
        input_dim: int,
    ):
        # Create a dropout instance with the given dropout rate
        dropout = Dropout(dropout_rate, input_dim=input_dim)
        # Call the forward method of the dropout instance with the input array and the training flag as arguments
        output_array = dropout.forward(random_input, training=training)
        print("Output array: ", output_array)
        print("Dropout Mask: ", dropout.mask)
        # Check that the output array is an array of zeros and the mask attribute of the dropout instance is an array of zeros
        assert np.all(output_array == 0) and np.all(dropout.mask == 0)

    # Define a parametrized test for testing the dropout rate when the training flag is False and the dropout rate is zero
    @pytest.mark.parametrize("dropout_rate", [0])
    @pytest.mark.parametrize("training", [False])
    def test_dropout_rate_zero_training_false(
        self,
        dropout_rate: float,
        random_input: np.ndarray,
        training: bool,
        input_dim: int,
    ):
        # Create a dropout instance with the given dropout rate
        dropout = Dropout(dropout_rate, input_dim=input_dim)
        # Call the forward method of the dropout instance with the input array and the training flag as arguments
        output_array = dropout.forward(random_input, training=training)
        # Check that the output array is equal to the input array
        assert np.allclose(output_array, random_input)

    # Define a parametrized test for testing the dropout rate when the training flag is False and the dropout rate is one
    @pytest.mark.parametrize("dropout_rate", [1])
    @pytest.mark.parametrize("training", [False])
    def test_dropout_rate_one_training_false(
        self,
        dropout_rate: float,
        random_input: np.ndarray,
        training: bool,
        input_dim: int,
    ):
        # Create a dropout instance with the given dropout rate
        dropout = Dropout(dropout_rate, input_dim=input_dim)
        # Call the forward method of the dropout instance with the input array and the training flag as arguments
        output_array = dropout.forward(random_input, training=training)
        # Check that the output array is equal to the input array
        assert np.allclose(output_array, random_input)

    # Define a parametrized test for checking the input validation
    @pytest.mark.parametrize("dropout_rate", [0.1, 0.5, 0.9])
    @pytest.mark.parametrize("training", [True, False])
    @pytest.mark.parametrize("invalid_input", [42, [1, 2, 3], "Hello, world!"])
    def test_input_validation(
        self, dropout: Dropout, training: bool, invalid_input: Any
    ):
        # Call the forward method of the dropout instance with the invalid input and the training flag as arguments
        with pytest.raises(ValueError):
            dropout.forward(invalid_input, training=training)

    # Define a parametrized test for checking the error validation
    @pytest.mark.parametrize("dropout_rate", [0.1, 0.5, 0.9])
    @pytest.mark.parametrize("learning_rate", [0.01, 0.1, 0.5])
    @pytest.mark.parametrize("invalid_error", [np.random.randn(2, 3, 4)])
    def test_error_validation(
        self, dropout: Dropout, learning_rate: float, invalid_error: Any, training: bool
    ):
        # Create a random input array of shape (3, 4)
        input_array = np.random.randn(3, 4)
        # Call the forward method of the dropout instance with the input array and the training flag set to True
        dropout.forward(input_array, training=True)
        # Call the backward method of the dropout instance with the invalid error and the learning rate as arguments
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
        # Pass the input_dim, output_dim, and activation arguments to the BatchNormalization class
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
            # Compute the batch mean and variance of the input array
            batch_mean = np.mean(random_input, axis=0)
            batch_var = np.var(random_input, axis=0)
            # Compute the normalized input array
            normalized_input = (random_input - batch_mean) / np.sqrt(
                batch_var + batch_normalization.epsilon
            )
            # Compute the expected output array
            expected_output = (
                batch_normalization.gamma * normalized_input + batch_normalization.beta
            )
            # Compare the output and expected_output arrays
            assert np.allclose(forward_output_train, expected_output)

        def test_forward_inference_mode_without_activation(
            self,
            batch_normalization: BatchNormalization,
            random_input: np.ndarray,
            forward_output_inference: np.ndarray,
        ):
            # Use the running_mean and running_var parameters to normalize the input array
            normalized_input = (
                random_input - batch_normalization.running_mean
            ) / np.sqrt(batch_normalization.running_var + batch_normalization.epsilon)
            # Compute the expected output array using the gamma and beta parameters
            expected_output = (
                batch_normalization.gamma * normalized_input + batch_normalization.beta
            )
            # Compare the output and expected_output arrays using np.allclose
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
            # Compute the batch mean and variance of the input array
            batch_mean = np.mean(random_input, axis=0)
            batch_var = np.var(random_input, axis=0)
            # Compute the normalized input array
            normalized_input = (random_input - batch_mean) / np.sqrt(
                batch_var + batch_normalization.epsilon
            )
            # Compute the expected output array
            expected_output = (
                gradients
                - np.mean(gradients, axis=0)
                - normalized_input * np.mean(gradients * normalized_input, axis=0)
            ) / np.sqrt(batch_var + epsilon)
            batch_normalization.forward(random_input, training=True)
            # Call the backward method of the batch_normalization instance with the gradients and learning rate
            output = batch_normalization.backward(
                gradients, learning_rate=learning_rate
            )
            # Compare the batch_normalization backward output and the expected_output
            assert np.allclose(output, expected_output, rtol=0.1)

        def test_backward_gamma_without_activation(
            self,
            batch_normalization: BatchNormalization,
            random_input: np.ndarray,
            gradients: np.ndarray,
        ):
            batch_normalization.forward(random_input, training=True)
            batch_normalization.backward(gradients, learning_rate=0.01)

            # Compute the expected gradient with respect to the gamma parameter using the formula from the lecture
            batch_mean = np.mean(random_input, axis=0)
            batch_var = np.var(random_input, axis=0)
            normalized_input = (random_input - batch_mean) / np.sqrt(
                batch_var + batch_normalization.epsilon
            )
            expected_gradient_gamma = np.sum(gradients * normalized_input, axis=0)
            # Compare the gradient of the gamma parameter with the expected gradient using np.allclose
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

            # Compute the expected gradient with respect to the beta parameter using the formula from the lecture
            expected_gradient_beta = np.sum(gradients, axis=0)
            # Compare the gradient of the beta parameter with the expected gradient using np.allclose
            assert np.allclose(
                batch_normalization.gradients["beta"], expected_gradient_beta
            )
