from typing import Type

import numpy as np
import pytest

from inventronet.layers import BatchNormalization
from inventronet.optimizers.optimizer import Optimizer


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
