import numpy as np
import pytest
from hypothesis import given, strategies as st, settings

from inventronet.layers import Dropout
from inventronet.layers.shape_error import ShapeError
from tests.layers.conftest import generate_random_array


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
