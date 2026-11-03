from __future__ import annotations

import numpy as np
import pytest

from preproc_imdb.config import Hyperparameters
from preproc_imdb.models import MODELS, build_model

HP = Hyperparameters()


def layer_names(model):
    return [layer.name for layer in model.layers]


def test_cnn_lstm_has_the_layers_of_the_original_code():
    # sensecnn: Embedding, Dropout(0.5), Conv1D(50, 2, relu), MaxPooling1D(2), Bidirectional(LSTM(25)),
    # Dense(1, sigmoid)
    model = build_model("cnn-lstm", input_length=40, vocab_size=30, embedding_dim=8, hp=HP)
    assert layer_names(model) == ["word_ids", "embedding", "dropout", "conv", "max_pooling", "bilstm", "positive"]
    assert model.get_layer("conv").filters == 50 and model.get_layer("conv").kernel_size == (2,)
    assert model.get_layer("bilstm").forward_layer.units == 25
    assert model.output_shape == (None, 1)


def test_cnn_pools_over_time_straight_into_the_output():
    model = build_model("cnn", input_length=40, vocab_size=30, embedding_dim=8, hp=HP)
    assert layer_names(model) == ["word_ids", "embedding", "dropout", "conv", "max_over_time", "positive"]


@pytest.mark.parametrize("name", list(MODELS))
def test_pretrained_embeddings_are_loaded_and_can_be_frozen(name):
    weights = np.arange(30 * 8, dtype="float32").reshape(30, 8)
    model = build_model(
        name,
        input_length=40,
        vocab_size=30,
        embedding_dim=8,
        hp=HP,
        embedding_weights=weights,
        trainable_embeddings=False,
    )
    embedding = model.get_layer("embedding")
    np.testing.assert_array_equal(embedding.get_weights()[0], weights)
    assert not embedding.trainable


def test_embedding_shape_must_match():
    with pytest.raises(ValueError, match="shape"):
        build_model("cnn", input_length=40, vocab_size=30, embedding_dim=8, hp=HP, embedding_weights=np.zeros((29, 8)))


def test_unknown_model_name():
    with pytest.raises(ValueError, match="cnn-lstm"):
        build_model("transformer", input_length=40, vocab_size=30, embedding_dim=8, hp=HP)
