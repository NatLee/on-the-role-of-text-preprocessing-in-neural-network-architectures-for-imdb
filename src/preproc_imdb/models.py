"""The two classifiers of the paper (section 3.1), built with Keras.

Both start the same way: word ids -> embedding -> dropout -> 1D convolution with ReLU.

* ``cnn-lstm`` is the paper's CNN+LSTM, layer for layer the model in the original sensecnn code:
  max pooling, then a bidirectional LSTM, then one sigmoid unit.
* ``cnn`` is the paper's plain CNN "similar to that of Kim (2014)". Its code was never published, so
  this is our reading of that description: max-over-time pooling straight into the sigmoid unit.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from preproc_imdb.config import Hyperparameters

if TYPE_CHECKING:
    import keras
    import numpy as np

# Keras is imported inside the functions, so that the CLI (`--help`, `download`) starts without
# loading TensorFlow.
Head = Callable[["keras.KerasTensor", Hyperparameters], "keras.KerasTensor"]


def _cnn_head(x: keras.KerasTensor, hp: Hyperparameters) -> keras.KerasTensor:
    from keras import layers

    return layers.GlobalMaxPooling1D(name="max_over_time")(x)


def _cnn_lstm_head(x: keras.KerasTensor, hp: Hyperparameters) -> keras.KerasTensor:
    from keras import layers

    x = layers.MaxPooling1D(pool_size=hp.pool_size, name="max_pooling")(x)
    return layers.Bidirectional(layers.LSTM(hp.lstm_units), name="bilstm")(x)


MODELS: dict[str, Head] = {"cnn": _cnn_head, "cnn-lstm": _cnn_lstm_head}


def build_model(
    name: str,
    *,
    input_length: int,
    vocab_size: int,
    embedding_dim: int,
    hp: Hyperparameters,
    embedding_weights: np.ndarray | None = None,
    trainable_embeddings: bool = True,
) -> keras.Model:
    """A compiled binary classifier that maps padded word-id sequences to P(positive)."""
    import keras
    from keras import layers

    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose one of {', '.join(MODELS)}")
    if embedding_weights is not None and embedding_weights.shape != (vocab_size, embedding_dim):
        raise ValueError(
            f"embedding weights have shape {embedding_weights.shape}, expected {(vocab_size, embedding_dim)}"
        )
    inputs = keras.Input(shape=(input_length,), dtype="int32", name="word_ids")
    embedding = layers.Embedding(vocab_size, embedding_dim, trainable=trainable_embeddings, name="embedding")
    x = embedding(inputs)
    x = layers.Dropout(hp.dropout, name="dropout")(x)
    x = layers.Conv1D(hp.filters, hp.kernel_size, padding="valid", activation="relu", name="conv")(x)
    x = MODELS[name](x, hp)
    outputs = layers.Dense(1, activation="sigmoid", name="positive")(x)
    model = keras.Model(inputs, outputs, name=name)
    if embedding_weights is not None:
        embedding.set_weights([embedding_weights])
    model.compile(optimizer=keras.optimizers.Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    return model
