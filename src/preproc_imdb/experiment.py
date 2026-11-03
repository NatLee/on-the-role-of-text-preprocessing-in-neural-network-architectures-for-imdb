"""One experiment: read a preprocessing variant, build the vocabulary, train and test a model, and write
everything needed to check or repeat the run into one folder."""

from __future__ import annotations

import csv
import importlib
import json
import logging
import platform
import time
from datetime import datetime
from pathlib import Path

import keras
import numpy as np

from preproc_imdb import __version__
from preproc_imdb.config import ExperimentConfig
from preproc_imdb.data import LABELS, Reviews, hold_out, load_split, sample_per_class
from preproc_imdb.embeddings import embedding_matrix
from preproc_imdb.models import build_model
from preproc_imdb.vocab import PAD, Vocabulary

log = logging.getLogger(__name__)


def default_run_dir(config: ExperimentConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return config.output_dir / f"{config.preprocessing}-{config.model}-{stamp}"


def environment() -> dict[str, str]:
    """Versions that decide the numbers; every Keras backend name is also its module name."""
    backend = keras.config.backend()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "preproc_imdb": __version__,
        "keras": keras.__version__,
        "backend": backend,
        "backend_version": importlib.import_module(backend).__version__,
        "numpy": np.__version__,
    }


def _pad(vocab: Vocabulary, reviews: Reviews, length: int) -> np.ndarray:
    # Padding and cutting both happen at the start of a review, keeping its end, as in the original code.
    return keras.utils.pad_sequences(
        vocab.encode_all(reviews.docs), maxlen=length, padding="pre", truncating="pre", value=PAD
    )


def _write_predictions(path: Path, reviews: Reviews, probabilities: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["label", "line", "p_positive"])
        for label, line, p in zip(reviews.labels, reviews.lines, probabilities, strict=True):
            writer.writerow([LABELS[label], int(line), f"{p:.6f}"])


def run_experiment(config: ExperimentConfig, run_dir: Path) -> dict:
    """Train and test ``config.runs`` models and return (and save) the results.

    All runs see the same data; run ``i`` uses seed ``config.seed + i`` for weights and batch order.
    """
    rng = np.random.default_rng(config.seed)
    train = load_split(config.data_dir, config.preprocessing, "train")
    test = load_split(config.data_dir, config.preprocessing, "test")
    if config.per_class:
        train = sample_per_class(train, config.per_class, rng)
        test = sample_per_class(test, config.per_class, rng)
    valid = None
    if config.validation_split:
        train, valid = hold_out(train, config.validation_split, rng)

    vocab = Vocabulary.build(train.docs, config.vocab_size)
    splits = [train, test] + ([valid] if valid else [])
    max_len = config.max_len or max((len(doc) for part in splits for doc in part.docs), default=1)
    x_train, x_test = _pad(vocab, train, max_len), _pad(vocab, test, max_len)
    validation_data = (_pad(vocab, valid, max_len), valid.labels) if valid else None

    weights, dim, found = None, config.embedding_dim, None
    if config.embeddings:
        log.info("reading word vectors from %s", Path(config.embeddings).as_posix())
        matrix = embedding_matrix(vocab, config.embeddings)
        weights, dim, found = matrix.weights, matrix.dim, matrix.found
        log.info("found vectors for %d of %d vocabulary words", found, len(vocab.words))

    data = {
        "train_reviews": len(train),
        "validation_reviews": len(valid) if valid else 0,
        "test_reviews": len(test),
        "max_len": max_len,
        "vocabulary_ids": len(vocab),
        "test_token_coverage": round(vocab.coverage(test.docs), 4),
        "embedding_dim": dim,
        "embeddings_found": found,
    }
    log.info("data: %s", json.dumps(data))

    run_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for i in range(config.runs):
        seed = config.seed + i
        keras.utils.set_random_seed(seed)
        model = build_model(
            config.model,
            input_length=max_len,
            vocab_size=len(vocab),
            embedding_dim=dim,
            hp=config.hyperparameters,
            embedding_weights=weights,
            trainable_embeddings=not config.freeze_embeddings,
        )
        callbacks = []
        if config.patience:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_accuracy", patience=config.patience, restore_best_weights=True
                )
            )
        log.info("run %d/%d (seed %d): training %s", i + 1, config.runs, seed, config.model)
        started = time.perf_counter()
        history = model.fit(
            x_train,
            train.labels,
            batch_size=config.batch_size,
            epochs=config.epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=2,
        )
        seconds = time.perf_counter() - started
        loss, accuracy = model.evaluate(x_test, test.labels, batch_size=config.batch_size, verbose=0)
        probabilities = model.predict(x_test, batch_size=config.batch_size, verbose=0)[:, 0]

        out = run_dir / f"run-{i + 1}"
        out.mkdir(exist_ok=True)
        _write_predictions(out / "predictions.tsv", test, probabilities)
        model.save(out / "model.keras")
        runs.append(
            {
                "seed": seed,
                "test_accuracy": float(accuracy),
                "test_loss": float(loss),
                "epochs_trained": len(history.history["loss"]),
                "train_seconds": round(seconds, 1),
                "history": {key: [float(v) for v in values] for key, values in history.history.items()},
            }
        )
        log.info("run %d/%d: test accuracy %.4f", i + 1, config.runs, accuracy)

    results = {
        "config": config.to_dict(),
        "data": data,
        "runs": runs,
        "mean_test_accuracy": float(np.mean([run["test_accuracy"] for run in runs])),
        "environment": environment(),
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results
