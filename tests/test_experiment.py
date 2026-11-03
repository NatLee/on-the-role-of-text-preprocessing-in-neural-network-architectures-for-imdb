"""Smoke tests: train real (tiny) models on a small slice of the bundled IMDb data."""

from __future__ import annotations

import csv
import json

import keras
import numpy as np
import pytest
from conftest import BUNDLED_DATA
from test_embeddings import write_text

from preproc_imdb.cli import main
from preproc_imdb.config import ExperimentConfig, Hyperparameters
from preproc_imdb.data import load_split, sample_per_class
from preproc_imdb.experiment import run_experiment
from preproc_imdb.vocab import Vocabulary

SMALL = dict(
    data_dir=BUNDLED_DATA,
    per_class=40,
    max_len=60,
    vocab_size=2_000,
    embedding_dim=16,
    epochs=1,
    batch_size=20,
    hyperparameters=Hyperparameters(filters=8, lstm_units=4),
)


@pytest.mark.parametrize("model", ["cnn", "cnn-lstm"])
def test_a_run_writes_results_predictions_and_the_model(tmp_path, model):
    config = ExperimentConfig(model=model, output_dir=tmp_path, **SMALL)
    results = run_experiment(config, tmp_path / "run")

    saved = json.loads((tmp_path / "run" / "results.json").read_text(encoding="utf-8"))
    assert saved == results
    assert saved["config"]["model"] == model
    assert saved["data"]["train_reviews"] == saved["data"]["test_reviews"] == 80
    assert 0.0 <= saved["mean_test_accuracy"] <= 1.0
    assert saved["environment"]["backend"] == keras.config.backend()

    with open(tmp_path / "run" / "run-1" / "predictions.tsv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 80
    assert {row["label"] for row in rows} == {"neg", "pos"}
    assert keras.models.load_model(tmp_path / "run" / "run-1" / "model.keras").input_shape == (None, 60)


def test_max_len_is_honoured(tmp_path):
    # Regression: the old code replaced --max_len with the longest review, whatever was passed.
    results = run_experiment(ExperimentConfig(output_dir=tmp_path, **SMALL), tmp_path / "run")
    assert results["data"]["max_len"] == 60


def test_without_max_len_the_longest_review_sets_the_length(tmp_path):
    config = ExperimentConfig(output_dir=tmp_path, **{**SMALL, "max_len": None, "per_class": 5})
    results = run_experiment(config, tmp_path / "run")
    rng = np.random.default_rng(config.seed)
    train = sample_per_class(load_split(BUNDLED_DATA, "multiword", "train"), 5, rng)
    test = sample_per_class(load_split(BUNDLED_DATA, "multiword", "test"), 5, rng)
    assert results["data"]["max_len"] == max(len(doc) for doc in train.docs + test.docs)


def test_same_seed_same_numbers(tmp_path):
    # Regression: only numpy was seeded before, so two runs of the same command gave different models.
    config = ExperimentConfig(model="cnn-lstm", output_dir=tmp_path, **SMALL)
    first = run_experiment(config, tmp_path / "a")["runs"][0]
    second = run_experiment(config, tmp_path / "b")["runs"][0]
    assert first["history"] == second["history"]
    assert first["test_loss"] == second["test_loss"]


def test_runs_use_consecutive_seeds(tmp_path):
    results = run_experiment(ExperimentConfig(runs=2, seed=5, output_dir=tmp_path, **SMALL), tmp_path / "run")
    assert [run["seed"] for run in results["runs"]] == [5, 6]
    assert results["mean_test_accuracy"] == pytest.approx(np.mean([r["test_accuracy"] for r in results["runs"]]))


def test_early_stopping_watches_held_out_training_reviews(tmp_path):
    config = ExperimentConfig(output_dir=tmp_path, **{**SMALL, "epochs": 4, "validation_split": 0.25, "patience": 1})
    results = run_experiment(config, tmp_path / "run")
    assert results["data"]["validation_reviews"] == 20
    assert results["data"]["train_reviews"] == 60
    assert results["data"]["test_reviews"] == 80  # the test set is never used for stopping
    run = results["runs"][0]
    assert "val_accuracy" in run["history"]
    assert 1 <= run["epochs_trained"] <= 4


def test_pretrained_vectors_are_used(tmp_path):
    train = load_split(BUNDLED_DATA, "multiword", "train")
    vocab = Vocabulary.build(train.docs, 50)
    vectors = {word: [0.1 * (i % 7)] * 16 for i, word in enumerate(vocab.words[:20])}
    path = write_text(tmp_path / "vectors.txt", vectors, header=False)

    config = ExperimentConfig(output_dir=tmp_path, embeddings=path, freeze_embeddings=True, **SMALL)
    results = run_experiment(config, tmp_path / "run")
    assert results["data"]["embeddings_found"] >= 15  # the sample's vocabulary misses a few common words at most
    assert results["data"]["embedding_dim"] == 16


def test_cli_end_to_end(tmp_path, capsys):
    code = main(
        [
            "train",
            "--data-dir",
            str(BUNDLED_DATA),
            "--per-class",
            "20",
            "--max-len",
            "40",
            "--epochs",
            "1",
            "--vocab-size",
            "1000",
            "--embedding-dim",
            "8",
            "--filters",
            "4",
            "--lstm-units",
            "2",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "test accuracy:" in out
    (run_dir,) = tmp_path.iterdir()
    assert run_dir.name.startswith("multiword-cnn-lstm-")
    assert (run_dir / "results.json").is_file()
