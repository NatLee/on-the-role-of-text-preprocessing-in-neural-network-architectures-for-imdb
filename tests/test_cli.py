from __future__ import annotations

from dataclasses import fields

import pytest

from preproc_imdb.cli import _config, build_parser, main
from preproc_imdb.config import ExperimentConfig, Hyperparameters


def parse(*argv):
    return _config(build_parser().parse_args(["train", *argv]))


def test_defaults_are_the_config_defaults():
    assert parse() == ExperimentConfig()


def test_every_setting_has_an_option():
    config = parse(
        *("--preprocessing", "vanilla", "--model", "cnn", "--data-dir", "d", "--embeddings", "e.bin"),
        *("--embedding-dim", "50", "--freeze-embeddings", "--vocab-size", "500", "--max-len", "80"),
        *("--epochs", "2", "--batch-size", "8", "--validation-split", "0.2", "--patience", "1"),
        *("--per-class", "10", "--seed", "3", "--runs", "2", "--output-dir", "out"),
        *("--dropout", "0.1", "--filters", "4", "--kernel-size", "3", "--pool-size", "5", "--lstm-units", "6"),
    )
    defaults = ExperimentConfig()
    changed = [f.name for f in fields(ExperimentConfig) if getattr(config, f.name) != getattr(defaults, f.name)]
    assert changed == [f.name for f in fields(ExperimentConfig)]
    assert config.hyperparameters == Hyperparameters(dropout=0.1, filters=4, kernel_size=3, pool_size=5, lstm_units=6)


def test_freezing_is_a_switch():
    # Regression: the old `--static` used argparse type=bool, so `--static False` froze the embeddings.
    assert parse().freeze_embeddings is False
    assert parse("--freeze-embeddings").freeze_embeddings is True
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "--freeze-embeddings", "False"])


@pytest.mark.parametrize(
    "argv",
    [["--epochs", "0"], ["--patience", "2"], ["--validation-split", "1.5"], ["--max-len", "0"], ["--model", "rnn"]],
)
def test_bad_settings_are_usage_errors(argv, capsys):
    with pytest.raises(SystemExit) as stop:
        main(["train", *argv])
    assert stop.value.code == 2
    assert "error" in capsys.readouterr().err


def test_missing_variant_fails_with_a_message(tiny_data, tmp_path, capsys):
    code = main(["train", "--data-dir", str(tiny_data), "-p", "vanilla", "--output-dir", str(tmp_path / "runs")])
    assert code == 1
    err = capsys.readouterr().err
    assert "vanilla" in err and "preproc-imdb download" in err
