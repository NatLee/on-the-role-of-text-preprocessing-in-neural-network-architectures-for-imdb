"""``preproc-imdb``: download the paper's data and train or test a model on one preprocessing variant.

Exit codes: 0 success, 1 the run failed (missing files, bad download, ...), 2 wrong arguments.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import urllib.error
from dataclasses import fields
from pathlib import Path
from typing import Any

from preproc_imdb import __version__
from preproc_imdb.config import ExperimentConfig, Hyperparameters
from preproc_imdb.data import PAPER_VARIANTS
from preproc_imdb.embeddings import EmbeddingFileError
from preproc_imdb.models import MODELS
from preproc_imdb.sources import EMBEDDINGS, ChecksumError, download_variants, fetch

log = logging.getLogger("preproc_imdb")

DEFAULTS = ExperimentConfig()
DEFAULT_EMBEDDINGS_DIR = Path("data/embeddings")


def _add_train(subparsers: Any) -> None:
    variants = ", ".join(f"{name} ({about})" for name, about in PAPER_VARIANTS.items())
    p = subparsers.add_parser(
        "train",
        help="train and test a model on one preprocessing variant",
        description="Train a model on the IMDb training reviews of one preprocessing variant and report its "
        "accuracy on the test reviews. Defaults follow the code the paper's authors used.",
    )
    data = p.add_argument_group("data")
    data.add_argument(
        "-p",
        "--preprocessing",
        default=DEFAULTS.preprocessing,
        metavar="VARIANT",
        help=f"folder name under --data-dir. The paper's: {variants}. Default: %(default)s",
    )
    data.add_argument("--data-dir", type=Path, default=DEFAULTS.data_dir, help="default: %(default)s")
    data.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULTS.vocab_size,
        help="number of word ids, padding and unknown included (default: %(default)s)",
    )
    data.add_argument(
        "--max-len",
        type=int,
        default=DEFAULTS.max_len,
        help="cut reviews to their last N tokens (default: the longest review, as in the paper's code)",
    )
    data.add_argument(
        "--per-class",
        type=int,
        default=DEFAULTS.per_class,
        metavar="N",
        help="use only N random reviews per class from each split, for quick looks (default: all)",
    )

    model = p.add_argument_group("model")
    model.add_argument("-m", "--model", choices=list(MODELS), default=DEFAULTS.model, help="default: %(default)s")
    model.add_argument(
        "--embeddings",
        type=Path,
        default=DEFAULTS.embeddings,
        metavar="FILE",
        help="word2vec vectors to start from (.bin = binary, else text); default: random",
    )
    model.add_argument(
        "--embedding-dim",
        type=int,
        default=DEFAULTS.embedding_dim,
        help="size of random embeddings; --embeddings files bring their own (default: %(default)s)",
    )
    model.add_argument("--freeze-embeddings", action="store_true", help="do not update the embeddings in training")
    for hp in fields(Hyperparameters):
        model.add_argument(
            f"--{hp.name.replace('_', '-')}",
            type=type(hp.default),
            default=hp.default,
            help=f"{hp.metadata['help']} (default: {hp.default})",
        )

    training = p.add_argument_group("training")
    training.add_argument("--epochs", type=int, default=DEFAULTS.epochs, help="default: %(default)s")
    training.add_argument("--batch-size", type=int, default=DEFAULTS.batch_size, help="default: %(default)s")
    training.add_argument(
        "--validation-split",
        type=float,
        default=DEFAULTS.validation_split,
        metavar="FRACTION",
        help="hold out this share of the training reviews to watch progress (default: none)",
    )
    training.add_argument(
        "--patience",
        type=int,
        default=DEFAULTS.patience,
        help="stop after N epochs without better validation accuracy and keep the best "
        "weights; needs --validation-split (default: never stop early)",
    )
    training.add_argument("--seed", type=int, default=DEFAULTS.seed, help="default: %(default)s")
    training.add_argument(
        "--runs",
        type=int,
        default=DEFAULTS.runs,
        help="train N models with seeds seed, seed+1, ... and report the mean "
        "(the paper averages 2; default: %(default)s)",
    )
    training.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULTS.output_dir,
        help="results go to a new folder in here (default: %(default)s)",
    )
    p.set_defaults(handler=_train)


def _add_download(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "download",
        help="get the paper's other preprocessing variants and its word embeddings",
        description="Add the vanilla, lowercased and lemmatized IMDb variants from the authors' archive "
        "(96 MB), and optionally their UMBC word2vec embeddings (1.6-2.1 GB each). Every file is checked "
        "against a pinned SHA-256.",
    )
    p.add_argument("--data-dir", type=Path, default=DEFAULTS.data_dir, help="default: %(default)s")
    p.add_argument(
        "--archive",
        type=Path,
        metavar="IMDb.zip",
        help="use an IMDb.zip you already downloaded instead of downloading it",
    )
    p.add_argument(
        "--embeddings",
        nargs="+",
        choices=[*EMBEDDINGS, "all"],
        default=[],
        metavar="VARIANT",
        help=f"also download these embeddings: {', '.join(EMBEDDINGS)} or all",
    )
    p.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR, help="default: %(default)s")
    p.set_defaults(handler=_download)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="preproc-imdb",
        description="IMDb experiments of 'On the Role of Text Preprocessing in Neural Network Architectures'.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    _add_train(subparsers)
    _add_download(subparsers)
    return parser


def _config(args: argparse.Namespace) -> ExperimentConfig:
    """Option names match the field names, so the options map onto the dataclasses one to one."""
    hp = Hyperparameters(**{f.name: getattr(args, f.name) for f in fields(Hyperparameters)})
    settings = {f.name: getattr(args, f.name) for f in fields(ExperimentConfig) if f.name != "hyperparameters"}
    return ExperimentConfig(**settings, hyperparameters=hp)


def _train(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        config = _config(args)
    except ValueError as error:
        parser.error(str(error))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # keep TensorFlow's C++ start-up chatter out of the log
    from preproc_imdb.experiment import default_run_dir, run_experiment  # loads Keras and its backend

    run_dir = default_run_dir(config)
    results = run_experiment(config, run_dir)
    accuracies = ", ".join(f"{run['test_accuracy']:.4f}" for run in results["runs"])
    print(f"test accuracy: {results['mean_test_accuracy']:.4f} (runs: {accuracies})")
    print(f"results: {run_dir.as_posix()}/results.json")
    return 0


def _download(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    written = download_variants(args.data_dir, args.archive)
    if written:
        log.info("wrote %d variant files to %s", len(written), args.data_dir.as_posix())
    names = list(EMBEDDINGS) if "all" in args.embeddings else list(dict.fromkeys(args.embeddings))
    for name in names:
        path = fetch(EMBEDDINGS[name], args.embeddings_dir)
        log.info("embeddings %s: %s", name, path.as_posix())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        return args.handler(args, parser)
    except (FileNotFoundError, ChecksumError, EmbeddingFileError, urllib.error.URLError) as error:
        print(f"preproc-imdb: error: {error}", file=sys.stderr)
        return 1
