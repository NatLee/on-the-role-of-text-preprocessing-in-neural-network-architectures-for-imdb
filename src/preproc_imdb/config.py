"""Experiment settings.

The defaults are the settings of the original sensecnn code that the paper's authors used
(https://github.com/pilehvar/sensecnn, ``sentiment_analysis/__main__.py``), so a plain
``preproc-imdb train`` runs the paper's CNN+LSTM setup on the bundled multiword data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Hyperparameters:
    """Layer sizes shared by both models (``pool_size`` and ``lstm_units`` only matter for cnn-lstm)."""

    dropout: float = field(default=0.5, metadata={"help": "dropout rate after the embeddings"})
    filters: int = field(default=50, metadata={"help": "convolution filters"})
    kernel_size: int = field(default=2, metadata={"help": "convolution width in tokens"})
    pool_size: int = field(default=2, metadata={"help": "max pooling size before the LSTM (cnn-lstm only)"})
    lstm_units: int = field(default=25, metadata={"help": "LSTM units per direction (cnn-lstm only)"})


@dataclass(frozen=True)
class ExperimentConfig:
    preprocessing: str = "multiword"
    model: str = "cnn-lstm"
    data_dir: Path = Path("data/IMDb")
    embeddings: Path | None = None
    embedding_dim: int = 300
    freeze_embeddings: bool = False
    vocab_size: int = 20_000
    max_len: int | None = None
    epochs: int = 3
    batch_size: int = 100
    validation_split: float = 0.0
    patience: int = 0
    per_class: int | None = None
    seed: int = 1337
    runs: int = 1
    output_dir: Path = Path("runs")
    hyperparameters: Hyperparameters = field(default_factory=Hyperparameters)

    def __post_init__(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "runs": self.runs,
        }
        optional_positive = {"max_len": self.max_len, "per_class": self.per_class}
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
        for name, value in optional_positive.items():
            if value is not None and value < 1:
                raise ValueError(f"{name} must be at least 1 when given, got {value}")
        if not 0.0 <= self.validation_split < 1.0:
            raise ValueError(f"validation_split must be in [0, 1), got {self.validation_split}")
        if self.patience < 0:
            raise ValueError(f"patience must not be negative, got {self.patience}")
        if self.patience and not self.validation_split:
            raise ValueError("early stopping (patience > 0) needs a validation split taken from the training data")

    def to_dict(self) -> dict:
        """JSON-friendly copy, written next to every result so a run can be repeated exactly."""
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, Path):
                data[key] = value.as_posix()
        return data
