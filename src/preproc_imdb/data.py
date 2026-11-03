"""IMDb reviews on disk: one folder per preprocessing variant, one review per line.

Layout, relative to the data directory (``data/IMDb`` by default)::

    <variant>/<split>/<label>.txt        e.g. multiword/train/pos.txt

Every line is one review, tokens separated by spaces. The paper's authors preprocessed the text with
external tools (Stanford CoreNLP for tokenizing and lemmatizing, the phrase list of the Google News
word2vec vectors for multiwords), so a preprocessing variant here is simply a folder of files. The repo
ships ``multiword``; ``preproc-imdb download`` adds the other three. To try a preprocessing of your own,
write the four files into a new folder and pass its name to ``--preprocessing``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SPLITS = ("train", "test")
LABELS = ("neg", "pos")  # the position is the class id: neg = 0, pos = 1, as in the original code

PAPER_VARIANTS = {
    "vanilla": "tokenized only",
    "lowercased": "tokenized, then lowercased",
    "lemmatized": "tokenized, then every token replaced by its lemma",
    "multiword": "tokenized, then known multiword expressions joined with '_'; ships with the repo",
}


class VariantNotFoundError(FileNotFoundError):
    """The requested preprocessing variant has no complete set of files in the data directory."""


@dataclass(frozen=True)
class Reviews:
    """Tokenized reviews with their class ids and where each one came from."""

    docs: list[list[str]]
    labels: np.ndarray  # class ids, see LABELS
    lines: np.ndarray  # 1-based line number of each review in its <label>.txt file

    def __len__(self) -> int:
        return len(self.docs)

    def take(self, indices: np.ndarray) -> Reviews:
        return Reviews([self.docs[i] for i in indices], self.labels[indices], self.lines[indices])


def variant_file(data_dir: Path, variant: str, split: str, label: str) -> Path:
    return Path(data_dir) / variant / split / f"{label}.txt"


def available_variants(data_dir: Path) -> list[str]:
    """Names of the folders in ``data_dir`` that hold all four split/label files."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return []
    return sorted(
        folder.name
        for folder in data_dir.iterdir()
        if all(variant_file(data_dir, folder.name, s, label).is_file() for s in SPLITS for label in LABELS)
    )


def read_reviews(path: Path) -> list[list[str]]:
    """One token list per line.

    Only ``\\n`` ends a line: some reviews contain U+0085 (NEXT LINE), which ``str.splitlines()``
    would treat as a line break and so shift every later review.
    """
    with open(path, encoding="utf-8", newline="\n") as handle:
        return [line.split() for line in handle]


def load_split(data_dir: Path, variant: str, split: str) -> Reviews:
    files = [variant_file(data_dir, variant, split, label) for label in LABELS]
    missing = [path for path in files if not path.is_file()]
    if missing:
        found = ", ".join(available_variants(data_dir)) or "none"
        hint = " Run `preproc-imdb download` to add the paper's variants." if variant in PAPER_VARIANTS else ""
        raise VariantNotFoundError(
            f"no {split} files for preprocessing {variant!r} (missing {missing[0].as_posix()}). "
            f"Variants found in {Path(data_dir).as_posix()}: {found}.{hint}"
        )
    docs: list[list[str]] = []
    labels: list[int] = []
    lines: list[int] = []
    for class_id, path in enumerate(files):
        reviews = read_reviews(path)
        docs.extend(reviews)
        labels.extend([class_id] * len(reviews))
        lines.extend(range(1, len(reviews) + 1))
    return Reviews(docs, np.asarray(labels, dtype="int32"), np.asarray(lines, dtype="int64"))


def sample_per_class(reviews: Reviews, per_class: int, rng: np.random.Generator) -> Reviews:
    """Keep at most ``per_class`` random reviews of each class (for smoke tests and quick looks)."""
    keep = []
    for class_id in range(len(LABELS)):
        members = np.flatnonzero(reviews.labels == class_id)
        keep.append(rng.choice(members, size=min(per_class, len(members)), replace=False))
    return reviews.take(np.sort(np.concatenate(keep)))


def hold_out(reviews: Reviews, fraction: float, rng: np.random.Generator) -> tuple[Reviews, Reviews]:
    """Split off a random ``fraction`` of the reviews, e.g. a validation set taken from the training data."""
    order = rng.permutation(len(reviews))
    cut = round(len(reviews) * fraction)
    return reviews.take(np.sort(order[cut:])), reviews.take(np.sort(order[:cut]))
