"""Pre-trained word vectors in word2vec format, read as a stream.

The paper's UMBC vectors are 1.6 to 2.1 GB each, while a run needs at most ``vocab_size`` (20,000)
of them. So instead of loading the whole file, the reader walks through it once and keeps only the
vectors of words in the vocabulary.

Format (https://code.google.com/archive/p/word2vec/): a header line ``<count> <dim>``, then one record
per word. In the binary format (``.bin``) a record is the word, a space and ``dim`` little-endian
float32 values, optionally followed by a newline. In the text format (any other suffix) a record is a
line ``word v1 v2 ...``; the header is optional there, as in GloVe files.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import BinaryIO

import numpy as np

from preproc_imdb.vocab import Vocabulary

_CHUNK = 1 << 20


class EmbeddingFileError(ValueError):
    """The vectors file does not follow the word2vec format."""


@dataclass(frozen=True)
class EmbeddingMatrix:
    weights: np.ndarray  # (len(vocab), dim) float32; rows of PAD, UNK and missing words stay zero
    found: int  # vocabulary words that had a vector in the file

    @property
    def dim(self) -> int:
        return self.weights.shape[1]


def is_binary(path: Path) -> bool:
    return Path(path).suffix.lower() == ".bin"


def _parse_header(line: bytes) -> tuple[int, int] | None:
    parts = line.split()
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        return int(parts[0]), int(parts[1])
    return None


def _binary_records(handle: BinaryIO, count: int, dim: int) -> Iterator[tuple[bytes, bytes]]:
    """Yield (word, raw vector bytes) for each record without decoding vectors nobody asked for."""
    size = dim * 4
    buffer, start = b"", 0
    for _ in range(count):
        while True:
            space = buffer.find(b" ", start)
            if space != -1 and len(buffer) - space - 1 >= size:
                break
            chunk = handle.read(_CHUNK)
            if not chunk:
                raise EmbeddingFileError(f"word2vec file ends early: expected {count} vectors")
            buffer, start = buffer[start:] + chunk, 0
        yield buffer[start:space].lstrip(b"\n"), buffer[space + 1 : space + 1 + size]
        start = space + 1 + size


def read_vectors(path: Path, words: set[str]) -> tuple[dict[str, np.ndarray], int]:
    """Vectors of ``words`` found in the file, and the file's vector size."""
    wanted = {word.encode("utf-8"): word for word in words}
    vectors: dict[str, np.ndarray] = {}
    with open(path, "rb") as handle:
        first = handle.readline()
        header = _parse_header(first)
        if is_binary(path):
            if header is None:
                raise EmbeddingFileError(f"{Path(path).as_posix()} has no '<count> <dim>' header line")
            count, dim = header
            for word, raw in _binary_records(handle, count, dim):
                if word in wanted:
                    vectors[wanted[word]] = np.frombuffer(raw, dtype="<f4").astype(np.float32)
            return vectors, dim
        dim = header[1] if header else len(first.split()) - 1
        lines = handle if header else chain([first], handle)
        for number, line in enumerate(lines, start=2 if header else 1):
            word, _, rest = line.rstrip(b"\r\n").partition(b" ")
            if word in wanted:
                values = np.array(rest.split(), dtype=np.float32)
                if values.shape != (dim,):
                    raise EmbeddingFileError(f"line {number}: expected {dim} values, got {values.size}")
                vectors[wanted[word]] = values
    return vectors, dim


def embedding_matrix(vocab: Vocabulary, path: Path) -> EmbeddingMatrix:
    """Embedding layer weights for ``vocab``: row ``vocab.index[word]`` holds that word's vector."""
    vectors, dim = read_vectors(path, set(vocab.words))
    weights = np.zeros((len(vocab), dim), dtype=np.float32)
    for word, vector in vectors.items():
        weights[vocab.index[word]] = vector
    return EmbeddingMatrix(weights, found=len(vectors))
