from __future__ import annotations

import numpy as np
import pytest

from preproc_imdb import embeddings
from preproc_imdb.embeddings import EmbeddingFileError, embedding_matrix, read_vectors
from preproc_imdb.vocab import Vocabulary

VECTORS = {
    "zebra": [0.5, -1.0, 2.0],
    "the": [1.0, 2.0, 3.0],
    "film": [-0.25, 0.0, 4.5],
    "café": [9.0, 8.0, 7.0],
}


def write_binary(path, vectors, newline=True):
    with open(path, "wb") as out:
        out.write(f"{len(vectors)} 3\n".encode())
        for word, values in vectors.items():
            out.write(word.encode() + b" " + np.asarray(values, dtype="<f4").tobytes() + (b"\n" if newline else b""))
    return path


def write_text(path, vectors, header=True):
    lines = [f"{len(vectors)} 3"] if header else []
    lines += [word + " " + " ".join(str(v) for v in values) for word, values in vectors.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("newline", [True, False])
def test_binary_rows_line_up_with_word_ids(tmp_path, newline):
    # Regression: the 2019 port filled row i with the i-th word of the dictionary loop instead of row
    # vocab.index[word], so words and vectors did not match.
    vocab = Vocabulary.build([["the", "the", "the", "film", "film", "café", "missing"]], size=10)
    matrix = embedding_matrix(vocab, write_binary(tmp_path / "v.bin", VECTORS, newline))
    for word in ("the", "film", "café"):
        assert matrix.weights[vocab.index[word]].tolist() == VECTORS[word]
    assert not matrix.weights[vocab.index["missing"]].any()
    assert not matrix.weights[:2].any()  # PAD and UNK
    assert matrix.found == 3 and matrix.dim == 3
    assert matrix.weights.shape == (len(vocab), 3) and matrix.weights.dtype == np.float32


def test_binary_reader_survives_records_split_across_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "_CHUNK", 5)
    vectors, dim = read_vectors(write_binary(tmp_path / "v.bin", VECTORS), {"zebra", "café"})
    assert dim == 3
    assert {word: v.tolist() for word, v in vectors.items()} == {"zebra": VECTORS["zebra"], "café": VECTORS["café"]}


@pytest.mark.parametrize("header", [True, False])
def test_text_format_with_or_without_header(tmp_path, header):
    vectors, dim = read_vectors(write_text(tmp_path / "v.txt", VECTORS, header), {"film", "the"})
    assert dim == 3
    assert vectors["film"].tolist() == VECTORS["film"]
    assert set(vectors) == {"film", "the"}


def test_truncated_binary_file_is_an_error(tmp_path):
    path = write_binary(tmp_path / "v.bin", VECTORS)
    path.write_bytes(path.read_bytes()[:-10])
    with pytest.raises(EmbeddingFileError, match="ends early"):
        read_vectors(path, {"the"})


def test_text_line_with_wrong_size_is_an_error(tmp_path):
    path = tmp_path / "v.txt"
    path.write_text("2 3\nthe 1 2 3\nfilm 1 2\n", encoding="utf-8")
    with pytest.raises(EmbeddingFileError, match="line 3"):
        read_vectors(path, {"film"})
