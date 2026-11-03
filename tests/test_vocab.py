from __future__ import annotations

import pytest

from preproc_imdb.vocab import PAD, UNK, Vocabulary

DOCS = [["the", "film", "the", "end"], ["the", "film", "was", "fine"], ["the", "end"]]


def test_ids_follow_frequency_rank():
    # Regression: the 2019 port gave every word the id "count + 2", so words with the same count shared
    # an id and the most frequent words (count above the vocabulary size) became UNK.
    vocab = Vocabulary.build(DOCS, size=100)
    assert vocab.index["the"] == 2  # 4 times
    assert vocab.index["end"] == 3 and vocab.index["film"] == 4  # twice each, alphabetical
    assert vocab.index["fine"] == 5 and vocab.index["was"] == 6  # once each
    assert len(set(vocab.index.values())) == len(vocab.words)


def test_most_frequent_words_survive_a_small_vocabulary():
    vocab = Vocabulary.build(DOCS, size=4)  # PAD, UNK and the two most frequent words
    assert vocab.words == ("the", "end")
    assert vocab.encode(["the", "film", "end", "unseen"]) == [2, UNK, 3, UNK]
    assert len(vocab) == 4


def test_every_id_is_below_the_size_and_reserved_ids_are_not_words():
    vocab = Vocabulary.build(DOCS, size=5)
    ids = [i for doc in vocab.encode_all(DOCS) for i in doc]
    assert max(ids) < 5
    assert PAD not in ids


def test_coverage_counts_known_tokens():
    vocab = Vocabulary.build(DOCS, size=4)
    assert vocab.coverage([["the", "end", "film", "x"]]) == pytest.approx(0.5)
    assert vocab.coverage([]) == 0.0


def test_size_must_leave_room_for_words():
    with pytest.raises(ValueError):
        Vocabulary.build(DOCS, size=2)
