from __future__ import annotations

import numpy as np
import pytest
from conftest import BUNDLED_DATA, write_variant

from preproc_imdb.data import (
    LABELS,
    SPLITS,
    VariantNotFoundError,
    available_variants,
    hold_out,
    load_split,
    read_reviews,
    sample_per_class,
    variant_file,
)


def test_bundled_multiword_variant_is_complete():
    assert "multiword" in available_variants(BUNDLED_DATA)
    for split in SPLITS:
        reviews = load_split(BUNDLED_DATA, "multiword", split)
        assert len(reviews) == 25_000
        assert np.bincount(reviews.labels).tolist() == [12_500, 12_500]


def test_only_newline_ends_a_review(tmp_path):
    # U+0085 (NEXT LINE) appears inside real IMDb reviews; str.splitlines() would split there.
    path = tmp_path / "reviews.txt"
    path.write_text("first review\x85 still first\nsecond review\n", encoding="utf-8")
    assert read_reviews(path) == [["first", "review", "still", "first"], ["second", "review"]]


def test_load_split_gives_class_ids_and_line_numbers(tmp_path):
    data_dir = write_variant(tmp_path, "toy", {("train", "neg"): ["bad", "awful film"], ("train", "pos"): ["good"]})
    reviews = load_split(data_dir, "toy", "train")
    assert reviews.docs == [["bad"], ["awful", "film"], ["good"]]
    assert reviews.labels.tolist() == [0, 0, 1]
    assert reviews.lines.tolist() == [1, 2, 1]
    assert LABELS[reviews.labels[2]] == "pos"


def test_available_variants_needs_all_four_files(tiny_data):
    write_variant(tiny_data, "other", {})
    variant_file(tiny_data, "other", "test", "pos").unlink()
    assert available_variants(tiny_data) == ["toy"]


def test_missing_variant_says_what_is_there_and_how_to_get_more(tiny_data):
    with pytest.raises(VariantNotFoundError) as error:
        load_split(tiny_data, "vanilla", "train")
    message = str(error.value)
    assert "toy" in message and "preproc-imdb download" in message


def test_sample_per_class_is_balanced_and_repeatable():
    reviews = load_split(BUNDLED_DATA, "multiword", "test")
    first = sample_per_class(reviews, 30, np.random.default_rng(7))
    again = sample_per_class(reviews, 30, np.random.default_rng(7))
    assert np.bincount(first.labels).tolist() == [30, 30]
    assert first.lines.tolist() == again.lines.tolist()


def test_hold_out_splits_without_overlap(tiny_data):
    reviews = load_split(tiny_data, "toy", "train")
    kept, held = hold_out(reviews, 0.25, np.random.default_rng(0))
    assert (len(kept), len(held)) == (3, 1)
    pairs = {(int(c), int(n)) for c, n in zip(reviews.labels, reviews.lines, strict=True)}
    kept_pairs = {(int(c), int(n)) for c, n in zip(kept.labels, kept.lines, strict=True)}
    held_pairs = {(int(c), int(n)) for c, n in zip(held.labels, held.lines, strict=True)}
    assert kept_pairs | held_pairs == pairs and not kept_pairs & held_pairs
