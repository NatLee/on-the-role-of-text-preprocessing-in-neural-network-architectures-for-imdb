from __future__ import annotations

import hashlib
import logging
import zipfile

import pytest
from conftest import write_variant

from preproc_imdb import sources
from preproc_imdb.data import LABELS, PAPER_VARIANTS, SPLITS, available_variants, variant_file
from preproc_imdb.sources import ARCHIVE_FOLDERS, ChecksumError, Source, download_variants, fetch, install_variants


def make_archive(path):
    """A small IMDb.zip with the authors' layout: IMDb/<split>/<folder>/<label>/<label>_<folder>.txt."""
    with zipfile.ZipFile(path, "w") as bundle:
        for folder in ARCHIVE_FOLDERS:
            for split in SPLITS:
                for label in LABELS:
                    bundle.writestr(f"IMDb/{split}/{folder}/{label}/{label}_{folder}.txt", f"{folder} {label}\n")
    return path


def source_for(path):
    data = path.read_bytes()
    return Source(path.name, path.as_uri(), hashlib.sha256(data).hexdigest(), len(data))


def test_install_variants_uses_the_paper_names(tmp_path):
    written = install_variants(make_archive(tmp_path / "IMDb.zip"), tmp_path / "data")
    assert len(written) == 16
    assert available_variants(tmp_path / "data") == ["lemmatized", "lowercased", "multiword", "vanilla"]
    assert variant_file(tmp_path / "data", "lowercased", "test", "pos").read_text() == "lower pos\n"


def test_install_variants_never_overwrites(tmp_path, caplog):
    data_dir = write_variant(tmp_path / "data", "multiword", {("train", "neg"): ["my own edit"]})
    with caplog.at_level(logging.WARNING):
        written = install_variants(make_archive(tmp_path / "IMDb.zip"), data_dir)
    assert len(written) == 12  # the three other variants
    assert variant_file(data_dir, "multiword", "train", "neg").read_text() == "my own edit\n"
    assert "multiword/train/neg.txt" in caplog.text


def test_fetch_downloads_and_verifies(tmp_path):
    remote = tmp_path / "remote" / "file.bin"
    remote.parent.mkdir()
    remote.write_bytes(b"x" * 3000)
    target = fetch(source_for(remote), tmp_path / "local")
    assert target.read_bytes() == remote.read_bytes()
    assert fetch(source_for(remote), tmp_path / "local") == target  # second call: already there


def test_fetch_rejects_a_changed_file_and_leaves_nothing_behind(tmp_path):
    remote = tmp_path / "file.bin"
    remote.write_bytes(b"original")
    expected = source_for(remote)
    remote.write_bytes(b"replaced")
    with pytest.raises(ChecksumError, match="SHA-256"):
        fetch(expected, tmp_path / "local")
    assert not any((tmp_path / "local").iterdir())


def test_download_variants_skips_the_network_when_everything_is_there(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    for variant in ARCHIVE_FOLDERS.values():
        write_variant(data_dir, variant, {})

    def no_network(*args, **kwargs):
        raise AssertionError("should not download")

    monkeypatch.setattr(sources, "fetch", no_network)
    assert download_variants(data_dir) == []


def test_download_variants_checks_a_hand_downloaded_archive(tmp_path):
    with pytest.raises(ChecksumError, match="bytes"):
        download_variants(tmp_path / "data", make_archive(tmp_path / "IMDb.zip"))


def test_pinned_sources_look_sane():
    for source in [sources.VARIANTS_ARCHIVE, *sources.EMBEDDINGS.values()]:
        assert len(source.sha256) == 64 and source.size > 0
        assert source.url.startswith("https://")
    # the four paper variants are named the same everywhere: data folders, archive mapping, embeddings
    assert set(sources.EMBEDDINGS) == set(ARCHIVE_FOLDERS.values()) == set(PAPER_VARIANTS)
