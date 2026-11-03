"""The paper authors' published files, pinned by size and SHA-256.

Camacho-Collados and Pilehvar share the preprocessed datasets and the UMBC word embeddings on Google
Drive (listed in https://github.com/pedrada88/preproc-textclassification). The checksums below are of
the files as downloaded on 2026-11-04, so a changed or truncated download fails loudly instead of
quietly giving different numbers.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from preproc_imdb.data import LABELS, SPLITS, variant_file

log = logging.getLogger(__name__)

DRIVE_URL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
_CHUNK = 1 << 20


@dataclass(frozen=True)
class Source:
    file_name: str
    url: str
    sha256: str
    size: int  # bytes


VARIANTS_ARCHIVE = Source(
    "IMDb.zip",
    DRIVE_URL.format("1DT7WRBA1aXxz2arB90_wielI5FudxHKF"),
    "30e75414711b13acd5e0271b544cb21c835be197f731f7664fb3de3ce083c13d",
    100_202_193,
)

# Folder names inside IMDb.zip -> the variant names used here and in the paper.
ARCHIVE_FOLDERS = {"tokenized": "vanilla", "lower": "lowercased", "lemmatized": "lemmatized", "multiwords": "multiword"}

# 300-dimensional CBOW word2vec vectors trained on the 3B-word UMBC corpus, one per preprocessing.
EMBEDDINGS = {
    "vanilla": Source(
        "umbc-vanilla.bin",
        DRIVE_URL.format("1dBnoeLR20LjNy4hVjR2P5U923A1uqQbR"),
        "fad8983fc067290e189aa33f62d1feeb3dc8786d4e39975a9adeae715eb22827",
        1_773_277_493,
    ),
    "lowercased": Source(
        "umbc-lowercased.bin",
        DRIVE_URL.format("1vObC9pjcWiKpahziolHyTi5-lhiRw7xF"),
        "7d6a08da64c7edc2f75c276d5156d80a1f040a902cdffb59132bab213ee4c998",
        1_566_705_534,
    ),
    "lemmatized": Source(
        "umbc-lemmatized.bin",
        DRIVE_URL.format("1E-oVlug-Pu6LA7Ke6ZHHDMmL7wjz9P1l"),
        "8deac877d21b4661ea0447e59101e3129b4a20abd74f7079a2a630cd1b074fa0",
        1_671_719_899,
    ),
    "multiword": Source(
        "umbc-multiword.bin",
        DRIVE_URL.format("13fGiYIeXEpBQaNuKb3wRnfhCyxF02rDu"),
        "f1c2b12284ccffc46d03d665d0752623502d5a2ddfee20b97150ddcf4cafd74f",
        2_124_419_216,
    ),
}


class ChecksumError(RuntimeError):
    """A downloaded file is not the one the checksums were taken from."""


def _sha256(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(_CHUNK):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_of(path: Path) -> str:
    with open(path, "rb") as handle:
        return _sha256(handle)


def verify(path: Path, source: Source) -> None:
    size = Path(path).stat().st_size
    if size != source.size:
        raise ChecksumError(f"{Path(path).as_posix()} has {size:,} bytes, expected {source.size:,}")
    actual = sha256_of(path)
    if actual != source.sha256:
        raise ChecksumError(f"{Path(path).as_posix()} has SHA-256 {actual}, expected {source.sha256}")


def fetch(source: Source, directory: Path) -> Path:
    """Download ``source`` into ``directory`` (skipped when a verified copy is already there)."""
    target = Path(directory) / source.file_name
    if target.is_file():
        verify(target, source)
        log.info("%s is already there and verified", target.as_posix())
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    log.info("downloading %s (%.1f MB)", source.file_name, source.size / 1e6)
    with urllib.request.urlopen(source.url) as response, open(partial, "wb") as out:
        done, next_report = 0, 0.1
        while chunk := response.read(_CHUNK):
            out.write(chunk)
            done += len(chunk)
            if source.size and done / source.size >= next_report:
                log.info("  %3.0f%%", 100 * done / source.size)
                next_report += 0.1
    try:
        verify(partial, source)
    except ChecksumError:
        partial.unlink()
        raise
    partial.replace(target)
    return target


def install_variants(archive: Path, data_dir: Path) -> list[Path]:
    """Copy the four text variants out of the authors' IMDb.zip into the ``data_dir`` layout.

    Files that already exist are left alone (a warning says so if they differ from the archive).
    Returns the files that were written.
    """
    written = []
    with zipfile.ZipFile(archive) as bundle:
        for folder, variant in ARCHIVE_FOLDERS.items():
            for split in SPLITS:
                for label in LABELS:
                    member = f"IMDb/{split}/{folder}/{label}/{label}_{folder}.txt"
                    target = variant_file(data_dir, variant, split, label)
                    if target.is_file():
                        with bundle.open(member) as packed:
                            differs = _sha256(packed) != sha256_of(target)
                        if differs:
                            log.warning("kept %s, which differs from %s in the archive", target.as_posix(), member)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(member) as src, open(target, "wb") as out:
                        shutil.copyfileobj(src, out, _CHUNK)
                    written.append(target)
    return written


def download_variants(data_dir: Path, archive: Path | None = None) -> list[Path]:
    """Add the paper's four text variants to ``data_dir`` from the authors' archive.

    ``archive`` is an IMDb.zip downloaded by hand; without it the archive is downloaded to a temporary
    folder and deleted afterwards. Nothing is downloaded when every file is already in place.
    """
    needed = [
        variant_file(data_dir, variant, split, label)
        for variant in ARCHIVE_FOLDERS.values()
        for split in SPLITS
        for label in LABELS
    ]
    if archive is None and all(path.is_file() for path in needed):
        log.info("all four variants are already in %s", Path(data_dir).as_posix())
        return []
    if archive is not None:
        verify(archive, VARIANTS_ARCHIVE)
        return install_variants(archive, data_dir)
    with tempfile.TemporaryDirectory() as scratch:
        return install_variants(fetch(VARIANTS_ARCHIVE, Path(scratch)), data_dir)
