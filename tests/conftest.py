from __future__ import annotations

from pathlib import Path

import pytest

from preproc_imdb.data import LABELS, SPLITS, variant_file

REPO = Path(__file__).resolve().parents[1]
BUNDLED_DATA = REPO / "data" / "IMDb"


def write_variant(data_dir: Path, variant: str, reviews: dict[tuple[str, str], list[str]]) -> Path:
    """Write one review per line for every (split, label) pair; missing pairs get two short reviews."""
    for split in SPLITS:
        for label in LABELS:
            path = variant_file(data_dir, variant, split, label)
            path.parent.mkdir(parents=True, exist_ok=True)
            lines = reviews.get((split, label), [f"{label} review one", f"another {label} review"])
            path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return data_dir


@pytest.fixture
def tiny_data(tmp_path: Path) -> Path:
    return write_variant(tmp_path / "IMDb", "toy", {})
